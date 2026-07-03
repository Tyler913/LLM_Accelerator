`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed Layer 0 MLP gate/up projection wrapper:
//
//   post_attention_norm[1024] + persistent gate/up Q4 weight/scale
//       -> two q4_gemv_row_1024 cores in parallel
//       -> write gate_out[3072] and up_out[3072]
//
// The large gate/up matrices stay in persistent DDR regions. The runtime QMAP
// packet carries descriptors, the 1024-word activation, output buffers, and
// golden debug vectors for simulation.
module qmap_mlp_gate_up_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 10,
    parameter int NUM_LAYERS       = 28,
    parameter int INPUT_SIZE       = 1024,
    parameter int OUT_FEATURES     = 3072,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL   = 8,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int OUT_WIDTH        = 24,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024,
    parameter int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic                              o_saturation,
    output logic [31 : 0]                     o_rows_done,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_last_gate_row_sum_q26,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_last_up_row_sum_q26,
    output logic signed [31 : 0]              o_last_gate_output_q12_12,
    output logic signed [31 : 0]              o_last_up_output_q12_12,
    output logic [31 : 0]                     o_gate_write_word_count,
    output logic [31 : 0]                     o_up_write_word_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,
    output logic [7 : 0]                      o_state_debug,
    output logic [31 : 0]                     o_row_index_debug,
    output logic [1 : 0]                      o_write_slot_debug,

    output logic                              o_mem_rd_req_valid,
    input  wire logic                         i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_rd_req_addr,
    output logic [15 : 0]                     o_mem_rd_req_len_bytes,

    input  wire logic                         i_mem_rd_rsp_valid,
    output logic                              o_mem_rd_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]  i_mem_rd_rsp_data,
    input  wire logic                         i_mem_rd_rsp_last,

    output logic                              o_mem_wr_req_valid,
    input  wire logic                         i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_wr_req_addr,
    output logic [15 : 0]                     o_mem_wr_req_len_bytes,

    output logic [31 : 0]                     o_mem_wr_data,
    output logic                              o_mem_wr_data_valid,
    input  wire logic                         i_mem_wr_data_ready,
    output logic                              o_mem_wr_data_last,
    input  wire logic                         i_mem_wr_done,
    input  wire logic                         i_mem_wr_error
);

    localparam int SLOT_METADATA      = 0;
    localparam int SLOT_ACTIVATION    = 1;
    localparam int SLOT_GATE_WEIGHT   = 2;
    localparam int SLOT_GATE_SCALE    = 3;
    localparam int SLOT_UP_WEIGHT     = 4;
    localparam int SLOT_UP_SCALE      = 5;
    localparam int SLOT_GATE_OUTPUT   = 6;
    localparam int SLOT_UP_OUTPUT     = 7;
    localparam int SLOT_GATE_EXPECTED = 8;
    localparam int SLOT_UP_EXPECTED   = 9;

    localparam int MEM_DATA_BYTES         = MEM_DATA_WIDTH / 8;
    localparam int ACTIVATION_BYTES       = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int ACTIVATION_CHUNK_BYTES = MAX_READ_BYTES;
    localparam int ACTIVATION_CHUNK_WORDS = ACTIVATION_CHUNK_BYTES / MEM_DATA_BYTES;
    localparam int ACTIVATION_CHUNK_COUNT = ACTIVATION_BYTES / ACTIVATION_CHUNK_BYTES;
    localparam int WEIGHT_ROW_BYTES       = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int WEIGHT_ROW_WORDS       = WEIGHT_ROW_BYTES / MEM_DATA_BYTES;
    localparam int SCALE_ROW_BYTES        = GROUP_COUNT * (SCALE_WIDTH / 8);
    localparam int SCALE_ROW_WORDS        = SCALE_ROW_BYTES / MEM_DATA_BYTES;
    localparam int WEIGHT_BYTES           = OUT_FEATURES * WEIGHT_ROW_BYTES;
    localparam int SCALE_BYTES            = OUT_FEATURES * SCALE_ROW_BYTES;
    localparam int OUTPUT_BYTES           = OUT_FEATURES * MEM_DATA_BYTES;

    localparam logic [15 : 0] ACTIVATION_CHUNK_LEN_BYTES = ACTIVATION_CHUNK_BYTES;
    localparam logic [15 : 0] WEIGHT_ROW_LEN_BYTES       = WEIGHT_ROW_BYTES;
    localparam logic [15 : 0] SCALE_ROW_LEN_BYTES        = SCALE_ROW_BYTES;
    localparam logic [15 : 0] OUTPUT_BYTES_U16           = OUTPUT_BYTES;

    localparam logic signed [ROW_ACC_WIDTH-1 : 0] Q12_12_MAX_EXT = 8388607;
    localparam logic signed [ROW_ACC_WIDTH-1 : 0] Q12_12_MIN_EXT = -8388608;

    typedef enum logic [4 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_ACT_REQ,
        S_ACT_READ,
        S_ROW_SETUP,
        S_GATE_WEIGHT_REQ,
        S_GATE_WEIGHT_READ,
        S_GATE_SCALE_REQ,
        S_GATE_SCALE_READ,
        S_UP_WEIGHT_REQ,
        S_UP_WEIGHT_READ,
        S_UP_SCALE_REQ,
        S_UP_SCALE_READ,
        S_COMPUTE_START,
        S_COMPUTE_WAIT,
        S_STORE_ROW,
        S_NEXT_ROW,
        S_GATE_WRITE_REQ,
        S_GATE_WRITE_DATA,
        S_GATE_WRITE_WAIT,
        S_UP_WRITE_REQ,
        S_UP_WRITE_DATA,
        S_UP_WRITE_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [1 : 0] {
        W_GATE,
        W_UP
    } write_slot_t;

    state_t state;
    write_slot_t active_write_slot;

    logic reader_start;
    logic reader_done;
    logic reader_error;
    logic reader_req_valid;
    logic reader_req_ready;
    logic [ADDR_WIDTH-1 : 0] reader_req_addr;
    logic [15 : 0] reader_req_len_bytes;
    logic reader_rsp_valid;
    logic reader_rsp_ready;
    logic [31 : 0] reader_rsp_data;
    logic reader_rsp_last;
    logic [31 : 0] reader_descriptor_count;
    logic [31 : 0] reader_descriptor_capacity;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_role_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dtype_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_rank_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_flags_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_element_bits_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_group_size_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_scale_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_base_addr_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_nbytes_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux3_flat;

    logic [31 : 0] desc_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_role [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dtype [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_rank [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_flags [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_group_size [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_scale_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux2 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux3 [0 : DESCRIPTOR_SLOTS-1];

    logic [INPUT_SIZE*ACT_WIDTH-1 : 0] activation_flat;
    logic [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] gate_weight_packed;
    logic [GROUP_COUNT*SCALE_WIDTH-1 : 0] gate_scale_flat;
    logic [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] up_weight_packed;
    logic [GROUP_COUNT*SCALE_WIDTH-1 : 0] up_scale_flat;
    logic [31 : 0] gate_output_words [0 : OUT_FEATURES-1];
    logic [31 : 0] up_output_words [0 : OUT_FEATURES-1];

    logic [31 : 0] row_index;
    logic [31 : 0] activation_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] activation_element_index;
    logic [31 : 0] write_word_index;
    logic [ADDR_WIDTH-1 : 0] payload_req_addr;
    logic [15 : 0] payload_req_len_bytes;
    logic expected_payload_last;
    logic payload_protocol_error;
    logic validate_error;

    logic row_start;
    logic gate_row_done;
    logic up_row_done;
    logic signed [ROW_ACC_WIDTH-1 : 0] gate_row_sum_q26;
    logic signed [ROW_ACC_WIDTH-1 : 0] up_row_sum_q26;
    logic signed [ROW_ACC_WIDTH-1 : 0] gate_shifted_q12_12;
    logic signed [ROW_ACC_WIDTH-1 : 0] up_shifted_q12_12;
    logic signed [31 : 0] converted_gate_output_q12_12;
    logic signed [31 : 0] converted_up_output_q12_12;
    logic converted_gate_saturated;
    logic converted_up_saturated;

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]       = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]            = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]           = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_rank[desc_index]            = reader_desc_rank_flat[desc_index*32 +: 32];
            assign desc_flags[desc_index]           = reader_desc_flags_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index]    = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_group_size[desc_index]      = reader_desc_group_size_flat[desc_index*32 +: 32];
            assign desc_scale_tensor_id[desc_index] = reader_desc_scale_tensor_id_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]       = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]          = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]            = reader_desc_dim0_flat[desc_index*32 +: 32];
            assign desc_dim1[desc_index]            = reader_desc_dim1_flat[desc_index*32 +: 32];
            assign desc_aux0[desc_index]            = reader_desc_aux0_flat[desc_index*32 +: 32];
            assign desc_aux1[desc_index]            = reader_desc_aux1_flat[desc_index*32 +: 32];
            assign desc_aux2[desc_index]            = reader_desc_aux2_flat[desc_index*32 +: 32];
            assign desc_aux3[desc_index]            = reader_desc_aux3_flat[desc_index*32 +: 32];
        end
    endgenerate

    qmap_dot64_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS)
    ) qmap_reader (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(reader_start),
        .i_qmap_base_addr(i_qmap_base_addr),
        .o_busy(),
        .o_done(reader_done),
        .o_error(reader_error),
        .o_mem_req_valid(reader_req_valid),
        .i_mem_req_ready(reader_req_ready),
        .o_mem_req_addr(reader_req_addr),
        .o_mem_req_len_bytes(reader_req_len_bytes),
        .i_mem_rsp_valid(reader_rsp_valid),
        .o_mem_rsp_ready(reader_rsp_ready),
        .i_mem_rsp_data(reader_rsp_data),
        .i_mem_rsp_last(reader_rsp_last),
        .o_header_magic(),
        .o_header_version(),
        .o_header_bytes(),
        .o_descriptor_bytes(),
        .o_descriptor_count(reader_descriptor_count),
        .o_descriptor_capacity(reader_descriptor_capacity),
        .o_descriptor_table_addr(),
        .o_payload_base_addr(),
        .o_image_base_addr(),
        .o_image_bytes(),
        .o_desc_tensor_id_flat(reader_desc_tensor_id_flat),
        .o_desc_role_flat(reader_desc_role_flat),
        .o_desc_dtype_flat(reader_desc_dtype_flat),
        .o_desc_rank_flat(reader_desc_rank_flat),
        .o_desc_flags_flat(reader_desc_flags_flat),
        .o_desc_element_bits_flat(reader_desc_element_bits_flat),
        .o_desc_group_size_flat(reader_desc_group_size_flat),
        .o_desc_scale_tensor_id_flat(reader_desc_scale_tensor_id_flat),
        .o_desc_base_addr_flat(reader_desc_base_addr_flat),
        .o_desc_nbytes_flat(reader_desc_nbytes_flat),
        .o_desc_dim0_flat(reader_desc_dim0_flat),
        .o_desc_dim1_flat(reader_desc_dim1_flat),
        .o_desc_dim2_flat(),
        .o_desc_dim3_flat(),
        .o_desc_aux0_flat(reader_desc_aux0_flat),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(reader_desc_aux2_flat),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    q4_gemv_row_1024 #(
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH     (ACT_WIDTH),
        .ACT_FRAC      (ACT_FRAC),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .SCALE_FRAC    (SCALE_FRAC),
        .PARTIAL_WIDTH (PARTIAL_WIDTH),
        .SCALED_WIDTH  (SCALED_WIDTH),
        .ROW_ACC_WIDTH (ROW_ACC_WIDTH)
    ) gate_row_gemv (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(row_start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(gate_weight_packed),
        .i_scale_flat(gate_scale_flat),
        .o_busy(),
        .o_done(gate_row_done),
        .o_row_sum_q26(gate_row_sum_q26)
    );

    q4_gemv_row_1024 #(
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH     (ACT_WIDTH),
        .ACT_FRAC      (ACT_FRAC),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .SCALE_FRAC    (SCALE_FRAC),
        .PARTIAL_WIDTH (PARTIAL_WIDTH),
        .SCALED_WIDTH  (SCALED_WIDTH),
        .ROW_ACC_WIDTH (ROW_ACC_WIDTH)
    ) up_row_gemv (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(row_start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(up_weight_packed),
        .i_scale_flat(up_scale_flat),
        .o_busy(),
        .o_done(up_row_done),
        .o_row_sum_q26(up_row_sum_q26)
    );

    assign reader_start = (state == S_READER_START);
    assign row_start = (state == S_COMPUTE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {3'd0, state};
    assign o_row_index_debug = row_index;
    assign o_write_slot_debug = active_write_slot;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data = i_mem_rd_rsp_data;
    assign reader_rsp_last = i_mem_rd_rsp_last;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        ((state == S_ACT_REQ) ||
         (state == S_GATE_WEIGHT_REQ) ||
         (state == S_GATE_SCALE_REQ) ||
         (state == S_UP_WEIGHT_REQ) ||
         (state == S_UP_SCALE_REQ)) ? 1'b1 :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr : payload_req_addr;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes : payload_req_len_bytes;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        ((state == S_ACT_READ) ||
         (state == S_GATE_WEIGHT_READ) ||
         (state == S_GATE_SCALE_READ) ||
         (state == S_UP_WEIGHT_READ) ||
         (state == S_UP_SCALE_READ)) ? 1'b1 :
        1'b0;

    assign o_mem_wr_req_valid =
        (state == S_GATE_WRITE_REQ) || (state == S_UP_WRITE_REQ);
    assign o_mem_wr_req_addr =
        (state == S_GATE_WRITE_REQ) ? desc_base_addr[SLOT_GATE_OUTPUT] :
        (state == S_UP_WRITE_REQ)   ? desc_base_addr[SLOT_UP_OUTPUT] :
        (active_write_slot == W_GATE) ? desc_base_addr[SLOT_GATE_OUTPUT] :
        desc_base_addr[SLOT_UP_OUTPUT];
    assign o_mem_wr_req_len_bytes = OUTPUT_BYTES_U16;
    assign o_mem_wr_data_valid =
        (state == S_GATE_WRITE_DATA) || (state == S_UP_WRITE_DATA);
    assign o_mem_wr_data =
        (state == S_GATE_WRITE_DATA) ? gate_output_words[write_word_index] :
        up_output_words[write_word_index];
    assign o_mem_wr_data_last =
        ((state == S_GATE_WRITE_DATA) || (state == S_UP_WRITE_DATA)) &&
        (write_word_index == (OUT_FEATURES - 1));

    assign activation_element_index =
        (activation_chunk_index * ACTIVATION_CHUNK_WORDS) + read_word_index;

    always @* begin
        payload_req_addr = desc_base_addr[SLOT_ACTIVATION];
        payload_req_len_bytes = 16'd0;

        case (state)
            S_ACT_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_ACTIVATION] +
                    ({{(ADDR_WIDTH-32){1'b0}}, activation_chunk_index} * ACTIVATION_CHUNK_BYTES);
                payload_req_len_bytes = ACTIVATION_CHUNK_LEN_BYTES;
            end

            S_GATE_WEIGHT_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_GATE_WEIGHT] +
                    ({{(ADDR_WIDTH-32){1'b0}}, row_index} * WEIGHT_ROW_BYTES);
                payload_req_len_bytes = WEIGHT_ROW_LEN_BYTES;
            end

            S_GATE_SCALE_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_GATE_SCALE] +
                    ({{(ADDR_WIDTH-32){1'b0}}, row_index} * SCALE_ROW_BYTES);
                payload_req_len_bytes = SCALE_ROW_LEN_BYTES;
            end

            S_UP_WEIGHT_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_UP_WEIGHT] +
                    ({{(ADDR_WIDTH-32){1'b0}}, row_index} * WEIGHT_ROW_BYTES);
                payload_req_len_bytes = WEIGHT_ROW_LEN_BYTES;
            end

            S_UP_SCALE_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_UP_SCALE] +
                    ({{(ADDR_WIDTH-32){1'b0}}, row_index} * SCALE_ROW_BYTES);
                payload_req_len_bytes = SCALE_ROW_LEN_BYTES;
            end

            default: begin
                payload_req_addr = desc_base_addr[SLOT_ACTIVATION];
                payload_req_len_bytes = 16'd0;
            end
        endcase
    end

    always @* begin
        expected_payload_last = 1'b0;
        case (state)
            S_ACT_READ: begin
                expected_payload_last = (read_word_index == (ACTIVATION_CHUNK_WORDS - 1));
            end
            S_GATE_WEIGHT_READ,
            S_UP_WEIGHT_READ: begin
                expected_payload_last = (read_word_index == (WEIGHT_ROW_WORDS - 1));
            end
            S_GATE_SCALE_READ,
            S_UP_SCALE_READ: begin
                expected_payload_last = (read_word_index == (SCALE_ROW_WORDS - 1));
            end
            default: begin
                expected_payload_last = 1'b0;
            end
        endcase
    end

    assign payload_protocol_error =
        ((state == S_ACT_READ) ||
         (state == S_GATE_WEIGHT_READ) ||
         (state == S_GATE_SCALE_READ) ||
         (state == S_UP_WEIGHT_READ) ||
         (state == S_UP_SCALE_READ)) &&
        i_mem_rd_rsp_valid &&
        (i_mem_rd_rsp_last != expected_payload_last);

    always @* begin
        gate_shifted_q12_12 = gate_row_sum_q26 >>> SCALE_FRAC;
        up_shifted_q12_12 = up_row_sum_q26 >>> SCALE_FRAC;
        converted_gate_saturated = 1'b0;
        converted_up_saturated = 1'b0;

        if (gate_shifted_q12_12 > Q12_12_MAX_EXT) begin
            converted_gate_output_q12_12 = 32'sd8388607;
            converted_gate_saturated = 1'b1;
        end
        else if (gate_shifted_q12_12 < Q12_12_MIN_EXT) begin
            converted_gate_output_q12_12 = -32'sd8388608;
            converted_gate_saturated = 1'b1;
        end
        else begin
            converted_gate_output_q12_12 = $signed(gate_shifted_q12_12[31 : 0]);
        end

        if (up_shifted_q12_12 > Q12_12_MAX_EXT) begin
            converted_up_output_q12_12 = 32'sd8388607;
            converted_up_saturated = 1'b1;
        end
        else if (up_shifted_q12_12 < Q12_12_MIN_EXT) begin
            converted_up_output_q12_12 = -32'sd8388608;
            converted_up_saturated = 1'b1;
        end
        else begin
            converted_up_output_q12_12 = $signed(up_shifted_q12_12[31 : 0]);
        end
    end

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count < `QMAP_MLP_GATE_UP_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity < `QMAP_MLP_GATE_UP_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_MLP_GATE_UP_METADATA) ||
            (desc_tensor_id[SLOT_ACTIVATION] != `QMAP_TENSOR_ID_MLP_GATE_UP_ACTIVATION) ||
            (desc_tensor_id[SLOT_GATE_WEIGHT] != `QMAP_TENSOR_ID_MLP_GATE_WEIGHT) ||
            (desc_tensor_id[SLOT_GATE_SCALE] != `QMAP_TENSOR_ID_MLP_GATE_SCALE) ||
            (desc_tensor_id[SLOT_UP_WEIGHT] != `QMAP_TENSOR_ID_MLP_UP_WEIGHT) ||
            (desc_tensor_id[SLOT_UP_SCALE] != `QMAP_TENSOR_ID_MLP_UP_SCALE) ||
            (desc_tensor_id[SLOT_GATE_OUTPUT] != `QMAP_TENSOR_ID_MLP_GATE_OUTPUT) ||
            (desc_tensor_id[SLOT_UP_OUTPUT] != `QMAP_TENSOR_ID_MLP_UP_OUTPUT) ||
            (desc_tensor_id[SLOT_GATE_EXPECTED] != `QMAP_TENSOR_ID_MLP_GATE_EXPECTED) ||
            (desc_tensor_id[SLOT_UP_EXPECTED] != `QMAP_TENSOR_ID_MLP_UP_EXPECTED) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_ACTIVATION] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_GATE_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
            (desc_role[SLOT_GATE_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
            (desc_role[SLOT_UP_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
            (desc_role[SLOT_UP_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
            (desc_role[SLOT_GATE_OUTPUT] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_UP_OUTPUT] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_GATE_EXPECTED] != `QMAP_ROLE_EXPECTED) ||
            (desc_role[SLOT_UP_EXPECTED] != `QMAP_ROLE_EXPECTED) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_ACTIVATION] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_GATE_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
            (desc_dtype[SLOT_GATE_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
            (desc_dtype[SLOT_UP_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
            (desc_dtype[SLOT_UP_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
            (desc_dtype[SLOT_GATE_OUTPUT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_UP_OUTPUT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_GATE_EXPECTED] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_UP_EXPECTED] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_ACTIVATION] != 32'd1) ||
            (desc_rank[SLOT_GATE_WEIGHT] != 32'd2) ||
            (desc_rank[SLOT_GATE_SCALE] != 32'd2) ||
            (desc_rank[SLOT_UP_WEIGHT] != 32'd2) ||
            (desc_rank[SLOT_UP_SCALE] != 32'd2) ||
            (desc_rank[SLOT_GATE_OUTPUT] != 32'd1) ||
            (desc_rank[SLOT_UP_OUTPUT] != 32'd1) ||
            (desc_rank[SLOT_GATE_EXPECTED] != 32'd1) ||
            (desc_rank[SLOT_UP_EXPECTED] != 32'd1) ||
            (desc_element_bits[SLOT_METADATA] != 32'd32) ||
            (desc_element_bits[SLOT_ACTIVATION] != ACT_WIDTH) ||
            (desc_element_bits[SLOT_GATE_WEIGHT] != WEIGHT_WIDTH) ||
            (desc_element_bits[SLOT_GATE_SCALE] != SCALE_WIDTH) ||
            (desc_element_bits[SLOT_UP_WEIGHT] != WEIGHT_WIDTH) ||
            (desc_element_bits[SLOT_UP_SCALE] != SCALE_WIDTH) ||
            (desc_element_bits[SLOT_GATE_OUTPUT] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_UP_OUTPUT] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_GATE_EXPECTED] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_UP_EXPECTED] != OUT_WIDTH) ||
            (desc_group_size[SLOT_ACTIVATION] != 32'd0) ||
            (desc_group_size[SLOT_GATE_WEIGHT] != GROUP_SIZE) ||
            (desc_group_size[SLOT_GATE_SCALE] != 32'd0) ||
            (desc_group_size[SLOT_UP_WEIGHT] != GROUP_SIZE) ||
            (desc_group_size[SLOT_UP_SCALE] != 32'd0) ||
            (desc_scale_tensor_id[SLOT_ACTIVATION] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_GATE_WEIGHT] != `QMAP_TENSOR_ID_MLP_GATE_SCALE) ||
            (desc_scale_tensor_id[SLOT_GATE_SCALE] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_UP_WEIGHT] != `QMAP_TENSOR_ID_MLP_UP_SCALE) ||
            (desc_scale_tensor_id[SLOT_UP_SCALE] != `QMAP_NO_TENSOR_ID) ||
            (desc_dim0[SLOT_ACTIVATION] != INPUT_SIZE) ||
            (desc_dim1[SLOT_ACTIVATION] != 32'd0) ||
            (desc_dim0[SLOT_GATE_WEIGHT] != OUT_FEATURES) ||
            (desc_dim1[SLOT_GATE_WEIGHT] != INPUT_SIZE) ||
            (desc_dim0[SLOT_GATE_SCALE] != OUT_FEATURES) ||
            (desc_dim1[SLOT_GATE_SCALE] != GROUP_COUNT) ||
            (desc_dim0[SLOT_UP_WEIGHT] != OUT_FEATURES) ||
            (desc_dim1[SLOT_UP_WEIGHT] != INPUT_SIZE) ||
            (desc_dim0[SLOT_UP_SCALE] != OUT_FEATURES) ||
            (desc_dim1[SLOT_UP_SCALE] != GROUP_COUNT) ||
            (desc_dim0[SLOT_GATE_OUTPUT] != OUT_FEATURES) ||
            (desc_dim0[SLOT_UP_OUTPUT] != OUT_FEATURES) ||
            (desc_dim0[SLOT_GATE_EXPECTED] != OUT_FEATURES) ||
            (desc_dim0[SLOT_UP_EXPECTED] != OUT_FEATURES) ||
            (desc_nbytes[SLOT_ACTIVATION] != ACTIVATION_BYTES) ||
            (desc_nbytes[SLOT_GATE_WEIGHT] != WEIGHT_BYTES) ||
            (desc_nbytes[SLOT_GATE_SCALE] != SCALE_BYTES) ||
            (desc_nbytes[SLOT_UP_WEIGHT] != WEIGHT_BYTES) ||
            (desc_nbytes[SLOT_UP_SCALE] != SCALE_BYTES) ||
            (desc_nbytes[SLOT_GATE_OUTPUT] != OUTPUT_BYTES) ||
            (desc_nbytes[SLOT_UP_OUTPUT] != OUTPUT_BYTES) ||
            (desc_nbytes[SLOT_GATE_EXPECTED] != OUTPUT_BYTES) ||
            (desc_nbytes[SLOT_UP_EXPECTED] != OUTPUT_BYTES) ||
            (desc_base_addr[SLOT_GATE_WEIGHT] != `QMAP_MLP_GATE_WEIGHT_BASE_ADDR) ||
            (desc_base_addr[SLOT_GATE_SCALE] != `QMAP_MLP_GATE_SCALE_BASE_ADDR) ||
            (desc_base_addr[SLOT_UP_WEIGHT] != `QMAP_MLP_UP_WEIGHT_BASE_ADDR) ||
            (desc_base_addr[SLOT_UP_SCALE] != `QMAP_MLP_UP_SCALE_BASE_ADDR) ||
            ((desc_flags[SLOT_ACTIVATION] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GATE_WEIGHT] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GATE_WEIGHT] & `QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN) == 32'd0) ||
            ((desc_flags[SLOT_GATE_SCALE] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_UP_WEIGHT] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_UP_WEIGHT] & `QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN) == 32'd0) ||
            ((desc_flags[SLOT_UP_SCALE] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GATE_OUTPUT] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_UP_OUTPUT] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GATE_EXPECTED] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_UP_EXPECTED] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            (desc_aux0[SLOT_METADATA] != `QMAP_STAGE_ID_MLP_GATE_UP) ||
            (desc_aux1[SLOT_METADATA] >= NUM_LAYERS) ||
            (desc_aux2[SLOT_METADATA] != OUT_FEATURES) ||
            (desc_aux3[SLOT_METADATA] != INPUT_SIZE) ||
            (desc_aux0[SLOT_ACTIVATION] != `QMAP_STAGE_ID_MLP_GATE_UP) ||
            (desc_aux1[SLOT_ACTIVATION] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux0[SLOT_GATE_WEIGHT] != `QMAP_MATRIX_ID_GATE_PROJ) ||
            (desc_aux0[SLOT_GATE_SCALE] != `QMAP_MATRIX_ID_GATE_PROJ) ||
            (desc_aux0[SLOT_UP_WEIGHT] != `QMAP_MATRIX_ID_UP_PROJ) ||
            (desc_aux0[SLOT_UP_SCALE] != `QMAP_MATRIX_ID_UP_PROJ) ||
            (desc_aux0[SLOT_GATE_OUTPUT] != `QMAP_MATRIX_ID_GATE_PROJ) ||
            (desc_aux0[SLOT_UP_OUTPUT] != `QMAP_MATRIX_ID_UP_PROJ) ||
            (desc_aux0[SLOT_GATE_EXPECTED] != `QMAP_MATRIX_ID_GATE_PROJ) ||
            (desc_aux0[SLOT_UP_EXPECTED] != `QMAP_MATRIX_ID_UP_PROJ) ||
            (desc_aux1[SLOT_GATE_WEIGHT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_GATE_SCALE] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_UP_WEIGHT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_UP_SCALE] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_GATE_OUTPUT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_UP_OUTPUT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux3[SLOT_GATE_WEIGHT] != OUT_FEATURES) ||
            (desc_aux3[SLOT_GATE_SCALE] != OUT_FEATURES) ||
            (desc_aux3[SLOT_UP_WEIGHT] != OUT_FEATURES) ||
            (desc_aux3[SLOT_UP_SCALE] != OUT_FEATURES) ||
            (desc_aux3[SLOT_GATE_OUTPUT] != OUT_FEATURES) ||
            (desc_aux3[SLOT_UP_OUTPUT] != OUT_FEATURES)) begin
            validate_error = 1'b1;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_write_slot <= W_GATE;
            row_index <= 32'd0;
            activation_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            write_word_index <= 32'd0;
            activation_flat <= '0;
            gate_weight_packed <= '0;
            gate_scale_flat <= '0;
            up_weight_packed <= '0;
            up_scale_flat <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_rows_done <= 32'd0;
            o_last_gate_row_sum_q26 <= '0;
            o_last_up_row_sum_q26 <= '0;
            o_last_gate_output_q12_12 <= 32'sd0;
            o_last_up_output_q12_12 <= 32'sd0;
            o_gate_write_word_count <= 32'd0;
            o_up_write_word_count <= 32'd0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count <= 32'd0;
            o_mem_write_req_count <= 32'd0;
            o_mem_write_word_count <= 32'd0;
        end
        else begin
            o_done <= 1'b0;

            if (o_mem_rd_req_valid && i_mem_rd_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end
            if (i_mem_rd_rsp_valid && o_mem_rd_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end
            if (o_mem_wr_req_valid && i_mem_wr_req_ready) begin
                o_mem_write_req_count <= o_mem_write_req_count + 1'b1;
            end
            if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                o_mem_write_word_count <= o_mem_write_word_count + 1'b1;
                if (active_write_slot == W_GATE) begin
                    o_gate_write_word_count <= o_gate_write_word_count + 1'b1;
                end
                else begin
                    o_up_write_word_count <= o_up_write_word_count + 1'b1;
                end
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        active_write_slot <= W_GATE;
                        row_index <= 32'd0;
                        activation_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        write_word_index <= 32'd0;
                        activation_flat <= '0;
                        gate_weight_packed <= '0;
                        gate_scale_flat <= '0;
                        up_weight_packed <= '0;
                        up_scale_flat <= '0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_rows_done <= 32'd0;
                        o_last_gate_row_sum_q26 <= '0;
                        o_last_up_row_sum_q26 <= '0;
                        o_last_gate_output_q12_12 <= 32'sd0;
                        o_last_up_output_q12_12 <= 32'sd0;
                        o_gate_write_word_count <= 32'd0;
                        o_up_write_word_count <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state <= S_READER_START;
                    end
                end

                S_READER_START: begin
                    state <= S_READER_WAIT;
                end

                S_READER_WAIT: begin
                    if (reader_done) begin
                        if (reader_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_VALIDATE;
                        end
                    end
                end

                S_VALIDATE: begin
                    if (validate_error) begin
                        o_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else begin
                        activation_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        state <= S_ACT_REQ;
                    end
                end

                S_ACT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_ACT_READ;
                    end
                end

                S_ACT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            activation_flat[activation_element_index*ACT_WIDTH +: ACT_WIDTH] <=
                                i_mem_rd_rsp_data[ACT_WIDTH-1 : 0];
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                if (activation_chunk_index == (ACTIVATION_CHUNK_COUNT - 1)) begin
                                    row_index <= 32'd0;
                                    state <= S_ROW_SETUP;
                                end
                                else begin
                                    activation_chunk_index <= activation_chunk_index + 1'b1;
                                    state <= S_ACT_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_ROW_SETUP: begin
                    read_word_index <= 32'd0;
                    gate_weight_packed <= '0;
                    gate_scale_flat <= '0;
                    up_weight_packed <= '0;
                    up_scale_flat <= '0;
                    state <= S_GATE_WEIGHT_REQ;
                end

                S_GATE_WEIGHT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_GATE_WEIGHT_READ;
                    end
                end

                S_GATE_WEIGHT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            gate_weight_packed[read_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <=
                                i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state <= S_GATE_SCALE_REQ;
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_GATE_SCALE_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_GATE_SCALE_READ;
                    end
                end

                S_GATE_SCALE_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            gate_scale_flat[read_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <=
                                i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state <= S_UP_WEIGHT_REQ;
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_UP_WEIGHT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_UP_WEIGHT_READ;
                    end
                end

                S_UP_WEIGHT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            up_weight_packed[read_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <=
                                i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state <= S_UP_SCALE_REQ;
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_UP_SCALE_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_UP_SCALE_READ;
                    end
                end

                S_UP_SCALE_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            up_scale_flat[read_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <=
                                i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state <= S_COMPUTE_START;
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_COMPUTE_START: begin
                    state <= S_COMPUTE_WAIT;
                end

                S_COMPUTE_WAIT: begin
                    if (gate_row_done && up_row_done) begin
                        state <= S_STORE_ROW;
                    end
                end

                S_STORE_ROW: begin
                    gate_output_words[row_index] <= converted_gate_output_q12_12;
                    up_output_words[row_index] <= converted_up_output_q12_12;
                    o_last_gate_row_sum_q26 <= gate_row_sum_q26;
                    o_last_up_row_sum_q26 <= up_row_sum_q26;
                    o_last_gate_output_q12_12 <= converted_gate_output_q12_12;
                    o_last_up_output_q12_12 <= converted_up_output_q12_12;
                    o_saturation <=
                        o_saturation || converted_gate_saturated || converted_up_saturated;
                    o_rows_done <= o_rows_done + 1'b1;
                    state <= S_NEXT_ROW;
                end

                S_NEXT_ROW: begin
                    if ((row_index + 1'b1) < OUT_FEATURES) begin
                        row_index <= row_index + 1'b1;
                        state <= S_ROW_SETUP;
                    end
                    else begin
                        active_write_slot <= W_GATE;
                        write_word_index <= 32'd0;
                        state <= S_GATE_WRITE_REQ;
                    end
                end

                S_GATE_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        write_word_index <= 32'd0;
                        state <= S_GATE_WRITE_DATA;
                    end
                end

                S_GATE_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        if (o_mem_wr_data_last) begin
                            state <= S_GATE_WRITE_WAIT;
                        end
                        else begin
                            write_word_index <= write_word_index + 1'b1;
                        end
                    end
                end

                S_GATE_WRITE_WAIT: begin
                    if (i_mem_wr_done || i_mem_wr_error) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            active_write_slot <= W_UP;
                            write_word_index <= 32'd0;
                            state <= S_UP_WRITE_REQ;
                        end
                    end
                end

                S_UP_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        write_word_index <= 32'd0;
                        state <= S_UP_WRITE_DATA;
                    end
                end

                S_UP_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        if (o_mem_wr_data_last) begin
                            state <= S_UP_WRITE_WAIT;
                        end
                        else begin
                            write_word_index <= write_word_index + 1'b1;
                        end
                    end
                end

                S_UP_WRITE_WAIT: begin
                    if (i_mem_wr_done || i_mem_wr_error) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                        end
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    o_error <= 1'b1;
                    state <= S_DONE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
