`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed Layer 0 MLP SiLU/multiply wrapper:
//
//   gate[3072] + up[3072] + sigmoid LUT -> silu(gate) * up -> hidden[3072]
//
// The runtime QMAP packet carries descriptors, gate/up inputs, the sigmoid LUT,
// one writable hidden output buffer, and a golden debug vector for simulation.
module qmap_mlp_silu_mul_compute_path #(
    parameter int ADDR_WIDTH            = 64,
    parameter int DESCRIPTOR_SLOTS      = 6,
    parameter int NUM_LAYERS            = 28,
    parameter int FEATURES              = 3072,
    parameter int IN_WIDTH              = 24,
    parameter int IN_FRAC               = 12,
    parameter int SIGMOID_WIDTH         = 16,
    parameter int SIGMOID_FRAC          = 16,
    parameter int SIGMOID_LUT_INDEX_FRAC = 6,
    parameter int SIGMOID_LUT_MIN_INT   = -8,
    parameter int SIGMOID_LUT_MAX_INT   = 8,
    parameter int OUT_WIDTH             = 24,
    parameter int OUT_FRAC              = 12,
    parameter int MEM_DATA_WIDTH        = 32,
    parameter int MAX_READ_BYTES        = 1024
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
    output logic [31 : 0]                     o_stage_input_count,
    output logic [31 : 0]                     o_stage_output_count,
    output logic [31 : 0]                     o_stage_cycle_count,
    output logic [31 : 0]                     o_output_write_word_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,
    output logic [7 : 0]                      o_state_debug,
    output logic [31 : 0]                     o_read_slot_debug,

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

    localparam int SLOT_METADATA = 0;
    localparam int SLOT_GATE     = 1;
    localparam int SLOT_UP       = 2;
    localparam int SLOT_LUT      = 3;
    localparam int SLOT_HIDDEN   = 4;
    localparam int SLOT_EXPECTED = 5;

    localparam int MEM_DATA_BYTES         = MEM_DATA_WIDTH / 8;
    localparam int VECTOR_BYTES           = FEATURES * MEM_DATA_BYTES;
    localparam int VECTOR_CHUNK_BYTES     = MAX_READ_BYTES;
    localparam int VECTOR_CHUNK_WORDS     = VECTOR_CHUNK_BYTES / MEM_DATA_BYTES;
    localparam int VECTOR_CHUNK_COUNT     = VECTOR_BYTES / VECTOR_CHUNK_BYTES;
    localparam int SIGMOID_LUT_SIZE       = ((SIGMOID_LUT_MAX_INT - SIGMOID_LUT_MIN_INT) << SIGMOID_LUT_INDEX_FRAC) + 1;
    localparam int LUT_BYTES              = SIGMOID_LUT_SIZE * MEM_DATA_BYTES;
    localparam int LUT_CHUNK_COUNT        = (LUT_BYTES + MAX_READ_BYTES - 1) / MAX_READ_BYTES;
    localparam int LUT_LAST_CHUNK_BYTES   = LUT_BYTES - ((LUT_CHUNK_COUNT - 1) * MAX_READ_BYTES);
    localparam int LUT_LAST_CHUNK_WORDS   = LUT_LAST_CHUNK_BYTES / MEM_DATA_BYTES;
    localparam int METADATA_WORDS         = 13;
    localparam int METADATA_BYTES         = METADATA_WORDS * MEM_DATA_BYTES;
    localparam int ROW_INDEX_W            = (FEATURES <= 1) ? 1 : $clog2(FEATURES);

    localparam logic [15 : 0] VECTOR_CHUNK_LEN_BYTES = VECTOR_CHUNK_BYTES;
    localparam logic [15 : 0] VECTOR_BYTES_U16       = VECTOR_BYTES;

    typedef enum logic [4 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_GATE_REQ,
        S_GATE_READ,
        S_UP_REQ,
        S_UP_READ,
        S_LUT_REQ,
        S_LUT_READ,
        S_STAGE_START,
        S_STAGE_RUN,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [1 : 0] {
        R_GATE,
        R_UP,
        R_LUT
    } read_slot_t;

    state_t state;
    read_slot_t active_read_slot;

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

    logic signed [31 : 0] gate_words [0 : FEATURES-1];
    logic signed [31 : 0] up_words [0 : FEATURES-1];
    logic signed [31 : 0] hidden_words [0 : FEATURES-1];
    logic [SIGMOID_LUT_SIZE*SIGMOID_WIDTH-1 : 0] sigmoid_lut_flat;

    logic [31 : 0] read_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] read_element_index;
    logic [31 : 0] active_read_words;
    logic [ADDR_WIDTH-1 : 0] payload_req_addr;
    logic [15 : 0] payload_req_len_bytes;
    logic expected_payload_last;
    logic payload_protocol_error;
    logic validate_error;

    logic stage_start;
    logic stage_in_valid;
    logic stage_in_ready;
    logic [ROW_INDEX_W-1 : 0] stage_in_index;
    logic signed [IN_WIDTH-1 : 0] stage_gate_data;
    logic signed [IN_WIDTH-1 : 0] stage_up_data;
    logic stage_in_last;
    logic stage_busy;
    logic stage_done;
    logic stage_error;
    logic stage_saturation;
    logic stage_out_valid;
    logic stage_out_ready;
    logic [ROW_INDEX_W-1 : 0] stage_out_index;
    logic signed [OUT_WIDTH-1 : 0] stage_hidden_data;
    logic stage_out_last;
    logic [31 : 0] stage_core_input_count;
    logic [31 : 0] stage_core_output_count;
    logic [31 : 0] stage_input_index;
    logic [31 : 0] stage_output_index;
    logic [31 : 0] stage_feed_index;
    logic signed [31 : 0] stage_hidden_word_ext;
    logic stage_output_fire;
    logic [31 : 0] write_word_index;

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

    mlp_silu_mul_stage #(
        .FEATURES(FEATURES),
        .IN_WIDTH(IN_WIDTH),
        .IN_FRAC(IN_FRAC),
        .SIGMOID_WIDTH(SIGMOID_WIDTH),
        .SIGMOID_FRAC(SIGMOID_FRAC),
        .SIGMOID_LUT_INDEX_FRAC(SIGMOID_LUT_INDEX_FRAC),
        .SIGMOID_LUT_MIN_INT(SIGMOID_LUT_MIN_INT),
        .SIGMOID_LUT_MAX_INT(SIGMOID_LUT_MAX_INT),
        .SIGMOID_LUT_SIZE(SIGMOID_LUT_SIZE),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC(OUT_FRAC),
        .ROW_INDEX_W(ROW_INDEX_W)
    ) silu_mul_stage (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(stage_start),
        .i_sigmoid_lut_flat(sigmoid_lut_flat),
        .i_in_valid(stage_in_valid),
        .o_in_ready(stage_in_ready),
        .i_in_index(stage_in_index),
        .i_gate_data(stage_gate_data),
        .i_up_data(stage_up_data),
        .i_in_last(stage_in_last),
        .o_busy(stage_busy),
        .o_done(stage_done),
        .o_error(stage_error),
        .o_saturation(stage_saturation),
        .o_out_valid(stage_out_valid),
        .i_out_ready(stage_out_ready),
        .o_out_index(stage_out_index),
        .o_hidden_data(stage_hidden_data),
        .o_out_last(stage_out_last),
        .o_input_count(stage_core_input_count),
        .o_output_count(stage_core_output_count)
    );

    assign reader_start = (state == S_READER_START);
    assign stage_start = (state == S_STAGE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {3'd0, state};
    assign o_read_slot_debug = {30'd0, active_read_slot};
    assign o_stage_input_count = stage_core_input_count;
    assign o_stage_output_count = stage_core_output_count;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data = i_mem_rd_rsp_data;
    assign reader_rsp_last = i_mem_rd_rsp_last;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        ((state == S_GATE_REQ) || (state == S_UP_REQ) || (state == S_LUT_REQ)) ? 1'b1 :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr : payload_req_addr;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes : payload_req_len_bytes;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        ((state == S_GATE_READ) || (state == S_UP_READ) || (state == S_LUT_READ)) ? 1'b1 :
        1'b0;

    assign o_mem_wr_req_valid = (state == S_WRITE_REQ);
    assign o_mem_wr_req_addr = desc_base_addr[SLOT_HIDDEN];
    assign o_mem_wr_req_len_bytes = VECTOR_BYTES_U16;
    assign o_mem_wr_data_valid = (state == S_WRITE_DATA);
    assign o_mem_wr_data = hidden_words[write_word_index];
    assign o_mem_wr_data_last = (state == S_WRITE_DATA) && (write_word_index == (FEATURES - 1));

    assign stage_feed_index = (stage_input_index < FEATURES) ? stage_input_index : 32'd0;
    assign stage_in_valid = (state == S_STAGE_RUN) && (stage_input_index < FEATURES);
    assign stage_in_index = stage_input_index[ROW_INDEX_W-1 : 0];
    assign stage_gate_data = gate_words[stage_feed_index][IN_WIDTH-1 : 0];
    assign stage_up_data = up_words[stage_feed_index][IN_WIDTH-1 : 0];
    assign stage_in_last = (stage_input_index == (FEATURES - 1));
    assign stage_out_ready = (state == S_STAGE_RUN);
    assign stage_output_fire = stage_out_valid && stage_out_ready;
    assign stage_hidden_word_ext = {{(32-OUT_WIDTH){stage_hidden_data[OUT_WIDTH-1]}}, stage_hidden_data};

    always @* begin
        active_read_words = VECTOR_CHUNK_WORDS;
        if (((state == S_LUT_REQ) || (state == S_LUT_READ)) &&
            (read_chunk_index == (LUT_CHUNK_COUNT - 1))) begin
            active_read_words = LUT_LAST_CHUNK_WORDS;
        end
    end

    assign read_element_index = (read_chunk_index * VECTOR_CHUNK_WORDS) + read_word_index;
    assign expected_payload_last = (read_word_index == (active_read_words - 1));
    assign payload_protocol_error = (i_mem_rd_rsp_last != expected_payload_last);

    always @* begin
        payload_req_addr = desc_base_addr[SLOT_GATE];
        payload_req_len_bytes = VECTOR_CHUNK_LEN_BYTES;

        case (state)
            S_GATE_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_GATE] +
                    ({{(ADDR_WIDTH-32){1'b0}}, read_chunk_index} * MAX_READ_BYTES);
                payload_req_len_bytes = VECTOR_CHUNK_LEN_BYTES;
            end

            S_UP_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_UP] +
                    ({{(ADDR_WIDTH-32){1'b0}}, read_chunk_index} * MAX_READ_BYTES);
                payload_req_len_bytes = VECTOR_CHUNK_LEN_BYTES;
            end

            S_LUT_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_LUT] +
                    ({{(ADDR_WIDTH-32){1'b0}}, read_chunk_index} * MAX_READ_BYTES);
                payload_req_len_bytes = active_read_words[15 : 0] * MEM_DATA_BYTES;
            end

            default: begin
                payload_req_addr = desc_base_addr[SLOT_GATE];
                payload_req_len_bytes = 16'd0;
            end
        endcase
    end

    always @* begin
        validate_error = 1'b0;

        if ((reader_descriptor_count != `QMAP_MLP_SILU_MUL_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity != `QMAP_MLP_SILU_MUL_DESCRIPTOR_CAPACITY) ||
            (DESCRIPTOR_SLOTS != `QMAP_MLP_SILU_MUL_DESCRIPTOR_COUNT) ||
            (SIGMOID_LUT_SIZE != `QMAP_MLP_SILU_MUL_LUT_ENTRIES) ||
            (VECTOR_BYTES != `QMAP_MLP_SILU_MUL_VECTOR_BYTES) ||
            (LUT_BYTES != `QMAP_MLP_SILU_MUL_LUT_BYTES) ||
            (MAX_READ_BYTES != `QMAP_MLP_SILU_MUL_MAX_READ_BYTES) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_MLP_SILU_MUL_METADATA) ||
            (desc_tensor_id[SLOT_GATE] != `QMAP_TENSOR_ID_MLP_SILU_GATE) ||
            (desc_tensor_id[SLOT_UP] != `QMAP_TENSOR_ID_MLP_SILU_UP) ||
            (desc_tensor_id[SLOT_LUT] != `QMAP_TENSOR_ID_MLP_SILU_LUT) ||
            (desc_tensor_id[SLOT_HIDDEN] != `QMAP_TENSOR_ID_MLP_SILU_HIDDEN) ||
            (desc_tensor_id[SLOT_EXPECTED] != `QMAP_TENSOR_ID_MLP_SILU_EXPECTED) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_GATE] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_UP] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_LUT] != `QMAP_ROLE_PARAMETER) ||
            (desc_role[SLOT_HIDDEN] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_EXPECTED] != `QMAP_ROLE_EXPECTED) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_GATE] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_UP] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_LUT] != `QMAP_DTYPE_U16_Q0_16) ||
            (desc_dtype[SLOT_HIDDEN] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_EXPECTED] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_GATE] != 32'd1) ||
            (desc_rank[SLOT_UP] != 32'd1) ||
            (desc_rank[SLOT_LUT] != 32'd1) ||
            (desc_rank[SLOT_HIDDEN] != 32'd1) ||
            (desc_rank[SLOT_EXPECTED] != 32'd1) ||
            (desc_element_bits[SLOT_METADATA] != 32'd32) ||
            (desc_element_bits[SLOT_GATE] != IN_WIDTH) ||
            (desc_element_bits[SLOT_UP] != IN_WIDTH) ||
            (desc_element_bits[SLOT_LUT] != SIGMOID_WIDTH) ||
            (desc_element_bits[SLOT_HIDDEN] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_EXPECTED] != OUT_WIDTH) ||
            (desc_group_size[SLOT_METADATA] != 32'd0) ||
            (desc_group_size[SLOT_GATE] != 32'd0) ||
            (desc_group_size[SLOT_UP] != 32'd0) ||
            (desc_group_size[SLOT_LUT] != 32'd0) ||
            (desc_group_size[SLOT_HIDDEN] != 32'd0) ||
            (desc_group_size[SLOT_EXPECTED] != 32'd0) ||
            (desc_scale_tensor_id[SLOT_METADATA] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_GATE] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_UP] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_LUT] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_HIDDEN] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_EXPECTED] != `QMAP_NO_TENSOR_ID) ||
            (desc_dim0[SLOT_METADATA] != METADATA_WORDS) ||
            (desc_dim0[SLOT_GATE] != FEATURES) ||
            (desc_dim0[SLOT_UP] != FEATURES) ||
            (desc_dim0[SLOT_LUT] != SIGMOID_LUT_SIZE) ||
            (desc_dim0[SLOT_HIDDEN] != FEATURES) ||
            (desc_dim0[SLOT_EXPECTED] != FEATURES) ||
            (desc_dim1[SLOT_METADATA] != 32'd0) ||
            (desc_dim1[SLOT_GATE] != 32'd0) ||
            (desc_dim1[SLOT_UP] != 32'd0) ||
            (desc_dim1[SLOT_LUT] != 32'd0) ||
            (desc_dim1[SLOT_HIDDEN] != 32'd0) ||
            (desc_dim1[SLOT_EXPECTED] != 32'd0) ||
            (desc_nbytes[SLOT_METADATA] != METADATA_BYTES) ||
            (desc_nbytes[SLOT_GATE] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_UP] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_LUT] != LUT_BYTES) ||
            (desc_nbytes[SLOT_HIDDEN] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_EXPECTED] != VECTOR_BYTES) ||
            ((desc_flags[SLOT_METADATA] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GATE] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_UP] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_LUT] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_HIDDEN] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_EXPECTED] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            (desc_aux0[SLOT_METADATA] != `QMAP_STAGE_ID_MLP_SILU_MUL) ||
            (desc_aux1[SLOT_METADATA] >= NUM_LAYERS) ||
            (desc_aux2[SLOT_METADATA] != FEATURES) ||
            (desc_aux3[SLOT_METADATA] != SIGMOID_LUT_SIZE) ||
            (desc_aux0[SLOT_GATE] != `QMAP_MATRIX_ID_GATE_PROJ) ||
            (desc_aux0[SLOT_UP] != `QMAP_MATRIX_ID_UP_PROJ) ||
            (desc_aux0[SLOT_LUT] != `QMAP_STAGE_ID_MLP_SILU_MUL) ||
            (desc_aux0[SLOT_HIDDEN] != `QMAP_STAGE_ID_MLP_SILU_MUL) ||
            (desc_aux0[SLOT_EXPECTED] != `QMAP_STAGE_ID_MLP_SILU_MUL) ||
            (desc_aux1[SLOT_GATE] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_UP] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_LUT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_HIDDEN] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_EXPECTED] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux3[SLOT_GATE] != FEATURES) ||
            (desc_aux3[SLOT_UP] != FEATURES) ||
            (desc_aux3[SLOT_LUT] != SIGMOID_LUT_SIZE) ||
            (desc_aux3[SLOT_HIDDEN] != SIGMOID_LUT_SIZE) ||
            (desc_aux3[SLOT_EXPECTED] != SIGMOID_LUT_SIZE)) begin
            validate_error = 1'b1;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_read_slot <= R_GATE;
            read_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            stage_input_index <= 32'd0;
            stage_output_index <= 32'd0;
            write_word_index <= 32'd0;
            sigmoid_lut_flat <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_stage_cycle_count <= 32'd0;
            o_output_write_word_count <= 32'd0;
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
                o_output_write_word_count <= o_output_write_word_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        active_read_slot <= R_GATE;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        stage_input_index <= 32'd0;
                        stage_output_index <= 32'd0;
                        write_word_index <= 32'd0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_stage_cycle_count <= 32'd0;
                        o_output_write_word_count <= 32'd0;
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
                        active_read_slot <= R_GATE;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        state <= S_GATE_REQ;
                    end
                end

                S_GATE_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_GATE_READ;
                    end
                end

                S_GATE_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            gate_words[read_element_index] <= i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                if (read_chunk_index == (VECTOR_CHUNK_COUNT - 1)) begin
                                    active_read_slot <= R_UP;
                                    read_chunk_index <= 32'd0;
                                    state <= S_UP_REQ;
                                end
                                else begin
                                    read_chunk_index <= read_chunk_index + 1'b1;
                                    state <= S_GATE_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_UP_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_UP_READ;
                    end
                end

                S_UP_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            up_words[read_element_index] <= i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                if (read_chunk_index == (VECTOR_CHUNK_COUNT - 1)) begin
                                    active_read_slot <= R_LUT;
                                    read_chunk_index <= 32'd0;
                                    state <= S_LUT_REQ;
                                end
                                else begin
                                    read_chunk_index <= read_chunk_index + 1'b1;
                                    state <= S_UP_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_LUT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_LUT_READ;
                    end
                end

                S_LUT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (payload_protocol_error || (read_element_index >= SIGMOID_LUT_SIZE)) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            sigmoid_lut_flat[read_element_index*SIGMOID_WIDTH +: SIGMOID_WIDTH] <=
                                i_mem_rd_rsp_data[SIGMOID_WIDTH-1 : 0];
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                if (read_chunk_index == (LUT_CHUNK_COUNT - 1)) begin
                                    stage_input_index <= 32'd0;
                                    stage_output_index <= 32'd0;
                                    o_stage_cycle_count <= 32'd0;
                                    state <= S_STAGE_START;
                                end
                                else begin
                                    read_chunk_index <= read_chunk_index + 1'b1;
                                    state <= S_LUT_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_STAGE_START: begin
                    state <= S_STAGE_RUN;
                end

                S_STAGE_RUN: begin
                    o_stage_cycle_count <= o_stage_cycle_count + 1'b1;

                    if (stage_in_valid && stage_in_ready) begin
                        stage_input_index <= stage_input_index + 1'b1;
                    end

                    if (stage_output_fire) begin
                        if ((stage_out_index != stage_output_index[ROW_INDEX_W-1 : 0]) ||
                            (stage_out_last != (stage_output_index == (FEATURES - 1)))) begin
                            o_error <= 1'b1;
                        end
                        hidden_words[stage_output_index] <= stage_hidden_word_ext;
                        stage_output_index <= stage_output_index + 1'b1;
                    end

                    if (stage_done) begin
                        if (stage_error ||
                            (stage_core_input_count != FEATURES) ||
                            (stage_core_output_count != FEATURES) ||
                            (stage_input_index != FEATURES) ||
                            (stage_output_index != FEATURES)) begin
                            o_error <= 1'b1;
                        end
                        o_saturation <= o_saturation || stage_saturation;
                        write_word_index <= 32'd0;
                        state <= S_WRITE_REQ;
                    end
                end

                S_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        write_word_index <= 32'd0;
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        if (o_mem_wr_data_last) begin
                            state <= S_WRITE_WAIT;
                        end
                        else begin
                            write_word_index <= write_word_index + 1'b1;
                        end
                    end
                end

                S_WRITE_WAIT: begin
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
