`default_nettype none

`include "qmap_defs.svh"

// Descriptor-driven Layer 0 Q/K/V projection compute path.
//
// This module consumes a QMAP runtime work packet with 12 active descriptors:
// metadata, input activation, Q/K/V packed Q4 weights, Q/K/V scales,
// Q/K/V output buffers, and an optional expected/debug tensor. It reads the
// descriptor-provided tensors through the project-local memory read interface,
// reuses q4_gemv_row_1024 for every output row, converts Q26 row sums to
// I32_Q12_12 words, and writes those words through the project-local write
// stream.
module qmap_qkv_projection_compute_path #(
    parameter int ADDR_WIDTH     = 64,
    parameter int INPUT_SIZE     = 1024,
    parameter int GROUP_SIZE     = 64,
    parameter int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL = 4,
    parameter int ACT_WIDTH      = 24,
    parameter int ACT_FRAC       = 12,
    parameter int WEIGHT_WIDTH   = 4,
    parameter int SCALE_WIDTH    = 16,
    parameter int SCALE_FRAC     = 14,
    parameter int PARTIAL_WIDTH  = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH   = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH  = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_qmap_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [31 : 0]                     o_rows_done,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_last_row_sum_q26,
    output logic signed [31 : 0]              o_last_output_q12_12,

    output logic                              o_mem_rd_req_valid,
    input  wire logic                              i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_rd_req_addr,
    output logic [15 : 0]                     o_mem_rd_req_len_bytes,

    input  wire logic                              i_mem_rd_rsp_valid,
    output logic                              o_mem_rd_rsp_ready,
    input  wire logic [31 : 0]                     i_mem_rd_rsp_data,
    input  wire logic                              i_mem_rd_rsp_last,

    output logic                              o_mem_wr_req_valid,
    input  wire logic                              i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_wr_req_addr,
    output logic [15 : 0]                     o_mem_wr_req_len_bytes,

    output logic [31 : 0]                     o_mem_wr_data,
    output logic                              o_mem_wr_data_valid,
    input  wire logic                              i_mem_wr_data_ready,
    output logic                              o_mem_wr_data_last,
    input  wire logic                              i_mem_wr_done,
    input  wire logic                              i_mem_wr_error
);

    localparam int DESCRIPTOR_SLOTS        = 12;
    localparam int SLOT_METADATA           = 0;
    localparam int SLOT_ACTIVATION         = 1;
    localparam int SLOT_Q_WEIGHT           = 2;
    localparam int SLOT_Q_SCALE            = 3;
    localparam int SLOT_K_WEIGHT           = 4;
    localparam int SLOT_K_SCALE            = 5;
    localparam int SLOT_V_WEIGHT           = 6;
    localparam int SLOT_V_SCALE            = 7;
    localparam int SLOT_Q_OUT              = 8;
    localparam int SLOT_K_OUT              = 9;
    localparam int SLOT_V_OUT              = 10;
    localparam int SLOT_EXPECTED           = 11;

    localparam int DATA_BYTES              = 4;
    localparam int ACTIVATION_BYTES        = INPUT_SIZE * DATA_BYTES;
    localparam int ACTIVATION_CHUNK_BYTES  = `QMAP_QKV_MAX_READ_BYTES;
    localparam int ACTIVATION_CHUNK_WORDS  = ACTIVATION_CHUNK_BYTES / DATA_BYTES;
    localparam int ACTIVATION_CHUNK_COUNT  = ACTIVATION_BYTES / ACTIVATION_CHUNK_BYTES;
    localparam int WEIGHT_ROW_BYTES        = INPUT_SIZE / 2;
    localparam int WEIGHT_ROW_WORDS        = WEIGHT_ROW_BYTES / DATA_BYTES;
    localparam int SCALE_ROW_BYTES         = GROUP_COUNT * (SCALE_WIDTH / 8);
    localparam int SCALE_ROW_WORDS         = SCALE_ROW_BYTES / DATA_BYTES;
    localparam logic [15 : 0] ACTIVATION_CHUNK_LEN_BYTES = ACTIVATION_CHUNK_BYTES;
    localparam logic [15 : 0] WEIGHT_ROW_LEN_BYTES       = WEIGHT_ROW_BYTES;
    localparam logic [15 : 0] SCALE_ROW_LEN_BYTES        = SCALE_ROW_BYTES;

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
        S_WEIGHT_REQ,
        S_WEIGHT_READ,
        S_SCALE_REQ,
        S_SCALE_READ,
        S_COMPUTE_START,
        S_COMPUTE_WAIT,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_NEXT_ROW,
        S_DONE
    } state_t;

    typedef enum logic [1 : 0] {
        MATRIX_Q,
        MATRIX_K,
        MATRIX_V
    } matrix_t;

    state_t state;
    matrix_t matrix_index;

    logic [31 : 0] row_index;
    logic [31 : 0] activation_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] activation_element_index;
    logic [31 : 0] current_row_count;
    logic [31 : 0] next_row_count;
    logic [ADDR_WIDTH-1 : 0] current_weight_base;
    logic [ADDR_WIDTH-1 : 0] current_scale_base;
    logic [ADDR_WIDTH-1 : 0] current_out_base;
    logic [ADDR_WIDTH-1 : 0] payload_req_addr;
    logic [15 : 0] payload_req_len_bytes;
    logic payload_req_valid;
    logic payload_rsp_ready;
    logic expected_payload_last;
    logic protocol_error;
    logic signed [31 : 0] converted_output_q12_12;

    logic [INPUT_SIZE*ACT_WIDTH-1 : 0]    activation_flat;
    logic [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] weight_packed;
    logic [GROUP_COUNT*SCALE_WIDTH-1 : 0] scale_flat;

    logic reader_start;
    logic reader_busy;
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
    logic [31 : 0] reader_header_magic;
    logic [31 : 0] reader_header_version;
    logic [31 : 0] reader_header_bytes;
    logic [31 : 0] reader_descriptor_bytes;
    logic [31 : 0] reader_descriptor_count;
    logic [31 : 0] reader_descriptor_capacity;
    logic [63 : 0] reader_descriptor_table_addr;
    logic [63 : 0] reader_payload_base_addr;
    logic [63 : 0] reader_image_base_addr;
    logic [63 : 0] reader_image_bytes;
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
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim3_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux3_flat;

    logic [31 : 0] desc_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_role [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dtype [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_flags [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_group_size [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_scale_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux0 [0 : DESCRIPTOR_SLOTS-1];

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]       = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]            = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]           = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_flags[desc_index]           = reader_desc_flags_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index]    = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_group_size[desc_index]      = reader_desc_group_size_flat[desc_index*32 +: 32];
            assign desc_scale_tensor_id[desc_index] = reader_desc_scale_tensor_id_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]       = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]          = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]            = reader_desc_dim0_flat[desc_index*32 +: 32];
            assign desc_dim1[desc_index]            = reader_desc_dim1_flat[desc_index*32 +: 32];
            assign desc_aux0[desc_index]            = reader_desc_aux0_flat[desc_index*32 +: 32];
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
        .o_busy(reader_busy),
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
        .o_header_magic(reader_header_magic),
        .o_header_version(reader_header_version),
        .o_header_bytes(reader_header_bytes),
        .o_descriptor_bytes(reader_descriptor_bytes),
        .o_descriptor_count(reader_descriptor_count),
        .o_descriptor_capacity(reader_descriptor_capacity),
        .o_descriptor_table_addr(reader_descriptor_table_addr),
        .o_payload_base_addr(reader_payload_base_addr),
        .o_image_base_addr(reader_image_base_addr),
        .o_image_bytes(reader_image_bytes),
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
        .o_desc_dim2_flat(reader_desc_dim2_flat),
        .o_desc_dim3_flat(reader_desc_dim3_flat),
        .o_desc_aux0_flat(reader_desc_aux0_flat),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(reader_desc_aux2_flat),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    logic row_start;
    logic row_busy;
    logic row_done;
    logic signed [ROW_ACC_WIDTH-1 : 0] row_sum_q26;

    q4_gemv_row_1024 #(
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH)
    ) row_gemv (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(row_start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(weight_packed),
        .i_scale_flat(scale_flat),
        .o_busy(row_busy),
        .o_done(row_done),
        .o_row_sum_q26(row_sum_q26)
    );

    function automatic logic signed [31 : 0] q26_to_q12_12(
        input logic signed [ROW_ACC_WIDTH-1 : 0] value
    );
        logic signed [ROW_ACC_WIDTH-1 : 0] shifted;
        begin
            shifted = value >>> SCALE_FRAC;
            if (shifted > Q12_12_MAX_EXT) begin
                q26_to_q12_12 = 32'sd8388607;
            end else if (shifted < Q12_12_MIN_EXT) begin
                q26_to_q12_12 = -32'sd8388608;
            end else begin
                q26_to_q12_12 = $signed(shifted[31:0]);
            end
        end
    endfunction

    assign o_busy = (state != S_IDLE);
    assign reader_start = (state == S_READER_START);
    assign row_start = (state == S_COMPUTE_START);

    assign payload_req_valid =
        (state == S_ACT_REQ) || (state == S_WEIGHT_REQ) || (state == S_SCALE_REQ);
    assign payload_rsp_ready =
        (state == S_ACT_READ) || (state == S_WEIGHT_READ) || (state == S_SCALE_READ);

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid : payload_req_valid;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr : payload_req_addr;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes : payload_req_len_bytes;
    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;

    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready : payload_rsp_ready;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data  = i_mem_rd_rsp_data;
    assign reader_rsp_last  = i_mem_rd_rsp_last;

    assign o_mem_wr_req_valid     = (state == S_WRITE_REQ);
    assign o_mem_wr_req_addr      = current_out_base + ({{(ADDR_WIDTH-32){1'b0}}, row_index} << 2);
    assign o_mem_wr_req_len_bytes = 16'd4;
    assign o_mem_wr_data          = o_last_output_q12_12;
    assign o_mem_wr_data_valid    = (state == S_WRITE_DATA);
    assign o_mem_wr_data_last     = 1'b1;

    always @* begin
        current_weight_base = desc_base_addr[SLOT_Q_WEIGHT];
        current_scale_base  = desc_base_addr[SLOT_Q_SCALE];
        current_out_base    = desc_base_addr[SLOT_Q_OUT];
        current_row_count   = desc_dim0[SLOT_Q_WEIGHT];
        next_row_count      = desc_dim0[SLOT_K_WEIGHT];

        case (matrix_index)
            MATRIX_Q: begin
                current_weight_base = desc_base_addr[SLOT_Q_WEIGHT];
                current_scale_base  = desc_base_addr[SLOT_Q_SCALE];
                current_out_base    = desc_base_addr[SLOT_Q_OUT];
                current_row_count   = desc_dim0[SLOT_Q_WEIGHT];
                next_row_count      = desc_dim0[SLOT_K_WEIGHT];
            end

            MATRIX_K: begin
                current_weight_base = desc_base_addr[SLOT_K_WEIGHT];
                current_scale_base  = desc_base_addr[SLOT_K_SCALE];
                current_out_base    = desc_base_addr[SLOT_K_OUT];
                current_row_count   = desc_dim0[SLOT_K_WEIGHT];
                next_row_count      = desc_dim0[SLOT_V_WEIGHT];
            end

            MATRIX_V: begin
                current_weight_base = desc_base_addr[SLOT_V_WEIGHT];
                current_scale_base  = desc_base_addr[SLOT_V_SCALE];
                current_out_base    = desc_base_addr[SLOT_V_OUT];
                current_row_count   = desc_dim0[SLOT_V_WEIGHT];
                next_row_count      = 32'd0;
            end

            default: begin
                current_weight_base = desc_base_addr[SLOT_Q_WEIGHT];
                current_scale_base  = desc_base_addr[SLOT_Q_SCALE];
                current_out_base    = desc_base_addr[SLOT_Q_OUT];
                current_row_count   = desc_dim0[SLOT_Q_WEIGHT];
                next_row_count      = desc_dim0[SLOT_K_WEIGHT];
            end
        endcase
    end

    always @* begin
        payload_req_addr      = desc_base_addr[SLOT_ACTIVATION];
        payload_req_len_bytes = 16'd0;

        case (state)
            S_ACT_REQ: begin
                payload_req_addr =
                    desc_base_addr[SLOT_ACTIVATION] +
                    ({{(ADDR_WIDTH-32){1'b0}}, activation_chunk_index} * ACTIVATION_CHUNK_BYTES);
                payload_req_len_bytes = ACTIVATION_CHUNK_LEN_BYTES;
            end

            S_WEIGHT_REQ: begin
                payload_req_addr =
                    current_weight_base + ({{(ADDR_WIDTH-32){1'b0}}, row_index} * WEIGHT_ROW_BYTES);
                payload_req_len_bytes = WEIGHT_ROW_LEN_BYTES;
            end

            S_SCALE_REQ: begin
                payload_req_addr =
                    current_scale_base + ({{(ADDR_WIDTH-32){1'b0}}, row_index} * SCALE_ROW_BYTES);
                payload_req_len_bytes = SCALE_ROW_LEN_BYTES;
            end

            default: begin
                payload_req_addr      = desc_base_addr[SLOT_ACTIVATION];
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
            S_WEIGHT_READ: begin
                expected_payload_last = (read_word_index == (WEIGHT_ROW_WORDS - 1));
            end
            S_SCALE_READ: begin
                expected_payload_last = (read_word_index == (SCALE_ROW_WORDS - 1));
            end
            default: begin
                expected_payload_last = 1'b0;
            end
        endcase
    end

    always @* begin
        activation_element_index =
            (activation_chunk_index * ACTIVATION_CHUNK_WORDS) + read_word_index;
        converted_output_q12_12 = q26_to_q12_12(row_sum_q26);
        protocol_error = 1'b0;

        if ((state == S_ACT_READ) || (state == S_WEIGHT_READ) || (state == S_SCALE_READ)) begin
            protocol_error = i_mem_rd_rsp_valid && (i_mem_rd_rsp_last != expected_payload_last);
        end
    end

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                <= S_IDLE;
            matrix_index         <= MATRIX_Q;
            row_index            <= 32'd0;
            activation_chunk_index <= 32'd0;
            read_word_index      <= 32'd0;
            activation_flat      <= 'd0;
            weight_packed        <= 'd0;
            scale_flat           <= 'd0;
            o_done               <= 1'b0;
            o_error              <= 1'b0;
            o_rows_done          <= 32'd0;
            o_last_row_sum_q26   <= 'd0;
            o_last_output_q12_12 <= 32'sd0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        matrix_index           <= MATRIX_Q;
                        row_index              <= 32'd0;
                        activation_chunk_index <= 32'd0;
                        read_word_index        <= 32'd0;
                        o_error                <= 1'b0;
                        o_rows_done            <= 32'd0;
                        o_last_row_sum_q26     <= 'd0;
                        o_last_output_q12_12   <= 32'sd0;
                        state                  <= S_READER_START;
                    end
                end

                S_READER_START: begin
                    state <= S_READER_WAIT;
                end

                S_READER_WAIT: begin
                    if (reader_done) begin
                        if (reader_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            state <= S_VALIDATE;
                        end
                    end
                end

                S_VALIDATE: begin
                    if ((reader_descriptor_count < DESCRIPTOR_SLOTS) ||
                        (reader_descriptor_capacity < `QMAP_QKV_DESCRIPTOR_CAPACITY) ||
                        (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_QKV_METADATA) ||
                        (desc_tensor_id[SLOT_ACTIVATION] != `QMAP_TENSOR_ID_QKV_ACTIVATION) ||
                        (desc_tensor_id[SLOT_Q_WEIGHT] != `QMAP_TENSOR_ID_Q_WEIGHT) ||
                        (desc_tensor_id[SLOT_Q_SCALE] != `QMAP_TENSOR_ID_Q_SCALE) ||
                        (desc_tensor_id[SLOT_K_WEIGHT] != `QMAP_TENSOR_ID_K_WEIGHT) ||
                        (desc_tensor_id[SLOT_K_SCALE] != `QMAP_TENSOR_ID_K_SCALE) ||
                        (desc_tensor_id[SLOT_V_WEIGHT] != `QMAP_TENSOR_ID_V_WEIGHT) ||
                        (desc_tensor_id[SLOT_V_SCALE] != `QMAP_TENSOR_ID_V_SCALE) ||
                        (desc_tensor_id[SLOT_Q_OUT] != `QMAP_TENSOR_ID_Q_OUT) ||
                        (desc_tensor_id[SLOT_K_OUT] != `QMAP_TENSOR_ID_K_OUT) ||
                        (desc_tensor_id[SLOT_V_OUT] != `QMAP_TENSOR_ID_V_OUT) ||
                        (desc_role[SLOT_ACTIVATION] != `QMAP_ROLE_ACTIVATION) ||
                        (desc_dtype[SLOT_ACTIVATION] != `QMAP_DTYPE_I32_Q12_12) ||
                        (desc_dim0[SLOT_ACTIVATION] != INPUT_SIZE) ||
                        (desc_nbytes[SLOT_ACTIVATION] != ACTIVATION_BYTES) ||
                        (desc_role[SLOT_Q_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
                        (desc_role[SLOT_K_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
                        (desc_role[SLOT_V_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
                        (desc_dtype[SLOT_Q_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
                        (desc_dtype[SLOT_K_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
                        (desc_dtype[SLOT_V_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
                        (desc_dim1[SLOT_Q_WEIGHT] != INPUT_SIZE) ||
                        (desc_dim1[SLOT_K_WEIGHT] != INPUT_SIZE) ||
                        (desc_dim1[SLOT_V_WEIGHT] != INPUT_SIZE) ||
                        (desc_role[SLOT_Q_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
                        (desc_role[SLOT_K_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
                        (desc_role[SLOT_V_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
                        (desc_dtype[SLOT_Q_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
                        (desc_dtype[SLOT_K_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
                        (desc_dtype[SLOT_V_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
                        (desc_dim1[SLOT_Q_SCALE] != GROUP_COUNT) ||
                        (desc_dim1[SLOT_K_SCALE] != GROUP_COUNT) ||
                        (desc_dim1[SLOT_V_SCALE] != GROUP_COUNT) ||
                        (desc_dim0[SLOT_Q_OUT] != desc_dim0[SLOT_Q_WEIGHT]) ||
                        (desc_dim0[SLOT_K_OUT] != desc_dim0[SLOT_K_WEIGHT]) ||
                        (desc_dim0[SLOT_V_OUT] != desc_dim0[SLOT_V_WEIGHT]) ||
                        (desc_role[SLOT_Q_OUT] != `QMAP_ROLE_OUTPUT) ||
                        (desc_role[SLOT_K_OUT] != `QMAP_ROLE_OUTPUT) ||
                        (desc_role[SLOT_V_OUT] != `QMAP_ROLE_OUTPUT) ||
                        (desc_dtype[SLOT_Q_OUT] != `QMAP_DTYPE_I32_Q12_12) ||
                        (desc_dtype[SLOT_K_OUT] != `QMAP_DTYPE_I32_Q12_12) ||
                        (desc_dtype[SLOT_V_OUT] != `QMAP_DTYPE_I32_Q12_12)) begin
                        o_error <= 1'b1;
                        state   <= S_DONE;
                    end else begin
                        activation_chunk_index <= 32'd0;
                        read_word_index        <= 32'd0;
                        state                  <= S_ACT_REQ;
                    end
                end

                S_ACT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state           <= S_ACT_READ;
                    end
                end

                S_ACT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (protocol_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            activation_flat[activation_element_index*ACT_WIDTH +: ACT_WIDTH]
                                <= i_mem_rd_rsp_data[ACT_WIDTH-1 : 0];

                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                if (activation_chunk_index == (ACTIVATION_CHUNK_COUNT - 1)) begin
                                    matrix_index <= MATRIX_Q;
                                    row_index    <= 32'd0;
                                    state        <= S_ROW_SETUP;
                                end else begin
                                    activation_chunk_index <= activation_chunk_index + 1'b1;
                                    state                  <= S_ACT_REQ;
                                end
                            end else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_ROW_SETUP: begin
                    read_word_index <= 32'd0;
                    state           <= S_WEIGHT_REQ;
                end

                S_WEIGHT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state           <= S_WEIGHT_READ;
                    end
                end

                S_WEIGHT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (protocol_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            weight_packed[read_word_index*32 +: 32] <= i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state           <= S_SCALE_REQ;
                            end else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_SCALE_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state           <= S_SCALE_READ;
                    end
                end

                S_SCALE_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (protocol_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            scale_flat[read_word_index*32 +: 32] <= i_mem_rd_rsp_data;
                            if (expected_payload_last) begin
                                read_word_index <= 32'd0;
                                state           <= S_COMPUTE_START;
                            end else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_COMPUTE_START: begin
                    state <= S_COMPUTE_WAIT;
                end

                S_COMPUTE_WAIT: begin
                    if (row_done) begin
                        o_last_row_sum_q26   <= row_sum_q26;
                        o_last_output_q12_12 <= converted_output_q12_12;
                        state                <= S_WRITE_REQ;
                    end
                end

                S_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        state <= S_WRITE_WAIT;
                    end
                end

                S_WRITE_WAIT: begin
                    if (i_mem_wr_done) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            o_rows_done <= o_rows_done + 1'b1;
                            state       <= S_NEXT_ROW;
                        end
                    end
                end

                S_NEXT_ROW: begin
                    if ((row_index + 1'b1) < current_row_count) begin
                        row_index <= row_index + 1'b1;
                        state     <= S_ROW_SETUP;
                    end else begin
                        row_index <= 32'd0;
                        case (matrix_index)
                            MATRIX_Q: begin
                                matrix_index <= MATRIX_K;
                                state        <= S_ROW_SETUP;
                            end
                            MATRIX_K: begin
                                matrix_index <= MATRIX_V;
                                state        <= S_ROW_SETUP;
                            end
                            MATRIX_V: begin
                                state <= S_DONE;
                            end
                            default: begin
                                state <= S_DONE;
                            end
                        endcase
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
