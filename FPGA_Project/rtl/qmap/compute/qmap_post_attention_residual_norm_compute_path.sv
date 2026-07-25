`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed per-layer post-attention residual/RMSNorm wrapper:
//
//   residual_input[1024] + o_proj_out[1024]
//       -> post_attention_residual_norm_stage
//       -> write post_attention_hidden[1024]
//       -> write post_norm[1024]
//
// The wrapper keeps the arithmetic inside the already-validated focused stage.
// This file only adds descriptor validation, memory reads/writes, and counters.
module qmap_post_attention_residual_norm_compute_path #(
    parameter int ADDR_WIDTH      = 64,
    parameter int DESCRIPTOR_SLOTS = 8,
    parameter int INPUT_SIZE      = 1024,
    parameter int RESIDUAL_WIDTH  = 24,
    parameter int RESIDUAL_FRAC   = 10,
    parameter int O_PROJ_WIDTH    = 24,
    parameter int O_PROJ_FRAC     = 12,
    parameter int GAMMA_WIDTH     = 16,
    parameter int GAMMA_FRAC      = 7,
    parameter int INV_RMS_WIDTH   = 24,
    parameter int INV_RMS_FRAC    = 16,
    parameter int NORM_OUT_WIDTH  = 24,
    parameter int NORM_OUT_FRAC   = 12,
    parameter int SUM_WIDTH       = 64,
    parameter int SUM_FRAC        = 2 * RESIDUAL_FRAC,
    parameter int MEAN_SHIFT      = $clog2(INPUT_SIZE),
    parameter int RMS_WIDTH       = RESIDUAL_WIDTH,
    parameter int RMS_FRAC        = RESIDUAL_FRAC,
    parameter int DIV_NUM_WIDTH   = 48,
    parameter int DIV_NUM_SHIFT   = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q20         = 1,
    parameter int MEM_DATA_WIDTH  = 32,
    parameter int MAX_READ_BYTES  = 1024
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,
    input  wire logic                         i_residual_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_residual_base_override_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [ADDR_WIDTH-1 : 0]           o_effective_residual_base_addr,
    output logic                              o_residual_saturation,
    output logic                              o_norm_saturation,
    output logic [31 : 0]                     o_residual_count,
    output logic [31 : 0]                     o_stage_cycle_count,
    output logic [SUM_WIDTH-1 : 0]            o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]            o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]        o_inv_rms,
    output logic [31 : 0]                     o_post_hidden_write_word_count,
    output logic [31 : 0]                     o_post_norm_write_word_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,
    output logic [7 : 0]                      o_state_debug,
    output logic [1 : 0]                      o_read_slot_debug,
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

    localparam int SLOT_METADATA        = 0;
    localparam int SLOT_RESIDUAL        = 1;
    localparam int SLOT_O_PROJ          = 2;
    localparam int SLOT_GAMMA           = 3;
    localparam int SLOT_HIDDEN          = 4;
    localparam int SLOT_NORM            = 5;
    localparam int SLOT_EXPECTED_HIDDEN = 6;
    localparam int SLOT_EXPECTED_NORM   = 7;

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int CHUNK_WORDS    = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int VECTOR_BYTES   = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int CHUNK_COUNT    = VECTOR_BYTES / MAX_READ_BYTES;
    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] VECTOR_BYTES_U16   = VECTOR_BYTES;

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_READ_REQ,
        S_READ_DATA,
        S_STAGE_START,
        S_STAGE_WAIT,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [1 : 0] {
        R_RESIDUAL,
        R_O_PROJ,
        R_GAMMA
    } read_slot_t;

    typedef enum logic [1 : 0] {
        W_HIDDEN,
        W_NORM
    } write_slot_t;

    state_t state;
    read_slot_t active_read_slot;
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

    logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0] residual_flat;
    logic [INPUT_SIZE*O_PROJ_WIDTH-1 : 0] o_proj_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0] gamma_flat;
    logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0] post_hidden_flat;
    logic [INPUT_SIZE*NORM_OUT_WIDTH-1 : 0] post_norm_flat;

    logic [31 : 0] read_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] write_word_index;
    logic [31 : 0] element_index;
    logic [ADDR_WIDTH-1 : 0] active_read_base_addr;
    logic [ADDR_WIDTH-1 : 0] active_write_base_addr;
    logic residual_base_override_valid_reg;
    logic [ADDR_WIDTH-1 : 0] residual_base_override_addr_reg;
    logic [31 : 0] write_data_word;
    logic read_protocol_error;
    logic validate_error;

    logic stage_start;
    logic stage_busy;
    logic stage_done;
    logic stage_error;
    logic stage_residual_saturation;
    logic stage_norm_saturation;
    logic [31 : 0] stage_residual_count;
    logic [31 : 0] stage_cycle_count;
    logic [SUM_WIDTH-1 : 0] stage_sum_squares;
    logic [SUM_WIDTH-1 : 0] stage_mean_square;
    logic [INV_RMS_WIDTH-1 : 0] stage_inv_rms;

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

    post_attention_residual_norm_stage #(
        .INPUT_SIZE(INPUT_SIZE),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .RESIDUAL_FRAC(RESIDUAL_FRAC),
        .O_PROJ_WIDTH(O_PROJ_WIDTH),
        .O_PROJ_FRAC(O_PROJ_FRAC),
        .GAMMA_WIDTH(GAMMA_WIDTH),
        .GAMMA_FRAC(GAMMA_FRAC),
        .INV_RMS_WIDTH(INV_RMS_WIDTH),
        .INV_RMS_FRAC(INV_RMS_FRAC),
        .NORM_OUT_WIDTH(NORM_OUT_WIDTH),
        .NORM_OUT_FRAC(NORM_OUT_FRAC),
        .SUM_WIDTH(SUM_WIDTH),
        .SUM_FRAC(SUM_FRAC),
        .MEAN_SHIFT(MEAN_SHIFT),
        .RMS_WIDTH(RMS_WIDTH),
        .RMS_FRAC(RMS_FRAC),
        .DIV_NUM_WIDTH(DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT(DIV_NUM_SHIFT),
        .EPS_Q20(EPS_Q20)
    ) post_attention_stage (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(stage_start),
        .i_residual_flat(residual_flat),
        .i_o_proj_flat(o_proj_flat),
        .i_gamma_flat(gamma_flat),
        .o_busy(stage_busy),
        .o_done(stage_done),
        .o_error(stage_error),
        .o_residual_saturation(stage_residual_saturation),
        .o_norm_saturation(stage_norm_saturation),
        .o_residual_count(stage_residual_count),
        .o_cycle_count(stage_cycle_count),
        .o_post_attention_hidden_flat(post_hidden_flat),
        .o_post_norm_flat(post_norm_flat),
        .o_sum_squares(stage_sum_squares),
        .o_mean_square(stage_mean_square),
        .o_inv_rms(stage_inv_rms)
    );

    assign reader_start = (state == S_READER_START);
    assign stage_start = (state == S_STAGE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {4'd0, state};
    assign o_read_slot_debug = active_read_slot;
    assign o_write_slot_debug = active_write_slot;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data  = i_mem_rd_rsp_data;
    assign reader_rsp_last  = i_mem_rd_rsp_last;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_READ_REQ)    ? 1'b1 :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        active_read_base_addr + ({{(ADDR_WIDTH-32){1'b0}}, read_chunk_index} * MAX_READ_BYTES);
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_READ_REQ)    ? MAX_READ_BYTES_U16 :
        16'd0;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_READ_DATA)   ? 1'b1 :
        1'b0;

    assign o_mem_wr_req_valid = (state == S_WRITE_REQ);
    assign o_mem_wr_req_addr = active_write_base_addr;
    assign o_mem_wr_req_len_bytes = VECTOR_BYTES_U16;
    assign o_mem_wr_data_valid = (state == S_WRITE_DATA);
    assign o_mem_wr_data = write_data_word;
    assign o_mem_wr_data_last = (state == S_WRITE_DATA) && (write_word_index == (INPUT_SIZE - 1));

    assign element_index = (state == S_WRITE_DATA) ?
        write_word_index :
        ((read_chunk_index * CHUNK_WORDS) + read_word_index);

    assign read_protocol_error = (i_mem_rd_rsp_last != (read_word_index == (CHUNK_WORDS - 1)));
    assign o_effective_residual_base_addr =
        (residual_base_override_valid_reg === 1'b1) ?
        residual_base_override_addr_reg :
        desc_base_addr[SLOT_RESIDUAL][ADDR_WIDTH-1 : 0];

    always @* begin
        case (active_read_slot)
            R_RESIDUAL: active_read_base_addr = o_effective_residual_base_addr;
            R_O_PROJ:   active_read_base_addr = desc_base_addr[SLOT_O_PROJ];
            R_GAMMA:    active_read_base_addr = desc_base_addr[SLOT_GAMMA];
            default:    active_read_base_addr = o_effective_residual_base_addr;
        endcase
    end

    always @* begin
        case (active_write_slot)
            W_HIDDEN: active_write_base_addr = desc_base_addr[SLOT_HIDDEN];
            W_NORM:   active_write_base_addr = desc_base_addr[SLOT_NORM];
            default:  active_write_base_addr = desc_base_addr[SLOT_HIDDEN];
        endcase
    end

    always @* begin
        write_data_word = 32'd0;
        if (active_write_slot == W_HIDDEN) begin
            write_data_word = {
                {8{post_hidden_flat[(element_index*RESIDUAL_WIDTH) + (RESIDUAL_WIDTH-1)]}},
                post_hidden_flat[element_index*RESIDUAL_WIDTH +: RESIDUAL_WIDTH]
            };
        end
        else begin
            write_data_word = {
                {8{post_norm_flat[(element_index*NORM_OUT_WIDTH) + (NORM_OUT_WIDTH-1)]}},
                post_norm_flat[element_index*NORM_OUT_WIDTH +: NORM_OUT_WIDTH]
            };
        end
    end

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count < `QMAP_POST_ATTN_NORM_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity < `QMAP_POST_ATTN_NORM_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_POST_ATTN_METADATA) ||
            (desc_tensor_id[SLOT_RESIDUAL] != `QMAP_TENSOR_ID_POST_ATTN_RESIDUAL) ||
            (desc_tensor_id[SLOT_O_PROJ] != `QMAP_TENSOR_ID_POST_ATTN_O_PROJ) ||
            (desc_tensor_id[SLOT_GAMMA] != `QMAP_TENSOR_ID_POST_ATTN_GAMMA) ||
            (desc_tensor_id[SLOT_HIDDEN] != `QMAP_TENSOR_ID_POST_ATTN_HIDDEN) ||
            (desc_tensor_id[SLOT_NORM] != `QMAP_TENSOR_ID_POST_ATTN_NORM) ||
            (desc_tensor_id[SLOT_EXPECTED_HIDDEN] != `QMAP_TENSOR_ID_POST_ATTN_EXPECTED_HIDDEN) ||
            (desc_tensor_id[SLOT_EXPECTED_NORM] != `QMAP_TENSOR_ID_POST_ATTN_EXPECTED_NORM) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_RESIDUAL] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_O_PROJ] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_GAMMA] != `QMAP_ROLE_PARAMETER) ||
            (desc_role[SLOT_HIDDEN] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_NORM] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_EXPECTED_HIDDEN] != `QMAP_ROLE_EXPECTED) ||
            (desc_role[SLOT_EXPECTED_NORM] != `QMAP_ROLE_EXPECTED) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_RESIDUAL] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_O_PROJ] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_GAMMA] != `QMAP_DTYPE_I16_Q8_7) ||
            (desc_dtype[SLOT_HIDDEN] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_NORM] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_EXPECTED_HIDDEN] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_EXPECTED_NORM] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_RESIDUAL] != 32'd1) ||
            (desc_rank[SLOT_O_PROJ] != 32'd1) ||
            (desc_rank[SLOT_GAMMA] != 32'd1) ||
            (desc_rank[SLOT_HIDDEN] != 32'd1) ||
            (desc_rank[SLOT_NORM] != 32'd1) ||
            (desc_rank[SLOT_EXPECTED_HIDDEN] != 32'd1) ||
            (desc_rank[SLOT_EXPECTED_NORM] != 32'd1) ||
            (desc_element_bits[SLOT_METADATA] != 32'd32) ||
            (desc_element_bits[SLOT_RESIDUAL] != RESIDUAL_WIDTH) ||
            (desc_element_bits[SLOT_O_PROJ] != O_PROJ_WIDTH) ||
            (desc_element_bits[SLOT_GAMMA] != GAMMA_WIDTH) ||
            (desc_element_bits[SLOT_HIDDEN] != RESIDUAL_WIDTH) ||
            (desc_element_bits[SLOT_NORM] != NORM_OUT_WIDTH) ||
            (desc_element_bits[SLOT_EXPECTED_HIDDEN] != RESIDUAL_WIDTH) ||
            (desc_element_bits[SLOT_EXPECTED_NORM] != NORM_OUT_WIDTH) ||
            (desc_group_size[SLOT_RESIDUAL] != 32'd0) ||
            (desc_group_size[SLOT_O_PROJ] != 32'd0) ||
            (desc_group_size[SLOT_GAMMA] != 32'd0) ||
            (desc_scale_tensor_id[SLOT_RESIDUAL] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_O_PROJ] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_GAMMA] != `QMAP_NO_TENSOR_ID) ||
            (desc_dim0[SLOT_RESIDUAL] != INPUT_SIZE) ||
            (desc_dim0[SLOT_O_PROJ] != INPUT_SIZE) ||
            (desc_dim0[SLOT_GAMMA] != INPUT_SIZE) ||
            (desc_dim0[SLOT_HIDDEN] != INPUT_SIZE) ||
            (desc_dim0[SLOT_NORM] != INPUT_SIZE) ||
            (desc_dim0[SLOT_EXPECTED_HIDDEN] != INPUT_SIZE) ||
            (desc_dim0[SLOT_EXPECTED_NORM] != INPUT_SIZE) ||
            (desc_dim1[SLOT_RESIDUAL] != 32'd0) ||
            (desc_dim1[SLOT_O_PROJ] != 32'd0) ||
            (desc_dim1[SLOT_GAMMA] != 32'd0) ||
            (desc_nbytes[SLOT_RESIDUAL] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_O_PROJ] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_GAMMA] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_HIDDEN] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_NORM] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_EXPECTED_HIDDEN] != VECTOR_BYTES) ||
            (desc_nbytes[SLOT_EXPECTED_NORM] != VECTOR_BYTES) ||
            (o_effective_residual_base_addr == '0) ||
            (o_effective_residual_base_addr[1 : 0] != 2'b00) ||
            ((desc_flags[SLOT_RESIDUAL] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_O_PROJ] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_GAMMA] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_HIDDEN] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_NORM] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_EXPECTED_HIDDEN] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_EXPECTED_NORM] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            (desc_aux0[SLOT_METADATA] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux2[SLOT_METADATA] != INPUT_SIZE) ||
            (desc_aux0[SLOT_RESIDUAL] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux0[SLOT_O_PROJ] != `QMAP_MATRIX_ID_O_PROJ) ||
            (desc_aux0[SLOT_GAMMA] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux0[SLOT_HIDDEN] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux0[SLOT_NORM] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux1[SLOT_RESIDUAL] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_O_PROJ] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_GAMMA] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_HIDDEN] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_NORM] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_EXPECTED_HIDDEN] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux1[SLOT_EXPECTED_NORM] != desc_aux1[SLOT_METADATA])) begin
            validate_error = 1'b1;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_read_slot <= R_RESIDUAL;
            active_write_slot <= W_HIDDEN;
            read_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            write_word_index <= 32'd0;
            residual_flat <= '0;
            o_proj_flat <= '0;
            gamma_flat <= '0;
            residual_base_override_valid_reg <= 1'b0;
            residual_base_override_addr_reg <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_residual_saturation <= 1'b0;
            o_norm_saturation <= 1'b0;
            o_residual_count <= 32'd0;
            o_stage_cycle_count <= 32'd0;
            o_sum_squares <= '0;
            o_mean_square <= '0;
            o_inv_rms <= '0;
            o_post_hidden_write_word_count <= 32'd0;
            o_post_norm_write_word_count <= 32'd0;
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
                if (active_write_slot == W_HIDDEN) begin
                    o_post_hidden_write_word_count <= o_post_hidden_write_word_count + 1'b1;
                end
                else begin
                    o_post_norm_write_word_count <= o_post_norm_write_word_count + 1'b1;
                end
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        o_error <= 1'b0;
                        o_residual_saturation <= 1'b0;
                        o_norm_saturation <= 1'b0;
                        o_residual_count <= 32'd0;
                        o_stage_cycle_count <= 32'd0;
                        o_sum_squares <= '0;
                        o_mean_square <= '0;
                        o_inv_rms <= '0;
                        o_post_hidden_write_word_count <= 32'd0;
                        o_post_norm_write_word_count <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        active_read_slot <= R_RESIDUAL;
                        active_write_slot <= W_HIDDEN;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        write_word_index <= 32'd0;
                        residual_base_override_valid_reg <=
                            (i_residual_base_override_valid === 1'b1);
                        residual_base_override_addr_reg <= i_residual_base_override_addr;
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
                        active_read_slot <= R_RESIDUAL;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        state <= S_READ_REQ;
                    end
                end

                S_READ_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_READ_DATA;
                    end
                end

                S_READ_DATA: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (read_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            if (active_read_slot == R_RESIDUAL) begin
                                residual_flat[element_index*RESIDUAL_WIDTH +: RESIDUAL_WIDTH] <=
                                    i_mem_rd_rsp_data[RESIDUAL_WIDTH-1 : 0];
                            end
                            else if (active_read_slot == R_O_PROJ) begin
                                o_proj_flat[element_index*O_PROJ_WIDTH +: O_PROJ_WIDTH] <=
                                    i_mem_rd_rsp_data[O_PROJ_WIDTH-1 : 0];
                            end
                            else begin
                                gamma_flat[element_index*GAMMA_WIDTH +: GAMMA_WIDTH] <=
                                    i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0];
                            end

                            if (i_mem_rd_rsp_last) begin
                                if (read_chunk_index == (CHUNK_COUNT - 1)) begin
                                    read_chunk_index <= 32'd0;
                                    read_word_index <= 32'd0;
                                    if (active_read_slot == R_RESIDUAL) begin
                                        active_read_slot <= R_O_PROJ;
                                        state <= S_READ_REQ;
                                    end
                                    else if (active_read_slot == R_O_PROJ) begin
                                        active_read_slot <= R_GAMMA;
                                        state <= S_READ_REQ;
                                    end
                                    else begin
                                        state <= S_STAGE_START;
                                    end
                                end
                                else begin
                                    read_chunk_index <= read_chunk_index + 1'b1;
                                    read_word_index <= 32'd0;
                                    state <= S_READ_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_STAGE_START: begin
                    state <= S_STAGE_WAIT;
                end

                S_STAGE_WAIT: begin
                    if (stage_done) begin
                        o_residual_saturation <= stage_residual_saturation;
                        o_norm_saturation <= stage_norm_saturation;
                        o_residual_count <= stage_residual_count;
                        o_stage_cycle_count <= stage_cycle_count;
                        o_sum_squares <= stage_sum_squares;
                        o_mean_square <= stage_mean_square;
                        o_inv_rms <= stage_inv_rms;
                        if (stage_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            active_write_slot <= W_HIDDEN;
                            write_word_index <= 32'd0;
                            state <= S_WRITE_REQ;
                        end
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
                            state <= S_DONE;
                        end
                        else if (active_write_slot == W_HIDDEN) begin
                            active_write_slot <= W_NORM;
                            write_word_index <= 32'd0;
                            state <= S_WRITE_REQ;
                        end
                        else begin
                            state <= S_DONE;
                        end
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
