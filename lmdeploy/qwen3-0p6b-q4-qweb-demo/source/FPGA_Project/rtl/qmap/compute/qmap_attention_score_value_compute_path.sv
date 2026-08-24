`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed attention score + softmax/value wrapper:
//
//   q_rope[2048] + K/V cache + exp LUT
//       -> attention_score_stage
//       -> attention_softmax_value_stage
//       -> write attn_out[2048]
//
// K and V cache accesses are converted into project-local 4-byte memory reads
// using the same KV-cache address generator as the cache append path.
module qmap_attention_score_value_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 5,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int IN_WIDTH         = 24,
    parameter int SCORE_WIDTH      = 64,
    parameter int SCALE_WIDTH      = 32,
    parameter int EXP_WIDTH        = 24,
    parameter int EXP_LUT_SIZE     = 257,
    parameter int OUT_WIDTH        = 24,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024,
    parameter int USE_BRAM_STREAMING = 1,
    parameter int LAYER_INDEX_W    = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int Q_HEAD_INDEX_W   = (NUM_Q_HEADS <= 1) ? 1 : $clog2(NUM_Q_HEADS),
    parameter int KV_HEAD_INDEX_W  = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int CACHE_LENGTH_W   = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT + 1),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,
    input  wire logic                         i_runtime_context_valid,
    input  wire logic [LAYER_INDEX_W-1 : 0]  i_runtime_layer_id,
    input  wire logic [POSITION_INDEX_W-1 : 0] i_runtime_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_runtime_kv_cache_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic                              o_saturation,
    output logic [31 : 0]                     o_score_count,
    output logic [31 : 0]                     o_k_read_count,
    output logic [31 : 0]                     o_v_read_count,
    output logic [31 : 0]                     o_attn_out_capture_count,
    output logic [31 : 0]                     o_attn_out_write_word_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,
    output logic [7 : 0]                      o_state_debug,
    output logic [3 : 0]                      o_read_slot_debug,

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
    localparam int SLOT_Q_ROPE   = 1;
    localparam int SLOT_KV_CACHE = 2;
    localparam int SLOT_EXP_LUT  = 3;
    localparam int SLOT_ATTN_OUT = 4;

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int CHUNK_WORDS    = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int Q_COUNT        = NUM_Q_HEADS * HEAD_DIM;
    localparam int Q_ADDR_WIDTH   =
        (Q_COUNT <= 1) ? 1 : $clog2(Q_COUNT);
    localparam int EXP_LUT_INDEX_W =
        (EXP_LUT_SIZE <= 1) ? 1 : $clog2(EXP_LUT_SIZE);
    localparam int KV_CACHE_BYTES = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_KIND_STRIDE_BYTES = NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_HEAD_STRIDE_BYTES = MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_POSITION_STRIDE_BYTES = HEAD_DIM * MEM_DATA_BYTES;
    localparam int Q_BYTES        = Q_COUNT * MEM_DATA_BYTES;
    localparam int EXP_LUT_BYTES  = EXP_LUT_SIZE * MEM_DATA_BYTES;
    localparam int ATTN_OUT_BYTES = Q_COUNT * MEM_DATA_BYTES;
    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] CACHE_READ_BYTES_U16 = 16'd4;
    localparam logic [15 : 0] ATTN_OUT_BYTES_U16 = ATTN_OUT_BYTES;

    typedef enum logic [4 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_READ_REQ,
        S_READ_DATA,
        S_STAGE_START,
        S_STAGE_RUN,
        S_CACHE_RD_REQ,
        S_CACHE_RD_DATA,
        S_ATTN_WR_REQ,
        S_ATTN_WR_DATA,
        S_ATTN_WR_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [1 : 0] {
        R_Q_ROPE,
        R_EXP_LUT,
        R_DONE
    } read_slot_t;

    state_t state;
    read_slot_t active_read_slot;
    logic active_cache_kind;

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
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_element_bits_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_base_addr_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_nbytes_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim3_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_stride0_bytes_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_stride1_bytes_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_stride2_bytes_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_stride3_bytes_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux3_flat;

    logic [31 : 0] desc_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_role [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dtype [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_rank [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim2 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim3 [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_stride0_bytes [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_stride1_bytes [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_stride2_bytes [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_stride3_bytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux2 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux3 [0 : DESCRIPTOR_SLOTS-1];

    logic [Q_COUNT*IN_WIDTH-1 : 0] q_rope_flat;
    logic [EXP_LUT_SIZE*EXP_WIDTH-1 : 0] exp_lut_flat;
    logic [Q_COUNT*OUT_WIDTH-1 : 0] attn_out_flat;
    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] q_rope_mem [0 : Q_COUNT-1];
    (* ram_style = "block" *)
    logic [EXP_WIDTH-1 : 0] exp_lut_mem [0 : EXP_LUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0] attn_out_mem [0 : Q_COUNT-1];

    logic [31 : 0] read_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] read_element_index;
    logic [63 : 0] current_read_base_addr;
    logic [63 : 0] current_read_total_bytes;
    logic [63 : 0] current_read_offset_bytes;
    logic [63 : 0] current_read_remaining_bytes;
    logic [15 : 0] current_read_len_bytes;
    logic [31 : 0] current_chunk_words;
    logic read_protocol_error;
    logic cache_read_protocol_error;
    logic validate_error;

    logic stage_start;
    logic score_busy;
    logic score_done;
    logic score_error;
    logic score_k_req_valid;
    logic score_k_req_ready;
    logic [KV_HEAD_INDEX_W-1 : 0] score_k_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] score_k_req_position;
    logic [DIM_INDEX_W-1 : 0] score_k_req_dim;
    logic score_k_rsp_valid;
    logic score_k_rsp_ready;
    logic signed [IN_WIDTH-1 : 0] score_k_rsp_data;
    logic [Q_ADDR_WIDTH-1 : 0] score_q_read_addr;
    logic signed [IN_WIDTH-1 : 0] score_q_read_data;
    logic score_valid;
    logic score_ready;
    logic [Q_HEAD_INDEX_W-1 : 0] score_q_head;
    logic [KV_HEAD_INDEX_W-1 : 0] score_kv_head;
    logic [POSITION_INDEX_W-1 : 0] score_position;
    logic signed [SCORE_WIDTH-1 : 0] score_raw;
    logic signed [SCORE_WIDTH-1 : 0] score_scaled;
    logic score_last;
    logic [31 : 0] score_k_request_count;
    logic [31 : 0] score_k_response_count;

    logic value_busy;
    logic value_done;
    logic value_error;
    logic value_saturation;
    logic value_v_req_valid;
    logic value_v_req_ready;
    logic [KV_HEAD_INDEX_W-1 : 0] value_v_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] value_v_req_position;
    logic [DIM_INDEX_W-1 : 0] value_v_req_dim;
    logic [23 : 0] value_v_req_prob;
    logic value_v_rsp_valid;
    logic value_v_rsp_ready;
    logic signed [IN_WIDTH-1 : 0] value_v_rsp_data;
    logic value_out_valid;
    logic value_out_ready;
    logic [Q_HEAD_INDEX_W-1 : 0] value_out_q_head;
    logic [DIM_INDEX_W-1 : 0] value_out_dim;
    logic signed [OUT_WIDTH-1 : 0] value_out_data;
    logic value_out_last;
    logic [31 : 0] value_score_count;
    logic [31 : 0] value_v_request_count;
    logic [31 : 0] value_v_response_count;
    logic [31 : 0] value_output_count;

    logic [LAYER_INDEX_W-1 : 0] metadata_layer_id;
    logic [CACHE_LENGTH_W-1 : 0] metadata_cache_length;
    logic signed [SCALE_WIDTH-1 : 0] metadata_score_scale_q0_31;
    logic runtime_context_valid_reg;
    logic [LAYER_INDEX_W-1 : 0] runtime_layer_id_reg;
    logic [POSITION_INDEX_W-1 : 0] runtime_position_reg;
    logic [ADDR_WIDTH-1 : 0] runtime_kv_cache_base_addr_reg;
    logic [CACHE_LENGTH_W-1 : 0] runtime_cache_length;
    logic [ADDR_WIDTH-1 : 0] effective_kv_cache_base_addr;

    logic [KV_HEAD_INDEX_W-1 : 0] cache_req_head;
    logic [POSITION_INDEX_W-1 : 0] cache_req_position;
    logic [DIM_INDEX_W-1 : 0] cache_req_dim;
    logic cache_addr_valid;
    logic [ADDR_WIDTH-1 : 0] cache_read_addr;
    logic [ADDR_WIDTH-1 : 0] cache_offset_bytes;

    logic score_done_seen;
    logic value_done_seen;
    logic score_error_seen;
    logic value_error_seen;
    logic value_saturation_seen;
    logic [31 : 0] attn_write_word_index;
    logic [31 : 0] attn_write_data_word;
    logic [Q_ADDR_WIDTH-1 : 0] attn_out_read_addr;
    logic signed [OUT_WIDTH-1 : 0] attn_out_read_data;
    logic [EXP_LUT_INDEX_W-1 : 0] value_exp_lut_read_addr;
    logic [EXP_WIDTH-1 : 0] value_exp_lut_read_data;

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]    = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]         = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]        = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_rank[desc_index]         = reader_desc_rank_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index] = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]    = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]       = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]         = reader_desc_dim0_flat[desc_index*32 +: 32];
            assign desc_dim1[desc_index]         = reader_desc_dim1_flat[desc_index*32 +: 32];
            assign desc_dim2[desc_index]         = reader_desc_dim2_flat[desc_index*32 +: 32];
            assign desc_dim3[desc_index]         = reader_desc_dim3_flat[desc_index*32 +: 32];
            assign desc_stride0_bytes[desc_index] = reader_desc_stride0_bytes_flat[desc_index*64 +: 64];
            assign desc_stride1_bytes[desc_index] = reader_desc_stride1_bytes_flat[desc_index*64 +: 64];
            assign desc_stride2_bytes[desc_index] = reader_desc_stride2_bytes_flat[desc_index*64 +: 64];
            assign desc_stride3_bytes[desc_index] = reader_desc_stride3_bytes_flat[desc_index*64 +: 64];
            assign desc_aux0[desc_index]         = reader_desc_aux0_flat[desc_index*32 +: 32];
            assign desc_aux1[desc_index]         = reader_desc_aux1_flat[desc_index*32 +: 32];
            assign desc_aux2[desc_index]         = reader_desc_aux2_flat[desc_index*32 +: 32];
            assign desc_aux3[desc_index]         = reader_desc_aux3_flat[desc_index*32 +: 32];
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
        .o_desc_flags_flat(),
        .o_desc_element_bits_flat(reader_desc_element_bits_flat),
        .o_desc_group_size_flat(),
        .o_desc_scale_tensor_id_flat(),
        .o_desc_base_addr_flat(reader_desc_base_addr_flat),
        .o_desc_nbytes_flat(reader_desc_nbytes_flat),
        .o_desc_dim0_flat(reader_desc_dim0_flat),
        .o_desc_dim1_flat(reader_desc_dim1_flat),
        .o_desc_dim2_flat(reader_desc_dim2_flat),
        .o_desc_dim3_flat(reader_desc_dim3_flat),
        .o_desc_stride0_bytes_flat(reader_desc_stride0_bytes_flat),
        .o_desc_stride1_bytes_flat(reader_desc_stride1_bytes_flat),
        .o_desc_stride2_bytes_flat(reader_desc_stride2_bytes_flat),
        .o_desc_stride3_bytes_flat(reader_desc_stride3_bytes_flat),
        .o_desc_aux0_flat(reader_desc_aux0_flat),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(reader_desc_aux2_flat),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    attention_score_stage #(
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CONTEXT(MAX_CONTEXT),
        .IN_WIDTH(IN_WIDTH),
        .SCORE_WIDTH(SCORE_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .USE_Q_READ_PORT(USE_BRAM_STREAMING),
        .Q_COUNT(Q_COUNT),
        .Q_ADDR_WIDTH(Q_ADDR_WIDTH),
        .Q_HEAD_INDEX_W(Q_HEAD_INDEX_W),
        .KV_HEAD_INDEX_W(KV_HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .CACHE_LENGTH_W(CACHE_LENGTH_W),
        .DIM_INDEX_W(DIM_INDEX_W)
    ) score_stage (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(stage_start),
        .i_cache_length(metadata_cache_length),
        .i_score_scale_q0_31(metadata_score_scale_q0_31),
        .i_q_rope_flat(q_rope_flat),
        .o_q_read_addr(score_q_read_addr),
        .i_q_read_data(score_q_read_data),
        .o_busy(score_busy),
        .o_done(score_done),
        .o_error(score_error),
        .o_k_req_valid(score_k_req_valid),
        .i_k_req_ready(score_k_req_ready),
        .o_k_req_kv_head(score_k_req_kv_head),
        .o_k_req_position(score_k_req_position),
        .o_k_req_dim(score_k_req_dim),
        .i_k_rsp_valid(score_k_rsp_valid),
        .o_k_rsp_ready(score_k_rsp_ready),
        .i_k_rsp_data(score_k_rsp_data),
        .o_score_valid(score_valid),
        .i_score_ready(score_ready),
        .o_score_q_head(score_q_head),
        .o_score_kv_head(score_kv_head),
        .o_score_position(score_position),
        .o_score_raw(score_raw),
        .o_score_scaled(score_scaled),
        .o_score_last(score_last),
        .o_k_request_count(score_k_request_count),
        .o_k_response_count(score_k_response_count),
        .o_score_count(o_score_count)
    );

    attention_softmax_value_stage #(
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CONTEXT(MAX_CONTEXT),
        .SCORE_WIDTH(SCORE_WIDTH),
        .VALUE_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .EXP_WIDTH(EXP_WIDTH),
        .EXP_LUT_SIZE(EXP_LUT_SIZE),
        .USE_EXP_READ_PORT(USE_BRAM_STREAMING),
        .Q_HEAD_INDEX_W(Q_HEAD_INDEX_W),
        .KV_HEAD_INDEX_W(KV_HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .CACHE_LENGTH_W(CACHE_LENGTH_W),
        .DIM_INDEX_W(DIM_INDEX_W)
    ) value_stage (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(stage_start),
        .i_cache_length(metadata_cache_length),
        .i_exp_lut_flat(exp_lut_flat),
        .o_exp_lut_rd_addr(value_exp_lut_read_addr),
        .i_exp_lut_rd_data(value_exp_lut_read_data),
        .i_score_valid(score_valid),
        .o_score_ready(score_ready),
        .i_score_q_head(score_q_head),
        .i_score_kv_head(score_kv_head),
        .i_score_position(score_position),
        .i_score_scaled(score_scaled),
        .i_score_last(score_last),
        .o_v_req_valid(value_v_req_valid),
        .i_v_req_ready(value_v_req_ready),
        .o_v_req_kv_head(value_v_req_kv_head),
        .o_v_req_position(value_v_req_position),
        .o_v_req_dim(value_v_req_dim),
        .o_v_req_prob(value_v_req_prob),
        .i_v_rsp_valid(value_v_rsp_valid),
        .o_v_rsp_ready(value_v_rsp_ready),
        .i_v_rsp_data(value_v_rsp_data),
        .o_out_valid(value_out_valid),
        .i_out_ready(value_out_ready),
        .o_out_q_head(value_out_q_head),
        .o_out_dim(value_out_dim),
        .o_out_data(value_out_data),
        .o_out_last(value_out_last),
        .o_busy(value_busy),
        .o_done(value_done),
        .o_error(value_error),
        .o_saturation(value_saturation),
        .o_score_count(value_score_count),
        .o_v_request_count(value_v_request_count),
        .o_v_response_count(value_v_response_count),
        .o_output_count(value_output_count)
    );

    kv_cache_addr_gen #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CONTEXT(MAX_CONTEXT),
        .ELEMENT_BYTES(MEM_DATA_BYTES),
        .LAYER_INDEX_W(LAYER_INDEX_W),
        .HEAD_INDEX_W(KV_HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .DIM_INDEX_W(DIM_INDEX_W)
    ) cache_addr_gen (
        .i_base_addr(effective_kv_cache_base_addr),
        .i_layer_id(metadata_layer_id),
        .i_kv_kind(active_cache_kind),
        .i_head_id(cache_req_head),
        .i_position(cache_req_position),
        .i_dim(cache_req_dim),
        .o_valid(cache_addr_valid),
        .o_offset_bytes(cache_offset_bytes),
        .o_byte_addr(cache_read_addr)
    );

    assign reader_start = (state == S_READER_START);
    assign stage_start = (state == S_STAGE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {3'd0, state};
    assign o_read_slot_debug = {2'd0, active_read_slot};
    assign o_saturation = value_saturation_seen || value_saturation;
    assign o_k_read_count = score_k_response_count;
    assign o_v_read_count = value_v_response_count;

    assign runtime_cache_length =
        {{(CACHE_LENGTH_W-POSITION_INDEX_W){1'b0}}, runtime_position_reg} + 1'b1;
    assign metadata_layer_id =
        runtime_context_valid_reg ?
        runtime_layer_id_reg :
        desc_aux1[SLOT_METADATA][LAYER_INDEX_W-1 : 0];
    assign metadata_cache_length =
        runtime_context_valid_reg ?
        runtime_cache_length :
        desc_aux2[SLOT_METADATA][CACHE_LENGTH_W-1 : 0];
    assign metadata_score_scale_q0_31 = desc_aux0[SLOT_METADATA];
    assign effective_kv_cache_base_addr =
        runtime_context_valid_reg ?
        runtime_kv_cache_base_addr_reg :
        desc_base_addr[SLOT_KV_CACHE];

    assign cache_req_head =
        (active_cache_kind == 1'b0) ? score_k_req_kv_head : value_v_req_kv_head;
    assign cache_req_position =
        (active_cache_kind == 1'b0) ? score_k_req_position : value_v_req_position;
    assign cache_req_dim =
        (active_cache_kind == 1'b0) ? score_k_req_dim : value_v_req_dim;

    assign score_k_req_ready =
        (state == S_CACHE_RD_REQ) &&
        (active_cache_kind == 1'b0) &&
        cache_addr_valid &&
        i_mem_rd_req_ready;
    assign value_v_req_ready =
        (state == S_CACHE_RD_REQ) &&
        (active_cache_kind == 1'b1) &&
        cache_addr_valid &&
        i_mem_rd_req_ready;

    assign score_k_rsp_valid =
        (state == S_CACHE_RD_DATA) &&
        (active_cache_kind == 1'b0) &&
        i_mem_rd_rsp_valid &&
        i_mem_rd_rsp_last;
    assign value_v_rsp_valid =
        (state == S_CACHE_RD_DATA) &&
        (active_cache_kind == 1'b1) &&
        i_mem_rd_rsp_valid &&
        i_mem_rd_rsp_last;
    assign score_k_rsp_data = i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
    assign value_v_rsp_data = i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
    assign value_out_ready = (state == S_STAGE_RUN);
    assign attn_out_read_addr =
        ((state == S_ATTN_WR_DATA) &&
         i_mem_wr_data_ready &&
         (attn_write_word_index < (Q_COUNT - 1))) ?
        Q_ADDR_WIDTH'(attn_write_word_index + 1'b1) :
        Q_ADDR_WIDTH'(attn_write_word_index);

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data = i_mem_rd_rsp_data;
    assign reader_rsp_last = i_mem_rd_rsp_last;

    always @* begin
        current_read_base_addr = '0;
        current_read_total_bytes = '0;
        case (active_read_slot)
            R_Q_ROPE: begin
                current_read_base_addr = desc_base_addr[SLOT_Q_ROPE];
                current_read_total_bytes = Q_BYTES;
            end
            R_EXP_LUT: begin
                current_read_base_addr = desc_base_addr[SLOT_EXP_LUT];
                current_read_total_bytes = EXP_LUT_BYTES;
            end
            default: begin
                current_read_base_addr = '0;
                current_read_total_bytes = '0;
            end
        endcase
    end

    assign current_read_offset_bytes = read_chunk_index * MAX_READ_BYTES;
    assign current_read_remaining_bytes = current_read_total_bytes - current_read_offset_bytes;
    assign current_read_len_bytes =
        (current_read_remaining_bytes > MAX_READ_BYTES) ?
        MAX_READ_BYTES_U16 :
        current_read_remaining_bytes[15 : 0];
    assign current_chunk_words = current_read_len_bytes >> 2;
    assign read_element_index = (read_chunk_index * CHUNK_WORDS) + read_word_index;
    assign read_protocol_error =
        (read_word_index >= current_chunk_words) ||
        (i_mem_rd_rsp_last && (read_word_index != (current_chunk_words - 1'b1))) ||
        (!i_mem_rd_rsp_last && (read_word_index == (current_chunk_words - 1'b1)));
    assign cache_read_protocol_error = !i_mem_rd_rsp_last;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_READ_REQ) ? 1'b1 :
        (state == S_CACHE_RD_REQ) ? cache_addr_valid :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        (state == S_READ_REQ) ? (current_read_base_addr + current_read_offset_bytes) :
        (state == S_CACHE_RD_REQ) ? cache_read_addr :
        '0;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_READ_REQ) ? current_read_len_bytes :
        (state == S_CACHE_RD_REQ) ? CACHE_READ_BYTES_U16 :
        16'd0;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_READ_DATA) ? 1'b1 :
        (state == S_CACHE_RD_DATA && active_cache_kind == 1'b0) ? score_k_rsp_ready :
        (state == S_CACHE_RD_DATA && active_cache_kind == 1'b1) ? value_v_rsp_ready :
        1'b0;

    generate
        if (USE_BRAM_STREAMING != 0) begin : gen_attn_write_bram
            assign attn_write_data_word = {
                {(MEM_DATA_WIDTH-OUT_WIDTH){
                    attn_out_read_data[OUT_WIDTH-1]
                }},
                attn_out_read_data
            };
        end
        else begin : gen_attn_write_flat
            assign attn_write_data_word = {
                {(MEM_DATA_WIDTH-OUT_WIDTH){
                    attn_out_flat[
                        (attn_write_word_index*OUT_WIDTH) +
                        (OUT_WIDTH-1)
                    ]
                }},
                attn_out_flat[
                    attn_write_word_index*OUT_WIDTH +: OUT_WIDTH
                ]
            };
        end
    endgenerate

    assign o_mem_wr_req_valid = (state == S_ATTN_WR_REQ);
    assign o_mem_wr_req_addr =
        (state == S_ATTN_WR_REQ) ? desc_base_addr[SLOT_ATTN_OUT] : '0;
    assign o_mem_wr_req_len_bytes =
        (state == S_ATTN_WR_REQ) ? ATTN_OUT_BYTES_U16 : 16'd0;
    assign o_mem_wr_data_valid = (state == S_ATTN_WR_DATA);
    assign o_mem_wr_data = (state == S_ATTN_WR_DATA) ? attn_write_data_word : 32'd0;
    assign o_mem_wr_data_last =
        (state == S_ATTN_WR_DATA) ? (attn_write_word_index == (Q_COUNT - 1)) : 1'b0;

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count < `QMAP_ATTN_SCORE_VALUE_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity < `QMAP_ATTN_SCORE_VALUE_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_ATTN_SV_METADATA) ||
            (desc_tensor_id[SLOT_Q_ROPE] != `QMAP_TENSOR_ID_ATTN_SV_Q_ROPE) ||
            (desc_tensor_id[SLOT_KV_CACHE] != `QMAP_TENSOR_ID_ATTN_SV_KV_CACHE) ||
            (desc_tensor_id[SLOT_EXP_LUT] != `QMAP_TENSOR_ID_ATTN_SV_EXP_LUT) ||
            (desc_tensor_id[SLOT_ATTN_OUT] != `QMAP_TENSOR_ID_ATTN_SV_ATTN_OUT) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_Q_ROPE] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_KV_CACHE] != `QMAP_ROLE_KV_CACHE) ||
            (desc_role[SLOT_EXP_LUT] != `QMAP_ROLE_PARAMETER) ||
            (desc_role[SLOT_ATTN_OUT] != `QMAP_ROLE_OUTPUT) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_Q_ROPE] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_KV_CACHE] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_EXP_LUT] != `QMAP_DTYPE_U32_Q0_20) ||
            (desc_dtype[SLOT_ATTN_OUT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_Q_ROPE] != 32'd1) ||
            (desc_rank[SLOT_KV_CACHE] != 32'd4) ||
            (desc_rank[SLOT_EXP_LUT] != 32'd1) ||
            (desc_rank[SLOT_ATTN_OUT] != 32'd1) ||
            (desc_element_bits[SLOT_METADATA] != 32'd32) ||
            (desc_element_bits[SLOT_Q_ROPE] != IN_WIDTH) ||
            (desc_element_bits[SLOT_KV_CACHE] != IN_WIDTH) ||
            (desc_element_bits[SLOT_EXP_LUT] != EXP_WIDTH) ||
            (desc_element_bits[SLOT_ATTN_OUT] != OUT_WIDTH) ||
            (desc_dim0[SLOT_METADATA] != 32'd16) ||
            (desc_dim1[SLOT_METADATA] != 32'd0) ||
            (desc_dim2[SLOT_METADATA] != 32'd0) ||
            (desc_dim3[SLOT_METADATA] != 32'd0) ||
            (desc_dim0[SLOT_Q_ROPE] != Q_COUNT) ||
            (desc_dim1[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim2[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim3[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim0[SLOT_KV_CACHE] != 32'd2) ||
            (desc_dim1[SLOT_KV_CACHE] != NUM_KV_HEADS) ||
            (desc_dim2[SLOT_KV_CACHE] != MAX_CONTEXT) ||
            (desc_dim3[SLOT_KV_CACHE] != HEAD_DIM) ||
            (desc_dim0[SLOT_EXP_LUT] != EXP_LUT_SIZE) ||
            (desc_dim1[SLOT_EXP_LUT] != 32'd0) ||
            (desc_dim2[SLOT_EXP_LUT] != 32'd0) ||
            (desc_dim3[SLOT_EXP_LUT] != 32'd0) ||
            (desc_dim0[SLOT_ATTN_OUT] != Q_COUNT) ||
            (desc_dim1[SLOT_ATTN_OUT] != 32'd0) ||
            (desc_dim2[SLOT_ATTN_OUT] != 32'd0) ||
            (desc_dim3[SLOT_ATTN_OUT] != 32'd0) ||
            (desc_stride0_bytes[SLOT_METADATA] != 64'd4) ||
            (desc_stride1_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride2_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride3_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride0_bytes[SLOT_Q_ROPE] != 64'd4) ||
            (desc_stride1_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_stride2_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_stride3_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_stride0_bytes[SLOT_KV_CACHE] != KV_KIND_STRIDE_BYTES) ||
            (desc_stride1_bytes[SLOT_KV_CACHE] != KV_HEAD_STRIDE_BYTES) ||
            (desc_stride2_bytes[SLOT_KV_CACHE] != KV_POSITION_STRIDE_BYTES) ||
            (desc_stride3_bytes[SLOT_KV_CACHE] != 64'd4) ||
            (desc_stride0_bytes[SLOT_EXP_LUT] != 64'd4) ||
            (desc_stride1_bytes[SLOT_EXP_LUT] != 64'd0) ||
            (desc_stride2_bytes[SLOT_EXP_LUT] != 64'd0) ||
            (desc_stride3_bytes[SLOT_EXP_LUT] != 64'd0) ||
            (desc_stride0_bytes[SLOT_ATTN_OUT] != 64'd4) ||
            (desc_stride1_bytes[SLOT_ATTN_OUT] != 64'd0) ||
            (desc_stride2_bytes[SLOT_ATTN_OUT] != 64'd0) ||
            (desc_stride3_bytes[SLOT_ATTN_OUT] != 64'd0) ||
            (desc_base_addr[SLOT_Q_ROPE][1 : 0] != 2'b00) ||
            (desc_base_addr[SLOT_EXP_LUT][1 : 0] != 2'b00) ||
            (desc_base_addr[SLOT_ATTN_OUT][1 : 0] != 2'b00) ||
            (effective_kv_cache_base_addr[1 : 0] != 2'b00) ||
            (desc_nbytes[SLOT_METADATA] != 64'd64) ||
            (desc_nbytes[SLOT_Q_ROPE] != Q_BYTES) ||
            (desc_nbytes[SLOT_KV_CACHE] != KV_CACHE_BYTES) ||
            (desc_nbytes[SLOT_EXP_LUT] != EXP_LUT_BYTES) ||
            (desc_nbytes[SLOT_ATTN_OUT] != ATTN_OUT_BYTES) ||
            ((!runtime_context_valid_reg) &&
             ((desc_aux1[SLOT_METADATA] >= NUM_LAYERS) ||
              (desc_aux2[SLOT_METADATA] == 32'd0) ||
              (desc_aux2[SLOT_METADATA] > MAX_CONTEXT))) ||
            (runtime_context_valid_reg &&
             ((runtime_layer_id_reg >= NUM_LAYERS) ||
              (runtime_position_reg >= MAX_CONTEXT) ||
              (desc_aux1[SLOT_METADATA] != runtime_layer_id_reg)))) begin
            validate_error = 1'b1;
        end
    end

    function automatic read_slot_t next_read_slot(input read_slot_t slot);
        begin
            case (slot)
                R_Q_ROPE: next_read_slot = R_EXP_LUT;
                default: next_read_slot = R_DONE;
            endcase
        end
    endfunction

    // Board-mode Q RoPE and attention output stores. Both are written in full
    // and use synchronous reads, so no reset or whole-array assignment is
    // present in these RAM inference processes.
    always_ff @(posedge i_clk) begin
        if ((USE_BRAM_STREAMING != 0) &&
            (state == S_READ_DATA) &&
            i_mem_rd_rsp_valid &&
            !read_protocol_error &&
            (active_read_slot == R_Q_ROPE)) begin
            q_rope_mem[
                read_element_index[Q_ADDR_WIDTH-1 : 0]
            ] <= i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
        end
        score_q_read_data <= q_rope_mem[score_q_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if ((USE_BRAM_STREAMING != 0) &&
            (state == S_READ_DATA) &&
            i_mem_rd_rsp_valid &&
            !read_protocol_error &&
            (active_read_slot == R_EXP_LUT)) begin
            exp_lut_mem[
                read_element_index[EXP_LUT_INDEX_W-1 : 0]
            ] <= i_mem_rd_rsp_data[EXP_WIDTH-1 : 0];
        end
        value_exp_lut_read_data <=
            exp_lut_mem[value_exp_lut_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if ((USE_BRAM_STREAMING != 0) &&
            value_out_valid &&
            value_out_ready) begin
            attn_out_mem[
                Q_ADDR_WIDTH'(
                    (value_out_q_head * HEAD_DIM) + value_out_dim
                )
            ] <= value_out_data;
        end
        attn_out_read_data <= attn_out_mem[attn_out_read_addr];
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_read_slot <= R_Q_ROPE;
            active_cache_kind <= 1'b0;
            if (USE_BRAM_STREAMING == 0) begin
                q_rope_flat <= '0;
                exp_lut_flat <= '0;
                attn_out_flat <= '0;
            end
            read_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            score_done_seen <= 1'b0;
            value_done_seen <= 1'b0;
            score_error_seen <= 1'b0;
            value_error_seen <= 1'b0;
            value_saturation_seen <= 1'b0;
            attn_write_word_index <= 32'd0;
            runtime_context_valid_reg <= 1'b0;
            runtime_layer_id_reg <= '0;
            runtime_position_reg <= '0;
            runtime_kv_cache_base_addr_reg <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_attn_out_capture_count <= 32'd0;
            o_attn_out_write_word_count <= 32'd0;
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
            end
            if (score_done) begin
                score_done_seen <= 1'b1;
                score_error_seen <= score_error_seen || score_error;
            end
            if (value_done) begin
                value_done_seen <= 1'b1;
                value_error_seen <= value_error_seen || value_error;
                value_saturation_seen <= value_saturation_seen || value_saturation;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_READER_START;
                        active_read_slot <= R_Q_ROPE;
                        active_cache_kind <= 1'b0;
                        if (USE_BRAM_STREAMING == 0) begin
                            q_rope_flat <= '0;
                            exp_lut_flat <= '0;
                            attn_out_flat <= '0;
                        end
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        score_done_seen <= 1'b0;
                        value_done_seen <= 1'b0;
                        score_error_seen <= 1'b0;
                        value_error_seen <= 1'b0;
                        value_saturation_seen <= 1'b0;
                        attn_write_word_index <= 32'd0;
                        runtime_context_valid_reg <= (i_runtime_context_valid === 1'b1);
                        runtime_layer_id_reg <= i_runtime_layer_id;
                        runtime_position_reg <= i_runtime_position;
                        runtime_kv_cache_base_addr_reg <= i_runtime_kv_cache_base_addr;
                        o_error <= 1'b0;
                        o_attn_out_capture_count <= 32'd0;
                        o_attn_out_write_word_count <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
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
                        active_read_slot <= R_Q_ROPE;
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
                            case (active_read_slot)
                                R_Q_ROPE: begin
                                    if (USE_BRAM_STREAMING == 0) begin
                                        q_rope_flat[
                                            read_element_index*IN_WIDTH +:
                                            IN_WIDTH
                                        ] <= i_mem_rd_rsp_data[
                                            IN_WIDTH-1 : 0
                                        ];
                                    end
                                end
                                R_EXP_LUT: begin
                                    if (USE_BRAM_STREAMING == 0) begin
                                        exp_lut_flat[
                                            read_element_index*EXP_WIDTH +:
                                            EXP_WIDTH
                                        ] <= i_mem_rd_rsp_data[
                                            EXP_WIDTH-1 : 0
                                        ];
                                    end
                                end
                                default: begin
                                end
                            endcase

                            if (i_mem_rd_rsp_last) begin
                                if ((current_read_offset_bytes + current_read_len_bytes) >= current_read_total_bytes) begin
                                    if (next_read_slot(active_read_slot) == R_DONE) begin
                                        state <= S_STAGE_START;
                                    end
                                    else begin
                                        active_read_slot <= next_read_slot(active_read_slot);
                                        read_chunk_index <= 32'd0;
                                        read_word_index <= 32'd0;
                                        state <= S_READ_REQ;
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
                    score_done_seen <= 1'b0;
                    value_done_seen <= 1'b0;
                    score_error_seen <= 1'b0;
                    value_error_seen <= 1'b0;
                    value_saturation_seen <= 1'b0;
                    o_attn_out_capture_count <= 32'd0;
                    state <= S_STAGE_RUN;
                end

                S_STAGE_RUN: begin
                    if (value_out_valid && value_out_ready) begin
                        if (USE_BRAM_STREAMING == 0) begin
                            attn_out_flat[
                                (value_out_q_head*HEAD_DIM +
                                 value_out_dim)*OUT_WIDTH +:
                                OUT_WIDTH
                            ] <= value_out_data;
                        end
                        o_attn_out_capture_count <= o_attn_out_capture_count + 1'b1;
                    end

                    if (score_k_req_valid) begin
                        active_cache_kind <= 1'b0;
                        state <= S_CACHE_RD_REQ;
                    end
                    else if (value_v_req_valid) begin
                        active_cache_kind <= 1'b1;
                        state <= S_CACHE_RD_REQ;
                    end
                    else if (value_done_seen) begin
                        if (score_error_seen ||
                            value_error_seen ||
                            value_saturation_seen ||
                            (o_attn_out_capture_count != Q_COUNT)) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            attn_write_word_index <= 32'd0;
                            state <= S_ATTN_WR_REQ;
                        end
                    end
                end

                S_CACHE_RD_REQ: begin
                    if (!cache_addr_valid) begin
                        o_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else if (i_mem_rd_req_ready) begin
                        state <= S_CACHE_RD_DATA;
                    end
                end

                S_CACHE_RD_DATA: begin
                    if (i_mem_rd_rsp_valid && o_mem_rd_rsp_ready) begin
                        if (cache_read_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_STAGE_RUN;
                        end
                    end
                end

                S_ATTN_WR_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        attn_write_word_index <= 32'd0;
                        state <= S_ATTN_WR_DATA;
                    end
                end

                S_ATTN_WR_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        o_attn_out_write_word_count <= o_attn_out_write_word_count + 1'b1;
                        if (o_mem_wr_data_last) begin
                            state <= S_ATTN_WR_WAIT;
                        end
                        else begin
                            attn_write_word_index <= attn_write_word_index + 1'b1;
                        end
                    end
                end

                S_ATTN_WR_WAIT: begin
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
