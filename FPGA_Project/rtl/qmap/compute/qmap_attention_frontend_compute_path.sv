`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed attention front-end:
//
//   q_flat/k_flat/v_flat + q/k gamma + RoPE cos/sin
//       -> q/k norm + RoPE
//       -> write K RoPE and V into KV cache
//       -> write Q RoPE to an activation descriptor
//
// This is the first memory-mapped per-layer body boundary after the passing
// QKV projection write-back path.
module qmap_attention_frontend_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 10,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int IN_WIDTH         = 24,
    parameter int GAMMA_WIDTH      = 16,
    parameter int TRIG_WIDTH       = 16,
    parameter int OUT_WIDTH        = 24,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024,
    parameter int USE_BRAM_STREAMING = 1,
    parameter int LAYER_INDEX_W    = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int HEAD_INDEX_W     = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,
    input  wire logic                         i_runtime_context_valid,
    input  wire logic [LAYER_INDEX_W-1 : 0]   i_runtime_layer_id,
    input  wire logic [POSITION_INDEX_W-1 : 0] i_runtime_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_runtime_kv_cache_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic                              o_saturation,
    output logic                              o_norm_saturation,
    output logic                              o_rope_saturation,
    output logic [31 : 0]                     o_cache_write_count,
    output logic [31 : 0]                     o_q_rope_write_word_count,
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
    localparam int SLOT_Q_FLAT   = 1;
    localparam int SLOT_K_FLAT   = 2;
    localparam int SLOT_V_FLAT   = 3;
    localparam int SLOT_Q_GAMMA  = 4;
    localparam int SLOT_K_GAMMA  = 5;
    localparam int SLOT_COS      = 6;
    localparam int SLOT_SIN      = 7;
    localparam int SLOT_KV_CACHE = 8;
    localparam int SLOT_Q_ROPE   = 9;

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int CHUNK_WORDS    = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int Q_COUNT        = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT       = NUM_KV_HEADS * HEAD_DIM;
    localparam int Q_ADDR_WIDTH   =
        (Q_COUNT <= 1) ? 1 : $clog2(Q_COUNT);
    localparam int KV_ADDR_WIDTH  =
        (KV_COUNT <= 1) ? 1 : $clog2(KV_COUNT);
    localparam int PARAM_ADDR_WIDTH =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM);
    localparam int Q_BYTES        = Q_COUNT * MEM_DATA_BYTES;
    localparam int KV_BYTES       = KV_COUNT * MEM_DATA_BYTES;
    localparam int PARAM_BYTES    = HEAD_DIM * MEM_DATA_BYTES;
    localparam int Q_ROPE_BYTES   = Q_COUNT * MEM_DATA_BYTES;
    localparam int METADATA_WORDS = 16;
    localparam int METADATA_BYTES = METADATA_WORDS * MEM_DATA_BYTES;
    localparam int ROPE_TABLE_BYTES = MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_CACHE_BYTES = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_KIND_STRIDE_BYTES = NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_HEAD_STRIDE_BYTES = MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES;
    localparam int KV_POSITION_STRIDE_BYTES = HEAD_DIM * MEM_DATA_BYTES;
    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] Q_ROPE_BYTES_U16 = Q_ROPE_BYTES;

    typedef enum logic [4 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_READ_REQ,
        S_READ_DATA,
        S_STAGE_START,
        S_STAGE_RUN,
        S_CACHE_WR_REQ,
        S_CACHE_WR_DATA,
        S_CACHE_WR_WAIT,
        S_QROPE_WR_REQ,
        S_QROPE_WR_DATA,
        S_QROPE_WR_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [3 : 0] {
        R_Q_FLAT,
        R_K_FLAT,
        R_V_FLAT,
        R_Q_GAMMA,
        R_K_GAMMA,
        R_COS,
        R_SIN,
        R_DONE
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
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
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
    logic [31 : 0] desc_aux1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux3 [0 : DESCRIPTOR_SLOTS-1];

    logic [Q_COUNT*IN_WIDTH-1 : 0]      q_flat;
    logic [KV_COUNT*IN_WIDTH-1 : 0]     k_flat;
    logic [KV_COUNT*IN_WIDTH-1 : 0]     v_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]  q_gamma_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]  k_gamma_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]   cos_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]   sin_flat;

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
    logic validate_error;

    logic stage_start;
    logic stage_busy;
    logic stage_done;
    logic stage_error;
    logic stage_saturation;
    logic stage_norm_saturation;
    logic stage_rope_saturation;
    logic [Q_COUNT*OUT_WIDTH-1 : 0] q_rope_flat;
    logic [KV_COUNT*OUT_WIDTH-1 : 0] k_rope_flat;
    logic cache_wr_valid;
    logic cache_wr_ready;
    logic [ADDR_WIDTH-1 : 0] cache_wr_addr;
    logic [31 : 0] cache_wr_data;
    logic cache_wr_last;
    logic cache_wr_kind;
    logic [HEAD_INDEX_W-1 : 0] cache_wr_head;
    logic [DIM_INDEX_W-1 : 0] cache_wr_dim;
    logic [31 : 0] stage_cache_write_count;
    logic stage_done_seen;
    logic stage_error_seen;

    logic [ADDR_WIDTH-1 : 0] cache_wr_addr_reg;
    logic [31 : 0] cache_wr_data_reg;
    logic qrope_write_req_seen;
    logic [31 : 0] qrope_write_word_index;
    logic [31 : 0] qrope_write_data_word;
    logic [Q_ADDR_WIDTH-1 : 0] stage_qrope_read_addr;
    logic signed [OUT_WIDTH-1 : 0] stage_qrope_read_data;
    logic [LAYER_INDEX_W-1 : 0] metadata_layer_id;
    logic [POSITION_INDEX_W-1 : 0] metadata_position;
    logic active_runtime_context_valid;
    logic [LAYER_INDEX_W-1 : 0] active_runtime_layer_id;
    logic [POSITION_INDEX_W-1 : 0] active_runtime_position;
    logic [ADDR_WIDTH-1 : 0] active_runtime_kv_cache_base_addr;
    logic [LAYER_INDEX_W-1 : 0] effective_layer_id;
    logic [POSITION_INDEX_W-1 : 0] effective_position;
    logic [ADDR_WIDTH-1 : 0] effective_kv_cache_base_addr;
    logic [63 : 0] runtime_rope_row_offset_bytes;

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
            assign desc_aux1[desc_index]         = reader_desc_aux1_flat[desc_index*32 +: 32];
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
        .o_desc_flags_flat(reader_desc_flags_flat),
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
        .o_desc_aux0_flat(),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    generate
        if (USE_BRAM_STREAMING != 0) begin : gen_bram_frontend_stage
            attention_frontend_bram_stage #(
                .ADDR_WIDTH(ADDR_WIDTH),
                .DATA_WIDTH(MEM_DATA_WIDTH),
                .NUM_LAYERS(NUM_LAYERS),
                .NUM_Q_HEADS(NUM_Q_HEADS),
                .NUM_KV_HEADS(NUM_KV_HEADS),
                .HEAD_DIM(HEAD_DIM),
                .MAX_CONTEXT(MAX_CONTEXT),
                .IN_WIDTH(IN_WIDTH),
                .GAMMA_WIDTH(GAMMA_WIDTH),
                .TRIG_WIDTH(TRIG_WIDTH),
                .OUT_WIDTH(OUT_WIDTH),
                .Q_COUNT(Q_COUNT),
                .KV_COUNT(KV_COUNT),
                .Q_ADDR_WIDTH(Q_ADDR_WIDTH),
                .KV_ADDR_WIDTH(KV_ADDR_WIDTH),
                .PARAM_ADDR_WIDTH(PARAM_ADDR_WIDTH),
                .LAYER_INDEX_W(LAYER_INDEX_W),
                .POSITION_INDEX_W(POSITION_INDEX_W),
                .HEAD_INDEX_W(HEAD_INDEX_W),
                .DIM_INDEX_W(DIM_INDEX_W)
            ) frontend_stage (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_q_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_Q_FLAT)
                ),
                .i_q_wr_addr(
                    read_element_index[Q_ADDR_WIDTH-1 : 0]
                ),
                .i_q_wr_data(i_mem_rd_rsp_data[IN_WIDTH-1 : 0]),
                .i_k_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_K_FLAT)
                ),
                .i_k_wr_addr(
                    read_element_index[KV_ADDR_WIDTH-1 : 0]
                ),
                .i_k_wr_data(i_mem_rd_rsp_data[IN_WIDTH-1 : 0]),
                .i_v_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_V_FLAT)
                ),
                .i_v_wr_addr(
                    read_element_index[KV_ADDR_WIDTH-1 : 0]
                ),
                .i_v_wr_data(i_mem_rd_rsp_data[IN_WIDTH-1 : 0]),
                .i_q_gamma_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_Q_GAMMA)
                ),
                .i_q_gamma_wr_addr(
                    read_element_index[PARAM_ADDR_WIDTH-1 : 0]
                ),
                .i_q_gamma_wr_data(
                    i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0]
                ),
                .i_k_gamma_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_K_GAMMA)
                ),
                .i_k_gamma_wr_addr(
                    read_element_index[PARAM_ADDR_WIDTH-1 : 0]
                ),
                .i_k_gamma_wr_data(
                    i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0]
                ),
                .i_cos_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_COS)
                ),
                .i_cos_wr_addr(
                    read_element_index[PARAM_ADDR_WIDTH-1 : 0]
                ),
                .i_cos_wr_data(
                    i_mem_rd_rsp_data[TRIG_WIDTH-1 : 0]
                ),
                .i_sin_wr_valid(
                    (state == S_READ_DATA) &&
                    i_mem_rd_rsp_valid &&
                    !read_protocol_error &&
                    (active_read_slot == R_SIN)
                ),
                .i_sin_wr_addr(
                    read_element_index[PARAM_ADDR_WIDTH-1 : 0]
                ),
                .i_sin_wr_data(
                    i_mem_rd_rsp_data[TRIG_WIDTH-1 : 0]
                ),
                .i_start(stage_start),
                .i_cache_base_addr(effective_kv_cache_base_addr),
                .i_layer_id(effective_layer_id),
                .i_position(effective_position),
                .o_busy(stage_busy),
                .o_done(stage_done),
                .o_error(stage_error),
                .o_saturation(stage_saturation),
                .o_norm_saturation(stage_norm_saturation),
                .o_rope_saturation(stage_rope_saturation),
                .o_cache_wr_valid(cache_wr_valid),
                .i_cache_wr_ready(cache_wr_ready),
                .o_cache_wr_addr(cache_wr_addr),
                .o_cache_wr_data(cache_wr_data),
                .o_cache_wr_last(cache_wr_last),
                .o_cache_wr_kind(cache_wr_kind),
                .o_cache_wr_head(cache_wr_head),
                .o_cache_wr_dim(cache_wr_dim),
                .o_cache_write_count(stage_cache_write_count),
                .i_qrope_rd_addr(stage_qrope_read_addr),
                .o_qrope_rd_data(stage_qrope_read_data)
            );

            assign q_rope_flat = '0;
            assign k_rope_flat = '0;
            assign qrope_write_data_word = {
                {(MEM_DATA_WIDTH-OUT_WIDTH){
                    stage_qrope_read_data[OUT_WIDTH-1]
                }},
                stage_qrope_read_data
            };
        end
        else begin : gen_flat_frontend_stage
            qk_norm_rope_kv_cache_stage #(
                .ADDR_WIDTH(ADDR_WIDTH),
                .DATA_WIDTH(MEM_DATA_WIDTH),
                .NUM_LAYERS(NUM_LAYERS),
                .NUM_Q_HEADS(NUM_Q_HEADS),
                .NUM_KV_HEADS(NUM_KV_HEADS),
                .HEAD_DIM(HEAD_DIM),
                .MAX_CONTEXT(MAX_CONTEXT),
                .IN_WIDTH(IN_WIDTH),
                .GAMMA_WIDTH(GAMMA_WIDTH),
                .TRIG_WIDTH(TRIG_WIDTH),
                .OUT_WIDTH(OUT_WIDTH)
            ) frontend_stage (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_start(stage_start),
                .i_cache_base_addr(effective_kv_cache_base_addr),
                .i_layer_id(effective_layer_id),
                .i_position(effective_position),
                .i_q_flat(q_flat),
                .i_k_flat(k_flat),
                .i_v_flat(v_flat),
                .i_q_gamma_flat(q_gamma_flat),
                .i_k_gamma_flat(k_gamma_flat),
                .i_cos_flat(cos_flat),
                .i_sin_flat(sin_flat),
                .o_busy(stage_busy),
                .o_done(stage_done),
                .o_error(stage_error),
                .o_saturation(stage_saturation),
                .o_norm_saturation(stage_norm_saturation),
                .o_rope_saturation(stage_rope_saturation),
                .o_q_rope_flat(q_rope_flat),
                .o_k_rope_flat(k_rope_flat),
                .o_cache_wr_valid(cache_wr_valid),
                .i_cache_wr_ready(cache_wr_ready),
                .o_cache_wr_addr(cache_wr_addr),
                .o_cache_wr_data(cache_wr_data),
                .o_cache_wr_last(cache_wr_last),
                .o_cache_wr_kind(cache_wr_kind),
                .o_cache_wr_head(cache_wr_head),
                .o_cache_wr_dim(cache_wr_dim),
                .o_cache_write_count(stage_cache_write_count)
            );

            assign stage_qrope_read_data = '0;
            assign qrope_write_data_word = {
                {(MEM_DATA_WIDTH-OUT_WIDTH){
                    q_rope_flat[
                        (qrope_write_word_index*OUT_WIDTH) +
                        (OUT_WIDTH-1)
                    ]
                }},
                q_rope_flat[
                    qrope_write_word_index*OUT_WIDTH +: OUT_WIDTH
                ]
            };
        end
    endgenerate

    assign reader_start = (state == S_READER_START);
    assign stage_start = (state == S_STAGE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {3'd0, state};
    assign o_read_slot_debug = active_read_slot;
    assign metadata_layer_id = desc_aux1[SLOT_METADATA][LAYER_INDEX_W-1 : 0];
    assign metadata_position = desc_aux3[SLOT_METADATA][POSITION_INDEX_W-1 : 0];
    assign effective_layer_id = active_runtime_context_valid ? active_runtime_layer_id : metadata_layer_id;
    assign effective_position = active_runtime_context_valid ? active_runtime_position : metadata_position;
    assign effective_kv_cache_base_addr =
        active_runtime_context_valid ? active_runtime_kv_cache_base_addr : desc_base_addr[SLOT_KV_CACHE];
    assign runtime_rope_row_offset_bytes =
        active_runtime_context_valid ? (active_runtime_position * PARAM_BYTES) : 64'd0;
    assign stage_qrope_read_addr =
        ((state == S_QROPE_WR_DATA) &&
         i_mem_wr_data_ready &&
         (qrope_write_word_index < (Q_COUNT - 1))) ?
        Q_ADDR_WIDTH'(qrope_write_word_index + 1'b1) :
        Q_ADDR_WIDTH'(qrope_write_word_index);
    assign o_cache_write_count = stage_cache_write_count;
    assign o_saturation = stage_saturation;
    assign o_norm_saturation = stage_norm_saturation;
    assign o_rope_saturation = stage_rope_saturation;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data = i_mem_rd_rsp_data;
    assign reader_rsp_last = i_mem_rd_rsp_last;

    always @* begin
        current_read_base_addr = '0;
        current_read_total_bytes = '0;
        case (active_read_slot)
            R_Q_FLAT: begin
                current_read_base_addr = desc_base_addr[SLOT_Q_FLAT];
                current_read_total_bytes = Q_BYTES;
            end
            R_K_FLAT: begin
                current_read_base_addr = desc_base_addr[SLOT_K_FLAT];
                current_read_total_bytes = KV_BYTES;
            end
            R_V_FLAT: begin
                current_read_base_addr = desc_base_addr[SLOT_V_FLAT];
                current_read_total_bytes = KV_BYTES;
            end
            R_Q_GAMMA: begin
                current_read_base_addr = desc_base_addr[SLOT_Q_GAMMA];
                current_read_total_bytes = PARAM_BYTES;
            end
            R_K_GAMMA: begin
                current_read_base_addr = desc_base_addr[SLOT_K_GAMMA];
                current_read_total_bytes = PARAM_BYTES;
            end
            R_COS: begin
                current_read_base_addr = desc_base_addr[SLOT_COS] + runtime_rope_row_offset_bytes;
                current_read_total_bytes = PARAM_BYTES;
            end
            R_SIN: begin
                current_read_base_addr = desc_base_addr[SLOT_SIN] + runtime_rope_row_offset_bytes;
                current_read_total_bytes = PARAM_BYTES;
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

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_READ_REQ) ? 1'b1 :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        (state == S_READ_REQ) ? (current_read_base_addr + current_read_offset_bytes) :
        '0;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_READ_REQ) ? current_read_len_bytes :
        16'd0;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_READ_DATA) ? 1'b1 :
        1'b0;

    assign cache_wr_ready = (state == S_CACHE_WR_REQ) && i_mem_wr_req_ready;
    assign o_mem_wr_req_valid =
        (state == S_CACHE_WR_REQ) ? cache_wr_valid :
        (state == S_QROPE_WR_REQ) ? 1'b1 :
        1'b0;
    assign o_mem_wr_req_addr =
        (state == S_CACHE_WR_REQ) ? cache_wr_addr :
        (state == S_QROPE_WR_REQ) ? desc_base_addr[SLOT_Q_ROPE] :
        '0;
    assign o_mem_wr_req_len_bytes =
        (state == S_CACHE_WR_REQ) ? 16'd4 :
        (state == S_QROPE_WR_REQ) ? Q_ROPE_BYTES_U16 :
        16'd0;
    assign o_mem_wr_data_valid =
        (state == S_CACHE_WR_DATA) || (state == S_QROPE_WR_DATA);
    assign o_mem_wr_data =
        (state == S_CACHE_WR_DATA) ? cache_wr_data_reg :
        (state == S_QROPE_WR_DATA) ? qrope_write_data_word :
        32'd0;
    assign o_mem_wr_data_last =
        (state == S_CACHE_WR_DATA) ? 1'b1 :
        (state == S_QROPE_WR_DATA) ? (qrope_write_word_index == (Q_COUNT - 1)) :
        1'b0;

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count < `QMAP_ATTN_FRONTEND_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity < `QMAP_ATTN_FRONTEND_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_ATTN_METADATA) ||
            (desc_tensor_id[SLOT_Q_FLAT] != `QMAP_TENSOR_ID_ATTN_Q_FLAT) ||
            (desc_tensor_id[SLOT_K_FLAT] != `QMAP_TENSOR_ID_ATTN_K_FLAT) ||
            (desc_tensor_id[SLOT_V_FLAT] != `QMAP_TENSOR_ID_ATTN_V_FLAT) ||
            (desc_tensor_id[SLOT_Q_GAMMA] != `QMAP_TENSOR_ID_ATTN_Q_GAMMA) ||
            (desc_tensor_id[SLOT_K_GAMMA] != `QMAP_TENSOR_ID_ATTN_K_GAMMA) ||
            (desc_tensor_id[SLOT_COS] != `QMAP_TENSOR_ID_ATTN_ROPE_COS) ||
            (desc_tensor_id[SLOT_SIN] != `QMAP_TENSOR_ID_ATTN_ROPE_SIN) ||
            (desc_tensor_id[SLOT_KV_CACHE] != `QMAP_TENSOR_ID_ATTN_KV_CACHE) ||
            (desc_tensor_id[SLOT_Q_ROPE] != `QMAP_TENSOR_ID_ATTN_Q_ROPE) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_Q_FLAT] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_K_FLAT] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_V_FLAT] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_Q_GAMMA] != `QMAP_ROLE_PARAMETER) ||
            (desc_role[SLOT_K_GAMMA] != `QMAP_ROLE_PARAMETER) ||
            (desc_role[SLOT_COS] != `QMAP_ROLE_ROPE_TABLE) ||
            (desc_role[SLOT_SIN] != `QMAP_ROLE_ROPE_TABLE) ||
            (desc_role[SLOT_KV_CACHE] != `QMAP_ROLE_KV_CACHE) ||
            (desc_role[SLOT_Q_ROPE] != `QMAP_ROLE_OUTPUT) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_Q_FLAT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_K_FLAT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_V_FLAT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_Q_GAMMA] != `QMAP_DTYPE_I16_Q8_7) ||
            (desc_dtype[SLOT_K_GAMMA] != `QMAP_DTYPE_I16_Q8_7) ||
            (desc_dtype[SLOT_COS] != `QMAP_DTYPE_I16_Q1_15) ||
            (desc_dtype[SLOT_SIN] != `QMAP_DTYPE_I16_Q1_15) ||
            (desc_dtype[SLOT_KV_CACHE] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_Q_ROPE] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_Q_FLAT] != 32'd1) ||
            (desc_rank[SLOT_K_FLAT] != 32'd1) ||
            (desc_rank[SLOT_V_FLAT] != 32'd1) ||
            (desc_rank[SLOT_Q_GAMMA] != 32'd1) ||
            (desc_rank[SLOT_K_GAMMA] != 32'd1) ||
            (desc_rank[SLOT_Q_ROPE] != 32'd1) ||
            (desc_rank[SLOT_KV_CACHE] != 32'd4) ||
            (desc_element_bits[SLOT_METADATA] != 32) ||
            (desc_element_bits[SLOT_Q_FLAT] != IN_WIDTH) ||
            (desc_element_bits[SLOT_K_FLAT] != IN_WIDTH) ||
            (desc_element_bits[SLOT_V_FLAT] != IN_WIDTH) ||
            (desc_element_bits[SLOT_Q_GAMMA] != GAMMA_WIDTH) ||
            (desc_element_bits[SLOT_K_GAMMA] != GAMMA_WIDTH) ||
            (desc_element_bits[SLOT_COS] != TRIG_WIDTH) ||
            (desc_element_bits[SLOT_SIN] != TRIG_WIDTH) ||
            (desc_element_bits[SLOT_KV_CACHE] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_Q_ROPE] != OUT_WIDTH) ||
            (desc_dim0[SLOT_METADATA] != METADATA_WORDS) ||
            (desc_dim1[SLOT_METADATA] != 32'd0) ||
            (desc_dim2[SLOT_METADATA] != 32'd0) ||
            (desc_dim3[SLOT_METADATA] != 32'd0) ||
            (desc_dim0[SLOT_Q_FLAT] != Q_COUNT) ||
            (desc_dim1[SLOT_Q_FLAT] != 32'd0) ||
            (desc_dim2[SLOT_Q_FLAT] != 32'd0) ||
            (desc_dim3[SLOT_Q_FLAT] != 32'd0) ||
            (desc_dim0[SLOT_K_FLAT] != KV_COUNT) ||
            (desc_dim1[SLOT_K_FLAT] != 32'd0) ||
            (desc_dim2[SLOT_K_FLAT] != 32'd0) ||
            (desc_dim3[SLOT_K_FLAT] != 32'd0) ||
            (desc_dim0[SLOT_V_FLAT] != KV_COUNT) ||
            (desc_dim1[SLOT_V_FLAT] != 32'd0) ||
            (desc_dim2[SLOT_V_FLAT] != 32'd0) ||
            (desc_dim3[SLOT_V_FLAT] != 32'd0) ||
            (desc_dim0[SLOT_Q_GAMMA] != HEAD_DIM) ||
            (desc_dim1[SLOT_Q_GAMMA] != 32'd0) ||
            (desc_dim2[SLOT_Q_GAMMA] != 32'd0) ||
            (desc_dim3[SLOT_Q_GAMMA] != 32'd0) ||
            (desc_dim0[SLOT_K_GAMMA] != HEAD_DIM) ||
            (desc_dim1[SLOT_K_GAMMA] != 32'd0) ||
            (desc_dim2[SLOT_K_GAMMA] != 32'd0) ||
            (desc_dim3[SLOT_K_GAMMA] != 32'd0) ||
            (desc_dim0[SLOT_Q_ROPE] != Q_COUNT) ||
            (desc_dim1[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim2[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim3[SLOT_Q_ROPE] != 32'd0) ||
            (desc_dim0[SLOT_KV_CACHE] != 32'd2) ||
            (desc_dim1[SLOT_KV_CACHE] != NUM_KV_HEADS) ||
            (desc_dim2[SLOT_KV_CACHE] != MAX_CONTEXT) ||
            (desc_dim3[SLOT_KV_CACHE] != HEAD_DIM) ||
            (desc_stride0_bytes[SLOT_METADATA] != 64'd4) ||
            (desc_stride1_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride2_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride3_bytes[SLOT_METADATA] != 64'd0) ||
            (desc_stride0_bytes[SLOT_Q_FLAT] != 64'd4) ||
            (desc_stride1_bytes[SLOT_Q_FLAT] != 64'd0) ||
            (desc_stride2_bytes[SLOT_Q_FLAT] != 64'd0) ||
            (desc_stride3_bytes[SLOT_Q_FLAT] != 64'd0) ||
            (desc_stride0_bytes[SLOT_K_FLAT] != 64'd4) ||
            (desc_stride1_bytes[SLOT_K_FLAT] != 64'd0) ||
            (desc_stride2_bytes[SLOT_K_FLAT] != 64'd0) ||
            (desc_stride3_bytes[SLOT_K_FLAT] != 64'd0) ||
            (desc_stride0_bytes[SLOT_V_FLAT] != 64'd4) ||
            (desc_stride1_bytes[SLOT_V_FLAT] != 64'd0) ||
            (desc_stride2_bytes[SLOT_V_FLAT] != 64'd0) ||
            (desc_stride3_bytes[SLOT_V_FLAT] != 64'd0) ||
            (desc_stride0_bytes[SLOT_Q_GAMMA] != 64'd4) ||
            (desc_stride1_bytes[SLOT_Q_GAMMA] != 64'd0) ||
            (desc_stride2_bytes[SLOT_Q_GAMMA] != 64'd0) ||
            (desc_stride3_bytes[SLOT_Q_GAMMA] != 64'd0) ||
            (desc_stride0_bytes[SLOT_K_GAMMA] != 64'd4) ||
            (desc_stride1_bytes[SLOT_K_GAMMA] != 64'd0) ||
            (desc_stride2_bytes[SLOT_K_GAMMA] != 64'd0) ||
            (desc_stride3_bytes[SLOT_K_GAMMA] != 64'd0) ||
            (desc_stride0_bytes[SLOT_KV_CACHE] != KV_KIND_STRIDE_BYTES) ||
            (desc_stride1_bytes[SLOT_KV_CACHE] != KV_HEAD_STRIDE_BYTES) ||
            (desc_stride2_bytes[SLOT_KV_CACHE] != KV_POSITION_STRIDE_BYTES) ||
            (desc_stride3_bytes[SLOT_KV_CACHE] != 64'd4) ||
            (desc_stride0_bytes[SLOT_Q_ROPE] != 64'd4) ||
            (desc_stride1_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_stride2_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_stride3_bytes[SLOT_Q_ROPE] != 64'd0) ||
            (desc_nbytes[SLOT_METADATA] != METADATA_BYTES) ||
            (desc_nbytes[SLOT_Q_FLAT] != Q_BYTES) ||
            (desc_nbytes[SLOT_K_FLAT] != KV_BYTES) ||
            (desc_nbytes[SLOT_V_FLAT] != KV_BYTES) ||
            (desc_nbytes[SLOT_Q_GAMMA] != PARAM_BYTES) ||
            (desc_nbytes[SLOT_K_GAMMA] != PARAM_BYTES) ||
            (desc_nbytes[SLOT_KV_CACHE] != KV_CACHE_BYTES) ||
            (desc_nbytes[SLOT_Q_ROPE] != Q_ROPE_BYTES) ||
            (desc_aux1[SLOT_METADATA] >= NUM_LAYERS) ||
            (desc_aux3[SLOT_METADATA] >= MAX_CONTEXT)) begin
            validate_error = 1'b1;
        end

        if (active_runtime_context_valid) begin
            if ((active_runtime_layer_id >= NUM_LAYERS) ||
                (active_runtime_position >= MAX_CONTEXT) ||
                (active_runtime_kv_cache_base_addr[1 : 0] != 2'b00) ||
                (desc_base_addr[SLOT_COS][1 : 0] != 2'b00) ||
                (desc_base_addr[SLOT_SIN][1 : 0] != 2'b00) ||
                (desc_aux1[SLOT_METADATA] != active_runtime_layer_id) ||
                (desc_rank[SLOT_COS] != 32'd2) ||
                (desc_rank[SLOT_SIN] != 32'd2) ||
                (desc_dim0[SLOT_COS] != MAX_CONTEXT) ||
                (desc_dim1[SLOT_COS] != HEAD_DIM) ||
                (desc_dim2[SLOT_COS] != 32'd0) ||
                (desc_dim3[SLOT_COS] != 32'd0) ||
                (desc_dim0[SLOT_SIN] != MAX_CONTEXT) ||
                (desc_dim1[SLOT_SIN] != HEAD_DIM) ||
                (desc_dim2[SLOT_SIN] != 32'd0) ||
                (desc_dim3[SLOT_SIN] != 32'd0) ||
                (desc_nbytes[SLOT_COS] != ROPE_TABLE_BYTES) ||
                (desc_nbytes[SLOT_SIN] != ROPE_TABLE_BYTES) ||
                (desc_stride0_bytes[SLOT_COS] != PARAM_BYTES) ||
                (desc_stride1_bytes[SLOT_COS] != 64'd4) ||
                (desc_stride2_bytes[SLOT_COS] != 64'd0) ||
                (desc_stride3_bytes[SLOT_COS] != 64'd0) ||
                (desc_stride0_bytes[SLOT_SIN] != PARAM_BYTES) ||
                (desc_stride1_bytes[SLOT_SIN] != 64'd4) ||
                (desc_stride2_bytes[SLOT_SIN] != 64'd0) ||
                (desc_stride3_bytes[SLOT_SIN] != 64'd0)) begin
                validate_error = 1'b1;
            end
        end
        else begin
            if ((desc_rank[SLOT_COS] != 32'd1) ||
                (desc_rank[SLOT_SIN] != 32'd1) ||
                (desc_dim0[SLOT_COS] != HEAD_DIM) ||
                (desc_dim1[SLOT_COS] != 32'd0) ||
                (desc_dim2[SLOT_COS] != 32'd0) ||
                (desc_dim3[SLOT_COS] != 32'd0) ||
                (desc_dim0[SLOT_SIN] != HEAD_DIM) ||
                (desc_dim1[SLOT_SIN] != 32'd0) ||
                (desc_dim2[SLOT_SIN] != 32'd0) ||
                (desc_dim3[SLOT_SIN] != 32'd0) ||
                (desc_nbytes[SLOT_COS] != PARAM_BYTES) ||
                (desc_nbytes[SLOT_SIN] != PARAM_BYTES) ||
                (desc_stride0_bytes[SLOT_COS] != 64'd4) ||
                (desc_stride1_bytes[SLOT_COS] != 64'd0) ||
                (desc_stride2_bytes[SLOT_COS] != 64'd0) ||
                (desc_stride3_bytes[SLOT_COS] != 64'd0) ||
                (desc_stride0_bytes[SLOT_SIN] != 64'd4) ||
                (desc_stride1_bytes[SLOT_SIN] != 64'd0) ||
                (desc_stride2_bytes[SLOT_SIN] != 64'd0) ||
                (desc_stride3_bytes[SLOT_SIN] != 64'd0)) begin
                validate_error = 1'b1;
            end
        end
    end

    function automatic read_slot_t next_read_slot(input read_slot_t slot);
        begin
            case (slot)
                R_Q_FLAT: next_read_slot = R_K_FLAT;
                R_K_FLAT: next_read_slot = R_V_FLAT;
                R_V_FLAT: next_read_slot = R_Q_GAMMA;
                R_Q_GAMMA: next_read_slot = R_K_GAMMA;
                R_K_GAMMA: next_read_slot = R_COS;
                R_COS: next_read_slot = R_SIN;
                default: next_read_slot = R_DONE;
            endcase
        end
    endfunction

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_read_slot <= R_Q_FLAT;
            q_flat <= '0;
            k_flat <= '0;
            v_flat <= '0;
            q_gamma_flat <= '0;
            k_gamma_flat <= '0;
            cos_flat <= '0;
            sin_flat <= '0;
            read_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            stage_done_seen <= 1'b0;
            stage_error_seen <= 1'b0;
            cache_wr_addr_reg <= '0;
            cache_wr_data_reg <= 32'd0;
            qrope_write_req_seen <= 1'b0;
            qrope_write_word_index <= 32'd0;
            active_runtime_context_valid <= 1'b0;
            active_runtime_layer_id <= '0;
            active_runtime_position <= '0;
            active_runtime_kv_cache_base_addr <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_q_rope_write_word_count <= 32'd0;
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
            if (stage_done) begin
                stage_done_seen <= 1'b1;
                stage_error_seen <= stage_error_seen || stage_error;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_READER_START;
                        active_read_slot <= R_Q_FLAT;
                        q_flat <= '0;
                        k_flat <= '0;
                        v_flat <= '0;
                        q_gamma_flat <= '0;
                        k_gamma_flat <= '0;
                        cos_flat <= '0;
                        sin_flat <= '0;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        stage_done_seen <= 1'b0;
                        stage_error_seen <= 1'b0;
                        qrope_write_req_seen <= 1'b0;
                        qrope_write_word_index <= 32'd0;
                        active_runtime_context_valid <= (i_runtime_context_valid === 1'b1);
                        active_runtime_layer_id <= i_runtime_layer_id;
                        active_runtime_position <= i_runtime_position;
                        active_runtime_kv_cache_base_addr <= i_runtime_kv_cache_base_addr;
                        o_error <= 1'b0;
                        o_q_rope_write_word_count <= 32'd0;
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
                        active_read_slot <= R_Q_FLAT;
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
                                R_Q_FLAT: begin
                                    q_flat[read_element_index*IN_WIDTH +: IN_WIDTH] <=
                                        i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
                                end
                                R_K_FLAT: begin
                                    k_flat[read_element_index*IN_WIDTH +: IN_WIDTH] <=
                                        i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
                                end
                                R_V_FLAT: begin
                                    v_flat[read_element_index*IN_WIDTH +: IN_WIDTH] <=
                                        i_mem_rd_rsp_data[IN_WIDTH-1 : 0];
                                end
                                R_Q_GAMMA: begin
                                    q_gamma_flat[read_element_index*GAMMA_WIDTH +: GAMMA_WIDTH] <=
                                        i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0];
                                end
                                R_K_GAMMA: begin
                                    k_gamma_flat[read_element_index*GAMMA_WIDTH +: GAMMA_WIDTH] <=
                                        i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0];
                                end
                                R_COS: begin
                                    cos_flat[read_element_index*TRIG_WIDTH +: TRIG_WIDTH] <=
                                        i_mem_rd_rsp_data[TRIG_WIDTH-1 : 0];
                                end
                                R_SIN: begin
                                    sin_flat[read_element_index*TRIG_WIDTH +: TRIG_WIDTH] <=
                                        i_mem_rd_rsp_data[TRIG_WIDTH-1 : 0];
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
                    stage_done_seen <= 1'b0;
                    stage_error_seen <= 1'b0;
                    state <= S_STAGE_RUN;
                end

                S_STAGE_RUN: begin
                    if (cache_wr_valid) begin
                        state <= S_CACHE_WR_REQ;
                    end
                    else if (stage_done_seen) begin
                        if (stage_error_seen) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            qrope_write_req_seen <= 1'b0;
                            qrope_write_word_index <= 32'd0;
                            state <= S_QROPE_WR_REQ;
                        end
                    end
                end

                S_CACHE_WR_REQ: begin
                    if (cache_wr_valid && i_mem_wr_req_ready) begin
                        cache_wr_addr_reg <= cache_wr_addr;
                        cache_wr_data_reg <= cache_wr_data;
                        state <= S_CACHE_WR_DATA;
                    end
                end

                S_CACHE_WR_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        state <= S_CACHE_WR_WAIT;
                    end
                end

                S_CACHE_WR_WAIT: begin
                    if (i_mem_wr_done || i_mem_wr_error) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_STAGE_RUN;
                        end
                    end
                end

                S_QROPE_WR_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        qrope_write_req_seen <= 1'b1;
                        qrope_write_word_index <= 32'd0;
                        state <= S_QROPE_WR_DATA;
                    end
                end

                S_QROPE_WR_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        o_q_rope_write_word_count <= o_q_rope_write_word_count + 1'b1;
                        if (o_mem_wr_data_last) begin
                            state <= S_QROPE_WR_WAIT;
                        end
                        else begin
                            qrope_write_word_index <= qrope_write_word_index + 1'b1;
                        end
                    end
                end

                S_QROPE_WR_WAIT: begin
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
