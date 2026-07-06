`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_one_token_layer_scheduler;

    localparam int ADDR_WIDTH = 64;
    localparam int MEM_DATA_WIDTH = 32;
    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int MAX_LAYERS = 28;
    localparam int BASE_TABLE_BITS = MAX_LAYERS * ADDR_WIDTH;

    localparam int NUM_Q_HEADS = 16;
    localparam int NUM_KV_HEADS = 8;
    localparam int HEAD_DIM = 128;
    localparam int MAX_CONTEXT = 256;
    localparam int CACHE_LENGTH = 5;
    localparam int IN_WIDTH = 24;
    localparam int Q_COUNT = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT = NUM_KV_HEADS * HEAD_DIM;
    localparam int KV_REPEAT = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam int TOTAL_CACHE_WRITES = 2 * KV_COUNT;

    localparam int QKV_IMAGE_BYTES = 32'h0022_B000;
    localparam int FRONT_IMAGE_BYTES = 32'h0000_8000;
    localparam int SCORE_IMAGE_BYTES = 32'h0000_5000;
    localparam int OPROJ_IMAGE_BYTES = 32'h0000_5000;
    localparam int POST_IMAGE_BYTES = 32'h0000_8000;
    localparam int GATE_IMAGE_BYTES = 32'h0000_E000;
    localparam int SILU_IMAGE_BYTES = 32'h0000_E000;
    localparam int DOWN_IMAGE_BYTES = 32'h0000_6000;
    localparam int RESIDUAL_IMAGE_BYTES = 32'h0000_5000;

    localparam int QKV_WORDS = QKV_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int FRONT_WORDS = FRONT_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int SCORE_WORDS = SCORE_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int OPROJ_WORDS = OPROJ_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int POST_WORDS = POST_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int GATE_WORDS = GATE_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int SILU_WORDS = SILU_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DOWN_WORDS = DOWN_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int RESIDUAL_WORDS = RESIDUAL_IMAGE_BYTES / MEM_DATA_BYTES;

    localparam int VEC1024 = 1024;
    localparam int VEC2048 = 2048;
    localparam int VEC3072 = 3072;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;

    localparam int QKV_SLOT_ACTIVATION = 1;
    localparam int QKV_SLOT_Q_OUT = 8;
    localparam int QKV_SLOT_K_OUT = 9;
    localparam int QKV_SLOT_V_OUT = 10;
    localparam int FRONT_SLOT_Q_FLAT = 1;
    localparam int FRONT_SLOT_K_FLAT = 2;
    localparam int FRONT_SLOT_V_FLAT = 3;
    localparam int FRONT_SLOT_Q_ROPE = 9;
    localparam int SCORE_SLOT_Q_ROPE = 1;
    localparam int SCORE_SLOT_ATTN_OUT = 4;
    localparam int OPROJ_SLOT_ACTIVATION = 1;
    localparam int OPROJ_SLOT_OUTPUT = 4;
    localparam int POST_SLOT_RESIDUAL = 1;
    localparam int POST_SLOT_O_PROJ = 2;
    localparam int POST_SLOT_HIDDEN = 4;
    localparam int POST_SLOT_NORM = 5;
    localparam int GATE_SLOT_ACTIVATION = 1;
    localparam int GATE_SLOT_GATE_OUTPUT = 6;
    localparam int GATE_SLOT_UP_OUTPUT = 7;
    localparam int SILU_SLOT_GATE = 1;
    localparam int SILU_SLOT_UP = 2;
    localparam int SILU_SLOT_HIDDEN = 4;
    localparam int DOWN_SLOT_ACTIVATION = 1;
    localparam int DOWN_SLOT_OUTPUT = 4;
    localparam int RESIDUAL_SLOT_POST_ATTN = 1;
    localparam int RESIDUAL_SLOT_DOWN = 2;
    localparam int RESIDUAL_SLOT_OUTPUT = 3;

    localparam int OPROJ_WEIGHT_WORDS = (1024 * 1024) / 4;
    localparam int OPROJ_SCALE_WORDS = (1024 * 64) / 4;
    localparam int GATE_WEIGHT_WORDS = (3072 * 512) / 4;
    localparam int GATE_SCALE_WORDS = (3072 * 32) / 4;
    localparam int DOWN_WEIGHT_WORDS = (1024 * 1536) / 4;
    localparam int DOWN_SCALE_WORDS = (1024 * 96) / 4;

    localparam logic [ADDR_WIDTH-1 : 0] CACHE_BASE_ADDR = 64'h0000_0004_1410_0000;
    localparam int CACHE_WORDS_FULL = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM;
    localparam int MODELED_CACHE_LAYERS = 3;
    localparam int CACHE_LAYER_ACTIVE_WORDS = CACHE_LENGTH * KV_COUNT;

    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_QKV_BASE_ADDR = 64'h0000_0004_1008_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR = 64'h0000_0004_1502_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR = 64'h0000_0004_1503_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_O_PROJ_BASE_ADDR = 64'h0000_0004_1504_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR = 64'h0000_0004_1505_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR = 64'h0000_0004_1506_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR = 64'h0000_0004_1507_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_DOWN_BASE_ADDR = 64'h0000_0004_1508_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR = 64'h0000_0004_1509_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_O_PROJ_WEIGHT_BASE_ADDR = 64'h0000_0004_0700_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_O_PROJ_SCALE_BASE_ADDR = 64'h0000_0004_0710_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_GATE_WEIGHT_BASE_ADDR = 64'h0000_0004_0720_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_GATE_SCALE_BASE_ADDR = 64'h0000_0004_0738_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_UP_WEIGHT_BASE_ADDR = 64'h0000_0004_0740_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_UP_SCALE_BASE_ADDR = 64'h0000_0004_0758_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_DOWN_WEIGHT_BASE_ADDR = 64'h0000_0004_0760_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_DOWN_SCALE_BASE_ADDR = 64'h0000_0004_0778_0000;

    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_QKV_BASE_ADDR = 64'h0000_0004_2008_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR = 64'h0000_0004_2502_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR = 64'h0000_0004_2503_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_O_PROJ_BASE_ADDR = 64'h0000_0004_2504_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR = 64'h0000_0004_2505_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR = 64'h0000_0004_2506_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR = 64'h0000_0004_2507_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_DOWN_BASE_ADDR = 64'h0000_0004_2508_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR = 64'h0000_0004_2509_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_O_PROJ_WEIGHT_BASE_ADDR = 64'h0000_0004_0800_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_O_PROJ_SCALE_BASE_ADDR = 64'h0000_0004_0810_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_GATE_WEIGHT_BASE_ADDR = 64'h0000_0004_0820_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_GATE_SCALE_BASE_ADDR = 64'h0000_0004_0838_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_UP_WEIGHT_BASE_ADDR = 64'h0000_0004_0840_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_UP_SCALE_BASE_ADDR = 64'h0000_0004_0858_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_DOWN_WEIGHT_BASE_ADDR = 64'h0000_0004_0860_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_MLP_DOWN_SCALE_BASE_ADDR = 64'h0000_0004_0878_0000;

    localparam int WRITE_QKV_Q = 1;
    localparam int WRITE_QKV_K = 2;
    localparam int WRITE_QKV_V = 3;
    localparam int WRITE_CACHE = 4;
    localparam int WRITE_Q_ROPE = 5;
    localparam int WRITE_ATTN_OUT = 6;
    localparam int WRITE_O_PROJ = 7;
    localparam int WRITE_POST_HIDDEN = 8;
    localparam int WRITE_POST_NORM = 9;
    localparam int WRITE_GATE = 10;
    localparam int WRITE_UP = 11;
    localparam int WRITE_SILU_HIDDEN = 12;
    localparam int WRITE_DOWN = 13;
    localparam int WRITE_LAYER = 14;

    localparam int EXPECTED_QKV_RD_REQS = 8209;
    localparam int EXPECTED_QKV_RD_WORDS = 558480;
    localparam int EXPECTED_QKV_WR_REQS = 4096;
    localparam int EXPECTED_QKV_WR_WORDS = 4096;
    localparam int EXPECTED_FULL_RD_REQS = 38055;
    localparam int EXPECTED_FULL_RD_WORDS = 1579650;
    localparam int EXPECTED_FULL_WR_REQS = 2058;
    localparam int EXPECTED_FULL_WR_WORDS = 20480;
    localparam int EXPECTED_QKV_INVALID_RD_REQS = 13;
    localparam int EXPECTED_QKV_INVALID_RD_WORDS = 400;
    localparam int EXPECTED_FRONTEND_INVALID_RD_REQS = EXPECTED_QKV_RD_REQS + 11;
    localparam int EXPECTED_FRONTEND_INVALID_RD_WORDS = EXPECTED_QKV_RD_WORDS + 336;
    localparam int EXPECTED_NORMAL_RD_REQS = EXPECTED_QKV_RD_REQS + EXPECTED_FULL_RD_REQS;
    localparam int EXPECTED_NORMAL_RD_WORDS = EXPECTED_QKV_RD_WORDS + EXPECTED_FULL_RD_WORDS;
    localparam int EXPECTED_NORMAL_WR_REQS = EXPECTED_QKV_WR_REQS + EXPECTED_FULL_WR_REQS;
    localparam int EXPECTED_NORMAL_WR_WORDS = EXPECTED_QKV_WR_WORDS + EXPECTED_FULL_WR_WORDS;
    localparam int EXPECTED_TWO_LAYER_RD_REQS = 2 * EXPECTED_NORMAL_RD_REQS;
    localparam int EXPECTED_TWO_LAYER_RD_WORDS = 2 * EXPECTED_NORMAL_RD_WORDS;
    localparam int EXPECTED_TWO_LAYER_WR_REQS = 2 * EXPECTED_NORMAL_WR_REQS;
    localparam int EXPECTED_TWO_LAYER_WR_WORDS = 2 * EXPECTED_NORMAL_WR_WORDS;
    localparam int EXPECTED_THREE_LAYER_RD_REQS = 3 * EXPECTED_NORMAL_RD_REQS;
    localparam int EXPECTED_THREE_LAYER_RD_WORDS = 3 * EXPECTED_NORMAL_RD_WORDS;
    localparam int EXPECTED_THREE_LAYER_WR_REQS = 3 * EXPECTED_NORMAL_WR_REQS;
    localparam int EXPECTED_THREE_LAYER_WR_WORDS = 3 * EXPECTED_NORMAL_WR_WORDS;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic [1 : 0] active_stage_debug;
    logic [7 : 0] layer0_state_debug;
    logic [7 : 0] state_debug;
    logic [4 : 0] layer_start_index;
    logic [4 : 0] layer_count;
    logic [7 : 0] token_position;
    logic [ADDR_WIDTH-1 : 0] input_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] kv_cache_base_addr;
    logic [BASE_TABLE_BITS-1 : 0] qkv_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] attn_frontend_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] attn_score_value_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] o_proj_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] post_attn_norm_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_gate_up_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_silu_mul_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_down_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_residual_add_qmap_base_addr_table;
    logic [4 : 0] active_layer_index;
    logic [4 : 0] layers_started;
    logic [4 : 0] layers_completed;
    logic [27 : 0] layer_done_mask;
    logic [27 : 0] layer_error_mask;
    logic [1 : 0] stage_done_mask;
    logic [1 : 0] stage_error_mask;
    logic [3 : 0] layer0_full_stage_done_mask;
    logic [3 : 0] layer0_full_stage_error_mask;
    logic [4 : 0] body_stage_done_mask;
    logic [4 : 0] body_stage_error_mask;
    logic [31 : 0] qkv_rows_done;
    logic signed [55 : 0] qkv_last_row_sum_q26;
    logic signed [31 : 0] qkv_last_output_q12_12;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;

    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [31 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    logic [31 : 0] qkv_qmap [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap [0 : RESIDUAL_WORDS-1];

    logic [31 : 0] qkv_qmap_l1 [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap_l1 [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap_l1 [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap_l1 [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap_l1 [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap_l1 [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap_l1 [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap_l1 [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap_l1 [0 : RESIDUAL_WORDS-1];

    logic [31 : 0] qkv_qmap_l2 [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap_l2 [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap_l2 [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap_l2 [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap_l2 [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap_l2 [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap_l2 [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap_l2 [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap_l2 [0 : RESIDUAL_WORDS-1];

    logic signed [IN_WIDTH-1 : 0] k_cache_mem [0 : CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] v_cache_mem [0 : CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] k_cache_mem_l1 [0 : CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] v_cache_mem_l1 [0 : CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] k_cache_mem_l2 [0 : CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] v_cache_mem_l2 [0 : CACHE_LENGTH*KV_COUNT-1];

    logic [31 : 0] oproj_weight_mem [0 : OPROJ_WEIGHT_WORDS-1];
    logic [31 : 0] oproj_scale_mem [0 : OPROJ_SCALE_WORDS-1];
    logic [31 : 0] gate_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] gate_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] up_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] up_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] down_weight_mem [0 : DOWN_WEIGHT_WORDS-1];
    logic [31 : 0] down_scale_mem [0 : DOWN_SCALE_WORDS-1];

    logic [31 : 0] oproj_weight_mem_l1 [0 : OPROJ_WEIGHT_WORDS-1];
    logic [31 : 0] oproj_scale_mem_l1 [0 : OPROJ_SCALE_WORDS-1];
    logic [31 : 0] gate_weight_mem_l1 [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] gate_scale_mem_l1 [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] up_weight_mem_l1 [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] up_scale_mem_l1 [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] down_weight_mem_l1 [0 : DOWN_WEIGHT_WORDS-1];
    logic [31 : 0] down_scale_mem_l1 [0 : DOWN_SCALE_WORDS-1];

    logic [31 : 0] oproj_weight_mem_l2 [0 : OPROJ_WEIGHT_WORDS-1];
    logic [31 : 0] oproj_scale_mem_l2 [0 : OPROJ_SCALE_WORDS-1];
    logic [31 : 0] gate_weight_mem_l2 [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] gate_scale_mem_l2 [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] up_weight_mem_l2 [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] up_scale_mem_l2 [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] down_weight_mem_l2 [0 : DOWN_WEIGHT_WORDS-1];
    logic [31 : 0] down_scale_mem_l2 [0 : DOWN_SCALE_WORDS-1];

    logic [31 : 0] expected_qkv [0 : (VEC2048 + (2 * VEC1024))-1];
    logic [31 : 0] expected_q_rope [0 : VEC2048-1];
    logic [63 : 0] expected_cache_addr [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_cache_data [0 : TOTAL_CACHE_WRITES-1];
    logic [3 : 0] expected_cache_kind [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_attn_out [0 : VEC2048-1];
    logic [31 : 0] expected_o_proj [0 : VEC1024-1];
    logic [31 : 0] expected_post_hidden [0 : VEC1024-1];
    logic [31 : 0] expected_post_norm [0 : VEC1024-1];
    logic [31 : 0] expected_gate [0 : VEC3072-1];
    logic [31 : 0] expected_up [0 : VEC3072-1];
    logic [31 : 0] expected_silu_hidden [0 : VEC3072-1];
    logic [31 : 0] expected_down [0 : VEC1024-1];
    logic [31 : 0] expected_layer [0 : VEC1024-1];

    logic [31 : 0] expected_qkv_l1 [0 : (VEC2048 + (2 * VEC1024))-1];
    logic [31 : 0] expected_q_rope_l1 [0 : VEC2048-1];
    logic [63 : 0] expected_cache_addr_l1 [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_cache_data_l1 [0 : TOTAL_CACHE_WRITES-1];
    logic [3 : 0] expected_cache_kind_l1 [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_attn_out_l1 [0 : VEC2048-1];
    logic [31 : 0] expected_o_proj_l1 [0 : VEC1024-1];
    logic [31 : 0] expected_post_hidden_l1 [0 : VEC1024-1];
    logic [31 : 0] expected_post_norm_l1 [0 : VEC1024-1];
    logic [31 : 0] expected_gate_l1 [0 : VEC3072-1];
    logic [31 : 0] expected_up_l1 [0 : VEC3072-1];
    logic [31 : 0] expected_silu_hidden_l1 [0 : VEC3072-1];
    logic [31 : 0] expected_down_l1 [0 : VEC1024-1];
    logic [31 : 0] expected_layer_l1 [0 : VEC1024-1];

    logic [31 : 0] expected_qkv_l2 [0 : (VEC2048 + (2 * VEC1024))-1];
    logic [31 : 0] expected_q_rope_l2 [0 : VEC2048-1];
    logic [63 : 0] expected_cache_addr_l2 [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_cache_data_l2 [0 : TOTAL_CACHE_WRITES-1];
    logic [3 : 0] expected_cache_kind_l2 [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_attn_out_l2 [0 : VEC2048-1];
    logic [31 : 0] expected_o_proj_l2 [0 : VEC1024-1];
    logic [31 : 0] expected_post_hidden_l2 [0 : VEC1024-1];
    logic [31 : 0] expected_post_norm_l2 [0 : VEC1024-1];
    logic [31 : 0] expected_gate_l2 [0 : VEC3072-1];
    logic [31 : 0] expected_up_l2 [0 : VEC3072-1];
    logic [31 : 0] expected_silu_hidden_l2 [0 : VEC3072-1];
    logic [31 : 0] expected_down_l2 [0 : VEC1024-1];
    logic [31 : 0] expected_layer_l2 [0 : VEC1024-1];

    logic [ADDR_WIDTH-1 : 0] qkv_q_base;
    logic [ADDR_WIDTH-1 : 0] qkv_k_base;
    logic [ADDR_WIDTH-1 : 0] qkv_v_base;
    logic [ADDR_WIDTH-1 : 0] q_rope_base;
    logic [ADDR_WIDTH-1 : 0] attn_out_base;
    logic [ADDR_WIDTH-1 : 0] o_proj_base;
    logic [ADDR_WIDTH-1 : 0] post_hidden_base;
    logic [ADDR_WIDTH-1 : 0] post_norm_base;
    logic [ADDR_WIDTH-1 : 0] gate_output_base;
    logic [ADDR_WIDTH-1 : 0] up_output_base;
    logic [ADDR_WIDTH-1 : 0] silu_hidden_base;
    logic [ADDR_WIDTH-1 : 0] down_output_base;
    logic [ADDR_WIDTH-1 : 0] layer_output_base;

    logic [ADDR_WIDTH-1 : 0] qkv_q_base_l1;
    logic [ADDR_WIDTH-1 : 0] qkv_k_base_l1;
    logic [ADDR_WIDTH-1 : 0] qkv_v_base_l1;
    logic [ADDR_WIDTH-1 : 0] q_rope_base_l1;
    logic [ADDR_WIDTH-1 : 0] attn_out_base_l1;
    logic [ADDR_WIDTH-1 : 0] o_proj_base_l1;
    logic [ADDR_WIDTH-1 : 0] post_hidden_base_l1;
    logic [ADDR_WIDTH-1 : 0] post_norm_base_l1;
    logic [ADDR_WIDTH-1 : 0] gate_output_base_l1;
    logic [ADDR_WIDTH-1 : 0] up_output_base_l1;
    logic [ADDR_WIDTH-1 : 0] silu_hidden_base_l1;
    logic [ADDR_WIDTH-1 : 0] down_output_base_l1;
    logic [ADDR_WIDTH-1 : 0] layer_output_base_l1;

    logic [ADDR_WIDTH-1 : 0] qkv_q_base_l2;
    logic [ADDR_WIDTH-1 : 0] qkv_k_base_l2;
    logic [ADDR_WIDTH-1 : 0] qkv_v_base_l2;
    logic [ADDR_WIDTH-1 : 0] q_rope_base_l2;
    logic [ADDR_WIDTH-1 : 0] attn_out_base_l2;
    logic [ADDR_WIDTH-1 : 0] o_proj_base_l2;
    logic [ADDR_WIDTH-1 : 0] post_hidden_base_l2;
    logic [ADDR_WIDTH-1 : 0] post_norm_base_l2;
    logic [ADDR_WIDTH-1 : 0] gate_output_base_l2;
    logic [ADDR_WIDTH-1 : 0] up_output_base_l2;
    logic [ADDR_WIDTH-1 : 0] silu_hidden_base_l2;
    logic [ADDR_WIDTH-1 : 0] down_output_base_l2;
    logic [ADDR_WIDTH-1 : 0] layer_output_base_l2;

    string tracefile;
    string wavefile;
    integer trace_fd;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer total_fail_count;
    integer done_seen_count;
    integer normal_done_cycle;
    integer invalid_done_cycle;

    integer rd_req_accept_count;
    integer rd_rsp_accept_count;
    integer wr_req_accept_count;
    integer wr_data_accept_count;
    integer qkv_q_write_accept_count;
    integer qkv_k_write_accept_count;
    integer qkv_v_write_accept_count;
    integer cache_write_accept_count;
    integer q_rope_write_accept_count;
    integer attn_out_write_accept_count;
    integer o_proj_write_accept_count;
    integer post_hidden_write_accept_count;
    integer post_norm_write_accept_count;
    integer gate_write_accept_count;
    integer up_write_accept_count;
    integer silu_write_accept_count;
    integer down_write_accept_count;
    integer layer_write_accept_count;
    integer write_mismatch_count;
    longint signed max_abs_diff;

    integer normal_read_bursts;
    integer normal_read_words;
    integer normal_write_reqs;
    integer normal_write_words;
    logic [1 : 0] normal_stage_done_mask;
    logic [1 : 0] normal_stage_error_mask;
    logic [27 : 0] normal_layer_done_mask;
    logic [27 : 0] normal_layer_error_mask;
    logic [4 : 0] normal_layers_started;
    logic [4 : 0] normal_layers_completed;
    logic [3 : 0] normal_layer0_full_stage_done_mask;
    logic [3 : 0] normal_layer0_full_stage_error_mask;
    logic [4 : 0] normal_body_stage_done_mask;
    logic [4 : 0] normal_body_stage_error_mask;
    logic normal_error;
    integer normal_mismatch_count;
    integer normal_write_mismatch_count;
    longint signed normal_max_abs_diff;
    integer normal_qkv_q_write_accept_count;
    integer normal_qkv_k_write_accept_count;
    integer normal_qkv_v_write_accept_count;
    integer normal_cache_write_accept_count;
    integer normal_q_rope_write_accept_count;
    integer normal_attn_out_write_accept_count;
    integer normal_o_proj_write_accept_count;
    integer normal_post_hidden_write_accept_count;
    integer normal_post_norm_write_accept_count;
    integer normal_gate_write_accept_count;
    integer normal_up_write_accept_count;
    integer normal_silu_write_accept_count;
    integer normal_down_write_accept_count;
    integer normal_layer_write_accept_count;
    integer two_layer_done_cycle;
    integer two_layer_read_bursts;
    integer two_layer_read_words;
    integer two_layer_write_reqs;
    integer two_layer_write_words;
    logic [27 : 0] two_layer_done_mask;
    logic [27 : 0] two_layer_error_mask;
    logic [4 : 0] two_layer_layers_started;
    logic [4 : 0] two_layer_layers_completed;
    logic two_layer_error;
    integer two_layer_mismatch_count;
    integer two_layer_write_mismatch_count;
    longint signed two_layer_max_abs_diff;
    integer three_layer_done_cycle;
    integer three_layer_read_bursts;
    integer three_layer_read_words;
    integer three_layer_write_reqs;
    integer three_layer_write_words;
    logic [27 : 0] three_layer_done_mask;
    logic [27 : 0] three_layer_error_mask;
    logic [4 : 0] three_layer_layers_started;
    logic [4 : 0] three_layer_layers_completed;
    logic three_layer_error;
    integer three_layer_mismatch_count;
    integer three_layer_write_mismatch_count;
    longint signed three_layer_max_abs_diff;
    integer invalid_read_bursts;
    integer invalid_read_words;
    integer invalid_write_reqs;
    integer invalid_write_words;
    logic [1 : 0] invalid_stage_done_mask;
    logic [1 : 0] invalid_stage_error_mask;
    logic [3 : 0] invalid_layer0_full_stage_done_mask;
    logic [3 : 0] invalid_layer0_full_stage_error_mask;
    logic invalid_error;
    integer qkv_invalid_read_bursts;
    integer qkv_invalid_read_words;
    integer qkv_invalid_write_reqs;
    integer qkv_invalid_write_words;
    logic [1 : 0] qkv_invalid_stage_done_mask;
    logic [1 : 0] qkv_invalid_stage_error_mask;
    logic qkv_invalid_error;
    integer unsupported_read_bursts;
    integer unsupported_read_words;
    integer unsupported_write_reqs;
    integer unsupported_write_words;
    logic [27 : 0] unsupported_layer_done_mask;
    logic [27 : 0] unsupported_layer_error_mask;
    logic [4 : 0] unsupported_layers_started;
    logic [4 : 0] unsupported_layers_completed;
    logic unsupported_error;
    integer missing_base_read_bursts;
    integer missing_base_read_words;
    integer missing_base_write_reqs;
    integer missing_base_write_words;
    logic [27 : 0] missing_base_layer_done_mask;
    logic [27 : 0] missing_base_layer_error_mask;
    logic [4 : 0] missing_base_layers_started;
    logic [4 : 0] missing_base_layers_completed;
    logic missing_base_error;

    logic read_active;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer read_gap_count;
    logic [ADDR_WIDTH-1 : 0] active_read_addr;

    logic write_active;
    integer active_write_kind;
    integer active_write_index;
    integer active_write_words_left;
    integer active_write_total_words;
    integer write_done_delay;
    logic [ADDR_WIDTH-1 : 0] active_write_addr;
    integer active_write_layer;

    logic qkv_q_written;
    logic qkv_k_written;
    logic qkv_v_written;
    logic cache_written;
    logic q_rope_written;
    logic attn_out_written;
    logic o_proj_written;
    logic post_hidden_written;
    logic post_norm_written;
    logic gate_written;
    logic up_written;
    logic silu_hidden_written;
    logic down_written;
    logic layer_written;

    integer last_trace_stage;
    integer last_trace_state;
    integer last_trace_wr_words;
    integer scoreboard_layer_iterations;
    logic fastmem;

    qmap_one_token_layer_scheduler dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_layer_start_index(layer_start_index),
        .i_layer_count(layer_count),
        .i_position(token_position),
        .i_input_hidden_base_addr(input_hidden_base_addr),
        .i_output_hidden_base_addr(output_hidden_base_addr),
        .i_kv_cache_base_addr(kv_cache_base_addr),
        .i_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .i_attn_frontend_qmap_base_addr_table(attn_frontend_qmap_base_addr_table),
        .i_attn_score_value_qmap_base_addr_table(attn_score_value_qmap_base_addr_table),
        .i_o_proj_qmap_base_addr_table(o_proj_qmap_base_addr_table),
        .i_post_attn_norm_qmap_base_addr_table(post_attn_norm_qmap_base_addr_table),
        .i_mlp_gate_up_qmap_base_addr_table(mlp_gate_up_qmap_base_addr_table),
        .i_mlp_silu_mul_qmap_base_addr_table(mlp_silu_mul_qmap_base_addr_table),
        .i_mlp_down_qmap_base_addr_table(mlp_down_qmap_base_addr_table),
        .i_mlp_residual_add_qmap_base_addr_table(mlp_residual_add_qmap_base_addr_table),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_layer0_active_stage_debug(active_stage_debug),
        .o_layer0_state_debug(layer0_state_debug),
        .o_layer0_stage_done_mask(stage_done_mask),
        .o_layer0_stage_error_mask(stage_error_mask),
        .o_layer0_full_stage_done_mask(layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(layer0_full_stage_error_mask),
        .o_body_stage_done_mask(body_stage_done_mask),
        .o_body_stage_error_mask(body_stage_error_mask),
        .o_qkv_rows_done(qkv_rows_done),
        .o_qkv_last_row_sum_q26(qkv_last_row_sum_q26),
        .o_qkv_last_output_q12_12(qkv_last_output_q12_12),
        .o_mem_read_burst_count(dut_read_burst_count),
        .o_mem_read_word_count(dut_read_word_count),
        .o_mem_write_req_count(dut_write_req_count),
        .o_mem_write_word_count(dut_write_word_count),
        .o_mem_rd_req_valid(mem_rd_req_valid),
        .i_mem_rd_req_ready(mem_rd_req_ready),
        .o_mem_rd_req_addr(mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(mem_rd_rsp_data),
        .i_mem_rd_rsp_last(mem_rd_rsp_last),
        .o_mem_wr_req_valid(mem_wr_req_valid),
        .i_mem_wr_req_ready(mem_wr_req_ready),
        .o_mem_wr_req_addr(mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(mem_wr_req_len_bytes),
        .o_mem_wr_data(mem_wr_data),
        .o_mem_wr_data_valid(mem_wr_data_valid),
        .i_mem_wr_data_ready(mem_wr_data_ready),
        .o_mem_wr_data_last(mem_wr_data_last),
        .i_mem_wr_done(mem_wr_done),
        .i_mem_wr_error(mem_wr_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qmap_one_token_layer_scheduler.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        if ($test$plusargs("dumpwaves")) begin
            $dumpfile(wavefile);
            $dumpvars(0, tb_qmap_one_token_layer_scheduler);
        end
    end

    assign mem_rd_req_ready =
        fastmem ? ((!read_active) && (!mem_rd_rsp_valid)) :
        ((!read_active) &&
         (!mem_rd_rsp_valid) &&
         ((cycle_count % 11) != 3) &&
         (((cycle_count + rd_req_accept_count) % 29) != 7));
    assign mem_wr_req_ready =
        fastmem ? (!write_active) :
        ((!write_active) &&
         ((cycle_count % 13) != 5) &&
         (((cycle_count + wr_req_accept_count) % 31) != 9));
    assign mem_wr_data_ready =
        fastmem ? write_active :
        (write_active &&
         ((cycle_count % 7) != 2) &&
         (((cycle_count + wr_data_accept_count) % 23) != 10));

    function automatic integer desc_idx(input integer slot, input integer word_offset);
        begin
            desc_idx = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    function automatic logic in_range(
        input logic [ADDR_WIDTH-1 : 0] addr,
        input logic [ADDR_WIDTH-1 : 0] base_addr,
        input integer bytes
    );
        begin
            in_range = (addr >= base_addr) && (addr < (base_addr + bytes));
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] qkv_desc_base(input integer slot);
        begin
            qkv_desc_base = {
                qkv_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                qkv_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic is_cache_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            is_cache_addr =
                (addr >= CACHE_BASE_ADDR) &&
                (addr < (CACHE_BASE_ADDR + (MODELED_CACHE_LAYERS * CACHE_WORDS_FULL * MEM_DATA_BYTES)));
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] frontend_desc_base(input integer slot);
        begin
            frontend_desc_base = {
                frontend_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                frontend_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] score_desc_base(input integer slot);
        begin
            score_desc_base = {
                score_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                score_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] oproj_desc_base(input integer slot);
        begin
            oproj_desc_base = {
                oproj_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                oproj_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] post_desc_base(input integer slot);
        begin
            post_desc_base = {
                post_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                post_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] gate_desc_base(input integer slot);
        begin
            gate_desc_base = {
                gate_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                gate_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] silu_desc_base(input integer slot);
        begin
            silu_desc_base = {
                silu_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                silu_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] down_desc_base(input integer slot);
        begin
            down_desc_base = {
                down_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                down_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] residual_desc_base(input integer slot);
        begin
            residual_desc_base = {
                residual_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                residual_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] qkv_desc_base_l1(input integer slot);
        begin
            qkv_desc_base_l1 = {
                qkv_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                qkv_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] frontend_desc_base_l1(input integer slot);
        begin
            frontend_desc_base_l1 = {
                frontend_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                frontend_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] score_desc_base_l1(input integer slot);
        begin
            score_desc_base_l1 = {
                score_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                score_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] oproj_desc_base_l1(input integer slot);
        begin
            oproj_desc_base_l1 = {
                oproj_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                oproj_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] post_desc_base_l1(input integer slot);
        begin
            post_desc_base_l1 = {
                post_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                post_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] gate_desc_base_l1(input integer slot);
        begin
            gate_desc_base_l1 = {
                gate_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                gate_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] silu_desc_base_l1(input integer slot);
        begin
            silu_desc_base_l1 = {
                silu_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                silu_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] down_desc_base_l1(input integer slot);
        begin
            down_desc_base_l1 = {
                down_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                down_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] residual_desc_base_l1(input integer slot);
        begin
            residual_desc_base_l1 = {
                residual_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                residual_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] qkv_desc_base_l2(input integer slot);
        begin
            qkv_desc_base_l2 = {
                qkv_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                qkv_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] frontend_desc_base_l2(input integer slot);
        begin
            frontend_desc_base_l2 = {
                frontend_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                frontend_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] score_desc_base_l2(input integer slot);
        begin
            score_desc_base_l2 = {
                score_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                score_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] oproj_desc_base_l2(input integer slot);
        begin
            oproj_desc_base_l2 = {
                oproj_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                oproj_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] post_desc_base_l2(input integer slot);
        begin
            post_desc_base_l2 = {
                post_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                post_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] gate_desc_base_l2(input integer slot);
        begin
            gate_desc_base_l2 = {
                gate_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                gate_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] silu_desc_base_l2(input integer slot);
        begin
            silu_desc_base_l2 = {
                silu_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                silu_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] down_desc_base_l2(input integer slot);
        begin
            down_desc_base_l2 = {
                down_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                down_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] residual_desc_base_l2(input integer slot);
        begin
            residual_desc_base_l2 = {
                residual_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                residual_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    task patch_frontend_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            frontend_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            frontend_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_score_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            score_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            score_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_oproj_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            oproj_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            oproj_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_post_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            post_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            post_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_gate_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            gate_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            gate_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_silu_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            silu_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            silu_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_down_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            down_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            down_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_residual_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            residual_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            residual_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_qkv_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            qkv_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            qkv_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_frontend_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            frontend_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            frontend_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_score_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            score_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            score_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_oproj_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            oproj_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            oproj_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_post_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            post_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            post_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_gate_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            gate_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            gate_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_silu_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            silu_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            silu_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_down_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            down_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            down_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_residual_base_l1;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            residual_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            residual_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_qkv_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            qkv_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            qkv_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_frontend_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            frontend_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            frontend_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_score_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            score_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            score_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_oproj_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            oproj_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            oproj_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_post_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            post_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            post_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_gate_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            gate_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            gate_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_silu_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            silu_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            silu_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_down_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            down_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            down_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_residual_base_l2;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            residual_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            residual_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task fail_once(input string message);
        begin
            if (print_count < 64) begin
                $display("%s", message);
                print_count = print_count + 1;
            end
            mismatch_count = mismatch_count + 1;
            total_fail_count = total_fail_count + 1;
        end
    endtask

    task load_vectors;
        begin
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_full_image_words32.hex", qkv_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_frontend_image_words32.hex", frontend_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_score_value_image_words32.hex", score_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_o_proj_image_words32.hex", oproj_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_image_words32.hex", post_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_image_words32.hex", gate_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_silu_mul_image_words32.hex", silu_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_down_image_words32.hex", down_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_residual_add_image_words32.hex", residual_qmap);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer1_qkv_from_layer0_rtl_full_image_words32.hex", qkv_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_frontend_image_words32.hex", frontend_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_score_value_image_words32.hex", score_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_o_proj_image_words32.hex", oproj_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_post_attention_residual_norm_image_words32.hex", post_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_gate_up_image_words32.hex", gate_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_silu_mul_image_words32.hex", silu_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_down_image_words32.hex", down_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_residual_add_image_words32.hex", residual_qmap_l1);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer2_qkv_from_layer1_rtl_full_image_words32.hex", qkv_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_frontend_image_words32.hex", frontend_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_score_value_image_words32.hex", score_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_o_proj_image_words32.hex", oproj_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_post_attention_residual_norm_image_words32.hex", post_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_gate_up_image_words32.hex", gate_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_silu_mul_image_words32.hex", silu_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_down_image_words32.hex", down_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_residual_add_image_words32.hex", residual_qmap_l2);

            $readmemh("FPGA_Project/sim/vectors/attention_score_stage_real_k_cache.hex", k_cache_mem);
            $readmemh("FPGA_Project/sim/vectors/attention_softmax_value_stage_real_v_cache.hex", v_cache_mem);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_attention_score_stage_real_k_cache.hex", k_cache_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_attention_softmax_value_stage_real_v_cache.hex", v_cache_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_attention_score_stage_real_k_cache.hex", k_cache_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_attention_softmax_value_stage_real_v_cache.hex", v_cache_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/o_proj_stage_real_weight_words32.hex", oproj_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/o_proj_stage_real_scale_words32.hex", oproj_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_gate_weight_words32.hex", gate_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_gate_scale_words32.hex", gate_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_up_weight_words32.hex", up_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_up_scale_words32.hex", up_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_down_proj_stage_real_weight_words32.hex", down_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_down_proj_stage_real_scale_words32.hex", down_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_o_proj_stage_real_weight_words32.hex", oproj_weight_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_o_proj_stage_real_scale_words32.hex", oproj_scale_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_gate_up_proj_stage_real_gate_weight_words32.hex", gate_weight_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_gate_up_proj_stage_real_gate_scale_words32.hex", gate_scale_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_gate_up_proj_stage_real_up_weight_words32.hex", up_weight_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_gate_up_proj_stage_real_up_scale_words32.hex", up_scale_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_down_proj_stage_real_weight_words32.hex", down_weight_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_mlp_down_proj_stage_real_scale_words32.hex", down_scale_mem_l1);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_o_proj_stage_real_weight_words32.hex", oproj_weight_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_o_proj_stage_real_scale_words32.hex", oproj_scale_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_gate_up_proj_stage_real_gate_weight_words32.hex", gate_weight_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_gate_up_proj_stage_real_gate_scale_words32.hex", gate_scale_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_gate_up_proj_stage_real_up_weight_words32.hex", up_weight_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_gate_up_proj_stage_real_up_scale_words32.hex", up_scale_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_down_proj_stage_real_weight_words32.hex", down_weight_mem_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_mlp_down_proj_stage_real_scale_words32.hex", down_scale_mem_l2);

            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_full_expected_words32.hex", expected_qkv);
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_frontend_q_rope_expected_words32.hex", expected_q_rope);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_addr.hex", expected_cache_addr);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_data.hex", expected_cache_data);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_kind.hex", expected_cache_kind);
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_score_value_attn_out_expected_words32.hex", expected_attn_out);
            $readmemh("FPGA_Project/sim/vectors/qmap_o_proj_expected_words32.hex", expected_o_proj);
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_hidden_words32.hex", expected_post_hidden);
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_norm_words32.hex", expected_post_norm);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_expected_gate_words32.hex", expected_gate);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_expected_up_words32.hex", expected_up);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_silu_mul_expected_hidden_words32.hex", expected_silu_hidden);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_down_expected_words32.hex", expected_down);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_residual_add_expected_words32.hex", expected_layer);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer1_qkv_from_layer0_rtl_full_expected_words32.hex", expected_qkv_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_frontend_q_rope_expected_words32.hex", expected_q_rope_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_kv_cache_append_real_expected_addr.hex", expected_cache_addr_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_kv_cache_append_real_expected_data.hex", expected_cache_data_l1);
            $readmemh("FPGA_Project/sim/vectors/layer1_chained_kv_cache_append_real_expected_kind.hex", expected_cache_kind_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_score_value_attn_out_expected_words32.hex", expected_attn_out_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_o_proj_expected_words32.hex", expected_o_proj_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_post_attention_residual_norm_expected_hidden_words32.hex", expected_post_hidden_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_post_attention_residual_norm_expected_norm_words32.hex", expected_post_norm_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_gate_up_expected_gate_words32.hex", expected_gate_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_gate_up_expected_up_words32.hex", expected_up_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_silu_mul_expected_hidden_words32.hex", expected_silu_hidden_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_down_expected_words32.hex", expected_down_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_residual_add_expected_words32.hex", expected_layer_l1);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer2_qkv_from_layer1_rtl_full_expected_words32.hex", expected_qkv_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_frontend_q_rope_expected_words32.hex", expected_q_rope_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_kv_cache_append_real_expected_addr.hex", expected_cache_addr_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_kv_cache_append_real_expected_data.hex", expected_cache_data_l2);
            $readmemh("FPGA_Project/sim/vectors/layer2_chained_kv_cache_append_real_expected_kind.hex", expected_cache_kind_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_score_value_attn_out_expected_words32.hex", expected_attn_out_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_o_proj_expected_words32.hex", expected_o_proj_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_post_attention_residual_norm_expected_hidden_words32.hex", expected_post_hidden_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_post_attention_residual_norm_expected_norm_words32.hex", expected_post_norm_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_gate_up_expected_gate_words32.hex", expected_gate_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_gate_up_expected_up_words32.hex", expected_up_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_silu_mul_expected_hidden_words32.hex", expected_silu_hidden_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_down_expected_words32.hex", expected_down_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_residual_add_expected_words32.hex", expected_layer_l2);

            qkv_q_base = qkv_desc_base(QKV_SLOT_Q_OUT);
            qkv_k_base = qkv_desc_base(QKV_SLOT_K_OUT);
            qkv_v_base = qkv_desc_base(QKV_SLOT_V_OUT);
            q_rope_base = frontend_desc_base(FRONT_SLOT_Q_ROPE);
            attn_out_base = score_desc_base(SCORE_SLOT_ATTN_OUT);
            o_proj_base = oproj_desc_base(OPROJ_SLOT_OUTPUT);
            post_hidden_base = post_desc_base(POST_SLOT_HIDDEN);
            post_norm_base = post_desc_base(POST_SLOT_NORM);
            gate_output_base = gate_desc_base(GATE_SLOT_GATE_OUTPUT);
            up_output_base = gate_desc_base(GATE_SLOT_UP_OUTPUT);
            silu_hidden_base = silu_desc_base(SILU_SLOT_HIDDEN);
            down_output_base = down_desc_base(DOWN_SLOT_OUTPUT);
            layer_output_base = residual_desc_base(RESIDUAL_SLOT_OUTPUT);
            qkv_q_base_l1 = qkv_desc_base_l1(QKV_SLOT_Q_OUT);
            qkv_k_base_l1 = qkv_desc_base_l1(QKV_SLOT_K_OUT);
            qkv_v_base_l1 = qkv_desc_base_l1(QKV_SLOT_V_OUT);
            q_rope_base_l1 = frontend_desc_base_l1(FRONT_SLOT_Q_ROPE);
            attn_out_base_l1 = score_desc_base_l1(SCORE_SLOT_ATTN_OUT);
            o_proj_base_l1 = oproj_desc_base_l1(OPROJ_SLOT_OUTPUT);
            post_hidden_base_l1 = post_desc_base_l1(POST_SLOT_HIDDEN);
            post_norm_base_l1 = post_desc_base_l1(POST_SLOT_NORM);
            gate_output_base_l1 = gate_desc_base_l1(GATE_SLOT_GATE_OUTPUT);
            up_output_base_l1 = gate_desc_base_l1(GATE_SLOT_UP_OUTPUT);
            silu_hidden_base_l1 = silu_desc_base_l1(SILU_SLOT_HIDDEN);
            down_output_base_l1 = down_desc_base_l1(DOWN_SLOT_OUTPUT);
            layer_output_base_l1 = residual_desc_base_l1(RESIDUAL_SLOT_OUTPUT);
            qkv_q_base_l2 = qkv_desc_base_l2(QKV_SLOT_Q_OUT);
            qkv_k_base_l2 = qkv_desc_base_l2(QKV_SLOT_K_OUT);
            qkv_v_base_l2 = qkv_desc_base_l2(QKV_SLOT_V_OUT);
            q_rope_base_l2 = frontend_desc_base_l2(FRONT_SLOT_Q_ROPE);
            attn_out_base_l2 = score_desc_base_l2(SCORE_SLOT_ATTN_OUT);
            o_proj_base_l2 = oproj_desc_base_l2(OPROJ_SLOT_OUTPUT);
            post_hidden_base_l2 = post_desc_base_l2(POST_SLOT_HIDDEN);
            post_norm_base_l2 = post_desc_base_l2(POST_SLOT_NORM);
            gate_output_base_l2 = gate_desc_base_l2(GATE_SLOT_GATE_OUTPUT);
            up_output_base_l2 = gate_desc_base_l2(GATE_SLOT_UP_OUTPUT);
            silu_hidden_base_l2 = silu_desc_base_l2(SILU_SLOT_HIDDEN);
            down_output_base_l2 = down_desc_base_l2(DOWN_SLOT_OUTPUT);
            layer_output_base_l2 = residual_desc_base_l2(RESIDUAL_SLOT_OUTPUT);

            patch_frontend_base(FRONT_SLOT_Q_FLAT, qkv_q_base);
            patch_frontend_base(FRONT_SLOT_K_FLAT, qkv_k_base);
            patch_frontend_base(FRONT_SLOT_V_FLAT, qkv_v_base);
            patch_score_base(SCORE_SLOT_Q_ROPE, q_rope_base);
            patch_oproj_base(OPROJ_SLOT_ACTIVATION, attn_out_base);
            patch_post_base(POST_SLOT_O_PROJ, o_proj_base);
            patch_gate_base(GATE_SLOT_ACTIVATION, post_norm_base);
            patch_silu_base(SILU_SLOT_GATE, gate_output_base);
            patch_silu_base(SILU_SLOT_UP, up_output_base);
            patch_down_base(DOWN_SLOT_ACTIVATION, silu_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_POST_ATTN, post_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_DOWN, down_output_base);
            patch_qkv_base_l1(QKV_SLOT_ACTIVATION, layer_output_base);
            patch_frontend_base_l1(FRONT_SLOT_Q_FLAT, qkv_q_base_l1);
            patch_frontend_base_l1(FRONT_SLOT_K_FLAT, qkv_k_base_l1);
            patch_frontend_base_l1(FRONT_SLOT_V_FLAT, qkv_v_base_l1);
            patch_score_base_l1(SCORE_SLOT_Q_ROPE, q_rope_base_l1);
            patch_oproj_base_l1(OPROJ_SLOT_ACTIVATION, attn_out_base_l1);
            patch_post_base_l1(POST_SLOT_RESIDUAL, layer_output_base);
            patch_post_base_l1(POST_SLOT_O_PROJ, o_proj_base_l1);
            patch_gate_base_l1(GATE_SLOT_ACTIVATION, post_norm_base_l1);
            patch_silu_base_l1(SILU_SLOT_GATE, gate_output_base_l1);
            patch_silu_base_l1(SILU_SLOT_UP, up_output_base_l1);
            patch_down_base_l1(DOWN_SLOT_ACTIVATION, silu_hidden_base_l1);
            patch_residual_base_l1(RESIDUAL_SLOT_POST_ATTN, post_hidden_base_l1);
            patch_residual_base_l1(RESIDUAL_SLOT_DOWN, down_output_base_l1);
            patch_qkv_base_l2(QKV_SLOT_ACTIVATION, layer_output_base_l1);
            patch_frontend_base_l2(FRONT_SLOT_Q_FLAT, qkv_q_base_l2);
            patch_frontend_base_l2(FRONT_SLOT_K_FLAT, qkv_k_base_l2);
            patch_frontend_base_l2(FRONT_SLOT_V_FLAT, qkv_v_base_l2);
            patch_score_base_l2(SCORE_SLOT_Q_ROPE, q_rope_base_l2);
            patch_oproj_base_l2(OPROJ_SLOT_ACTIVATION, attn_out_base_l2);
            patch_post_base_l2(POST_SLOT_RESIDUAL, layer_output_base_l1);
            patch_post_base_l2(POST_SLOT_O_PROJ, o_proj_base_l2);
            patch_gate_base_l2(GATE_SLOT_ACTIVATION, post_norm_base_l2);
            patch_silu_base_l2(SILU_SLOT_GATE, gate_output_base_l2);
            patch_silu_base_l2(SILU_SLOT_UP, up_output_base_l2);
            patch_down_base_l2(DOWN_SLOT_ACTIVATION, silu_hidden_base_l2);
            patch_residual_base_l2(RESIDUAL_SLOT_POST_ATTN, post_hidden_base_l2);
            patch_residual_base_l2(RESIDUAL_SLOT_DOWN, down_output_base_l2);
        end
    endtask

    function automatic logic [31 : 0] cache_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer element_index;
        integer layer;
        integer layer_word_index;
        integer kind;
        integer rem0;
        integer head;
        integer rem1;
        integer position;
        integer dim;
        integer cache_index;
        logic signed [IN_WIDTH-1 : 0] value;
        begin
            element_index = (addr - CACHE_BASE_ADDR) >> 2;
            layer = element_index / CACHE_WORDS_FULL;
            layer_word_index = element_index % CACHE_WORDS_FULL;
            kind = layer_word_index / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            rem0 = layer_word_index % (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            head = rem0 / (MAX_CONTEXT * HEAD_DIM);
            rem1 = rem0 % (MAX_CONTEXT * HEAD_DIM);
            position = rem1 / HEAD_DIM;
            dim = rem1 % HEAD_DIM;
            cache_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;

            if ((layer < MODELED_CACHE_LAYERS) &&
                (position < CACHE_LENGTH) && (head < NUM_KV_HEADS) && (dim < HEAD_DIM)) begin
                if (kind == 0) begin
                    if (layer == 0) begin
                        value = k_cache_mem[cache_index];
                    end
                    else if (layer == 1) begin
                        value = k_cache_mem_l1[cache_index];
                    end
                    else begin
                        value = k_cache_mem_l2[cache_index];
                    end
                end
                else begin
                    if (layer == 0) begin
                        value = v_cache_mem[cache_index];
                    end
                    else if (layer == 1) begin
                        value = v_cache_mem_l1[cache_index];
                    end
                    else begin
                        value = v_cache_mem_l2[cache_index];
                    end
                end
                cache_word = {{8{value[IN_WIDTH-1]}}, value};
            end
            else begin
                cache_word = 32'hCAFE_BAD0;
                if (print_count < 64) begin
                    $display("FAIL: cache read outside modeled cache length");
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
                total_fail_count = total_fail_count + 1;
            end
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer index;
        begin
            memory_word = 32'hBAD0_BAD0;
            if (in_range(addr, `QMAP_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - `QMAP_QKV_BASE_ADDR) >> 2;
                memory_word = qkv_qmap[index];
            end
            else if (in_range(addr, `QMAP_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2;
                memory_word = frontend_qmap[index];
            end
            else if (in_range(addr, `QMAP_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                memory_word = score_qmap[index];
            end
            else if (in_range(addr, `QMAP_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
                memory_word = oproj_qmap[index];
            end
            else if (in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2;
                memory_word = post_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2;
                memory_word = gate_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2;
                memory_word = silu_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_BASE_ADDR) >> 2;
                memory_word = down_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                memory_word = residual_qmap[index];
            end
            else if (in_range(addr, QMAP_LAYER1_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_QKV_BASE_ADDR) >> 2;
                memory_word = qkv_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR) >> 2;
                memory_word = frontend_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                memory_word = score_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_O_PROJ_BASE_ADDR) >> 2;
                memory_word = oproj_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR) >> 2;
                memory_word = post_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR) >> 2;
                memory_word = gate_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR) >> 2;
                memory_word = silu_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_DOWN_BASE_ADDR) >> 2;
                memory_word = down_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                memory_word = residual_qmap_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER2_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_QKV_BASE_ADDR) >> 2;
                memory_word = qkv_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR) >> 2;
                memory_word = frontend_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                memory_word = score_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_O_PROJ_BASE_ADDR) >> 2;
                memory_word = oproj_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR) >> 2;
                memory_word = post_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR) >> 2;
                memory_word = gate_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR) >> 2;
                memory_word = silu_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_DOWN_BASE_ADDR) >> 2;
                memory_word = down_qmap_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                memory_word = residual_qmap_l2[index];
            end
            else if (is_cache_addr(addr)) begin
                memory_word = cache_word(addr);
            end
            else if (in_range(addr, `QMAP_O_PROJ_WEIGHT_BASE_ADDR, OPROJ_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_O_PROJ_WEIGHT_BASE_ADDR) >> 2;
                memory_word = oproj_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_O_PROJ_SCALE_BASE_ADDR, OPROJ_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_O_PROJ_SCALE_BASE_ADDR) >> 2;
                memory_word = oproj_scale_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_WEIGHT_BASE_ADDR) >> 2;
                memory_word = gate_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_SCALE_BASE_ADDR) >> 2;
                memory_word = gate_scale_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_UP_WEIGHT_BASE_ADDR) >> 2;
                memory_word = up_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_UP_SCALE_BASE_ADDR) >> 2;
                memory_word = up_scale_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR) >> 2;
                memory_word = down_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_SCALE_BASE_ADDR) >> 2;
                memory_word = down_scale_mem[index];
            end
            else if (in_range(addr, QMAP_LAYER1_O_PROJ_WEIGHT_BASE_ADDR, OPROJ_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_O_PROJ_WEIGHT_BASE_ADDR) >> 2;
                memory_word = oproj_weight_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_O_PROJ_SCALE_BASE_ADDR, OPROJ_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_O_PROJ_SCALE_BASE_ADDR) >> 2;
                memory_word = oproj_scale_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_GATE_WEIGHT_BASE_ADDR) >> 2;
                memory_word = gate_weight_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_GATE_SCALE_BASE_ADDR) >> 2;
                memory_word = gate_scale_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_UP_WEIGHT_BASE_ADDR) >> 2;
                memory_word = up_weight_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_UP_SCALE_BASE_ADDR) >> 2;
                memory_word = up_scale_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_DOWN_WEIGHT_BASE_ADDR) >> 2;
                memory_word = down_weight_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_DOWN_SCALE_BASE_ADDR) >> 2;
                memory_word = down_scale_mem_l1[index];
            end
            else if (in_range(addr, QMAP_LAYER2_O_PROJ_WEIGHT_BASE_ADDR, OPROJ_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_O_PROJ_WEIGHT_BASE_ADDR) >> 2;
                memory_word = oproj_weight_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_O_PROJ_SCALE_BASE_ADDR, OPROJ_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_O_PROJ_SCALE_BASE_ADDR) >> 2;
                memory_word = oproj_scale_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_GATE_WEIGHT_BASE_ADDR) >> 2;
                memory_word = gate_weight_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_GATE_SCALE_BASE_ADDR) >> 2;
                memory_word = gate_scale_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_UP_WEIGHT_BASE_ADDR) >> 2;
                memory_word = up_weight_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_UP_SCALE_BASE_ADDR) >> 2;
                memory_word = up_scale_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_DOWN_WEIGHT_BASE_ADDR) >> 2;
                memory_word = down_weight_mem_l2[index];
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_DOWN_SCALE_BASE_ADDR) >> 2;
                memory_word = down_scale_mem_l2[index];
            end
            else begin
                if (print_count < 64) begin
                    $display("FAIL: read from unknown address 0x%016h", addr);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
                total_fail_count = total_fail_count + 1;
            end
        end
    endfunction

    task write_qmap_word;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        integer index;
        begin
            if (in_range(addr, `QMAP_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - `QMAP_QKV_BASE_ADDR) >> 2;
                qkv_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2;
                frontend_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                score_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
                oproj_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2;
                post_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2;
                gate_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2;
                silu_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_BASE_ADDR) >> 2;
                down_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                residual_qmap[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_QKV_BASE_ADDR) >> 2;
                qkv_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR) >> 2;
                frontend_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                score_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_O_PROJ_BASE_ADDR) >> 2;
                oproj_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR) >> 2;
                post_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR) >> 2;
                gate_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR) >> 2;
                silu_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_DOWN_BASE_ADDR) >> 2;
                down_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                residual_qmap_l1[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_QKV_BASE_ADDR, QKV_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_QKV_BASE_ADDR) >> 2;
                qkv_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR) >> 2;
                frontend_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                score_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_O_PROJ_BASE_ADDR) >> 2;
                oproj_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR) >> 2;
                post_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR) >> 2;
                gate_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR) >> 2;
                silu_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_DOWN_BASE_ADDR) >> 2;
                down_qmap_l2[index] = data;
            end
            else if (in_range(addr, QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                residual_qmap_l2[index] = data;
            end
            else begin
                fail_once("FAIL: write to unknown QMAP address");
            end
        end
    endtask

    task prefill_layer0_output_for_l1_only;
        integer i;
        begin
            for (i = 0; i < VEC1024; i = i + 1) begin
                write_qmap_word(layer_output_base + (i * MEM_DATA_BYTES), expected_layer[i]);
            end
            layer_written = 1'b1;
        end
    endtask

    task prefill_layer1_output_for_l2_only;
        integer i;
        begin
            for (i = 0; i < VEC1024; i = i + 1) begin
                write_qmap_word(layer_output_base_l1 + (i * MEM_DATA_BYTES), expected_layer_l1[i]);
            end
            layer_written = 1'b1;
        end
    endtask

    task update_cache_word;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        integer element_index;
        integer layer;
        integer layer_word_index;
        integer kind;
        integer rem0;
        integer head;
        integer rem1;
        integer position;
        integer dim;
        integer cache_index;
        begin
            element_index = (addr - CACHE_BASE_ADDR) >> 2;
            layer = element_index / CACHE_WORDS_FULL;
            layer_word_index = element_index % CACHE_WORDS_FULL;
            kind = layer_word_index / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            rem0 = layer_word_index % (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            head = rem0 / (MAX_CONTEXT * HEAD_DIM);
            rem1 = rem0 % (MAX_CONTEXT * HEAD_DIM);
            position = rem1 / HEAD_DIM;
            dim = rem1 % HEAD_DIM;
            cache_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;

            if ((layer < MODELED_CACHE_LAYERS) &&
                (position < CACHE_LENGTH) && (head < NUM_KV_HEADS) && (dim < HEAD_DIM)) begin
                if (kind == 0) begin
                    if (layer == 0) begin
                        k_cache_mem[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                    else if (layer == 1) begin
                        k_cache_mem_l1[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                    else begin
                        k_cache_mem_l2[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                end
                else begin
                    if (layer == 0) begin
                        v_cache_mem[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                    else if (layer == 1) begin
                        v_cache_mem_l1[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                    else begin
                        v_cache_mem_l2[cache_index] = data[IN_WIDTH-1 : 0];
                    end
                end
            end
            else begin
                fail_once("FAIL: cache write outside modeled cache length");
            end
        end
    endtask

    function automatic integer classify_write_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            classify_write_addr = 0;
            if (in_range(addr, qkv_q_base, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_Q;
            end
            else if (in_range(addr, qkv_k_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_K;
            end
            else if (in_range(addr, qkv_v_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_V;
            end
            else if (is_cache_addr(addr)) begin
                classify_write_addr = WRITE_CACHE;
            end
            else if (in_range(addr, q_rope_base, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_Q_ROPE;
            end
            else if (in_range(addr, attn_out_base, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_ATTN_OUT;
            end
            else if (in_range(addr, o_proj_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_O_PROJ;
            end
            else if (in_range(addr, post_hidden_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_HIDDEN;
            end
            else if (in_range(addr, post_norm_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_NORM;
            end
            else if (in_range(addr, gate_output_base, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_GATE;
            end
            else if (in_range(addr, up_output_base, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_UP;
            end
            else if (in_range(addr, silu_hidden_base, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_SILU_HIDDEN;
            end
            else if (in_range(addr, down_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_DOWN;
            end
            else if (in_range(addr, layer_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_LAYER;
            end
            else if (in_range(addr, qkv_q_base_l1, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_Q;
            end
            else if (in_range(addr, qkv_k_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_K;
            end
            else if (in_range(addr, qkv_v_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_V;
            end
            else if (in_range(addr, q_rope_base_l1, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_Q_ROPE;
            end
            else if (in_range(addr, attn_out_base_l1, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_ATTN_OUT;
            end
            else if (in_range(addr, o_proj_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_O_PROJ;
            end
            else if (in_range(addr, post_hidden_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_HIDDEN;
            end
            else if (in_range(addr, post_norm_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_NORM;
            end
            else if (in_range(addr, gate_output_base_l1, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_GATE;
            end
            else if (in_range(addr, up_output_base_l1, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_UP;
            end
            else if (in_range(addr, silu_hidden_base_l1, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_SILU_HIDDEN;
            end
            else if (in_range(addr, down_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_DOWN;
            end
            else if (in_range(addr, layer_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_LAYER;
            end
            else if (in_range(addr, qkv_q_base_l2, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_Q;
            end
            else if (in_range(addr, qkv_k_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_K;
            end
            else if (in_range(addr, qkv_v_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_QKV_V;
            end
            else if (in_range(addr, q_rope_base_l2, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_Q_ROPE;
            end
            else if (in_range(addr, attn_out_base_l2, VEC2048 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_ATTN_OUT;
            end
            else if (in_range(addr, o_proj_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_O_PROJ;
            end
            else if (in_range(addr, post_hidden_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_HIDDEN;
            end
            else if (in_range(addr, post_norm_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_POST_NORM;
            end
            else if (in_range(addr, gate_output_base_l2, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_GATE;
            end
            else if (in_range(addr, up_output_base_l2, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_UP;
            end
            else if (in_range(addr, silu_hidden_base_l2, VEC3072 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_SILU_HIDDEN;
            end
            else if (in_range(addr, down_output_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_DOWN;
            end
            else if (in_range(addr, layer_output_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_LAYER;
            end
        end
    endfunction

    function automatic integer cache_layer_for_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        integer element_index;
        begin
            element_index = (addr - CACHE_BASE_ADDR) >> 2;
            cache_layer_for_addr = element_index / CACHE_WORDS_FULL;
        end
    endfunction

    function automatic integer write_layer_for_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            write_layer_for_addr = 0;
            if (is_cache_addr(addr)) begin
                write_layer_for_addr = cache_layer_for_addr(addr);
            end
            else if (in_range(addr, qkv_q_base_l1, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, qkv_k_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, qkv_v_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, q_rope_base_l1, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, attn_out_base_l1, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, o_proj_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, post_hidden_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, post_norm_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, gate_output_base_l1, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, up_output_base_l1, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, silu_hidden_base_l1, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, down_output_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, layer_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                write_layer_for_addr = 1;
            end
            else if (in_range(addr, qkv_q_base_l2, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, qkv_k_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, qkv_v_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, q_rope_base_l2, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, attn_out_base_l2, VEC2048 * MEM_DATA_BYTES) ||
                     in_range(addr, o_proj_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, post_hidden_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, post_norm_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, gate_output_base_l2, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, up_output_base_l2, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, silu_hidden_base_l2, VEC3072 * MEM_DATA_BYTES) ||
                     in_range(addr, down_output_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, layer_output_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                write_layer_for_addr = 2;
            end
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] expected_write_base(
        input integer kind,
        input integer layer
    );
        begin
            expected_write_base = '0;
            case (kind)
                WRITE_QKV_Q: expected_write_base = (layer == 0) ? qkv_q_base : ((layer == 1) ? qkv_q_base_l1 : qkv_q_base_l2);
                WRITE_QKV_K: expected_write_base = (layer == 0) ? qkv_k_base : ((layer == 1) ? qkv_k_base_l1 : qkv_k_base_l2);
                WRITE_QKV_V: expected_write_base = (layer == 0) ? qkv_v_base : ((layer == 1) ? qkv_v_base_l1 : qkv_v_base_l2);
                WRITE_Q_ROPE: expected_write_base = (layer == 0) ? q_rope_base : ((layer == 1) ? q_rope_base_l1 : q_rope_base_l2);
                WRITE_ATTN_OUT: expected_write_base = (layer == 0) ? attn_out_base : ((layer == 1) ? attn_out_base_l1 : attn_out_base_l2);
                WRITE_O_PROJ: expected_write_base = (layer == 0) ? o_proj_base : ((layer == 1) ? o_proj_base_l1 : o_proj_base_l2);
                WRITE_POST_HIDDEN: expected_write_base = (layer == 0) ? post_hidden_base : ((layer == 1) ? post_hidden_base_l1 : post_hidden_base_l2);
                WRITE_POST_NORM: expected_write_base = (layer == 0) ? post_norm_base : ((layer == 1) ? post_norm_base_l1 : post_norm_base_l2);
                WRITE_GATE: expected_write_base = (layer == 0) ? gate_output_base : ((layer == 1) ? gate_output_base_l1 : gate_output_base_l2);
                WRITE_UP: expected_write_base = (layer == 0) ? up_output_base : ((layer == 1) ? up_output_base_l1 : up_output_base_l2);
                WRITE_SILU_HIDDEN: expected_write_base = (layer == 0) ? silu_hidden_base : ((layer == 1) ? silu_hidden_base_l1 : silu_hidden_base_l2);
                WRITE_DOWN: expected_write_base = (layer == 0) ? down_output_base : ((layer == 1) ? down_output_base_l1 : down_output_base_l2);
                WRITE_LAYER: expected_write_base = (layer == 0) ? layer_output_base : ((layer == 1) ? layer_output_base_l1 : layer_output_base_l2);
                default: expected_write_base = '0;
            endcase
        end
    endfunction

    task track_diff;
        input logic [31 : 0] actual;
        input logic [31 : 0] expected;
        longint signed diff;
        longint signed abs_diff;
        begin
            diff = $signed(actual) - $signed(expected);
            abs_diff = (diff < 0) ? -diff : diff;
            if (abs_diff > max_abs_diff) begin
                max_abs_diff = abs_diff;
            end
        end
    endtask

    function automatic logic [31 : 0] expected_qkv_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_qkv_word = expected_qkv[index];
                1: expected_qkv_word = expected_qkv_l1[index];
                2: expected_qkv_word = expected_qkv_l2[index];
                default: expected_qkv_word = 32'hDEAD_E000;
            endcase
        end
    endfunction

    function automatic logic [63 : 0] expected_cache_addr_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_cache_addr_word = expected_cache_addr[index];
                1: expected_cache_addr_word = expected_cache_addr_l1[index];
                2: expected_cache_addr_word = expected_cache_addr_l2[index];
                default: expected_cache_addr_word = 64'hDEAD_E001_DEAD_E001;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_cache_data_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_cache_data_word = expected_cache_data[index];
                1: expected_cache_data_word = expected_cache_data_l1[index];
                2: expected_cache_data_word = expected_cache_data_l2[index];
                default: expected_cache_data_word = 32'hDEAD_E002;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_q_rope_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_q_rope_word = expected_q_rope[index];
                1: expected_q_rope_word = expected_q_rope_l1[index];
                2: expected_q_rope_word = expected_q_rope_l2[index];
                default: expected_q_rope_word = 32'hDEAD_E003;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_attn_out_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_attn_out_word = expected_attn_out[index];
                1: expected_attn_out_word = expected_attn_out_l1[index];
                2: expected_attn_out_word = expected_attn_out_l2[index];
                default: expected_attn_out_word = 32'hDEAD_E004;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_o_proj_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_o_proj_word = expected_o_proj[index];
                1: expected_o_proj_word = expected_o_proj_l1[index];
                2: expected_o_proj_word = expected_o_proj_l2[index];
                default: expected_o_proj_word = 32'hDEAD_E005;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_post_hidden_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_post_hidden_word = expected_post_hidden[index];
                1: expected_post_hidden_word = expected_post_hidden_l1[index];
                2: expected_post_hidden_word = expected_post_hidden_l2[index];
                default: expected_post_hidden_word = 32'hDEAD_E006;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_post_norm_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_post_norm_word = expected_post_norm[index];
                1: expected_post_norm_word = expected_post_norm_l1[index];
                2: expected_post_norm_word = expected_post_norm_l2[index];
                default: expected_post_norm_word = 32'hDEAD_E007;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_gate_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_gate_word = expected_gate[index];
                1: expected_gate_word = expected_gate_l1[index];
                2: expected_gate_word = expected_gate_l2[index];
                default: expected_gate_word = 32'hDEAD_E008;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_up_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_up_word = expected_up[index];
                1: expected_up_word = expected_up_l1[index];
                2: expected_up_word = expected_up_l2[index];
                default: expected_up_word = 32'hDEAD_E009;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_silu_hidden_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_silu_hidden_word = expected_silu_hidden[index];
                1: expected_silu_hidden_word = expected_silu_hidden_l1[index];
                2: expected_silu_hidden_word = expected_silu_hidden_l2[index];
                default: expected_silu_hidden_word = 32'hDEAD_E00A;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_down_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_down_word = expected_down[index];
                1: expected_down_word = expected_down_l1[index];
                2: expected_down_word = expected_down_l2[index];
                default: expected_down_word = 32'hDEAD_E00B;
            endcase
        end
    endfunction

    function automatic logic [31 : 0] expected_layer_word(input integer layer, input integer index);
        begin
            case (layer)
                0: expected_layer_word = expected_layer[index];
                1: expected_layer_word = expected_layer_l1[index];
                2: expected_layer_word = expected_layer_l2[index];
                default: expected_layer_word = 32'hDEAD_E00C;
            endcase
        end
    endfunction

    task check_write_request;
        input integer kind;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer words;
        integer q_index;
        integer k_index;
        integer v_index;
        integer cache_index;
        integer write_layer;
        logic [ADDR_WIDTH-1 : 0] base_addr;
        logic [63 : 0] expected_cache_request_addr;
        begin
            q_index = qkv_q_write_accept_count % VEC2048;
            k_index = qkv_k_write_accept_count % VEC1024;
            v_index = qkv_v_write_accept_count % VEC1024;
            cache_index = cache_write_accept_count % TOTAL_CACHE_WRITES;
            write_layer = write_layer_for_addr(addr);
            active_write_layer = write_layer;
            case (kind)
                WRITE_QKV_Q: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((words != 1) ||
                        (qkv_q_write_accept_count >= (scoreboard_layer_iterations * VEC2048)) ||
                        (addr !== (base_addr + (q_index * MEM_DATA_BYTES)))) begin
                        fail_once("FAIL: QKV Q output write request mismatch");
                    end
                    if (q_index == 0) begin
                        qkv_q_written = 1'b0;
                    end
                end
                WRITE_QKV_K: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((words != 1) ||
                        (qkv_k_write_accept_count >= (scoreboard_layer_iterations * VEC1024)) ||
                        (addr !== (base_addr + (k_index * MEM_DATA_BYTES)))) begin
                        fail_once("FAIL: QKV K output write request mismatch");
                    end
                    if (k_index == 0) begin
                        qkv_k_written = 1'b0;
                    end
                end
                WRITE_QKV_V: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((words != 1) ||
                        (qkv_v_write_accept_count >= (scoreboard_layer_iterations * VEC1024)) ||
                        (addr !== (base_addr + (v_index * MEM_DATA_BYTES)))) begin
                        fail_once("FAIL: QKV V output write request mismatch");
                    end
                    if (v_index == 0) begin
                        qkv_v_written = 1'b0;
                    end
                end
                WRITE_CACHE: begin
                    expected_cache_request_addr = expected_cache_addr_word(write_layer, cache_index);
                    if ((words != 1) ||
                        (cache_write_accept_count >= (scoreboard_layer_iterations * TOTAL_CACHE_WRITES)) ||
                        (addr !== expected_cache_request_addr)) begin
                        fail_once("FAIL: cache write request mismatch");
                    end
                    if (cache_index == 0) begin
                        cache_written = 1'b0;
                    end
                end
                WRITE_Q_ROPE: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC2048)) begin
                        fail_once("FAIL: Q RoPE write request mismatch");
                    end
                    else begin
                        q_rope_written = 1'b0;
                    end
                end
                WRITE_ATTN_OUT: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC2048)) begin
                        fail_once("FAIL: attention output write request mismatch");
                    end
                    else begin
                        attn_out_written = 1'b0;
                    end
                end
                WRITE_O_PROJ: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: o_proj write request mismatch");
                    end
                    else begin
                        o_proj_written = 1'b0;
                    end
                end
                WRITE_POST_HIDDEN: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: post-hidden write request mismatch");
                    end
                    else begin
                        post_hidden_written = 1'b0;
                    end
                end
                WRITE_POST_NORM: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: post-norm write request mismatch");
                    end
                    else begin
                        post_norm_written = 1'b0;
                    end
                end
                WRITE_GATE: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC3072)) begin
                        fail_once("FAIL: gate write request mismatch");
                    end
                    else begin
                        gate_written = 1'b0;
                    end
                end
                WRITE_UP: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC3072)) begin
                        fail_once("FAIL: up write request mismatch");
                    end
                    else begin
                        up_written = 1'b0;
                    end
                end
                WRITE_SILU_HIDDEN: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC3072)) begin
                        fail_once("FAIL: silu-hidden write request mismatch");
                    end
                    else begin
                        silu_hidden_written = 1'b0;
                    end
                end
                WRITE_DOWN: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: down write request mismatch");
                    end
                    else begin
                        down_written = 1'b0;
                    end
                end
                WRITE_LAYER: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: layer write request mismatch");
                    end
                    else begin
                        layer_written = 1'b0;
                    end
                end
                default: begin
                    fail_once("FAIL: write request to unexpected address");
                end
            endcase
        end
    endtask

    task check_write_word;
        input integer kind;
        input integer index;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        input logic is_last;
        logic [31 : 0] expected;
        integer q_index;
        integer k_index;
        integer v_index;
        integer cache_index;
        integer write_layer;
        logic [63 : 0] expected_cache_word_addr;
        begin
            expected = 32'h0;
            q_index = qkv_q_write_accept_count % VEC2048;
            k_index = qkv_k_write_accept_count % VEC1024;
            v_index = qkv_v_write_accept_count % VEC1024;
            cache_index = cache_write_accept_count % TOTAL_CACHE_WRITES;
            write_layer = active_write_layer;
            case (kind)
                WRITE_QKV_Q: begin
                    if (!is_last) begin
                        fail_once("FAIL: QKV Q single-word write missing last");
                    end
                    expected = expected_qkv_word(write_layer, q_index);
                    if (data !== expected) begin
                        fail_once("FAIL: QKV Q output write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    qkv_q_write_accept_count = qkv_q_write_accept_count + 1;
                    if ((qkv_q_write_accept_count % VEC2048) == 0) begin
                        qkv_q_written = 1'b1;
                    end
                end
                WRITE_QKV_K: begin
                    if (!is_last) begin
                        fail_once("FAIL: QKV K single-word write missing last");
                    end
                    expected = expected_qkv_word(write_layer, VEC2048 + k_index);
                    if (data !== expected) begin
                        fail_once("FAIL: QKV K output write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    qkv_k_write_accept_count = qkv_k_write_accept_count + 1;
                    if ((qkv_k_write_accept_count % VEC1024) == 0) begin
                        qkv_k_written = 1'b1;
                    end
                end
                WRITE_QKV_V: begin
                    if (!is_last) begin
                        fail_once("FAIL: QKV V single-word write missing last");
                    end
                    expected = expected_qkv_word(write_layer, VEC2048 + VEC1024 + v_index);
                    if (data !== expected) begin
                        fail_once("FAIL: QKV V output write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    qkv_v_write_accept_count = qkv_v_write_accept_count + 1;
                    if ((qkv_v_write_accept_count % VEC1024) == 0) begin
                        qkv_v_written = 1'b1;
                    end
                end
                WRITE_CACHE: begin
                    if (cache_write_accept_count >= (scoreboard_layer_iterations * TOTAL_CACHE_WRITES)) begin
                        fail_once("FAIL: extra cache write data");
                    end
                    else begin
                        expected = expected_cache_data_word(write_layer, cache_index);
                        expected_cache_word_addr = expected_cache_addr_word(write_layer, cache_index);
                        if ((addr !== expected_cache_word_addr) ||
                            (data !== expected)) begin
                            fail_once("FAIL: cache write data mismatch");
                            write_mismatch_count = write_mismatch_count + 1;
                        end
                        update_cache_word(addr, data);
                        cache_write_accept_count = cache_write_accept_count + 1;
                        if ((cache_write_accept_count % TOTAL_CACHE_WRITES) == 0) begin
                            cache_written = 1'b1;
                        end
                    end
                end
                WRITE_Q_ROPE: begin
                    expected = expected_q_rope_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: Q RoPE write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    q_rope_write_accept_count = q_rope_write_accept_count + 1;
                    if (is_last) begin
                        q_rope_written = 1'b1;
                    end
                end
                WRITE_ATTN_OUT: begin
                    expected = expected_attn_out_word(write_layer, index);
                    if (data !== expected) begin
                        if (write_mismatch_count < 16) begin
                            $display("FAIL: attn_out mismatch layer=%0d index=%0d addr=0x%016h actual=0x%08h expected=0x%08h diff=%0d",
                                     write_layer, index, addr, data, expected,
                                     ($signed(data) - $signed(expected)));
                        end
                        fail_once("FAIL: attention output write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    attn_out_write_accept_count = attn_out_write_accept_count + 1;
                    if (is_last) begin
                        attn_out_written = 1'b1;
                    end
                end
                WRITE_O_PROJ: begin
                    expected = expected_o_proj_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: o_proj write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    o_proj_write_accept_count = o_proj_write_accept_count + 1;
                    if (is_last) begin
                        o_proj_written = 1'b1;
                    end
                end
                WRITE_POST_HIDDEN: begin
                    expected = expected_post_hidden_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: post-hidden write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    post_hidden_write_accept_count = post_hidden_write_accept_count + 1;
                    if (is_last) begin
                        post_hidden_written = 1'b1;
                    end
                end
                WRITE_POST_NORM: begin
                    expected = expected_post_norm_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: post-norm write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    post_norm_write_accept_count = post_norm_write_accept_count + 1;
                    if (is_last) begin
                        post_norm_written = 1'b1;
                    end
                end
                WRITE_GATE: begin
                    expected = expected_gate_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: gate write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    gate_write_accept_count = gate_write_accept_count + 1;
                    if (is_last) begin
                        gate_written = 1'b1;
                    end
                end
                WRITE_UP: begin
                    expected = expected_up_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: up write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    up_write_accept_count = up_write_accept_count + 1;
                    if (is_last) begin
                        up_written = 1'b1;
                    end
                end
                WRITE_SILU_HIDDEN: begin
                    expected = expected_silu_hidden_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: silu-hidden write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    silu_write_accept_count = silu_write_accept_count + 1;
                    if (is_last) begin
                        silu_hidden_written = 1'b1;
                    end
                end
                WRITE_DOWN: begin
                    expected = expected_down_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: down write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    down_write_accept_count = down_write_accept_count + 1;
                    if (is_last) begin
                        down_written = 1'b1;
                    end
                end
                WRITE_LAYER: begin
                    expected = expected_layer_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: layer write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    layer_write_accept_count = layer_write_accept_count + 1;
                    if (is_last) begin
                        layer_written = 1'b1;
                    end
                end
                default: begin
                    fail_once("FAIL: write data for unknown write kind");
                end
            endcase
        end
    endtask

    task check_chained_read_ready;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer len_bytes;
        begin
            if (in_range(addr, qkv_q_base, VEC2048 * MEM_DATA_BYTES) && !qkv_q_written) begin
                fail_once("FAIL: QKV Q output read before producer write completed");
            end
            if (in_range(addr, qkv_k_base, VEC1024 * MEM_DATA_BYTES) && !qkv_k_written) begin
                fail_once("FAIL: QKV K output read before producer write completed");
            end
            if (in_range(addr, qkv_v_base, VEC1024 * MEM_DATA_BYTES) && !qkv_v_written) begin
                fail_once("FAIL: QKV V output read before producer write completed");
            end
            if (is_cache_addr(addr) && !cache_written && (active_stage_debug > 0)) begin
                fail_once("FAIL: K/V cache read before frontend cache writes completed");
            end
            if (in_range(addr, q_rope_base, VEC2048 * MEM_DATA_BYTES) && !q_rope_written) begin
                fail_once("FAIL: Q RoPE read before producer write completed");
            end
            if (in_range(addr, attn_out_base, VEC2048 * MEM_DATA_BYTES) && !attn_out_written) begin
                fail_once("FAIL: attention output read before producer write completed");
            end
            if (in_range(addr, o_proj_base, VEC1024 * MEM_DATA_BYTES) && !o_proj_written) begin
                fail_once("FAIL: o_proj output read before producer write completed");
            end
            if (in_range(addr, post_hidden_base, VEC1024 * MEM_DATA_BYTES) && !post_hidden_written) begin
                fail_once("FAIL: post_hidden read before producer write completed");
            end
            if (in_range(addr, post_norm_base, VEC1024 * MEM_DATA_BYTES) && !post_norm_written) begin
                fail_once("FAIL: post_norm read before producer write completed");
            end
            if (in_range(addr, gate_output_base, VEC3072 * MEM_DATA_BYTES) && !gate_written) begin
                fail_once("FAIL: gate output read before producer write completed");
            end
            if (in_range(addr, up_output_base, VEC3072 * MEM_DATA_BYTES) && !up_written) begin
                fail_once("FAIL: up output read before producer write completed");
            end
            if (in_range(addr, silu_hidden_base, VEC3072 * MEM_DATA_BYTES) && !silu_hidden_written) begin
                fail_once("FAIL: silu hidden read before producer write completed");
            end
            if (in_range(addr, down_output_base, VEC1024 * MEM_DATA_BYTES) && !down_written) begin
                fail_once("FAIL: down output read before producer write completed");
            end
            if (in_range(addr, layer_output_base, VEC1024 * MEM_DATA_BYTES) && !layer_written) begin
                fail_once("FAIL: layer output read before producer write completed");
            end
            if (in_range(addr, qkv_q_base_l1, VEC2048 * MEM_DATA_BYTES) && !qkv_q_written) begin
                fail_once("FAIL: Layer 1 QKV Q output read before producer write completed");
            end
            if (in_range(addr, qkv_k_base_l1, VEC1024 * MEM_DATA_BYTES) && !qkv_k_written) begin
                fail_once("FAIL: Layer 1 QKV K output read before producer write completed");
            end
            if (in_range(addr, qkv_v_base_l1, VEC1024 * MEM_DATA_BYTES) && !qkv_v_written) begin
                fail_once("FAIL: Layer 1 QKV V output read before producer write completed");
            end
            if (in_range(addr, q_rope_base_l1, VEC2048 * MEM_DATA_BYTES) && !q_rope_written) begin
                fail_once("FAIL: Layer 1 Q RoPE read before producer write completed");
            end
            if (in_range(addr, attn_out_base_l1, VEC2048 * MEM_DATA_BYTES) && !attn_out_written) begin
                fail_once("FAIL: Layer 1 attention output read before producer write completed");
            end
            if (in_range(addr, o_proj_base_l1, VEC1024 * MEM_DATA_BYTES) && !o_proj_written) begin
                fail_once("FAIL: Layer 1 o_proj output read before producer write completed");
            end
            if (in_range(addr, post_hidden_base_l1, VEC1024 * MEM_DATA_BYTES) && !post_hidden_written) begin
                fail_once("FAIL: Layer 1 post_hidden read before producer write completed");
            end
            if (in_range(addr, post_norm_base_l1, VEC1024 * MEM_DATA_BYTES) && !post_norm_written) begin
                fail_once("FAIL: Layer 1 post_norm read before producer write completed");
            end
            if (in_range(addr, gate_output_base_l1, VEC3072 * MEM_DATA_BYTES) && !gate_written) begin
                fail_once("FAIL: Layer 1 gate output read before producer write completed");
            end
            if (in_range(addr, up_output_base_l1, VEC3072 * MEM_DATA_BYTES) && !up_written) begin
                fail_once("FAIL: Layer 1 up output read before producer write completed");
            end
            if (in_range(addr, silu_hidden_base_l1, VEC3072 * MEM_DATA_BYTES) && !silu_hidden_written) begin
                fail_once("FAIL: Layer 1 silu hidden read before producer write completed");
            end
            if (in_range(addr, down_output_base_l1, VEC1024 * MEM_DATA_BYTES) && !down_written) begin
                fail_once("FAIL: Layer 1 down output read before producer write completed");
            end
            if (in_range(addr, layer_output_base_l1, VEC1024 * MEM_DATA_BYTES) && !layer_written) begin
                fail_once("FAIL: Layer 1 layer output read before producer write completed");
            end
            if (in_range(addr, qkv_q_base_l2, VEC2048 * MEM_DATA_BYTES) && !qkv_q_written) begin
                fail_once("FAIL: Layer 2 QKV Q output read before producer write completed");
            end
            if (in_range(addr, qkv_k_base_l2, VEC1024 * MEM_DATA_BYTES) && !qkv_k_written) begin
                fail_once("FAIL: Layer 2 QKV K output read before producer write completed");
            end
            if (in_range(addr, qkv_v_base_l2, VEC1024 * MEM_DATA_BYTES) && !qkv_v_written) begin
                fail_once("FAIL: Layer 2 QKV V output read before producer write completed");
            end
            if (in_range(addr, q_rope_base_l2, VEC2048 * MEM_DATA_BYTES) && !q_rope_written) begin
                fail_once("FAIL: Layer 2 Q RoPE read before producer write completed");
            end
            if (in_range(addr, attn_out_base_l2, VEC2048 * MEM_DATA_BYTES) && !attn_out_written) begin
                fail_once("FAIL: Layer 2 attention output read before producer write completed");
            end
            if (in_range(addr, o_proj_base_l2, VEC1024 * MEM_DATA_BYTES) && !o_proj_written) begin
                fail_once("FAIL: Layer 2 o_proj output read before producer write completed");
            end
            if (in_range(addr, post_hidden_base_l2, VEC1024 * MEM_DATA_BYTES) && !post_hidden_written) begin
                fail_once("FAIL: Layer 2 post_hidden read before producer write completed");
            end
            if (in_range(addr, post_norm_base_l2, VEC1024 * MEM_DATA_BYTES) && !post_norm_written) begin
                fail_once("FAIL: Layer 2 post_norm read before producer write completed");
            end
            if (in_range(addr, gate_output_base_l2, VEC3072 * MEM_DATA_BYTES) && !gate_written) begin
                fail_once("FAIL: Layer 2 gate output read before producer write completed");
            end
            if (in_range(addr, up_output_base_l2, VEC3072 * MEM_DATA_BYTES) && !up_written) begin
                fail_once("FAIL: Layer 2 up output read before producer write completed");
            end
            if (in_range(addr, silu_hidden_base_l2, VEC3072 * MEM_DATA_BYTES) && !silu_hidden_written) begin
                fail_once("FAIL: Layer 2 silu hidden read before producer write completed");
            end
            if (in_range(addr, down_output_base_l2, VEC1024 * MEM_DATA_BYTES) && !down_written) begin
                fail_once("FAIL: Layer 2 down output read before producer write completed");
            end
        end
    endtask

    task clear_scoreboard;
        begin
            cycle_count = 0;
            mismatch_count = 0;
            print_count = 0;
            done_seen_count = 0;
            rd_req_accept_count = 0;
            rd_rsp_accept_count = 0;
            wr_req_accept_count = 0;
            wr_data_accept_count = 0;
            qkv_q_write_accept_count = 0;
            qkv_k_write_accept_count = 0;
            qkv_v_write_accept_count = 0;
            cache_write_accept_count = 0;
            q_rope_write_accept_count = 0;
            attn_out_write_accept_count = 0;
            o_proj_write_accept_count = 0;
            post_hidden_write_accept_count = 0;
            post_norm_write_accept_count = 0;
            gate_write_accept_count = 0;
            up_write_accept_count = 0;
            silu_write_accept_count = 0;
            down_write_accept_count = 0;
            layer_write_accept_count = 0;
            write_mismatch_count = 0;
            max_abs_diff = 0;
            read_active = 1'b0;
            active_read_index = 0;
            active_words_left = 0;
            active_total_words = 0;
            read_gap_count = 0;
            active_read_addr = '0;
            write_active = 1'b0;
            active_write_kind = 0;
            active_write_index = 0;
            active_write_words_left = 0;
            active_write_total_words = 0;
            write_done_delay = 0;
            active_write_addr = '0;
            active_write_layer = 0;
            qkv_q_written = 1'b0;
            qkv_k_written = 1'b0;
            qkv_v_written = 1'b0;
            cache_written = 1'b0;
            q_rope_written = 1'b0;
            attn_out_written = 1'b0;
            o_proj_written = 1'b0;
            post_hidden_written = 1'b0;
            post_norm_written = 1'b0;
            gate_written = 1'b0;
            up_written = 1'b0;
            silu_hidden_written = 1'b0;
            down_written = 1'b0;
            layer_written = 1'b0;
            mem_rd_rsp_valid = 1'b0;
            mem_rd_rsp_data = 32'd0;
            mem_rd_rsp_last = 1'b0;
            mem_wr_done = 1'b0;
            mem_wr_error = 1'b0;
            last_trace_stage = -1;
            last_trace_state = -1;
            last_trace_wr_words = -1;
            scoreboard_layer_iterations = 1;
            fastmem = 1'b0;
        end
    endtask

    task set_layer_packet_bases;
        input integer layer_index;
        begin
            if (layer_index == 1) begin
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_QKV_BASE_ADDR;
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR;
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR;
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_O_PROJ_BASE_ADDR;
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR;
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR;
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR;
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_MLP_DOWN_BASE_ADDR;
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR;
            end
            else if (layer_index == 2) begin
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_QKV_BASE_ADDR;
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_ATTN_FRONTEND_BASE_ADDR;
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_ATTN_SCORE_VALUE_BASE_ADDR;
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_O_PROJ_BASE_ADDR;
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_POST_ATTN_NORM_BASE_ADDR;
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_MLP_GATE_UP_BASE_ADDR;
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_MLP_SILU_MUL_BASE_ADDR;
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_MLP_DOWN_BASE_ADDR;
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_MLP_RESIDUAL_ADD_BASE_ADDR;
            end
            else begin
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_QKV_BASE_ADDR;
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_FRONTEND_BASE_ADDR;
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_SCORE_VALUE_BASE_ADDR;
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_O_PROJ_BASE_ADDR;
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_POST_ATTN_NORM_BASE_ADDR;
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_GATE_UP_BASE_ADDR;
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_SILU_MUL_BASE_ADDR;
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_DOWN_BASE_ADDR;
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR;
            end
        end
    endtask

    task set_valid_loop_contract;
        begin
            layer_start_index = 5'd0;
            layer_count = 5'd1;
            token_position = CACHE_LENGTH - 1;
            input_hidden_base_addr = qkv_desc_base(QKV_SLOT_ACTIVATION);
            output_hidden_base_addr = layer_output_base;
            kv_cache_base_addr = CACHE_BASE_ADDR;
            qkv_qmap_base_addr_table = '0;
            attn_frontend_qmap_base_addr_table = '0;
            attn_score_value_qmap_base_addr_table = '0;
            o_proj_qmap_base_addr_table = '0;
            post_attn_norm_qmap_base_addr_table = '0;
            mlp_gate_up_qmap_base_addr_table = '0;
            mlp_silu_mul_qmap_base_addr_table = '0;
            mlp_down_qmap_base_addr_table = '0;
            mlp_residual_add_qmap_base_addr_table = '0;
            set_layer_packet_bases(0);
        end
    endtask

    task run_until_done;
        input integer timeout_cycles;
        integer next_progress_cycle;
        begin
            next_progress_cycle = 1000000;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            while ((done != 1'b1) && (cycle_count < timeout_cycles)) begin
                @(posedge clk);
                if ($test$plusargs("progress") && (cycle_count >= next_progress_cycle)) begin
                    $display(
                        "PROGRESS: cycle=%0d state=0x%0h active_layer=%0d started=%0d completed=%0d rd=%0d/%0d wr=%0d/%0d",
                        cycle_count,
                        state_debug,
                        active_layer_index,
                        layers_started,
                        layers_completed,
                        dut_read_burst_count,
                        dut_read_word_count,
                        dut_write_req_count,
                        dut_write_word_count
                    );
                    $fflush();
                    next_progress_cycle = next_progress_cycle + 1000000;
                end
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for qmap_one_token_layer_scheduler done");
                $finish(1);
            end
            @(posedge clk);
        end
    endtask

    task corrupt_qkv_activation_dtype;
        integer dtype_word_index;
        begin
            dtype_word_index =
                (`QMAP_QKV_DESCRIPTOR_TABLE_ADDR - `QMAP_QKV_BASE_ADDR +
                 (64'd1 * 64'd128) + (DESC_DTYPE_WORD * 4)) >> 2;
            qkv_qmap[dtype_word_index] = 32'd5;
        end
    endtask

    task corrupt_frontend_cos_dtype;
        integer cos_dtype_word_index;
        begin
            cos_dtype_word_index =
                (`QMAP_ATTN_FRONTEND_DESCRIPTOR_TABLE_ADDR - `QMAP_ATTN_FRONTEND_BASE_ADDR +
                 (64'd6 * 64'd128) + (DESC_DTYPE_WORD * 4)) >> 2;
            frontend_qmap[cos_dtype_word_index] = 32'd5;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            read_active <= 1'b0;
            write_active <= 1'b0;
            write_done_delay <= 0;
            cycle_count <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                if ((mem_rd_req_addr[1 : 0] != 2'd0) || (mem_rd_req_len_bytes[1 : 0] != 2'd0)) begin
                    fail_once("FAIL: unaligned read request");
                end
                check_chained_read_ready(mem_rd_req_addr, mem_rd_req_len_bytes);
                read_active <= 1'b1;
                active_read_addr <= mem_rd_req_addr;
                active_read_index <= 0;
                active_words_left <= mem_rd_req_len_bytes / MEM_DATA_BYTES;
                active_total_words <= mem_rd_req_len_bytes / MEM_DATA_BYTES;
                read_gap_count <= fastmem ? 0 : (rd_req_accept_count % 5);
                rd_req_accept_count = rd_req_accept_count + 1;
            end

            if (read_active) begin
                if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                end
                else if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                    rd_rsp_accept_count = rd_rsp_accept_count + 1;
                    if (mem_rd_rsp_last) begin
                        mem_rd_rsp_valid <= 1'b0;
                        mem_rd_rsp_last <= 1'b0;
                        read_active <= 1'b0;
                    end
                    else begin
                        active_read_index <= active_read_index + 1;
                        active_words_left <= active_words_left - 1;
                        mem_rd_rsp_valid <= 1'b0;
                        mem_rd_rsp_last <= 1'b0;
                        read_gap_count <= (fastmem || ((active_read_index % 17) != 3)) ? 0 : 2;
                    end
                end
                else if (!mem_rd_rsp_valid) begin
                    if (read_gap_count > 0) begin
                        read_gap_count <= read_gap_count - 1;
                    end
                    else begin
                        mem_rd_rsp_valid <= 1'b1;
                        mem_rd_rsp_data <= memory_word(active_read_addr + (active_read_index * MEM_DATA_BYTES));
                        mem_rd_rsp_last <= (active_words_left == 1);
                    end
                end
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                if ((mem_wr_req_addr[1 : 0] != 2'd0) || (mem_wr_req_len_bytes[1 : 0] != 2'd0)) begin
                    fail_once("FAIL: unaligned write request");
                end
                write_active <= 1'b1;
                active_write_addr <= mem_wr_req_addr;
                active_write_index <= 0;
                active_write_words_left <= mem_wr_req_len_bytes / MEM_DATA_BYTES;
                active_write_total_words <= mem_wr_req_len_bytes / MEM_DATA_BYTES;
                active_write_kind <= classify_write_addr(mem_wr_req_addr);
                check_write_request(classify_write_addr(mem_wr_req_addr), mem_wr_req_addr, mem_wr_req_len_bytes / MEM_DATA_BYTES);
                wr_req_accept_count = wr_req_accept_count + 1;
            end

            if (write_active && mem_wr_data_valid && mem_wr_data_ready) begin
                check_write_word(
                    active_write_kind,
                    active_write_index,
                    active_write_addr + (active_write_index * MEM_DATA_BYTES),
                    mem_wr_data,
                    mem_wr_data_last
                );
                wr_data_accept_count = wr_data_accept_count + 1;
                if (active_write_words_left == 1) begin
                    if (!mem_wr_data_last) begin
                        fail_once("FAIL: final write word missing last");
                    end
                    write_active <= 1'b0;
                    active_write_words_left <= 0;
                    write_done_delay <= 3;
                end
                else begin
                    if (mem_wr_data_last) begin
                        fail_once("FAIL: early write last");
                    end
                    active_write_words_left <= active_write_words_left - 1;
                    active_write_index <= active_write_index + 1;
                end
            end

            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) begin
                    mem_wr_done <= 1'b1;
                end
            end

            if (done) begin
                done_seen_count = done_seen_count + 1;
            end

            if (trace_fd != 0) begin
                if ((mem_rd_req_valid && mem_rd_req_ready) ||
                    (mem_rd_rsp_valid && mem_rd_rsp_ready && mem_rd_rsp_last) ||
                    (mem_wr_req_valid && mem_wr_req_ready) ||
                    (mem_wr_data_valid && mem_wr_data_ready && mem_wr_data_last) ||
                    mem_wr_done ||
                    done ||
                    (last_trace_stage != active_stage_debug) ||
                    (last_trace_state != state_debug) ||
                    (last_trace_wr_words != dut_write_word_count)) begin
                    $fwrite(
                        trace_fd,
                        "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%0h,0x%0h,%0d,%0d,0x%0h,0x%0h,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%0h,0x%0h\n",
                        cycle_count,
                        busy,
                        done,
                        error,
                        active_layer_index,
                        layers_started,
                        layers_completed,
                        layer_done_mask,
                        layer_error_mask,
                        active_stage_debug,
                        state_debug,
                        stage_done_mask,
                        stage_error_mask,
                        mem_rd_req_valid,
                        mem_rd_req_ready,
                        mem_rd_req_valid && mem_rd_req_ready,
                        mem_rd_req_addr,
                        mem_rd_req_len_bytes,
                        mem_rd_rsp_valid,
                        mem_rd_rsp_ready,
                        mem_rd_rsp_valid && mem_rd_rsp_ready && mem_rd_rsp_last,
                        mem_wr_req_valid,
                        mem_wr_req_addr,
                        mem_wr_req_len_bytes,
                        mem_wr_req_valid && mem_wr_req_ready,
                        mem_wr_data_valid,
                        mem_wr_data_ready,
                        mem_wr_data_valid && mem_wr_data_ready && mem_wr_data_last,
                        mem_wr_done,
                        dut_read_burst_count,
                        dut_read_word_count,
                        dut_write_req_count,
                        dut_write_word_count,
                        body_stage_done_mask,
                        body_stage_error_mask
                    );
                    last_trace_stage = active_stage_debug;
                    last_trace_state = state_debug;
                    last_trace_wr_words = dut_write_word_count;
                end
            end
        end
    end

    initial begin
        tracefile = "FPGA_Project/sim/qmap_one_token_layer_scheduler_trace.csv";
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        if ($test$plusargs("notrace")) begin
            trace_fd = 0;
        end
        else begin
            trace_fd = $fopen(tracefile, "w");
            if (trace_fd == 0) begin
                $display("FAIL: could not open trace file %s", tracefile);
                $finish(1);
            end
            $fwrite(
                trace_fd,
                "cycle,busy,done,error,active_layer,layers_started,layers_completed,layer_done,layer_error,layer0_stage,loop_state,stage_done,stage_error,rd_req_valid,rd_req_ready,rd_req_fire,rd_req_addr,rd_req_len,rd_rsp_valid,rd_rsp_ready,rd_rsp_last_fire,wr_req_valid,wr_req_addr,wr_req_len,wr_req_fire,wr_data_valid,wr_data_ready,wr_last_fire,wr_done,rd_bursts,rd_words,wr_reqs,wr_words,body_stage_done,body_stage_error\n"
            );
        end

        rst_n = 1'b0;
        start = 1'b0;
        total_fail_count = 0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        if ($test$plusargs("true2_only")) begin
            set_layer_packet_bases(1);
            layer_count = 5'd2;
            scoreboard_layer_iterations = 2;
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused true two-layer start");
            $fflush();
            run_until_done(40000000);
            two_layer_done_cycle = cycle_count;
            two_layer_read_bursts = dut_read_burst_count;
            two_layer_read_words = dut_read_word_count;
            two_layer_write_reqs = dut_write_req_count;
            two_layer_write_words = dut_write_word_count;
            two_layer_done_mask = layer_done_mask;
            two_layer_error_mask = layer_error_mask;
            two_layer_layers_started = layers_started;
            two_layer_layers_completed = layers_completed;
            two_layer_error = error;
            two_layer_mismatch_count = mismatch_count;
            two_layer_write_mismatch_count = write_mismatch_count;
            two_layer_max_abs_diff = max_abs_diff;

            if (two_layer_error) begin
                fail_once("FAIL: focused true two-layer run asserted error");
            end
            if ((two_layer_layers_started != 5'd2) ||
                (two_layer_layers_completed != 5'd2) ||
                (two_layer_done_mask != 28'h0000003) ||
                (two_layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: focused true two-layer layer masks mismatch");
            end
            if ((two_layer_read_bursts != EXPECTED_TWO_LAYER_RD_REQS) ||
                (two_layer_read_words != EXPECTED_TWO_LAYER_RD_WORDS) ||
                (two_layer_write_reqs != EXPECTED_TWO_LAYER_WR_REQS) ||
                (two_layer_write_words != EXPECTED_TWO_LAYER_WR_WORDS)) begin
                fail_once("FAIL: focused true two-layer memory counters mismatch");
            end
            if ((qkv_q_write_accept_count != (2 * VEC2048)) ||
                (qkv_k_write_accept_count != (2 * VEC1024)) ||
                (qkv_v_write_accept_count != (2 * VEC1024)) ||
                (cache_write_accept_count != (2 * TOTAL_CACHE_WRITES)) ||
                (q_rope_write_accept_count != (2 * VEC2048)) ||
                (attn_out_write_accept_count != (2 * VEC2048)) ||
                (o_proj_write_accept_count != (2 * VEC1024)) ||
                (post_hidden_write_accept_count != (2 * VEC1024)) ||
                (post_norm_write_accept_count != (2 * VEC1024)) ||
                (gate_write_accept_count != (2 * VEC3072)) ||
                (up_write_accept_count != (2 * VEC3072)) ||
                (silu_write_accept_count != (2 * VEC3072)) ||
                (down_write_accept_count != (2 * VEC1024)) ||
                (layer_write_accept_count != (2 * VEC1024))) begin
                fail_once("FAIL: focused true two-layer write-back word counts mismatch");
            end
            if ((two_layer_mismatch_count != 0) ||
                (two_layer_write_mismatch_count != 0) ||
                (two_layer_max_abs_diff != 0)) begin
                $display("FAIL: focused true two-layer found mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                         two_layer_mismatch_count, two_layer_write_mismatch_count, two_layer_max_abs_diff);
                $finish(1);
            end
            if (total_fail_count != 0) begin
                $display("FAIL: focused true two-layer accumulated %0d failure(s)", total_fail_count);
                $finish(1);
            end

            $display("qmap_one_token_layer_scheduler focused true Layer 0 -> Layer 1 test");
            $display("  true two-layer cycle   = %0d", two_layer_done_cycle);
            $display("  true two-layer rd/wr   = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                     two_layer_read_bursts, two_layer_read_words,
                     two_layer_write_reqs, two_layer_write_words,
                     two_layer_done_mask);
            $display("  producer writes        = qkv %0d/%0d/%0d cache %0d qrope %0d attn %0d oproj %0d post %0d/%0d gate/up %0d/%0d silu %0d down %0d layer %0d",
                     qkv_q_write_accept_count, qkv_k_write_accept_count, qkv_v_write_accept_count,
                     cache_write_accept_count, q_rope_write_accept_count, attn_out_write_accept_count,
                     o_proj_write_accept_count, post_hidden_write_accept_count, post_norm_write_accept_count,
                     gate_write_accept_count, up_write_accept_count, silu_write_accept_count,
                     down_write_accept_count, layer_write_accept_count);
            $display("  mismatches             = %0d write_mismatches=%0d max_abs=%0d",
                     two_layer_mismatch_count, two_layer_write_mismatch_count, two_layer_max_abs_diff);
            if (trace_fd != 0) begin
                $display("  trace                  = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace                  = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran a focused true Layer 0 -> Layer 1 QMAP loop with exact write-back.");
            $finish;
        end
        if ($test$plusargs("true3_only")) begin
            set_layer_packet_bases(1);
            set_layer_packet_bases(2);
            layer_count = 5'd3;
            scoreboard_layer_iterations = 3;
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused true three-layer start");
            $fflush();
            run_until_done(60000000);
            three_layer_done_cycle = cycle_count;
            three_layer_read_bursts = dut_read_burst_count;
            three_layer_read_words = dut_read_word_count;
            three_layer_write_reqs = dut_write_req_count;
            three_layer_write_words = dut_write_word_count;
            three_layer_done_mask = layer_done_mask;
            three_layer_error_mask = layer_error_mask;
            three_layer_layers_started = layers_started;
            three_layer_layers_completed = layers_completed;
            three_layer_error = error;
            three_layer_mismatch_count = mismatch_count;
            three_layer_write_mismatch_count = write_mismatch_count;
            three_layer_max_abs_diff = max_abs_diff;

            if (three_layer_error) begin
                fail_once("FAIL: focused true three-layer run asserted error");
            end
            if ((three_layer_layers_started != 5'd3) ||
                (three_layer_layers_completed != 5'd3) ||
                (three_layer_done_mask != 28'h0000007) ||
                (three_layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: focused true three-layer layer masks mismatch");
            end
            if ((three_layer_read_bursts != EXPECTED_THREE_LAYER_RD_REQS) ||
                (three_layer_read_words != EXPECTED_THREE_LAYER_RD_WORDS) ||
                (three_layer_write_reqs != EXPECTED_THREE_LAYER_WR_REQS) ||
                (three_layer_write_words != EXPECTED_THREE_LAYER_WR_WORDS)) begin
                fail_once("FAIL: focused true three-layer memory counters mismatch");
            end
            if ((qkv_q_write_accept_count != (3 * VEC2048)) ||
                (qkv_k_write_accept_count != (3 * VEC1024)) ||
                (qkv_v_write_accept_count != (3 * VEC1024)) ||
                (cache_write_accept_count != (3 * TOTAL_CACHE_WRITES)) ||
                (q_rope_write_accept_count != (3 * VEC2048)) ||
                (attn_out_write_accept_count != (3 * VEC2048)) ||
                (o_proj_write_accept_count != (3 * VEC1024)) ||
                (post_hidden_write_accept_count != (3 * VEC1024)) ||
                (post_norm_write_accept_count != (3 * VEC1024)) ||
                (gate_write_accept_count != (3 * VEC3072)) ||
                (up_write_accept_count != (3 * VEC3072)) ||
                (silu_write_accept_count != (3 * VEC3072)) ||
                (down_write_accept_count != (3 * VEC1024)) ||
                (layer_write_accept_count != (3 * VEC1024))) begin
                fail_once("FAIL: focused true three-layer write-back word counts mismatch");
            end
            if ((three_layer_mismatch_count != 0) ||
                (three_layer_write_mismatch_count != 0) ||
                (three_layer_max_abs_diff != 0)) begin
                $display("FAIL: focused true three-layer found mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                         three_layer_mismatch_count, three_layer_write_mismatch_count, three_layer_max_abs_diff);
                $finish(1);
            end
            if (total_fail_count != 0) begin
                $display("FAIL: focused true three-layer accumulated %0d failure(s)", total_fail_count);
                $finish(1);
            end

            $display("qmap_one_token_layer_scheduler focused true Layer 0 -> Layer 1 -> Layer 2 test");
            $display("  true three-layer cycle = %0d", three_layer_done_cycle);
            $display("  true three-layer rd/wr = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                     three_layer_read_bursts, three_layer_read_words,
                     three_layer_write_reqs, three_layer_write_words,
                     three_layer_done_mask);
            $display("  producer writes        = qkv %0d/%0d/%0d cache %0d qrope %0d attn %0d oproj %0d post %0d/%0d gate/up %0d/%0d silu %0d down %0d layer %0d",
                     qkv_q_write_accept_count, qkv_k_write_accept_count, qkv_v_write_accept_count,
                     cache_write_accept_count, q_rope_write_accept_count, attn_out_write_accept_count,
                     o_proj_write_accept_count, post_hidden_write_accept_count, post_norm_write_accept_count,
                     gate_write_accept_count, up_write_accept_count, silu_write_accept_count,
                     down_write_accept_count, layer_write_accept_count);
            $display("  mismatches             = %0d write_mismatches=%0d max_abs=%0d",
                     three_layer_mismatch_count, three_layer_write_mismatch_count, three_layer_max_abs_diff);
            if (trace_fd != 0) begin
                $display("  trace                  = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace                  = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran a focused true Layer 0 -> Layer 1 -> Layer 2 QMAP loop with exact write-back.");
            $finish;
        end
        if ($test$plusargs("l1_only")) begin
            set_layer_packet_bases(1);
            layer_start_index = 5'd1;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer0_output_for_l1_only();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer1-only chained start");
            $fflush();
            run_until_done(25000000);
            $display("SCENARIO: focused Layer1-only done cycle=%0d mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                     cycle_count, mismatch_count, write_mismatch_count, max_abs_diff);
            if ((mismatch_count != 0) || (write_mismatch_count != 0) || (max_abs_diff != 0) || error) begin
                $finish(1);
            end
            $finish;
        end
        if ($test$plusargs("l2_only")) begin
            set_layer_packet_bases(2);
            layer_start_index = 5'd2;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer1_output_for_l2_only();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer2-only chained start");
            $fflush();
            run_until_done(25000000);
            $display("SCENARIO: focused Layer2-only done cycle=%0d mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                     cycle_count, mismatch_count, write_mismatch_count, max_abs_diff);
            if ((layers_started != 5'd1) ||
                (layers_completed != 5'd1) ||
                (layer_done_mask != 28'h0000004) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: focused Layer2-only layer masks mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) || (max_abs_diff != 0) || error || (total_fail_count != 0)) begin
                $finish(1);
            end
            $finish;
        end
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: normal single-layer start");
        $fflush();
        run_until_done(20000000);
        normal_done_cycle = cycle_count;
        normal_read_bursts = dut_read_burst_count;
        normal_read_words = dut_read_word_count;
        normal_write_reqs = dut_write_req_count;
        normal_write_words = dut_write_word_count;
        normal_stage_done_mask = stage_done_mask;
        normal_stage_error_mask = stage_error_mask;
        normal_layer_done_mask = layer_done_mask;
        normal_layer_error_mask = layer_error_mask;
        normal_layers_started = layers_started;
        normal_layers_completed = layers_completed;
        normal_layer0_full_stage_done_mask = layer0_full_stage_done_mask;
        normal_layer0_full_stage_error_mask = layer0_full_stage_error_mask;
        normal_body_stage_done_mask = body_stage_done_mask;
        normal_body_stage_error_mask = body_stage_error_mask;
        normal_error = error;
        normal_mismatch_count = mismatch_count;
        normal_write_mismatch_count = write_mismatch_count;
        normal_max_abs_diff = max_abs_diff;
        normal_qkv_q_write_accept_count = qkv_q_write_accept_count;
        normal_qkv_k_write_accept_count = qkv_k_write_accept_count;
        normal_qkv_v_write_accept_count = qkv_v_write_accept_count;
        normal_cache_write_accept_count = cache_write_accept_count;
        normal_q_rope_write_accept_count = q_rope_write_accept_count;
        normal_attn_out_write_accept_count = attn_out_write_accept_count;
        normal_o_proj_write_accept_count = o_proj_write_accept_count;
        normal_post_hidden_write_accept_count = post_hidden_write_accept_count;
        normal_post_norm_write_accept_count = post_norm_write_accept_count;
        normal_gate_write_accept_count = gate_write_accept_count;
        normal_up_write_accept_count = up_write_accept_count;
        normal_silu_write_accept_count = silu_write_accept_count;
        normal_down_write_accept_count = down_write_accept_count;
        normal_layer_write_accept_count = layer_write_accept_count;

        if (normal_error) begin
            fail_once("FAIL: normal one-token layer scheduler run asserted error");
        end
        if ((normal_layers_started != 5'd1) ||
            (normal_layers_completed != 5'd1) ||
            (normal_layer_done_mask != 28'h0000001) ||
            (normal_layer_error_mask != 28'h0000000)) begin
            fail_once("FAIL: normal one-token layer-loop masks mismatch");
        end
        if ((normal_stage_done_mask != 2'b11) || (normal_stage_error_mask != 2'b00)) begin
            fail_once("FAIL: normal outer stage masks mismatch");
        end
        if (qkv_rows_done != (VEC2048 + (2 * VEC1024))) begin
            fail_once("FAIL: QKV rows_done mismatch");
        end
        if ((normal_layer0_full_stage_done_mask != 4'hf) ||
            (normal_layer0_full_stage_error_mask != 4'h0)) begin
            fail_once("FAIL: normal Layer 0 full scheduler stage masks mismatch");
        end
        if (normal_body_stage_done_mask != 5'h1f || normal_body_stage_error_mask != 5'h00) begin
            fail_once("FAIL: body sub-stage masks mismatch");
        end
        if ((normal_read_bursts != EXPECTED_NORMAL_RD_REQS) ||
            (normal_read_words != EXPECTED_NORMAL_RD_WORDS) ||
            (normal_write_reqs != EXPECTED_NORMAL_WR_REQS) ||
            (normal_write_words != EXPECTED_NORMAL_WR_WORDS)) begin
            fail_once("FAIL: Layer 0 compute scheduler memory counters mismatch");
        end
        if ((qkv_q_write_accept_count != VEC2048) ||
            (qkv_k_write_accept_count != VEC1024) ||
            (qkv_v_write_accept_count != VEC1024) ||
            (cache_write_accept_count != TOTAL_CACHE_WRITES) ||
            (q_rope_write_accept_count != VEC2048) ||
            (attn_out_write_accept_count != VEC2048) ||
            (o_proj_write_accept_count != VEC1024) ||
            (post_hidden_write_accept_count != VEC1024) ||
            (post_norm_write_accept_count != VEC1024) ||
            (gate_write_accept_count != VEC3072) ||
            (up_write_accept_count != VEC3072) ||
            (silu_write_accept_count != VEC3072) ||
            (down_write_accept_count != VEC1024) ||
            (layer_write_accept_count != VEC1024)) begin
            fail_once("FAIL: write-back word counts mismatch");
        end
        if ((mismatch_count != 0) || (write_mismatch_count != 0) || (max_abs_diff != 0)) begin
            $display("FAIL: normal run found mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                     mismatch_count, write_mismatch_count, max_abs_diff);
            $finish(1);
        end
        $display("SCENARIO: normal single-layer done cycle=%0d", normal_done_cycle);
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        set_layer_packet_bases(1);
        layer_count = 5'd2;
        scoreboard_layer_iterations = 2;
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: true two-layer start");
        $fflush();
        run_until_done(40000000);
        two_layer_done_cycle = cycle_count;
        two_layer_read_bursts = dut_read_burst_count;
        two_layer_read_words = dut_read_word_count;
        two_layer_write_reqs = dut_write_req_count;
        two_layer_write_words = dut_write_word_count;
        two_layer_done_mask = layer_done_mask;
        two_layer_error_mask = layer_error_mask;
        two_layer_layers_started = layers_started;
        two_layer_layers_completed = layers_completed;
        two_layer_error = error;
        two_layer_mismatch_count = mismatch_count;
        two_layer_write_mismatch_count = write_mismatch_count;
        two_layer_max_abs_diff = max_abs_diff;

        if (two_layer_error) begin
            fail_once("FAIL: true two-layer run asserted error");
        end
        if ((two_layer_layers_started != 5'd2) ||
            (two_layer_layers_completed != 5'd2) ||
            (two_layer_done_mask != 28'h0000003) ||
            (two_layer_error_mask != 28'h0000000)) begin
            fail_once("FAIL: true two-layer layer masks mismatch");
        end
        if ((two_layer_read_bursts != EXPECTED_TWO_LAYER_RD_REQS) ||
            (two_layer_read_words != EXPECTED_TWO_LAYER_RD_WORDS) ||
            (two_layer_write_reqs != EXPECTED_TWO_LAYER_WR_REQS) ||
            (two_layer_write_words != EXPECTED_TWO_LAYER_WR_WORDS)) begin
            fail_once("FAIL: true two-layer memory counters mismatch");
        end
        if ((qkv_q_write_accept_count != (2 * VEC2048)) ||
            (qkv_k_write_accept_count != (2 * VEC1024)) ||
            (qkv_v_write_accept_count != (2 * VEC1024)) ||
            (cache_write_accept_count != (2 * TOTAL_CACHE_WRITES)) ||
            (q_rope_write_accept_count != (2 * VEC2048)) ||
            (attn_out_write_accept_count != (2 * VEC2048)) ||
            (o_proj_write_accept_count != (2 * VEC1024)) ||
            (post_hidden_write_accept_count != (2 * VEC1024)) ||
            (post_norm_write_accept_count != (2 * VEC1024)) ||
            (gate_write_accept_count != (2 * VEC3072)) ||
            (up_write_accept_count != (2 * VEC3072)) ||
            (silu_write_accept_count != (2 * VEC3072)) ||
            (down_write_accept_count != (2 * VEC1024)) ||
            (layer_write_accept_count != (2 * VEC1024))) begin
            fail_once("FAIL: true two-layer write-back word counts mismatch");
        end
        if ((two_layer_mismatch_count != 0) ||
            (two_layer_write_mismatch_count != 0) ||
            (two_layer_max_abs_diff != 0)) begin
            $display("FAIL: true two-layer found mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                     two_layer_mismatch_count, two_layer_write_mismatch_count, two_layer_max_abs_diff);
            $finish(1);
        end
        $display("SCENARIO: true two-layer done cycle=%0d", two_layer_done_cycle);
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        corrupt_qkv_activation_dtype();
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: invalid QKV descriptor start");
        $fflush();
        run_until_done(200000);
        qkv_invalid_read_bursts = dut_read_burst_count;
        qkv_invalid_read_words = dut_read_word_count;
        qkv_invalid_write_reqs = dut_write_req_count;
        qkv_invalid_write_words = dut_write_word_count;
        qkv_invalid_stage_done_mask = stage_done_mask;
        qkv_invalid_stage_error_mask = stage_error_mask;
        qkv_invalid_error = error;

        if (!qkv_invalid_error) begin
            fail_once("FAIL: invalid QKV descriptor did not assert error");
        end
        if ((qkv_invalid_stage_done_mask != 2'b00) ||
            (qkv_invalid_stage_error_mask != 2'b01)) begin
            fail_once("FAIL: invalid QKV run stage masks mismatch");
        end
        if ((qkv_invalid_write_reqs != 0) || (qkv_invalid_write_words != 0) ||
            (wr_req_accept_count != 0) || (wr_data_accept_count != 0)) begin
            fail_once("FAIL: invalid QKV descriptor produced writes");
        end
        if ((qkv_invalid_read_bursts != EXPECTED_QKV_INVALID_RD_REQS) ||
            (qkv_invalid_read_words != EXPECTED_QKV_INVALID_RD_WORDS)) begin
            fail_once("FAIL: invalid QKV read counters mismatch");
        end
        if (mismatch_count != 0) begin
            $display("FAIL: invalid QKV run found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end
        $display("SCENARIO: invalid QKV descriptor done");
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        corrupt_frontend_cos_dtype();
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: invalid frontend descriptor start");
        $fflush();
        run_until_done(12000000);
        invalid_done_cycle = cycle_count;
        invalid_read_bursts = dut_read_burst_count;
        invalid_read_words = dut_read_word_count;
        invalid_write_reqs = dut_write_req_count;
        invalid_write_words = dut_write_word_count;
        invalid_stage_done_mask = stage_done_mask;
        invalid_stage_error_mask = stage_error_mask;
        invalid_layer0_full_stage_done_mask = layer0_full_stage_done_mask;
        invalid_layer0_full_stage_error_mask = layer0_full_stage_error_mask;
        invalid_error = error;

        if (!invalid_error) begin
            fail_once("FAIL: invalid frontend descriptor did not assert error");
        end
        if ((invalid_stage_done_mask != 2'b01) || (invalid_stage_error_mask != 2'b10) ||
            (invalid_layer0_full_stage_done_mask != 4'h0) ||
            (invalid_layer0_full_stage_error_mask != 4'h1)) begin
            fail_once("FAIL: invalid frontend run stage masks mismatch");
        end
        if ((qkv_q_write_accept_count != VEC2048) ||
            (qkv_k_write_accept_count != VEC1024) ||
            (qkv_v_write_accept_count != VEC1024) ||
            (cache_write_accept_count != 0) ||
            (q_rope_write_accept_count != 0) ||
            (attn_out_write_accept_count != 0) ||
            (o_proj_write_accept_count != 0) ||
            (post_hidden_write_accept_count != 0) ||
            (post_norm_write_accept_count != 0) ||
            (gate_write_accept_count != 0) ||
            (up_write_accept_count != 0) ||
            (silu_write_accept_count != 0) ||
            (down_write_accept_count != 0) ||
            (layer_write_accept_count != 0)) begin
            fail_once("FAIL: invalid frontend run produced unexpected downstream writes");
        end
        if ((invalid_write_reqs != EXPECTED_QKV_WR_REQS) ||
            (invalid_write_words != EXPECTED_QKV_WR_WORDS)) begin
            fail_once("FAIL: invalid frontend run write counters did not stop after QKV");
        end
        if ((invalid_read_bursts != EXPECTED_FRONTEND_INVALID_RD_REQS) ||
            (invalid_read_words != EXPECTED_FRONTEND_INVALID_RD_WORDS)) begin
            fail_once("FAIL: invalid frontend read counters mismatch");
        end
        $display("SCENARIO: invalid frontend descriptor done cycle=%0d", invalid_done_cycle);
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        qkv_qmap_base_addr_table[0*ADDR_WIDTH +: ADDR_WIDTH] = '0;
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: missing layer0 base start");
        $fflush();
        run_until_done(100);
        missing_base_read_bursts = dut_read_burst_count;
        missing_base_read_words = dut_read_word_count;
        missing_base_write_reqs = dut_write_req_count;
        missing_base_write_words = dut_write_word_count;
        missing_base_layer_done_mask = layer_done_mask;
        missing_base_layer_error_mask = layer_error_mask;
        missing_base_layers_started = layers_started;
        missing_base_layers_completed = layers_completed;
        missing_base_error = error;

        if (!missing_base_error) begin
            fail_once("FAIL: missing layer0 QKV base table entry did not assert error");
        end
        if ((missing_base_layers_started != 5'd0) ||
            (missing_base_layers_completed != 5'd0) ||
            (missing_base_layer_done_mask != 28'h0000000) ||
            (missing_base_layer_error_mask != 28'h0000001)) begin
            fail_once("FAIL: missing layer0 QKV base table entry masks mismatch");
        end
        if ((missing_base_read_bursts != 0) ||
            (missing_base_read_words != 0) ||
            (missing_base_write_reqs != 0) ||
            (missing_base_write_words != 0) ||
            (rd_req_accept_count != 0) ||
            (rd_rsp_accept_count != 0) ||
            (wr_req_accept_count != 0) ||
            (wr_data_accept_count != 0)) begin
            fail_once("FAIL: missing layer0 QKV base table entry issued memory traffic");
        end
        $display("SCENARIO: missing layer0 base done");
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        layer_count = 5'd2;
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: missing layer1 base start");
        $fflush();
        run_until_done(100);
        if (!error) begin
            fail_once("FAIL: missing layer1 base-table entries did not assert error");
        end
        if ((layers_started != 5'd0) ||
            (layers_completed != 5'd0) ||
            (layer_done_mask != 28'h0000000) ||
            (layer_error_mask != 28'h0000002)) begin
            fail_once("FAIL: missing layer1 base-table entry masks mismatch");
        end
        if ((dut_read_burst_count != 0) ||
            (dut_read_word_count != 0) ||
            (dut_write_req_count != 0) ||
            (dut_write_word_count != 0) ||
            (rd_req_accept_count != 0) ||
            (rd_rsp_accept_count != 0) ||
            (wr_req_accept_count != 0) ||
            (wr_data_accept_count != 0)) begin
            fail_once("FAIL: missing layer1 base-table entries issued memory traffic");
        end
        $display("SCENARIO: missing layer1 base done");
        $fflush();

        rst_n = 1'b0;
        load_vectors();
        clear_scoreboard();
        set_valid_loop_contract();
        layer_start_index = 5'd28;
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        $display("SCENARIO: out-of-range layer index start");
        $fflush();
        run_until_done(100);
        unsupported_read_bursts = dut_read_burst_count;
        unsupported_read_words = dut_read_word_count;
        unsupported_write_reqs = dut_write_req_count;
        unsupported_write_words = dut_write_word_count;
        unsupported_layer_done_mask = layer_done_mask;
        unsupported_layer_error_mask = layer_error_mask;
        unsupported_layers_started = layers_started;
        unsupported_layers_completed = layers_completed;
        unsupported_error = error;

        if (!unsupported_error) begin
            fail_once("FAIL: unsupported layer index did not assert error");
        end
        if ((unsupported_layers_started != 5'd0) ||
            (unsupported_layers_completed != 5'd0) ||
            (unsupported_layer_done_mask != 28'h0000000) ||
            (unsupported_layer_error_mask != 28'h8000000)) begin
            fail_once("FAIL: unsupported layer index masks mismatch");
        end
        if ((unsupported_read_bursts != 0) ||
            (unsupported_read_words != 0) ||
            (unsupported_write_reqs != 0) ||
            (unsupported_write_words != 0) ||
            (rd_req_accept_count != 0) ||
            (rd_rsp_accept_count != 0) ||
            (wr_req_accept_count != 0) ||
            (wr_data_accept_count != 0)) begin
            fail_once("FAIL: unsupported layer index issued memory traffic");
        end
        $display("SCENARIO: out-of-range layer index done");
        $fflush();

        $display("qmap_one_token_layer_scheduler Layer 0 loop-boundary test");
        $display("  normal done cycle      = %0d", normal_done_cycle);
        $display("  qkv invalid rd/wr      = %0d/%0d reads, %0d/%0d writes",
                 qkv_invalid_read_bursts, qkv_invalid_read_words,
                 qkv_invalid_write_reqs, qkv_invalid_write_words);
        $display("  frontend invalid cycle = %0d", invalid_done_cycle);
        $display("  normal rd/wr           = %0d/%0d reads, %0d/%0d writes",
                 normal_read_bursts, normal_read_words, normal_write_reqs, normal_write_words);
        $display("  true two-layer cycle   = %0d", two_layer_done_cycle);
        $display("  true two-layer rd/wr   = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                 two_layer_read_bursts, two_layer_read_words,
                 two_layer_write_reqs, two_layer_write_words,
                 two_layer_done_mask);
        $display("  frontend invalid rd/wr = %0d/%0d reads, %0d/%0d writes",
                 invalid_read_bursts, invalid_read_words, invalid_write_reqs, invalid_write_words);
        $display("  stage masks normal     = done 0x%0h error 0x%0h full_done 0x%0h full_error 0x%0h body_done 0x%0h body_error 0x%0h",
                 normal_stage_done_mask, normal_stage_error_mask,
                 normal_layer0_full_stage_done_mask, normal_layer0_full_stage_error_mask,
                 normal_body_stage_done_mask, normal_body_stage_error_mask);
        $display("  layer masks normal     = started %0d completed %0d done 0x%0h error 0x%0h",
                 normal_layers_started, normal_layers_completed,
                 normal_layer_done_mask, normal_layer_error_mask);
        $display("  stage masks qkv invalid= done 0x%0h error 0x%0h",
                 qkv_invalid_stage_done_mask, qkv_invalid_stage_error_mask);
        $display("  stage masks frontend invalid = done 0x%0h error 0x%0h full_done 0x%0h full_error 0x%0h",
                 invalid_stage_done_mask, invalid_stage_error_mask,
                 invalid_layer0_full_stage_done_mask, invalid_layer0_full_stage_error_mask);
        $display("  producer writes        = qkv %0d/%0d/%0d cache %0d qrope %0d attn %0d oproj %0d post %0d/%0d gate/up %0d/%0d silu %0d down %0d layer %0d",
                 normal_qkv_q_write_accept_count, normal_qkv_k_write_accept_count,
                 normal_qkv_v_write_accept_count, normal_cache_write_accept_count,
                 normal_q_rope_write_accept_count,
                 normal_attn_out_write_accept_count, normal_o_proj_write_accept_count,
                 normal_post_hidden_write_accept_count, normal_post_norm_write_accept_count,
                 normal_gate_write_accept_count, normal_up_write_accept_count,
                 normal_silu_write_accept_count, normal_down_write_accept_count, normal_layer_write_accept_count);
        $display("  normal mismatches      = %0d", normal_mismatch_count);
        $display("  write mismatches       = %0d", normal_write_mismatch_count);
        $display("  max_abs write diff     = %0d", normal_max_abs_diff);
        $display("  missing base rd/wr     = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                 missing_base_read_bursts, missing_base_read_words,
                 missing_base_write_reqs, missing_base_write_words,
                 missing_base_layer_error_mask);
        $display("  unsupported layer rd/wr= %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                 unsupported_read_bursts, unsupported_read_words,
                 unsupported_write_reqs, unsupported_write_words,
                 unsupported_layer_error_mask);
        if (trace_fd != 0) begin
            $display("  trace                  = %s", tracefile);
        end
        else begin
            $display("  trace                  = disabled by +notrace");
        end

        if ((total_fail_count != 0) || (mismatch_count != 0) ||
            (write_mismatch_count != 0) || (max_abs_diff != 0)) begin
            $display("FAIL: qmap_one_token_layer_scheduler found total_fail_count=%0d mismatch_count=%0d write_mismatches=%0d max_abs=%0d",
                     total_fail_count, mismatch_count, write_mismatch_count, max_abs_diff);
            $finish(1);
        end

        $display("PASS: qmap_one_token_layer_scheduler selected packet bases from per-layer tables, ran a true Layer 0 -> Layer 1 loop with exact write-back, and covered error propagation plus no-memory invalid-table/unsupported-loop exits.");
        if (trace_fd != 0) begin
            $fclose(trace_fd);
        end
        $finish;
    end

endmodule

`default_nettype wire
