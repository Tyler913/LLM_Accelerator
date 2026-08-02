`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_one_token_layer_scheduler;

    localparam int ADDR_WIDTH = 64;
    localparam int MEM_DATA_WIDTH = 32;
    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int MAX_LAYERS = 28;
    localparam int BASE_TABLE_BITS = MAX_LAYERS * ADDR_WIDTH;
    localparam int LAYER_INDEX_WIDTH = $clog2(MAX_LAYERS);
    localparam int LAYER_COUNT_WIDTH = $clog2(MAX_LAYERS + 1);
    localparam int POSITION_WIDTH = 8;

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
    localparam int CACHE_READ_WORDS_PER_POSITION = 2 * NUM_Q_HEADS * HEAD_DIM;

    localparam int QKV_IMAGE_BYTES = 32'h0022_B000;
    localparam int FRONT_IMAGE_BYTES = 32'h0000_8000;
    localparam int SCORE_IMAGE_BYTES = 32'h0000_5000;
    localparam int OPROJ_IMAGE_BYTES = 32'h0000_5000;
    localparam int POST_IMAGE_BYTES = 32'h0000_8000;
    localparam int GATE_IMAGE_BYTES = 32'h0000_E000;
    localparam int SILU_IMAGE_BYTES = 32'h0000_E000;
    localparam int DOWN_IMAGE_BYTES = 32'h0000_6000;
    localparam int RESIDUAL_IMAGE_BYTES = 32'h0000_5000;
    localparam int INPUT_NORM_IMAGE_BYTES = 32'h0000_5000;

    localparam int QKV_WORDS = QKV_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int FRONT_WORDS = FRONT_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int SCORE_WORDS = SCORE_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int OPROJ_WORDS = OPROJ_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int POST_WORDS = POST_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int GATE_WORDS = GATE_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int SILU_WORDS = SILU_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DOWN_WORDS = DOWN_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int RESIDUAL_WORDS = RESIDUAL_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int INPUT_NORM_WORDS = INPUT_NORM_IMAGE_BYTES / MEM_DATA_BYTES;

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
    localparam int INPUT_NORM_SLOT_OUTPUT = 3;

    localparam int OPROJ_WEIGHT_WORDS = (1024 * 1024) / 4;
    localparam int OPROJ_SCALE_WORDS = (1024 * 64) / 4;
    localparam int GATE_WEIGHT_WORDS = (3072 * 512) / 4;
    localparam int GATE_SCALE_WORDS = (3072 * 32) / 4;
    localparam int DOWN_WEIGHT_WORDS = (1024 * 1536) / 4;
    localparam int DOWN_SCALE_WORDS = (1024 * 96) / 4;

    localparam logic [ADDR_WIDTH-1 : 0] CACHE_BASE_ADDR = 64'h0000_0004_1410_0000;
    localparam int CACHE_WORDS_FULL = 2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM;
    localparam int ROPE_TABLE_WORDS = 2 * MAX_CONTEXT * HEAD_DIM;
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    localparam int MODELED_CACHE_LAYERS = MAX_LAYERS;
`else
    localparam int MODELED_CACHE_LAYERS = 3;
`endif
    localparam int CACHE_LAYER_ACTIVE_WORDS = CACHE_LENGTH * KV_COUNT;

    localparam logic [ADDR_WIDTH-1 : 0] EMBEDDING_WEIGHT_BASE_ADDR = 64'h0000_0004_0010_0000;
    localparam logic [ADDR_WIDTH-1 : 0] EMBEDDING_SCALE_BASE_ADDR = 64'h0000_0004_04B3_0000;
    localparam logic [ADDR_WIDTH-1 : 0] EMBEDDING_LAYER0_QKV_BASE_ADDR = 64'h0000_0004_1B40_0000;
    localparam int EMBEDDING_WEIGHT_WORDS = 128;
    localparam int EMBEDDING_SCALE_WORDS = 8;

    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_QKV_BASE_ADDR = 64'h0000_0004_1008_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_ATTN_FRONTEND_BASE_ADDR = 64'h0000_0004_1502_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_ATTN_SCORE_VALUE_BASE_ADDR = 64'h0000_0004_1503_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_O_PROJ_BASE_ADDR = 64'h0000_0004_1504_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_POST_ATTN_NORM_BASE_ADDR = 64'h0000_0004_1505_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_GATE_UP_BASE_ADDR = 64'h0000_0004_1506_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_SILU_MUL_BASE_ADDR = 64'h0000_0004_1507_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_DOWN_BASE_ADDR = 64'h0000_0004_1508_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_MLP_RESIDUAL_ADD_BASE_ADDR = 64'h0000_0004_1509_0000;
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER1_INPUT_NORM_BASE_ADDR = 64'h0000_0004_150A_0000;
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
    localparam logic [ADDR_WIDTH-1 : 0] QMAP_LAYER2_INPUT_NORM_BASE_ADDR = 64'h0000_0004_250A_0000;
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
    localparam int WRITE_INPUT_NORM = 15;
    localparam int WRITE_EMBEDDING = 16;

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
    localparam int EXPECTED_INPUT_NORM_QKV_RD_REQS = 8234;
    localparam int EXPECTED_INPUT_NORM_QKV_RD_WORDS = 561040;
    localparam int EXPECTED_INPUT_NORM_QKV_WR_REQS = 4097;
    localparam int EXPECTED_INPUT_NORM_QKV_WR_WORDS = 5120;
    localparam int EXPECTED_NORMAL_RD_REQS = EXPECTED_QKV_RD_REQS + EXPECTED_FULL_RD_REQS;
    localparam int EXPECTED_NORMAL_RD_WORDS = EXPECTED_QKV_RD_WORDS + EXPECTED_FULL_RD_WORDS;
    localparam int EXPECTED_NORMAL_WR_REQS = EXPECTED_QKV_WR_REQS + EXPECTED_FULL_WR_REQS;
    localparam int EXPECTED_NORMAL_WR_WORDS = EXPECTED_QKV_WR_WORDS + EXPECTED_FULL_WR_WORDS;
    localparam int EXPECTED_INPUT_NORM_FULL_RD_REQS = EXPECTED_NORMAL_RD_REQS + 14;
    localparam int EXPECTED_INPUT_NORM_FULL_RD_WORDS = EXPECTED_NORMAL_RD_WORDS + 2224;
    localparam int EXPECTED_INPUT_NORM_FULL_WR_REQS = EXPECTED_NORMAL_WR_REQS + 1;
    localparam int EXPECTED_INPUT_NORM_FULL_WR_WORDS = EXPECTED_NORMAL_WR_WORDS + VEC1024;
    localparam int EXPECTED_TWO_LAYER_RD_REQS = 2 * EXPECTED_NORMAL_RD_REQS;
    localparam int EXPECTED_TWO_LAYER_RD_WORDS = 2 * EXPECTED_NORMAL_RD_WORDS;
    localparam int EXPECTED_TWO_LAYER_WR_REQS = 2 * EXPECTED_NORMAL_WR_REQS;
    localparam int EXPECTED_TWO_LAYER_WR_WORDS = 2 * EXPECTED_NORMAL_WR_WORDS;
    localparam int EXPECTED_THREE_LAYER_RD_REQS = 3 * EXPECTED_NORMAL_RD_REQS;
    localparam int EXPECTED_THREE_LAYER_RD_WORDS = 3 * EXPECTED_NORMAL_RD_WORDS;
    localparam int EXPECTED_THREE_LAYER_WR_REQS = 3 * EXPECTED_NORMAL_WR_REQS;
    localparam int EXPECTED_THREE_LAYER_WR_WORDS = 3 * EXPECTED_NORMAL_WR_WORDS;
    localparam int EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_REQS = 2 * EXPECTED_INPUT_NORM_FULL_RD_REQS;
    localparam int EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_WORDS = 2 * EXPECTED_INPUT_NORM_FULL_RD_WORDS;
    localparam int EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_REQS = 2 * EXPECTED_INPUT_NORM_FULL_WR_REQS;
    localparam int EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_WORDS = 2 * EXPECTED_INPUT_NORM_FULL_WR_WORDS;
    localparam int EXPECTED_INPUT_NORM_THREE_LAYER_FULL_RD_REQS = 3 * EXPECTED_INPUT_NORM_FULL_RD_REQS;
    localparam int EXPECTED_INPUT_NORM_THREE_LAYER_FULL_RD_WORDS = 3 * EXPECTED_INPUT_NORM_FULL_RD_WORDS;
    localparam int EXPECTED_INPUT_NORM_THREE_LAYER_FULL_WR_REQS = 3 * EXPECTED_INPUT_NORM_FULL_WR_REQS;
    localparam int EXPECTED_INPUT_NORM_THREE_LAYER_FULL_WR_WORDS = 3 * EXPECTED_INPUT_NORM_FULL_WR_WORDS;
    localparam int EXPECTED_MIXED_THREE_LAYER_FULL_RD_REQS = EXPECTED_NORMAL_RD_REQS + EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_REQS;
    localparam int EXPECTED_MIXED_THREE_LAYER_FULL_RD_WORDS = EXPECTED_NORMAL_RD_WORDS + EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_WORDS;
    localparam int EXPECTED_MIXED_THREE_LAYER_FULL_WR_REQS = EXPECTED_NORMAL_WR_REQS + EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_REQS;
    localparam int EXPECTED_MIXED_THREE_LAYER_FULL_WR_WORDS = EXPECTED_NORMAL_WR_WORDS + EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_WORDS;
    localparam int EXPECTED_EMBEDDING_TRUE3_TOP_RD_REQS = 224330;
    localparam int EXPECTED_EMBEDDING_TRUE3_TOP_RD_WORDS = 27088110;
    localparam int EXPECTED_EMBEDDING_TRUE3_TOP_WR_REQS = 18468;
    localparam int EXPECTED_EMBEDDING_TRUE3_TOP_WR_WORDS = 78851;
    // The routed resource-reduced tail uses the BRAM row scheduler.  Each of
    // the 151936 vocabulary rows is fetched as one 512-byte weight request
    // followed by one 32-byte scale request.  The remaining 32 requests are
    // the two embedding-row requests plus QMAP/header and norm-side reads.
    localparam int EXPECTED_EMBEDDING_TAIL_EXTRA_RD_REQS = 303904;
    localparam int EXPECTED_EMBEDDING_TAIL_EXTRA_RD_WORDS = 20667048;
    localparam int EXPECTED_EMBEDDING_TAIL_EXTRA_WR_REQS = 3;
    localparam int EXPECTED_EMBEDDING_TAIL_EXTRA_WR_WORDS = 2051;

    localparam int TAIL_MAX_TILES = 9496;
    localparam int TAIL_TILE_ROWS = 16;
    localparam int TAIL_SCAN_ROWS = TAIL_MAX_TILES * TAIL_TILE_ROWS;
    localparam int TAIL_INPUT_SIZE = 1024;
    localparam int TAIL_HIDDEN_WIDTH = 24;
    localparam int TAIL_NORM_WIDTH = 24;
    localparam int TAIL_GROUP_SIZE = 64;
    localparam int TAIL_GROUP_COUNT = TAIL_INPUT_SIZE / TAIL_GROUP_SIZE;
    localparam int TAIL_WEIGHT_WIDTH = 4;
    localparam int TAIL_SCALE_WIDTH = 16;
    localparam int TAIL_PARTIAL_WIDTH = TAIL_NORM_WIDTH + TAIL_WEIGHT_WIDTH + $clog2(TAIL_GROUP_SIZE);
    localparam int TAIL_SCALED_WIDTH = TAIL_PARTIAL_WIDTH + TAIL_SCALE_WIDTH;
    localparam int TAIL_ROW_ACC_WIDTH = TAIL_SCALED_WIDTH + $clog2(TAIL_GROUP_COUNT) + 2;
    localparam int TAIL_WEIGHT_ROW_BYTES = (TAIL_INPUT_SIZE * TAIL_WEIGHT_WIDTH) / 8;
    localparam int TAIL_SCALE_ROW_BYTES = (TAIL_GROUP_COUNT * TAIL_SCALE_WIDTH) / 8;
    localparam int TAIL_TILE_WEIGHT_BYTES = TAIL_TILE_ROWS * TAIL_WEIGHT_ROW_BYTES;
    localparam int TAIL_TILE_SCALE_BYTES = TAIL_TILE_ROWS * TAIL_SCALE_ROW_BYTES;
    // The resource-reduced LM head fetches one packed-Q4 row per request even
    // though MAX_READ_BYTES is wider than a 512-byte row.
    localparam int TAIL_WEIGHT_BURSTS_PER_TILE = TAIL_TILE_ROWS;
    localparam int TAIL_SCALE_BURSTS_PER_TILE = TAIL_TILE_ROWS;
    localparam int TAIL_BURSTS_PER_TILE = TAIL_WEIGHT_BURSTS_PER_TILE + TAIL_SCALE_BURSTS_PER_TILE;
    localparam int TAIL_QMAP_IMAGE_BYTES = 32'h0000_4000;
    localparam int TAIL_QMAP_WORDS = TAIL_QMAP_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int TAIL_WEIGHT_WORDS = TAIL_SCAN_ROWS * TAIL_WEIGHT_ROW_BYTES / MEM_DATA_BYTES;
    localparam int TAIL_SCALE_WORDS = TAIL_SCAN_ROWS * TAIL_SCALE_ROW_BYTES / MEM_DATA_BYTES;
    localparam int TAIL_SLOT_NORM_OUTPUT = 1;
    localparam int TAIL_SLOT_WEIGHT = 2;
    localparam int TAIL_SLOT_SCALE = 3;
    localparam int TAIL_SLOT_OUTPUT = 4;
    localparam int TAIL_SLOT_FINAL_HIDDEN = 6;
    localparam int TAIL_SLOT_FINAL_GAMMA = 7;
    localparam int TAIL_WRITE_NONE = 100;
    localparam int TAIL_WRITE_NORM = 101;
    localparam int TAIL_WRITE_OUTPUT = 102;

    logic clk;
    logic rst_n;
    logic start;
    logic tail_start;
    logic use_tail_mem_manual;
    wire use_tail_mem;
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
    logic [BASE_TABLE_BITS-1 : 0] input_norm_qmap_base_addr_table;
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
    logic [ADDR_WIDTH-1 : 0] scheduler_last_layer_output_base;
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

`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
    logic [7 : 0] top_phase_debug;
    logic top_scheduler_done_pulse;
    logic top_tail_start_pulse;
    logic top_tail_done_pulse;
    logic top_tail_active;
    logic [ADDR_WIDTH-1 : 0] top_tail_effective_final_hidden_base;
    logic [31 : 0] top_mem_read_burst_count;
    logic [31 : 0] top_mem_read_word_count;
    logic [31 : 0] top_mem_write_req_count;
    logic [31 : 0] top_mem_write_word_count;
    integer top_scheduler_done_seen_count;
    integer top_tail_start_seen_count;

`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
    localparam logic [11 : 0] MMIO_REG_CTRL                 = 12'h000;
    localparam logic [11 : 0] MMIO_REG_STATUS               = 12'h004;
    localparam logic [11 : 0] MMIO_REG_LAYER_START          = 12'h008;
    localparam logic [11 : 0] MMIO_REG_LAYER_COUNT          = 12'h00C;
    localparam logic [11 : 0] MMIO_REG_POSITION             = 12'h010;
    localparam logic [11 : 0] MMIO_REG_INPUT_TOKEN          = 12'h014;
    localparam logic [11 : 0] MMIO_REG_INPUT_HIDDEN_LO      = 12'h020;
    localparam logic [11 : 0] MMIO_REG_INPUT_HIDDEN_HI      = 12'h024;
    localparam logic [11 : 0] MMIO_REG_OUTPUT_HIDDEN_LO     = 12'h028;
    localparam logic [11 : 0] MMIO_REG_OUTPUT_HIDDEN_HI     = 12'h02C;
    localparam logic [11 : 0] MMIO_REG_KV_CACHE_LO          = 12'h030;
    localparam logic [11 : 0] MMIO_REG_KV_CACHE_HI          = 12'h034;
    localparam logic [11 : 0] MMIO_REG_FINAL_TAIL_QMAP_LO   = 12'h038;
    localparam logic [11 : 0] MMIO_REG_FINAL_TAIL_QMAP_HI   = 12'h03C;
    localparam logic [11 : 0] MMIO_REG_FINAL_OVERRIDE_CTRL  = 12'h048;
    localparam logic [11 : 0] MMIO_REG_EMBEDDING_CTRL       = 12'h04C;
    localparam logic [11 : 0] MMIO_REG_TABLE_SELECT         = 12'h050;
    localparam logic [11 : 0] MMIO_REG_TABLE_DATA_LO        = 12'h054;
    localparam logic [11 : 0] MMIO_REG_TABLE_DATA_HI        = 12'h058;
    localparam logic [11 : 0] MMIO_REG_TABLE_COMMIT         = 12'h05C;
    localparam logic [11 : 0] MMIO_REG_OUT_TOKEN            = 12'h060;
    localparam logic [11 : 0] MMIO_REG_OUT_SCORE_LO         = 12'h064;
    localparam logic [11 : 0] MMIO_REG_OUT_SCORE_HI         = 12'h068;
    localparam logic [11 : 0] MMIO_REG_LAYERS               = 12'h06C;
    localparam logic [11 : 0] MMIO_REG_LAYER_DONE_MASK      = 12'h070;
    localparam logic [11 : 0] MMIO_REG_LAST_OUTPUT_LO       = 12'h078;
    localparam logic [11 : 0] MMIO_REG_TAIL_HIDDEN_LO       = 12'h080;
    localparam logic [11 : 0] MMIO_REG_MEM_RD_REQS          = 12'h090;
    localparam logic [11 : 0] MMIO_REG_MEM_RD_WORDS         = 12'h094;
    localparam logic [11 : 0] MMIO_REG_MEM_WR_REQS          = 12'h098;
    localparam logic [11 : 0] MMIO_REG_MEM_WR_WORDS         = 12'h09C;
    localparam logic [11 : 0] MMIO_REG_EMBED_WEIGHT_LO      = 12'h0A0;
    localparam logic [11 : 0] MMIO_REG_EMBED_WEIGHT_HI      = 12'h0A4;
    localparam logic [11 : 0] MMIO_REG_EMBED_SCALE_LO       = 12'h0A8;
    localparam logic [11 : 0] MMIO_REG_EMBED_SCALE_HI       = 12'h0AC;
    localparam logic [11 : 0] MMIO_REG_RUNTIME_CTRL         = 12'h0B0;

    localparam logic [7 : 0] MMIO_TABLE_QKV             = 8'd0;
    localparam logic [7 : 0] MMIO_TABLE_INPUT_NORM      = 8'd1;
    localparam logic [7 : 0] MMIO_TABLE_ATTN_FRONTEND   = 8'd2;
    localparam logic [7 : 0] MMIO_TABLE_ATTN_SCORE      = 8'd3;
    localparam logic [7 : 0] MMIO_TABLE_O_PROJ          = 8'd4;
    localparam logic [7 : 0] MMIO_TABLE_POST_ATTN_NORM  = 8'd5;
    localparam logic [7 : 0] MMIO_TABLE_MLP_GATE_UP     = 8'd6;
    localparam logic [7 : 0] MMIO_TABLE_MLP_SILU_MUL    = 8'd7;
    localparam logic [7 : 0] MMIO_TABLE_MLP_DOWN        = 8'd8;
    localparam logic [7 : 0] MMIO_TABLE_MLP_RESIDUAL    = 8'd9;

    logic mmio_reg_wr_valid;
    logic mmio_reg_wr_ready;
    logic mmio_reg_rd_valid;
    logic mmio_reg_rd_ready;
    logic [11 : 0] mmio_reg_addr;
    logic [31 : 0] mmio_reg_wdata;
    logic [31 : 0] mmio_reg_rdata;
    logic mmio_reg_error;
    logic [31 : 0] mmio_read_data;

`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
    localparam logic [1 : 0] AXIL_RESP_OKAY = 2'b00;

    logic [11 : 0] axil_awaddr;
    logic [2 : 0]  axil_awprot;
    logic          axil_awvalid;
    logic          axil_awready;
    logic [31 : 0] axil_wdata;
    logic [3 : 0]  axil_wstrb;
    logic          axil_wvalid;
    logic          axil_wready;
    logic [1 : 0]  axil_bresp;
    logic          axil_bvalid;
    logic          axil_bready;
    logic [11 : 0] axil_araddr;
    logic [2 : 0]  axil_arprot;
    logic          axil_arvalid;
    logic          axil_arready;
    logic [31 : 0] axil_rdata;
    logic [1 : 0]  axil_rresp;
    logic          axil_rvalid;
    logic          axil_rready;
    logic          axil_busy;
`endif

    logic mmio_start_pulse;
    logic [31 : 0] mmio_input_token_id;
    logic [LAYER_INDEX_WIDTH-1 : 0] mmio_layer_start_index;
    logic [LAYER_COUNT_WIDTH-1 : 0] mmio_layer_count;
    logic [POSITION_WIDTH-1 : 0] mmio_position;
    logic mmio_runtime_context_enable;
    logic [ADDR_WIDTH-1 : 0] mmio_input_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] mmio_output_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] mmio_kv_cache_base_addr;
    logic [ADDR_WIDTH-1 : 0] mmio_final_tail_qmap_base_addr;
    logic mmio_final_hidden_base_override_valid;
    logic [ADDR_WIDTH-1 : 0] mmio_final_hidden_base_override_addr;
    logic [BASE_TABLE_BITS-1 : 0] mmio_qkv_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_input_norm_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_attn_frontend_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_attn_score_value_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_o_proj_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_post_attn_norm_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_mlp_gate_up_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_mlp_silu_mul_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_mlp_down_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mmio_mlp_residual_add_qmap_base_addr_table;
`endif
`endif

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

    logic sched_mem_rd_req_valid;
    logic sched_mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] sched_mem_rd_req_addr;
    logic [15 : 0] sched_mem_rd_req_len_bytes;
    logic sched_mem_rd_rsp_valid;
    logic sched_mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] sched_mem_rd_rsp_data;
    logic sched_mem_rd_rsp_last;

    logic sched_mem_wr_req_valid;
    logic sched_mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] sched_mem_wr_req_addr;
    logic [15 : 0] sched_mem_wr_req_len_bytes;
    logic [31 : 0] sched_mem_wr_data;
    logic sched_mem_wr_data_valid;
    logic sched_mem_wr_data_ready;
    logic sched_mem_wr_data_last;
    logic sched_mem_wr_done;
    logic sched_mem_wr_error;

`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
    logic tail_busy;
    logic tail_done;
    logic tail_error;
    logic tail_norm_saturation;
    logic [31 : 0] tail_best_token_id;
    logic signed [TAIL_ROW_ACC_WIDTH-1 : 0] tail_best_score_q26;
    logic [31 : 0] tail_tiles_started;
    logic [31 : 0] tail_tiles_completed;
    logic [31 : 0] tail_norm_cycle_count;
    logic [31 : 0] tail_mem_read_burst_count;
    logic [31 : 0] tail_mem_read_word_count;
    logic [31 : 0] tail_mem_write_word_count;

    logic tail_mem_rd_req_valid;
    logic tail_mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] tail_mem_rd_req_addr;
    logic [15 : 0] tail_mem_rd_req_len_bytes;
    logic tail_mem_rd_rsp_valid;
    logic tail_mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] tail_mem_rd_rsp_data;
    logic tail_mem_rd_rsp_last;

    logic tail_mem_wr_req_valid;
    logic tail_mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] tail_mem_wr_req_addr;
    logic [15 : 0] tail_mem_wr_req_len_bytes;
    logic [31 : 0] tail_mem_wr_data;
    logic tail_mem_wr_data_valid;
    logic tail_mem_wr_data_ready;
    logic tail_mem_wr_data_last;
    logic tail_mem_wr_done;
    logic tail_mem_wr_error;
`endif

    logic [31 : 0] qkv_qmap [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap [0 : RESIDUAL_WORDS-1];
    logic [31 : 0] input_norm_qmap [0 : INPUT_NORM_WORDS-1];

    logic [31 : 0] qkv_qmap_l1 [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap_l1 [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap_l1 [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap_l1 [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap_l1 [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap_l1 [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap_l1 [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap_l1 [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap_l1 [0 : RESIDUAL_WORDS-1];
    logic [31 : 0] input_norm_qmap_l1 [0 : INPUT_NORM_WORDS-1];

    logic [31 : 0] qkv_qmap_l2 [0 : QKV_WORDS-1];
    logic [31 : 0] frontend_qmap_l2 [0 : FRONT_WORDS-1];
    logic [31 : 0] score_qmap_l2 [0 : SCORE_WORDS-1];
    logic [31 : 0] oproj_qmap_l2 [0 : OPROJ_WORDS-1];
    logic [31 : 0] post_qmap_l2 [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap_l2 [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap_l2 [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap_l2 [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap_l2 [0 : RESIDUAL_WORDS-1];
    logic [31 : 0] input_norm_qmap_l2 [0 : INPUT_NORM_WORDS-1];

    logic [31 : 0] embedding_weight_mem [0 : EMBEDDING_WEIGHT_WORDS-1];
    logic [31 : 0] embedding_scale_mem [0 : EMBEDDING_SCALE_WORDS-1];
    logic [31 : 0] embedding_expected [0 : VEC1024-1];
    logic [31 : 0] embedding_token_mem [0 : 0];

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
    logic [31 : 0] expected_input_norm [0 : VEC1024-1];
    logic [31 : 0] expected_qkv_input_norm [0 : (VEC2048 + (2 * VEC1024))-1];

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
    logic [31 : 0] expected_input_norm_l1 [0 : VEC1024-1];

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
    logic [31 : 0] expected_input_norm_l2 [0 : VEC1024-1];

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    logic [31 : 0] full_qkv_qmap [0 : (MAX_LAYERS*QKV_WORDS)-1];
    logic [31 : 0] full_frontend_qmap [0 : (MAX_LAYERS*FRONT_WORDS)-1];
    logic [31 : 0] full_score_qmap [0 : (MAX_LAYERS*SCORE_WORDS)-1];
    logic [31 : 0] full_oproj_qmap [0 : (MAX_LAYERS*OPROJ_WORDS)-1];
    logic [31 : 0] full_post_qmap [0 : (MAX_LAYERS*POST_WORDS)-1];
    logic [31 : 0] full_gate_qmap [0 : (MAX_LAYERS*GATE_WORDS)-1];
    logic [31 : 0] full_silu_qmap [0 : (MAX_LAYERS*SILU_WORDS)-1];
    logic [31 : 0] full_down_qmap [0 : (MAX_LAYERS*DOWN_WORDS)-1];
    logic [31 : 0] full_residual_qmap [0 : (MAX_LAYERS*RESIDUAL_WORDS)-1];
    logic [31 : 0] full_input_norm_qmap [0 : (MAX_LAYERS*INPUT_NORM_WORDS)-1];

    logic signed [IN_WIDTH-1 : 0] full_k_cache_mem [0 : (MAX_LAYERS*CACHE_LENGTH*KV_COUNT)-1];
    logic signed [IN_WIDTH-1 : 0] full_v_cache_mem [0 : (MAX_LAYERS*CACHE_LENGTH*KV_COUNT)-1];
    logic [31 : 0] full_oproj_weight_mem [0 : (MAX_LAYERS*OPROJ_WEIGHT_WORDS)-1];
    logic [31 : 0] full_oproj_scale_mem [0 : (MAX_LAYERS*OPROJ_SCALE_WORDS)-1];
    logic [31 : 0] full_gate_weight_mem [0 : (MAX_LAYERS*GATE_WEIGHT_WORDS)-1];
    logic [31 : 0] full_gate_scale_mem [0 : (MAX_LAYERS*GATE_SCALE_WORDS)-1];
    logic [31 : 0] full_up_weight_mem [0 : (MAX_LAYERS*GATE_WEIGHT_WORDS)-1];
    logic [31 : 0] full_up_scale_mem [0 : (MAX_LAYERS*GATE_SCALE_WORDS)-1];
    logic [31 : 0] full_down_weight_mem [0 : (MAX_LAYERS*DOWN_WEIGHT_WORDS)-1];
    logic [31 : 0] full_down_scale_mem [0 : (MAX_LAYERS*DOWN_SCALE_WORDS)-1];

    logic [31 : 0] full_expected_qkv [0 : (MAX_LAYERS*(VEC2048+(2*VEC1024)))-1];
    logic [31 : 0] full_expected_q_rope [0 : (MAX_LAYERS*VEC2048)-1];
    logic [63 : 0] full_expected_cache_addr [0 : (MAX_LAYERS*TOTAL_CACHE_WRITES)-1];
    logic [31 : 0] full_expected_cache_data [0 : (MAX_LAYERS*TOTAL_CACHE_WRITES)-1];
    logic [3 : 0] full_expected_cache_kind [0 : (MAX_LAYERS*TOTAL_CACHE_WRITES)-1];
    logic [31 : 0] full_expected_attn_out [0 : (MAX_LAYERS*VEC2048)-1];
    logic [31 : 0] full_expected_o_proj [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_expected_post_hidden [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_expected_post_norm [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_expected_gate [0 : (MAX_LAYERS*VEC3072)-1];
    logic [31 : 0] full_expected_up [0 : (MAX_LAYERS*VEC3072)-1];
    logic [31 : 0] full_expected_silu_hidden [0 : (MAX_LAYERS*VEC3072)-1];
    logic [31 : 0] full_expected_down [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_expected_layer [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_expected_input_norm [0 : (MAX_LAYERS*VEC1024)-1];
    logic [31 : 0] full_runtime_rope_mem [0 : ROPE_TABLE_WORDS-1];
    logic [31 : 0] full_runtime_hidden_mem [0 : (2*VEC1024)-1];

    logic [ADDR_WIDTH-1 : 0] full_qkv_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_input_norm_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_frontend_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_score_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_oproj_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_post_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_gate_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_silu_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_down_qmap_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_residual_qmap_base [0 : MAX_LAYERS-1];

    logic [ADDR_WIDTH-1 : 0] full_oproj_weight_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_oproj_scale_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_gate_weight_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_gate_scale_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_up_weight_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_up_scale_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_down_weight_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_down_scale_base [0 : MAX_LAYERS-1];

    logic [ADDR_WIDTH-1 : 0] full_input_hidden_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_input_norm_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_q_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_k_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_v_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_q_rope_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_attn_out_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_o_proj_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_post_hidden_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_post_norm_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_gate_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_up_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_silu_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_down_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_layer_output_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_actual_input_hidden_base [0 : MAX_LAYERS-1];
    logic [ADDR_WIDTH-1 : 0] full_actual_layer_output_base [0 : MAX_LAYERS-1];
`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
    logic signed [IN_WIDTH-1 : 0] persistent_step0_k_cache [0 : (MAX_LAYERS*KV_COUNT)-1];
    logic signed [IN_WIDTH-1 : 0] persistent_step0_v_cache [0 : (MAX_LAYERS*KV_COUNT)-1];
    logic [31 : 0] persistent_input_token [0 : 1];
    logic [31 : 0] persistent_expected_output_token [0 : 1];
    logic signed [TAIL_ROW_ACC_WIDTH-1 : 0] persistent_expected_output_score [0 : 1];
`endif
`endif

    logic [31 : 0] tail_qmap [0 : TAIL_QMAP_WORDS-1];
    logic [31 : 0] tail_weight_mem [0 : TAIL_WEIGHT_WORDS-1];
    logic [31 : 0] tail_scale_mem [0 : TAIL_SCALE_WORDS-1];
    logic signed [TAIL_ROW_ACC_WIDTH-1 : 0] tail_expected_logits [0 : TAIL_SCAN_ROWS-1];
    logic signed [TAIL_NORM_WIDTH-1 : 0] tail_final_norm_expected [0 : TAIL_INPUT_SIZE-1];
    logic [31 : 0] tail_expected_words [0 : 2];
    logic [31 : 0] tail_scan_base_token_mem [0 : 0];
    logic [ADDR_WIDTH-1 : 0] tail_weight_base_addr_mem [0 : 0];
    logic [ADDR_WIDTH-1 : 0] tail_scale_base_addr_mem [0 : 0];

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
    logic [ADDR_WIDTH-1 : 0] input_norm_output_base;

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
    logic [ADDR_WIDTH-1 : 0] input_norm_output_base_l1;

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
    logic [ADDR_WIDTH-1 : 0] input_norm_output_base_l2;

    logic [ADDR_WIDTH-1 : 0] layer0_qkv_packet_base_addr;
    logic [ADDR_WIDTH-1 : 0] embedding_weight_row_base;
    logic [ADDR_WIDTH-1 : 0] embedding_scale_row_base;
    integer full_chain_layer_count;
    integer full_chain_check_index;
    integer full_chain_manifest_runtime_context;
    integer full_chain_runtime_position;
    logic [ADDR_WIDTH-1 : 0] full_chain_kv_cache_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_embedding_output_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_runtime_hidden_a_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_runtime_hidden_b_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_runtime_rope_cos_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_runtime_rope_sin_base_addr;
    logic [ADDR_WIDTH-1 : 0] full_chain_runtime_last_hidden_base_addr;

    logic [ADDR_WIDTH-1 : 0] tail_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] tail_weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] tail_scale_base_addr;
    logic [ADDR_WIDTH-1 : 0] tail_norm_output_base;
    logic [ADDR_WIDTH-1 : 0] tail_output_base;
    logic [ADDR_WIDTH-1 : 0] tail_final_hidden_base;
    logic [ADDR_WIDTH-1 : 0] tail_descriptor_final_hidden_base;
    logic [ADDR_WIDTH-1 : 0] tail_final_gamma_base;

    string tracefile;
    string wavefile;
    string event_tracefile;
    integer trace_fd;
    integer event_trace_fd;
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
    integer input_norm_write_accept_count;
    integer embedding_write_accept_count;
    integer write_mismatch_count;
    longint signed max_abs_diff;

    integer tail_scheduler_done_cycle;
    integer tail_done_cycle;
    integer tail_mismatch_count;
    integer tail_write_mismatch_count;
    integer tail_read_req_count;
    integer tail_hidden_req_count;
    integer tail_gamma_req_count;
    integer tail_norm_read_req_count;
    integer tail_weight_req_count;
    integer tail_scale_req_count;
    integer tail_lm_req_count;
    integer tail_norm_write_word_count;
    integer tail_output_write_word_count;
    integer tail_done_seen_count;
    integer tail_first_hidden_read_cycle;
    integer tail_first_weight_read_cycle;
    integer tail_first_output_write_cycle;
    integer tail_start_cycle;
    logic [31 : 0] tail_expected_token;
    logic signed [TAIL_ROW_ACC_WIDTH-1 : 0] tail_expected_score_q26;

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
    integer input_norm_qkv_done_cycle;
    integer input_norm_qkv_read_bursts;
    integer input_norm_qkv_read_words;
    integer input_norm_qkv_write_reqs;
    integer input_norm_qkv_write_words;
    logic [1 : 0] invalid_stage_done_mask;
    logic [1 : 0] invalid_stage_error_mask;
    logic [1 : 0] input_norm_qkv_stage_done_mask;
    logic [1 : 0] input_norm_qkv_stage_error_mask;
    logic [3 : 0] invalid_layer0_full_stage_done_mask;
    logic [3 : 0] invalid_layer0_full_stage_error_mask;
    logic [3 : 0] input_norm_qkv_layer0_full_stage_done_mask;
    logic [3 : 0] input_norm_qkv_layer0_full_stage_error_mask;
    logic invalid_error;
    logic input_norm_qkv_error;
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
    integer pending_write_kind;
    integer pending_write_layer;
    logic [ADDR_WIDTH-1 : 0] pending_write_addr;

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
    logic input_norm_written;
    logic input_norm_written_l1;
    logic input_norm_written_l2;
    logic use_input_norm_qkv_expected;
    logic embedding_true3_mode;
    logic full_chain_mode;
    logic runtime_context_mode;
    logic persistent_two_token_mode;
    integer persistent_active_step;
    integer persistent_reset_release_count;
    integer persistent_prior_cache_read_req_count [0 : MAX_LAYERS-1];
    logic embedding_written;
    logic embedding_response_seen;
    logic [MAX_LAYERS-1 : 0] layer_response_seen;
    integer embedding_response_cycle;
    integer layer_response_cycle [0 : MAX_LAYERS-1];
    integer first_layer_hidden_read_cycle [0 : MAX_LAYERS-1];
    integer first_embedding_hidden_read_cycle;
    integer first_layer1_hidden_read_cycle;
    integer first_layer2_hidden_read_cycle;
    integer runtime_hidden_read_req_count [0 : MAX_LAYERS-1];
    integer runtime_layer_output_write_req_count [0 : MAX_LAYERS-1];
    integer runtime_rope_read_req_count [0 : MAX_LAYERS-1];
    integer runtime_cache_read_req_count [0 : MAX_LAYERS-1];
    integer runtime_cache_write_word_count [0 : MAX_LAYERS-1];

    integer last_trace_stage;
    integer last_trace_state;
    integer last_trace_wr_words;
    integer scoreboard_layer_iterations;
    logic fastmem;

`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
    assign use_tail_mem = top_tail_active;

`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
    qmap_one_token_control_regs #(
        .ADDR_WIDTH        (ADDR_WIDTH),
        .MAX_LAYERS        (MAX_LAYERS),
        .MAX_CONTEXT       (MAX_CONTEXT),
        .TOKEN_ID_WIDTH    (32),
        .SCORE_WIDTH       (TAIL_ROW_ACC_WIDTH)
    ) control_regs (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_reg_wr_valid(mmio_reg_wr_valid),
        .o_reg_wr_ready(mmio_reg_wr_ready),
        .i_reg_rd_valid(mmio_reg_rd_valid),
        .o_reg_rd_ready(mmio_reg_rd_ready),
        .i_reg_addr(mmio_reg_addr),
        .i_reg_wdata(mmio_reg_wdata),
        .o_reg_rdata(mmio_reg_rdata),
        .o_reg_error(mmio_reg_error),
        .o_start_pulse(mmio_start_pulse),
        .o_input_token_id(mmio_input_token_id),
        .o_layer_start_index(mmio_layer_start_index),
        .o_layer_count(mmio_layer_count),
        .o_position(mmio_position),
        .o_runtime_context_enable(mmio_runtime_context_enable),
        .o_input_hidden_base_addr(mmio_input_hidden_base_addr),
        .o_output_hidden_base_addr(mmio_output_hidden_base_addr),
        .o_kv_cache_base_addr(mmio_kv_cache_base_addr),
        .o_final_tail_qmap_base_addr(mmio_final_tail_qmap_base_addr),
        .o_final_hidden_base_override_valid(mmio_final_hidden_base_override_valid),
        .o_final_hidden_base_override_addr(mmio_final_hidden_base_override_addr),
        .o_qkv_qmap_base_addr_table(mmio_qkv_qmap_base_addr_table),
        .o_input_norm_qmap_base_addr_table(mmio_input_norm_qmap_base_addr_table),
        .o_attn_frontend_qmap_base_addr_table(mmio_attn_frontend_qmap_base_addr_table),
        .o_attn_score_value_qmap_base_addr_table(mmio_attn_score_value_qmap_base_addr_table),
        .o_o_proj_qmap_base_addr_table(mmio_o_proj_qmap_base_addr_table),
        .o_post_attn_norm_qmap_base_addr_table(mmio_post_attn_norm_qmap_base_addr_table),
        .o_mlp_gate_up_qmap_base_addr_table(mmio_mlp_gate_up_qmap_base_addr_table),
        .o_mlp_silu_mul_qmap_base_addr_table(mmio_mlp_silu_mul_qmap_base_addr_table),
        .o_mlp_down_qmap_base_addr_table(mmio_mlp_down_qmap_base_addr_table),
        .o_mlp_residual_add_qmap_base_addr_table(mmio_mlp_residual_add_qmap_base_addr_table),
        .i_top_busy(busy),
        .i_top_done(done),
        .i_top_error(error),
        .i_top_state_debug(state_debug),
        .i_top_phase_debug(top_phase_debug),
        .i_layers_started(layers_started),
        .i_layers_completed(layers_completed),
        .i_layer_done_mask(layer_done_mask),
        .i_layer_error_mask(layer_error_mask),
        .i_last_layer_output_base_addr(scheduler_last_layer_output_base),
        .i_tail_error(tail_error),
        .i_tail_norm_saturation(tail_norm_saturation),
        .i_tail_effective_final_hidden_base_addr(top_tail_effective_final_hidden_base),
        .i_tail_best_token_id(tail_best_token_id),
        .i_tail_best_score_q26(tail_best_score_q26),
        .i_tail_tiles_started(tail_tiles_started),
        .i_tail_tiles_completed(tail_tiles_completed),
        .i_mem_read_burst_count(top_mem_read_burst_count),
        .i_mem_read_word_count(top_mem_read_word_count),
        .i_mem_write_req_count(top_mem_write_req_count),
        .i_mem_write_word_count(top_mem_write_word_count)
    );
`endif

`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
    qmap_one_token_axil_top #(
        .AXI_ADDR_WIDTH (12),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH),
        .MAX_LAYERS     (MAX_LAYERS),
        .MAX_CONTEXT    (MAX_CONTEXT),
        .TAIL_MAX_TILES (TAIL_MAX_TILES),
        .TOKEN_ID_WIDTH (32)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_s_axi_awaddr(axil_awaddr),
        .i_s_axi_awprot(axil_awprot),
        .i_s_axi_awvalid(axil_awvalid),
        .o_s_axi_awready(axil_awready),
        .i_s_axi_wdata(axil_wdata),
        .i_s_axi_wstrb(axil_wstrb),
        .i_s_axi_wvalid(axil_wvalid),
        .o_s_axi_wready(axil_wready),
        .o_s_axi_bresp(axil_bresp),
        .o_s_axi_bvalid(axil_bvalid),
        .i_s_axi_bready(axil_bready),
        .i_s_axi_araddr(axil_araddr),
        .i_s_axi_arprot(axil_arprot),
        .i_s_axi_arvalid(axil_arvalid),
        .o_s_axi_arready(axil_arready),
        .o_s_axi_rdata(axil_rdata),
        .o_s_axi_rresp(axil_rresp),
        .o_s_axi_rvalid(axil_rvalid),
        .i_s_axi_rready(axil_rready),
        .o_axil_busy(axil_busy),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_phase_debug(top_phase_debug),
        .o_scheduler_done_pulse(top_scheduler_done_pulse),
        .o_tail_start_pulse(top_tail_start_pulse),
        .o_tail_done_pulse(top_tail_done_pulse),
        .o_tail_active(top_tail_active),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_last_layer_output_base_addr(scheduler_last_layer_output_base),
        .o_layer0_active_stage_debug(active_stage_debug),
        .o_layer0_state_debug(layer0_state_debug),
        .o_layer0_stage_done_mask(stage_done_mask),
        .o_layer0_stage_error_mask(stage_error_mask),
        .o_layer0_full_stage_done_mask(layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(layer0_full_stage_error_mask),
        .o_body_stage_done_mask(body_stage_done_mask),
        .o_body_stage_error_mask(body_stage_error_mask),
        .o_scheduler_mem_read_burst_count(dut_read_burst_count),
        .o_scheduler_mem_read_word_count(dut_read_word_count),
        .o_scheduler_mem_write_req_count(dut_write_req_count),
        .o_scheduler_mem_write_word_count(dut_write_word_count),
        .o_tail_error(tail_error),
        .o_tail_norm_saturation(tail_norm_saturation),
        .o_tail_effective_final_hidden_base_addr(top_tail_effective_final_hidden_base),
        .o_tail_best_token_id(tail_best_token_id),
        .o_tail_best_score_q26(tail_best_score_q26),
        .o_tail_tiles_started(tail_tiles_started),
        .o_tail_tiles_completed(tail_tiles_completed),
        .o_tail_norm_cycle_count(tail_norm_cycle_count),
        .o_tail_mem_read_burst_count(tail_mem_read_burst_count),
        .o_tail_mem_read_word_count(tail_mem_read_word_count),
        .o_tail_mem_write_word_count(tail_mem_write_word_count),
        .o_mem_read_burst_count(top_mem_read_burst_count),
        .o_mem_read_word_count(top_mem_read_word_count),
        .o_mem_write_req_count(top_mem_write_req_count),
        .o_mem_write_word_count(top_mem_write_word_count),
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
`else
`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
    wire logic top_start_signal = mmio_start_pulse;
    wire logic [LAYER_INDEX_WIDTH-1 : 0] top_layer_start_index_signal = mmio_layer_start_index;
    wire logic [LAYER_COUNT_WIDTH-1 : 0] top_layer_count_signal = mmio_layer_count;
    wire logic [POSITION_WIDTH-1 : 0] top_position_signal = mmio_position;
    wire logic top_runtime_context_enable_signal = mmio_runtime_context_enable;
    wire logic [ADDR_WIDTH-1 : 0] top_input_hidden_base_signal = mmio_input_hidden_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_output_hidden_base_signal = mmio_output_hidden_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_kv_cache_base_signal = mmio_kv_cache_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_final_tail_qmap_base_signal = mmio_final_tail_qmap_base_addr;
    wire logic top_final_hidden_override_valid_signal = mmio_final_hidden_base_override_valid;
    wire logic [ADDR_WIDTH-1 : 0] top_final_hidden_override_addr_signal = mmio_final_hidden_base_override_addr;
    wire logic [BASE_TABLE_BITS-1 : 0] top_qkv_qmap_base_addr_table_signal = mmio_qkv_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_input_norm_qmap_base_addr_table_signal = mmio_input_norm_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_attn_frontend_qmap_base_addr_table_signal = mmio_attn_frontend_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_attn_score_value_qmap_base_addr_table_signal = mmio_attn_score_value_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_o_proj_qmap_base_addr_table_signal = mmio_o_proj_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_post_attn_norm_qmap_base_addr_table_signal = mmio_post_attn_norm_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_gate_up_qmap_base_addr_table_signal = mmio_mlp_gate_up_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_silu_mul_qmap_base_addr_table_signal = mmio_mlp_silu_mul_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_down_qmap_base_addr_table_signal = mmio_mlp_down_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_residual_add_qmap_base_addr_table_signal = mmio_mlp_residual_add_qmap_base_addr_table;
`else
    wire logic top_start_signal = start;
    wire logic [LAYER_INDEX_WIDTH-1 : 0] top_layer_start_index_signal = layer_start_index;
    wire logic [LAYER_COUNT_WIDTH-1 : 0] top_layer_count_signal = layer_count;
    wire logic [POSITION_WIDTH-1 : 0] top_position_signal = token_position;
    wire logic top_runtime_context_enable_signal = 1'b0;
    wire logic [ADDR_WIDTH-1 : 0] top_input_hidden_base_signal = input_hidden_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_output_hidden_base_signal = output_hidden_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_kv_cache_base_signal = kv_cache_base_addr;
    wire logic [ADDR_WIDTH-1 : 0] top_final_tail_qmap_base_signal = tail_qmap_base_addr;
    wire logic top_final_hidden_override_valid_signal = 1'b0;
    wire logic [ADDR_WIDTH-1 : 0] top_final_hidden_override_addr_signal = {ADDR_WIDTH{1'b0}};
    wire logic [BASE_TABLE_BITS-1 : 0] top_qkv_qmap_base_addr_table_signal = qkv_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_input_norm_qmap_base_addr_table_signal = input_norm_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_attn_frontend_qmap_base_addr_table_signal = attn_frontend_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_attn_score_value_qmap_base_addr_table_signal = attn_score_value_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_o_proj_qmap_base_addr_table_signal = o_proj_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_post_attn_norm_qmap_base_addr_table_signal = post_attn_norm_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_gate_up_qmap_base_addr_table_signal = mlp_gate_up_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_silu_mul_qmap_base_addr_table_signal = mlp_silu_mul_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_down_qmap_base_addr_table_signal = mlp_down_qmap_base_addr_table;
    wire logic [BASE_TABLE_BITS-1 : 0] top_mlp_residual_add_qmap_base_addr_table_signal = mlp_residual_add_qmap_base_addr_table;
`endif

    qmap_one_token_top #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH),
        .MAX_LAYERS     (MAX_LAYERS),
        .MAX_CONTEXT    (MAX_CONTEXT),
        .TAIL_MAX_TILES (TAIL_MAX_TILES),
        .TOKEN_ID_WIDTH (32)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(top_start_signal),
        .i_embedding_enable(1'b0),
        .i_input_token_id(32'd0),
        .i_embedding_weight_base_addr('0),
        .i_embedding_scale_base_addr('0),
        .i_layer_start_index(top_layer_start_index_signal),
        .i_layer_count(top_layer_count_signal),
        .i_position(top_position_signal),
        .i_runtime_context_enable(top_runtime_context_enable_signal),
        .i_input_hidden_base_addr(top_input_hidden_base_signal),
        .i_output_hidden_base_addr(top_output_hidden_base_signal),
        .i_kv_cache_base_addr(top_kv_cache_base_signal),
        .i_final_tail_qmap_base_addr(top_final_tail_qmap_base_signal),
        .i_final_hidden_base_override_valid(top_final_hidden_override_valid_signal),
        .i_final_hidden_base_override_addr(top_final_hidden_override_addr_signal),
        .i_qkv_qmap_base_addr_table(top_qkv_qmap_base_addr_table_signal),
        .i_input_norm_qmap_base_addr_table(top_input_norm_qmap_base_addr_table_signal),
        .i_attn_frontend_qmap_base_addr_table(top_attn_frontend_qmap_base_addr_table_signal),
        .i_attn_score_value_qmap_base_addr_table(top_attn_score_value_qmap_base_addr_table_signal),
        .i_o_proj_qmap_base_addr_table(top_o_proj_qmap_base_addr_table_signal),
        .i_post_attn_norm_qmap_base_addr_table(top_post_attn_norm_qmap_base_addr_table_signal),
        .i_mlp_gate_up_qmap_base_addr_table(top_mlp_gate_up_qmap_base_addr_table_signal),
        .i_mlp_silu_mul_qmap_base_addr_table(top_mlp_silu_mul_qmap_base_addr_table_signal),
        .i_mlp_down_qmap_base_addr_table(top_mlp_down_qmap_base_addr_table_signal),
        .i_mlp_residual_add_qmap_base_addr_table(top_mlp_residual_add_qmap_base_addr_table_signal),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_phase_debug(top_phase_debug),
        .o_scheduler_done_pulse(top_scheduler_done_pulse),
        .o_tail_start_pulse(top_tail_start_pulse),
        .o_tail_done_pulse(top_tail_done_pulse),
        .o_tail_active(top_tail_active),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_last_layer_output_base_addr(scheduler_last_layer_output_base),
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
        .o_scheduler_mem_read_burst_count(dut_read_burst_count),
        .o_scheduler_mem_read_word_count(dut_read_word_count),
        .o_scheduler_mem_write_req_count(dut_write_req_count),
        .o_scheduler_mem_write_word_count(dut_write_word_count),
        .o_tail_error(tail_error),
        .o_tail_norm_saturation(tail_norm_saturation),
        .o_tail_effective_final_hidden_base_addr(top_tail_effective_final_hidden_base),
        .o_tail_best_token_id(tail_best_token_id),
        .o_tail_best_score_q26(tail_best_score_q26),
        .o_tail_tiles_started(tail_tiles_started),
        .o_tail_tiles_completed(tail_tiles_completed),
        .o_tail_norm_cycle_count(tail_norm_cycle_count),
        .o_tail_mem_read_burst_count(tail_mem_read_burst_count),
        .o_tail_mem_read_word_count(tail_mem_read_word_count),
        .o_tail_mem_write_word_count(tail_mem_write_word_count),
        .o_mem_read_burst_count(top_mem_read_burst_count),
        .o_mem_read_word_count(top_mem_read_word_count),
        .o_mem_write_req_count(top_mem_write_req_count),
        .o_mem_write_word_count(top_mem_write_word_count),
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
`endif
`else
    assign use_tail_mem = use_tail_mem_manual;

    qmap_one_token_layer_scheduler dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_layer_start_index(layer_start_index),
        .i_layer_count(layer_count),
        .i_position(token_position),
        .i_runtime_context_enable(1'b0),
        .i_input_hidden_base_addr(input_hidden_base_addr),
        .i_output_hidden_base_addr(output_hidden_base_addr),
        .i_kv_cache_base_addr(kv_cache_base_addr),
        .i_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .i_input_norm_qmap_base_addr_table(input_norm_qmap_base_addr_table),
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
        .o_last_layer_output_base_addr(scheduler_last_layer_output_base),
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
        .o_mem_rd_req_valid(sched_mem_rd_req_valid),
        .i_mem_rd_req_ready(sched_mem_rd_req_ready),
        .o_mem_rd_req_addr(sched_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(sched_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(sched_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(sched_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(sched_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(sched_mem_rd_rsp_last),
        .o_mem_wr_req_valid(sched_mem_wr_req_valid),
        .i_mem_wr_req_ready(sched_mem_wr_req_ready),
        .o_mem_wr_req_addr(sched_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(sched_mem_wr_req_len_bytes),
        .o_mem_wr_data(sched_mem_wr_data),
        .o_mem_wr_data_valid(sched_mem_wr_data_valid),
        .i_mem_wr_data_ready(sched_mem_wr_data_ready),
        .o_mem_wr_data_last(sched_mem_wr_data_last),
        .i_mem_wr_done(sched_mem_wr_done),
        .i_mem_wr_error(sched_mem_wr_error)
    );

`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
    qmap_final_token_tail_compute_path #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .MAX_TILES       (TAIL_MAX_TILES),
        .INPUT_SIZE      (TAIL_INPUT_SIZE),
        .MEM_DATA_WIDTH  (MEM_DATA_WIDTH)
    ) tail_dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(tail_start),
        .i_qmap_base_addr(tail_qmap_base_addr),
        .i_final_hidden_base_override_valid(1'b0),
        .i_final_hidden_base_override_addr({ADDR_WIDTH{1'b0}}),
        .o_busy(tail_busy),
        .o_done(tail_done),
        .o_error(tail_error),
        .o_norm_saturation(tail_norm_saturation),
        .o_effective_final_hidden_base_addr(),
        .o_best_token_id(tail_best_token_id),
        .o_best_score_q26(tail_best_score_q26),
        .o_tiles_started(tail_tiles_started),
        .o_tiles_completed(tail_tiles_completed),
        .o_norm_cycle_count(tail_norm_cycle_count),
        .o_mem_read_burst_count(tail_mem_read_burst_count),
        .o_mem_read_word_count(tail_mem_read_word_count),
        .o_mem_write_word_count(tail_mem_write_word_count),
        .o_mem_rd_req_valid(tail_mem_rd_req_valid),
        .i_mem_rd_req_ready(tail_mem_rd_req_ready),
        .o_mem_rd_req_addr(tail_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(tail_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(tail_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(tail_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(tail_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(tail_mem_rd_rsp_last),
        .o_mem_wr_req_valid(tail_mem_wr_req_valid),
        .i_mem_wr_req_ready(tail_mem_wr_req_ready),
        .o_mem_wr_req_addr(tail_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(tail_mem_wr_req_len_bytes),
        .o_mem_wr_data(tail_mem_wr_data),
        .o_mem_wr_data_valid(tail_mem_wr_data_valid),
        .i_mem_wr_data_ready(tail_mem_wr_data_ready),
        .o_mem_wr_data_last(tail_mem_wr_data_last),
        .i_mem_wr_done(tail_mem_wr_done),
        .i_mem_wr_error(tail_mem_wr_error)
    );
`endif

`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
    assign mem_rd_req_valid = use_tail_mem ? tail_mem_rd_req_valid : sched_mem_rd_req_valid;
    assign mem_rd_req_addr = use_tail_mem ? tail_mem_rd_req_addr : sched_mem_rd_req_addr;
    assign mem_rd_req_len_bytes = use_tail_mem ? tail_mem_rd_req_len_bytes : sched_mem_rd_req_len_bytes;
    assign sched_mem_rd_req_ready = use_tail_mem ? 1'b0 : mem_rd_req_ready;
    assign tail_mem_rd_req_ready = use_tail_mem ? mem_rd_req_ready : 1'b0;
    assign sched_mem_rd_rsp_valid = use_tail_mem ? 1'b0 : mem_rd_rsp_valid;
    assign tail_mem_rd_rsp_valid = use_tail_mem ? mem_rd_rsp_valid : 1'b0;
    assign sched_mem_rd_rsp_data = mem_rd_rsp_data;
    assign tail_mem_rd_rsp_data = mem_rd_rsp_data;
    assign sched_mem_rd_rsp_last = use_tail_mem ? 1'b0 : mem_rd_rsp_last;
    assign tail_mem_rd_rsp_last = use_tail_mem ? mem_rd_rsp_last : 1'b0;
    assign mem_rd_rsp_ready = use_tail_mem ? tail_mem_rd_rsp_ready : sched_mem_rd_rsp_ready;

    assign mem_wr_req_valid = use_tail_mem ? tail_mem_wr_req_valid : sched_mem_wr_req_valid;
    assign mem_wr_req_addr = use_tail_mem ? tail_mem_wr_req_addr : sched_mem_wr_req_addr;
    assign mem_wr_req_len_bytes = use_tail_mem ? tail_mem_wr_req_len_bytes : sched_mem_wr_req_len_bytes;
    assign sched_mem_wr_req_ready = use_tail_mem ? 1'b0 : mem_wr_req_ready;
    assign tail_mem_wr_req_ready = use_tail_mem ? mem_wr_req_ready : 1'b0;
    assign mem_wr_data = use_tail_mem ? tail_mem_wr_data : sched_mem_wr_data;
    assign mem_wr_data_valid = use_tail_mem ? tail_mem_wr_data_valid : sched_mem_wr_data_valid;
    assign mem_wr_data_last = use_tail_mem ? tail_mem_wr_data_last : sched_mem_wr_data_last;
    assign sched_mem_wr_data_ready = use_tail_mem ? 1'b0 : mem_wr_data_ready;
    assign tail_mem_wr_data_ready = use_tail_mem ? mem_wr_data_ready : 1'b0;
    assign sched_mem_wr_done = use_tail_mem ? 1'b0 : mem_wr_done;
    assign tail_mem_wr_done = use_tail_mem ? mem_wr_done : 1'b0;
    assign sched_mem_wr_error = use_tail_mem ? 1'b0 : mem_wr_error;
    assign tail_mem_wr_error = use_tail_mem ? mem_wr_error : 1'b0;
`else
    assign mem_rd_req_valid = sched_mem_rd_req_valid;
    assign mem_rd_req_addr = sched_mem_rd_req_addr;
    assign mem_rd_req_len_bytes = sched_mem_rd_req_len_bytes;
    assign sched_mem_rd_req_ready = mem_rd_req_ready;
    assign sched_mem_rd_rsp_valid = mem_rd_rsp_valid;
    assign sched_mem_rd_rsp_data = mem_rd_rsp_data;
    assign sched_mem_rd_rsp_last = mem_rd_rsp_last;
    assign mem_rd_rsp_ready = sched_mem_rd_rsp_ready;

    assign mem_wr_req_valid = sched_mem_wr_req_valid;
    assign mem_wr_req_addr = sched_mem_wr_req_addr;
    assign mem_wr_req_len_bytes = sched_mem_wr_req_len_bytes;
    assign sched_mem_wr_req_ready = mem_wr_req_ready;
    assign mem_wr_data = sched_mem_wr_data;
    assign mem_wr_data_valid = sched_mem_wr_data_valid;
    assign mem_wr_data_last = sched_mem_wr_data_last;
    assign sched_mem_wr_data_ready = mem_wr_data_ready;
    assign sched_mem_wr_done = mem_wr_done;
    assign sched_mem_wr_error = mem_wr_error;
`endif
`endif

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
    always @(posedge rst_n) begin
        persistent_reset_release_count = persistent_reset_release_count + 1;
    end
`endif

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
        logic [ADDR_WIDTH-1 : 0] selected_base;
        integer selected_layers;
        begin
            selected_base = full_chain_mode ? full_chain_kv_cache_base_addr : CACHE_BASE_ADDR;
            selected_layers = full_chain_mode ? full_chain_layer_count : MODELED_CACHE_LAYERS;
            is_cache_addr =
                (addr >= selected_base) &&
                (addr < (selected_base + (selected_layers * CACHE_WORDS_FULL * MEM_DATA_BYTES)));
        end
    endfunction

    function automatic logic [MAX_LAYERS-1 : 0] expected_layer_mask(
        input integer start_layer,
        input integer count
    );
        integer i;
        begin
            expected_layer_mask = '0;
            for (i = 0; i < count; i = i + 1) begin
                if ((start_layer + i) < MAX_LAYERS) begin
                    expected_layer_mask[start_layer + i] = 1'b1;
                end
            end
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

    function automatic logic [ADDR_WIDTH-1 : 0] tail_desc_base(input integer slot);
        begin
            tail_desc_base = {
                tail_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                tail_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] input_norm_desc_base(input integer slot);
        begin
            input_norm_desc_base = {
                input_norm_qmap[desc_idx(slot, DESC_BASE_HI_WORD)],
                input_norm_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] input_norm_desc_base_l1(input integer slot);
        begin
            input_norm_desc_base_l1 = {
                input_norm_qmap_l1[desc_idx(slot, DESC_BASE_HI_WORD)],
                input_norm_qmap_l1[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] input_norm_desc_base_l2(input integer slot);
        begin
            input_norm_desc_base_l2 = {
                input_norm_qmap_l2[desc_idx(slot, DESC_BASE_HI_WORD)],
                input_norm_qmap_l2[desc_idx(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [31 : 0] tail_norm_expected_word(input integer index);
        begin
            tail_norm_expected_word = {
                {8{tail_final_norm_expected[index][TAIL_NORM_WIDTH-1]}},
                tail_final_norm_expected[index]
            };
        end
    endfunction

    task patch_tail_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            tail_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            tail_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_qkv_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            qkv_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            qkv_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

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
            full_chain_mode = $test$plusargs("embedding_full_chain");
            runtime_context_mode = $test$plusargs("runtime_context");
            persistent_two_token_mode = $test$plusargs("persistent_two_token");
            embedding_true3_mode = $test$plusargs("embedding_true3") || full_chain_mode;
            persistent_active_step = 0;
            full_chain_layer_count = 0;
            full_chain_manifest_runtime_context = 0;
            full_chain_runtime_position = 0;
            full_chain_kv_cache_base_addr = CACHE_BASE_ADDR;
            full_chain_embedding_output_base_addr = '0;
            full_chain_runtime_hidden_a_base_addr = '0;
            full_chain_runtime_hidden_b_base_addr = '0;
            full_chain_runtime_rope_cos_base_addr = '0;
            full_chain_runtime_rope_sin_base_addr = '0;
            full_chain_runtime_last_hidden_base_addr = '0;
            tail_qmap_base_addr = `QMAP_FINAL_TOKEN_BASE_ADDR;
            layer0_qkv_packet_base_addr = embedding_true3_mode ?
                EMBEDDING_LAYER0_QKV_BASE_ADDR : `QMAP_QKV_BASE_ADDR;
            embedding_weight_row_base = '0;
            embedding_scale_row_base = '0;
            if (embedding_true3_mode) begin
                $readmemh("FPGA_Project/sim/vectors/embedding_weight_words32.hex", embedding_weight_mem);
                $readmemh("FPGA_Project/sim/vectors/embedding_scale_words32.hex", embedding_scale_mem);
                $readmemh("FPGA_Project/sim/vectors/embedding_expected_q14_10.hex", embedding_expected);
                $readmemh("FPGA_Project/sim/vectors/embedding_token_id.hex", embedding_token_mem);
                embedding_weight_row_base = EMBEDDING_WEIGHT_BASE_ADDR +
                    (embedding_token_mem[0] * (EMBEDDING_WEIGHT_WORDS * MEM_DATA_BYTES));
                embedding_scale_row_base = EMBEDDING_SCALE_BASE_ADDR +
                    (embedding_token_mem[0] * (EMBEDDING_SCALE_WORDS * MEM_DATA_BYTES));
            end
            if (!full_chain_mode) begin
            if ($test$plusargs("input_norm_qkv_only")) begin
                $readmemh("FPGA_Project/sim/vectors/qmap_qkv_projection_from_input_rmsnorm_image_words32.hex", qkv_qmap);
            end else begin
                $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_full_image_words32.hex", qkv_qmap);
            end
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_frontend_image_words32.hex", frontend_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_attention_score_value_image_words32.hex", score_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_o_proj_image_words32.hex", oproj_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_image_words32.hex", post_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_image_words32.hex", gate_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_silu_mul_image_words32.hex", silu_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_down_image_words32.hex", down_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_residual_add_image_words32.hex", residual_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_input_rmsnorm_image_words32.hex", input_norm_qmap);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer1_qkv_from_layer0_rtl_full_image_words32.hex", qkv_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_frontend_image_words32.hex", frontend_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_attention_score_value_image_words32.hex", score_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_o_proj_image_words32.hex", oproj_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_post_attention_residual_norm_image_words32.hex", post_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_gate_up_image_words32.hex", gate_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_silu_mul_image_words32.hex", silu_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_down_image_words32.hex", down_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_mlp_residual_add_image_words32.hex", residual_qmap_l1);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_input_rmsnorm_image_words32.hex", input_norm_qmap_l1);
            $readmemh("artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer2_qkv_from_layer1_rtl_full_image_words32.hex", qkv_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_frontend_image_words32.hex", frontend_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_attention_score_value_image_words32.hex", score_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_o_proj_image_words32.hex", oproj_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_post_attention_residual_norm_image_words32.hex", post_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_gate_up_image_words32.hex", gate_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_silu_mul_image_words32.hex", silu_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_down_image_words32.hex", down_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_mlp_residual_add_image_words32.hex", residual_qmap_l2);
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_input_rmsnorm_image_words32.hex", input_norm_qmap_l2);

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
            $readmemh("FPGA_Project/sim/vectors/qmap_input_rmsnorm_expected_words32.hex", expected_input_norm);
            $readmemh("FPGA_Project/sim/vectors/qmap_qkv_projection_expected_from_input_rmsnorm_words32.hex", expected_qkv_input_norm);
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
            $readmemh("FPGA_Project/sim/vectors/qmap_layer1_chained_input_rmsnorm_expected_words32.hex", expected_input_norm_l1);
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
            $readmemh("FPGA_Project/sim/vectors/qmap_layer2_chained_input_rmsnorm_expected_words32.hex", expected_input_norm_l2);
            end
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $readmemh("FPGA_Project/sim/vectors/qmap_final_token_tail_layer2_chained_full_vocab_image_words32.hex", tail_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_final_token_tail_layer2_chained_full_vocab_expected_words32.hex", tail_expected_words);
            $readmemh("FPGA_Project/sim/vectors/final_rmsnorm_layer2_chained_stage_real_expected.hex", tail_final_norm_expected);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_weight_words32.hex", tail_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_scale_words32.hex", tail_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_expected_scan_logits_q26.hex", tail_expected_logits);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_scan_base_token.hex", tail_scan_base_token_mem);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_weight_base_addr.hex", tail_weight_base_addr_mem);
            $readmemh("FPGA_Project/sim/vectors/lm_head_argmax_layer2_chained_full_vocab_real_scale_base_addr.hex", tail_scale_base_addr_mem);
`endif
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
`include "full_chain_vector_loads.svh"
`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
                if (persistent_two_token_mode) begin
`include "persistent_two_token_config.svh"
                end
`endif
                if ((full_chain_layer_count <= 0) || (full_chain_layer_count > MAX_LAYERS)) begin
                    $display("FAIL: generated full-chain layer count %0d is outside 1..%0d",
                             full_chain_layer_count, MAX_LAYERS);
                    $finish(1);
                end
                if (full_chain_manifest_runtime_context != runtime_context_mode) begin
                    $display("FAIL: runtime-context plusarg=%0d does not match generated manifest=%0d",
                             runtime_context_mode, full_chain_manifest_runtime_context);
                    $finish(1);
                end
                if (persistent_two_token_mode &&
                    (!runtime_context_mode || !full_chain_manifest_runtime_context)) begin
                    $display("FAIL: persistent two-token mode requires a RuntimeContext manifest");
                    $finish(1);
                end
                if (runtime_context_mode &&
                    ((full_chain_runtime_position < 0) ||
                     (full_chain_runtime_position >= MAX_CONTEXT) ||
                     (full_chain_embedding_output_base_addr !==
                      full_chain_runtime_hidden_a_base_addr) ||
                     (full_chain_runtime_hidden_a_base_addr == '0) ||
                     (full_chain_runtime_hidden_b_base_addr == '0) ||
                     (full_chain_runtime_hidden_a_base_addr ==
                      full_chain_runtime_hidden_b_base_addr) ||
                     (full_chain_runtime_rope_sin_base_addr !==
                      (full_chain_runtime_rope_cos_base_addr +
                       (MAX_CONTEXT * HEAD_DIM * MEM_DATA_BYTES))))) begin
                    $display("FAIL: generated runtime-context include violates RoPE/hidden contract");
                    $finish(1);
                end
                layer0_qkv_packet_base_addr = full_qkv_qmap_base[0];
            end
`endif

            if (!full_chain_mode) begin
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
            input_norm_output_base = input_norm_desc_base(INPUT_NORM_SLOT_OUTPUT);
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
            input_norm_output_base_l1 = input_norm_desc_base_l1(INPUT_NORM_SLOT_OUTPUT);
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
            input_norm_output_base_l2 = input_norm_desc_base_l2(INPUT_NORM_SLOT_OUTPUT);

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
            if ($test$plusargs("l1_only") ||
                $test$plusargs("l1_input_norm_only") ||
                $test$plusargs("l1_input_norm_full_only") ||
                $test$plusargs("l2_input_norm_only") ||
                $test$plusargs("l2_input_norm_full_only") ||
                $test$plusargs("l2_tail_only") ||
                $test$plusargs("l2_top_tail_only") ||
                $test$plusargs("l1_l2_top_tail_only") ||
                $test$plusargs("l1_l2_mmio_top_tail_only") ||
                $test$plusargs("true2_input_norm") ||
                $test$plusargs("true3_input_norm") ||
                $test$plusargs("true3_top_tail_only") ||
                $test$plusargs("true3_mmio_top_tail_only")) begin
                patch_qkv_base_l1(QKV_SLOT_ACTIVATION, input_norm_output_base_l1);
            end
            else begin
                patch_qkv_base_l1(QKV_SLOT_ACTIVATION, layer_output_base);
            end
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
            if ($test$plusargs("l2_only") ||
                $test$plusargs("l2_input_norm_only") ||
                $test$plusargs("l2_input_norm_full_only") ||
                $test$plusargs("l2_tail_only") ||
                $test$plusargs("l2_top_tail_only") ||
                $test$plusargs("l1_l2_top_tail_only") ||
                $test$plusargs("l1_l2_mmio_top_tail_only") ||
                $test$plusargs("true3_input_norm") ||
                $test$plusargs("true3_top_tail_only") ||
                $test$plusargs("true3_mmio_top_tail_only")) begin
                patch_qkv_base_l2(QKV_SLOT_ACTIVATION, input_norm_output_base_l2);
            end
            else begin
                patch_qkv_base_l2(QKV_SLOT_ACTIVATION, layer_output_base_l1);
            end
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

`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            tail_descriptor_final_hidden_base = tail_desc_base(TAIL_SLOT_FINAL_HIDDEN);
`ifndef QMAP_ONE_TOKEN_TB_USE_TOP
            patch_tail_base(TAIL_SLOT_FINAL_HIDDEN, layer_output_base_l2);
`endif
            tail_weight_base_addr = tail_weight_base_addr_mem[0];
            tail_scale_base_addr = tail_scale_base_addr_mem[0];
            tail_norm_output_base = tail_desc_base(TAIL_SLOT_NORM_OUTPUT);
            tail_output_base = tail_desc_base(TAIL_SLOT_OUTPUT);
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                tail_final_hidden_base = full_actual_layer_output_base[full_chain_layer_count - 1];
            end
            else begin
                tail_final_hidden_base = layer_output_base_l2;
            end
`else
            tail_final_hidden_base = layer_output_base_l2;
`endif
`else
            tail_final_hidden_base = tail_desc_base(TAIL_SLOT_FINAL_HIDDEN);
`endif
            tail_final_gamma_base = tail_desc_base(TAIL_SLOT_FINAL_GAMMA);
`endif
        end
    endtask

`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
    task load_persistent_step_vectors;
        input integer step;
        begin
            case (step)
                0: begin
`include "persistent_step0_vector_loads.svh"
                end
                1: begin
`include "persistent_step1_vector_loads.svh"
                end
                default: begin
                    $display("FAIL: persistent token step %0d is outside 0..1", step);
                    $finish(1);
                end
            endcase
            persistent_active_step = step;
            full_chain_runtime_position = step;
            embedding_weight_row_base = EMBEDDING_WEIGHT_BASE_ADDR +
                (embedding_token_mem[0] * (EMBEDDING_WEIGHT_WORDS * MEM_DATA_BYTES));
            embedding_scale_row_base = EMBEDDING_SCALE_BASE_ADDR +
                (embedding_token_mem[0] * (EMBEDDING_SCALE_WORDS * MEM_DATA_BYTES));
        end
    endtask
`endif

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
        integer full_index;
        logic [ADDR_WIDTH-1 : 0] selected_base;
        logic signed [IN_WIDTH-1 : 0] value;
        begin
            selected_base = full_chain_mode ? full_chain_kv_cache_base_addr : CACHE_BASE_ADDR;
            element_index = (addr - selected_base) >> 2;
            layer = element_index / CACHE_WORDS_FULL;
            layer_word_index = element_index % CACHE_WORDS_FULL;
            kind = layer_word_index / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            rem0 = layer_word_index % (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            head = rem0 / (MAX_CONTEXT * HEAD_DIM);
            rem1 = rem0 % (MAX_CONTEXT * HEAD_DIM);
            position = rem1 / HEAD_DIM;
            dim = rem1 % HEAD_DIM;
            cache_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;
            full_index = (layer * CACHE_LENGTH * KV_COUNT) + cache_index;

            if ((layer < MODELED_CACHE_LAYERS) &&
                (position < CACHE_LENGTH) && (head < NUM_KV_HEADS) && (dim < HEAD_DIM)) begin
                if (kind == 0) begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
                    if (full_chain_mode) begin
                        value = full_k_cache_mem[full_index];
                    end
                    else
`endif
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
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
                    if (full_chain_mode) begin
                        value = full_v_cache_mem[full_index];
                    end
                    else
`endif
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

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    function automatic logic [ADDR_WIDTH-1 : 0] full_chain_effective_input_hidden_base(
        input integer layer
    );
        begin
            full_chain_effective_input_hidden_base = full_actual_input_hidden_base[layer];
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] full_chain_effective_layer_output_base(
        input integer layer
    );
        begin
            full_chain_effective_layer_output_base = full_actual_layer_output_base[layer];
        end
    endfunction

    function automatic integer full_runtime_hidden_word_index(
        input logic [ADDR_WIDTH-1 : 0] addr
    );
        begin
            if (in_range(addr, full_chain_runtime_hidden_a_base_addr,
                         VEC1024 * MEM_DATA_BYTES)) begin
                full_runtime_hidden_word_index =
                    (addr - full_chain_runtime_hidden_a_base_addr) >> 2;
            end
            else begin
                full_runtime_hidden_word_index = VEC1024 +
                    ((addr - full_chain_runtime_hidden_b_base_addr) >> 2);
            end
        end
    endfunction

    function automatic integer full_chain_active_memory_layer;
        begin
            if (use_tail_mem) begin
                full_chain_active_memory_layer = full_chain_layer_count - 1;
            end
            else if (active_layer_index < full_chain_layer_count) begin
                full_chain_active_memory_layer = active_layer_index;
            end
            else begin
                full_chain_active_memory_layer = full_chain_layer_count - 1;
            end
        end
    endfunction

    function automatic logic full_chain_memory_hit(input logic [ADDR_WIDTH-1 : 0] addr);
        integer layer;
        begin
            layer = full_chain_active_memory_layer();
            full_chain_memory_hit = 1'b0;
            if ((layer >= 0) && (layer < full_chain_layer_count)) begin
                full_chain_memory_hit =
                    in_range(addr, full_chain_effective_input_hidden_base(layer),
                             VEC1024 * MEM_DATA_BYTES) ||
                    (runtime_context_mode &&
                     (in_range(addr, full_chain_runtime_hidden_a_base_addr,
                               VEC1024 * MEM_DATA_BYTES) ||
                      in_range(addr, full_chain_runtime_hidden_b_base_addr,
                               VEC1024 * MEM_DATA_BYTES) ||
                      in_range(addr, full_chain_runtime_rope_cos_base_addr,
                               ROPE_TABLE_WORDS * MEM_DATA_BYTES))) ||
                    in_range(addr, full_qkv_qmap_base[layer], QKV_IMAGE_BYTES) ||
                    in_range(addr, full_input_norm_qmap_base[layer], INPUT_NORM_IMAGE_BYTES) ||
                    in_range(addr, full_frontend_qmap_base[layer], FRONT_IMAGE_BYTES) ||
                    in_range(addr, full_score_qmap_base[layer], SCORE_IMAGE_BYTES) ||
                    in_range(addr, full_oproj_qmap_base[layer], OPROJ_IMAGE_BYTES) ||
                    in_range(addr, full_post_qmap_base[layer], POST_IMAGE_BYTES) ||
                    in_range(addr, full_gate_qmap_base[layer], GATE_IMAGE_BYTES) ||
                    in_range(addr, full_silu_qmap_base[layer], SILU_IMAGE_BYTES) ||
                    in_range(addr, full_down_qmap_base[layer], DOWN_IMAGE_BYTES) ||
                    in_range(addr, full_residual_qmap_base[layer], RESIDUAL_IMAGE_BYTES) ||
                    is_cache_addr(addr) ||
                    in_range(addr, full_oproj_weight_base[layer], OPROJ_WEIGHT_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_oproj_scale_base[layer], OPROJ_SCALE_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_gate_weight_base[layer], GATE_WEIGHT_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_gate_scale_base[layer], GATE_SCALE_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_up_weight_base[layer], GATE_WEIGHT_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_up_scale_base[layer], GATE_SCALE_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_down_weight_base[layer], DOWN_WEIGHT_WORDS * MEM_DATA_BYTES) ||
                    in_range(addr, full_down_scale_base[layer], DOWN_SCALE_WORDS * MEM_DATA_BYTES);
            end
        end
    endfunction

    function automatic logic [31 : 0] full_chain_memory_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer layer;
        integer source_layer;
        integer index;
        integer source_offset;
        begin
            layer = full_chain_active_memory_layer();
            full_chain_memory_word = 32'hBADF_C001;
            if (runtime_context_mode &&
                (in_range(addr, full_chain_runtime_hidden_a_base_addr,
                          VEC1024 * MEM_DATA_BYTES) ||
                 in_range(addr, full_chain_runtime_hidden_b_base_addr,
                          VEC1024 * MEM_DATA_BYTES))) begin
                index = full_runtime_hidden_word_index(addr);
                full_chain_memory_word = full_runtime_hidden_mem[index];
            end
            else if (runtime_context_mode &&
                     in_range(addr, full_chain_runtime_rope_cos_base_addr,
                              ROPE_TABLE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - full_chain_runtime_rope_cos_base_addr) >> 2;
                full_chain_memory_word = full_runtime_rope_mem[index];
            end
            else if (in_range(addr, full_input_hidden_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                source_layer = (layer == 0) ? 0 : (layer - 1);
                source_offset = (full_input_hidden_base[layer] - full_residual_qmap_base[source_layer]) >> 2;
                index = (source_layer * RESIDUAL_WORDS) + source_offset +
                    ((addr - full_input_hidden_base[layer]) >> 2);
                full_chain_memory_word = full_residual_qmap[index];
            end
            else if (in_range(addr, full_qkv_qmap_base[layer], QKV_IMAGE_BYTES)) begin
                index = (layer * QKV_WORDS) + ((addr - full_qkv_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_qkv_qmap[index];
            end
            else if (in_range(addr, full_input_norm_qmap_base[layer], INPUT_NORM_IMAGE_BYTES)) begin
                index = (layer * INPUT_NORM_WORDS) + ((addr - full_input_norm_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_input_norm_qmap[index];
            end
            else if (in_range(addr, full_frontend_qmap_base[layer], FRONT_IMAGE_BYTES)) begin
                index = (layer * FRONT_WORDS) + ((addr - full_frontend_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_frontend_qmap[index];
            end
            else if (in_range(addr, full_score_qmap_base[layer], SCORE_IMAGE_BYTES)) begin
                index = (layer * SCORE_WORDS) + ((addr - full_score_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_score_qmap[index];
            end
            else if (in_range(addr, full_oproj_qmap_base[layer], OPROJ_IMAGE_BYTES)) begin
                index = (layer * OPROJ_WORDS) + ((addr - full_oproj_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_oproj_qmap[index];
            end
            else if (in_range(addr, full_post_qmap_base[layer], POST_IMAGE_BYTES)) begin
                index = (layer * POST_WORDS) + ((addr - full_post_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_post_qmap[index];
            end
            else if (in_range(addr, full_gate_qmap_base[layer], GATE_IMAGE_BYTES)) begin
                index = (layer * GATE_WORDS) + ((addr - full_gate_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_gate_qmap[index];
            end
            else if (in_range(addr, full_silu_qmap_base[layer], SILU_IMAGE_BYTES)) begin
                index = (layer * SILU_WORDS) + ((addr - full_silu_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_silu_qmap[index];
            end
            else if (in_range(addr, full_down_qmap_base[layer], DOWN_IMAGE_BYTES)) begin
                index = (layer * DOWN_WORDS) + ((addr - full_down_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_down_qmap[index];
            end
            else if (in_range(addr, full_residual_qmap_base[layer], RESIDUAL_IMAGE_BYTES)) begin
                index = (layer * RESIDUAL_WORDS) + ((addr - full_residual_qmap_base[layer]) >> 2);
                full_chain_memory_word = full_residual_qmap[index];
            end
            else if (is_cache_addr(addr)) begin
                full_chain_memory_word = cache_word(addr);
            end
            else if (in_range(addr, full_oproj_weight_base[layer], OPROJ_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * OPROJ_WEIGHT_WORDS) + ((addr - full_oproj_weight_base[layer]) >> 2);
                full_chain_memory_word = full_oproj_weight_mem[index];
            end
            else if (in_range(addr, full_oproj_scale_base[layer], OPROJ_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * OPROJ_SCALE_WORDS) + ((addr - full_oproj_scale_base[layer]) >> 2);
                full_chain_memory_word = full_oproj_scale_mem[index];
            end
            else if (in_range(addr, full_gate_weight_base[layer], GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * GATE_WEIGHT_WORDS) + ((addr - full_gate_weight_base[layer]) >> 2);
                full_chain_memory_word = full_gate_weight_mem[index];
            end
            else if (in_range(addr, full_gate_scale_base[layer], GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * GATE_SCALE_WORDS) + ((addr - full_gate_scale_base[layer]) >> 2);
                full_chain_memory_word = full_gate_scale_mem[index];
            end
            else if (in_range(addr, full_up_weight_base[layer], GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * GATE_WEIGHT_WORDS) + ((addr - full_up_weight_base[layer]) >> 2);
                full_chain_memory_word = full_up_weight_mem[index];
            end
            else if (in_range(addr, full_up_scale_base[layer], GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * GATE_SCALE_WORDS) + ((addr - full_up_scale_base[layer]) >> 2);
                full_chain_memory_word = full_up_scale_mem[index];
            end
            else if (in_range(addr, full_down_weight_base[layer], DOWN_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * DOWN_WEIGHT_WORDS) + ((addr - full_down_weight_base[layer]) >> 2);
                full_chain_memory_word = full_down_weight_mem[index];
            end
            else if (in_range(addr, full_down_scale_base[layer], DOWN_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (layer * DOWN_SCALE_WORDS) + ((addr - full_down_scale_base[layer]) >> 2);
                full_chain_memory_word = full_down_scale_mem[index];
            end
        end
    endfunction
`endif

    function automatic logic [31 : 0] memory_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer index;
        begin
            memory_word = 32'hBAD0_BAD0;
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            if (use_tail_mem && in_range(addr, tail_qmap_base_addr, TAIL_QMAP_IMAGE_BYTES)) begin
                index = (addr - tail_qmap_base_addr) >> 2;
                memory_word = tail_qmap[index];
            end
            else if (use_tail_mem && in_range(addr, tail_weight_base_addr, TAIL_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - tail_weight_base_addr) >> 2;
                memory_word = tail_weight_mem[index];
            end
            else if (use_tail_mem && in_range(addr, tail_scale_base_addr, TAIL_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - tail_scale_base_addr) >> 2;
                memory_word = tail_scale_mem[index];
            end
            else
`endif
`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
            if (persistent_two_token_mode &&
                in_range(addr, EMBEDDING_WEIGHT_BASE_ADDR,
                         TAIL_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - EMBEDDING_WEIGHT_BASE_ADDR) >> 2;
                memory_word = tail_weight_mem[index];
            end
            else if (persistent_two_token_mode &&
                     in_range(addr, EMBEDDING_SCALE_BASE_ADDR,
                              TAIL_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - EMBEDDING_SCALE_BASE_ADDR) >> 2;
                memory_word = tail_scale_mem[index];
            end
            else
`endif
            if (embedding_true3_mode &&
                in_range(addr, embedding_weight_row_base, EMBEDDING_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - embedding_weight_row_base) >> 2;
                memory_word = embedding_weight_mem[index];
            end
            else if (embedding_true3_mode &&
                     in_range(addr, embedding_scale_row_base, EMBEDDING_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - embedding_scale_row_base) >> 2;
                memory_word = embedding_scale_mem[index];
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            else if (full_chain_mode && full_chain_memory_hit(addr)) begin
                memory_word = full_chain_memory_word(addr);
            end
`endif
            else if (in_range(addr, layer0_qkv_packet_base_addr, QKV_IMAGE_BYTES)) begin
                index = (addr - layer0_qkv_packet_base_addr) >> 2;
                memory_word = qkv_qmap[index];
            end
            else if (in_range(addr, `QMAP_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - `QMAP_INPUT_NORM_BASE_ADDR) >> 2;
                memory_word = input_norm_qmap[index];
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
            else if (in_range(addr, QMAP_LAYER1_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_INPUT_NORM_BASE_ADDR) >> 2;
                memory_word = input_norm_qmap_l1[index];
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
            else if (in_range(addr, QMAP_LAYER2_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_INPUT_NORM_BASE_ADDR) >> 2;
                memory_word = input_norm_qmap_l2[index];
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
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            else if (in_range(addr, tail_qmap_base_addr, TAIL_QMAP_IMAGE_BYTES)) begin
                index = (addr - tail_qmap_base_addr) >> 2;
                memory_word = tail_qmap[index];
            end
            else if (in_range(addr, tail_weight_base_addr, TAIL_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - tail_weight_base_addr) >> 2;
                memory_word = tail_weight_mem[index];
            end
            else if (in_range(addr, tail_scale_base_addr, TAIL_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - tail_scale_base_addr) >> 2;
                memory_word = tail_scale_mem[index];
            end
`endif
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

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    task write_full_chain_qmap_word;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        integer layer;
        integer index;
        begin
            layer = active_write_layer;
            if ((layer < 0) || (layer >= full_chain_layer_count)) begin
                fail_once("FAIL: full-chain write resolved to an invalid layer");
            end
            else if (runtime_context_mode &&
                     (in_range(addr, full_chain_runtime_hidden_a_base_addr,
                               VEC1024 * MEM_DATA_BYTES) ||
                      in_range(addr, full_chain_runtime_hidden_b_base_addr,
                               VEC1024 * MEM_DATA_BYTES))) begin
                index = full_runtime_hidden_word_index(addr);
                full_runtime_hidden_mem[index] = data;
            end
            else if (in_range(addr, full_qkv_qmap_base[layer], QKV_IMAGE_BYTES)) begin
                index = (layer * QKV_WORDS) + ((addr - full_qkv_qmap_base[layer]) >> 2);
                full_qkv_qmap[index] = data;
            end
            else if (in_range(addr, full_input_norm_qmap_base[layer], INPUT_NORM_IMAGE_BYTES)) begin
                index = (layer * INPUT_NORM_WORDS) + ((addr - full_input_norm_qmap_base[layer]) >> 2);
                full_input_norm_qmap[index] = data;
            end
            else if (in_range(addr, full_frontend_qmap_base[layer], FRONT_IMAGE_BYTES)) begin
                index = (layer * FRONT_WORDS) + ((addr - full_frontend_qmap_base[layer]) >> 2);
                full_frontend_qmap[index] = data;
            end
            else if (in_range(addr, full_score_qmap_base[layer], SCORE_IMAGE_BYTES)) begin
                index = (layer * SCORE_WORDS) + ((addr - full_score_qmap_base[layer]) >> 2);
                full_score_qmap[index] = data;
            end
            else if (in_range(addr, full_oproj_qmap_base[layer], OPROJ_IMAGE_BYTES)) begin
                index = (layer * OPROJ_WORDS) + ((addr - full_oproj_qmap_base[layer]) >> 2);
                full_oproj_qmap[index] = data;
            end
            else if (in_range(addr, full_post_qmap_base[layer], POST_IMAGE_BYTES)) begin
                index = (layer * POST_WORDS) + ((addr - full_post_qmap_base[layer]) >> 2);
                full_post_qmap[index] = data;
            end
            else if (in_range(addr, full_gate_qmap_base[layer], GATE_IMAGE_BYTES)) begin
                index = (layer * GATE_WORDS) + ((addr - full_gate_qmap_base[layer]) >> 2);
                full_gate_qmap[index] = data;
            end
            else if (in_range(addr, full_silu_qmap_base[layer], SILU_IMAGE_BYTES)) begin
                index = (layer * SILU_WORDS) + ((addr - full_silu_qmap_base[layer]) >> 2);
                full_silu_qmap[index] = data;
            end
            else if (in_range(addr, full_down_qmap_base[layer], DOWN_IMAGE_BYTES)) begin
                index = (layer * DOWN_WORDS) + ((addr - full_down_qmap_base[layer]) >> 2);
                full_down_qmap[index] = data;
            end
            else if (in_range(addr, full_residual_qmap_base[layer], RESIDUAL_IMAGE_BYTES)) begin
                index = (layer * RESIDUAL_WORDS) + ((addr - full_residual_qmap_base[layer]) >> 2);
                full_residual_qmap[index] = data;
            end
            else begin
                fail_once("FAIL: write to unknown full-chain QMAP address");
            end
        end
    endtask
`endif

    task write_qmap_word;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        integer index;
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode && !use_tail_mem) begin
                write_full_chain_qmap_word(addr, data);
            end
            else begin
`endif
            if (in_range(addr, layer0_qkv_packet_base_addr, QKV_IMAGE_BYTES)) begin
                index = (addr - layer0_qkv_packet_base_addr) >> 2;
                qkv_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - `QMAP_INPUT_NORM_BASE_ADDR) >> 2;
                input_norm_qmap[index] = data;
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
            else if (in_range(addr, QMAP_LAYER1_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER1_INPUT_NORM_BASE_ADDR) >> 2;
                input_norm_qmap_l1[index] = data;
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
            else if (in_range(addr, QMAP_LAYER2_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - QMAP_LAYER2_INPUT_NORM_BASE_ADDR) >> 2;
                input_norm_qmap_l2[index] = data;
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
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            else if (in_range(addr, tail_qmap_base_addr, TAIL_QMAP_IMAGE_BYTES)) begin
                index = (addr - tail_qmap_base_addr) >> 2;
                tail_qmap[index] = data;
            end
`endif
            else begin
                fail_once("FAIL: write to unknown QMAP address");
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
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
        integer full_index;
        logic [ADDR_WIDTH-1 : 0] selected_base;
        begin
            selected_base = full_chain_mode ? full_chain_kv_cache_base_addr : CACHE_BASE_ADDR;
            element_index = (addr - selected_base) >> 2;
            layer = element_index / CACHE_WORDS_FULL;
            layer_word_index = element_index % CACHE_WORDS_FULL;
            kind = layer_word_index / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            rem0 = layer_word_index % (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            head = rem0 / (MAX_CONTEXT * HEAD_DIM);
            rem1 = rem0 % (MAX_CONTEXT * HEAD_DIM);
            position = rem1 / HEAD_DIM;
            dim = rem1 % HEAD_DIM;
            cache_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;
            full_index = (layer * CACHE_LENGTH * KV_COUNT) + cache_index;

            if ((layer < MODELED_CACHE_LAYERS) &&
                (position < CACHE_LENGTH) && (head < NUM_KV_HEADS) && (dim < HEAD_DIM)) begin
                if (kind == 0) begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
                    if (full_chain_mode) begin
                        full_k_cache_mem[full_index] = data[IN_WIDTH-1 : 0];
                    end
                    else
`endif
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
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
                    if (full_chain_mode) begin
                        full_v_cache_mem[full_index] = data[IN_WIDTH-1 : 0];
                    end
                    else
`endif
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

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    function automatic integer classify_full_chain_write_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        integer layer;
        begin
            layer = active_layer_index;
            if ((layer < 0) || (layer >= full_chain_layer_count)) begin
                layer = 0;
            end
            classify_full_chain_write_addr = 0;
            if (embedding_true3_mode && !embedding_written &&
                in_range(addr, full_chain_embedding_output_base_addr,
                         VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_EMBEDDING;
            end
            else if (in_range(addr, full_input_norm_output_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_INPUT_NORM;
            end
            else if (in_range(addr, full_q_base[layer], VEC2048 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_QKV_Q;
            end
            else if (in_range(addr, full_k_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_QKV_K;
            end
            else if (in_range(addr, full_v_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_QKV_V;
            end
            else if (is_cache_addr(addr)) begin
                classify_full_chain_write_addr = WRITE_CACHE;
            end
            else if (in_range(addr, full_q_rope_base[layer], VEC2048 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_Q_ROPE;
            end
            else if (in_range(addr, full_attn_out_base[layer], VEC2048 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_ATTN_OUT;
            end
            else if (in_range(addr, full_o_proj_output_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_O_PROJ;
            end
            else if (in_range(addr, full_post_hidden_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_POST_HIDDEN;
            end
            else if (in_range(addr, full_post_norm_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_POST_NORM;
            end
            else if (in_range(addr, full_gate_output_base[layer], VEC3072 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_GATE;
            end
            else if (in_range(addr, full_up_output_base[layer], VEC3072 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_UP;
            end
            else if (in_range(addr, full_silu_output_base[layer], VEC3072 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_SILU_HIDDEN;
            end
            else if (in_range(addr, full_down_output_base[layer], VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_DOWN;
            end
            else if (in_range(addr, full_chain_effective_layer_output_base(layer),
                              VEC1024 * MEM_DATA_BYTES)) begin
                classify_full_chain_write_addr = WRITE_LAYER;
            end
        end
    endfunction
`endif

    function automatic integer classify_write_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                classify_write_addr = classify_full_chain_write_addr(addr);
            end
            else begin
`endif
            classify_write_addr = 0;
            if (in_range(addr, input_norm_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_INPUT_NORM;
            end
            else if (in_range(addr, qkv_q_base, VEC2048 * MEM_DATA_BYTES)) begin
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
            else if (embedding_true3_mode && !embedding_written &&
                     in_range(addr, layer_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_EMBEDDING;
            end
            else if (in_range(addr, layer_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_LAYER;
            end
            else if (in_range(addr, input_norm_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_INPUT_NORM;
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
            else if (in_range(addr, input_norm_output_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                classify_write_addr = WRITE_INPUT_NORM;
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
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic integer classify_completed_write_addr(
        input logic [ADDR_WIDTH-1 : 0] addr
    );
        begin
            classify_completed_write_addr = classify_write_addr(addr);
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            // The legacy embedding output aliases Layer 0's descriptor output.
            // At the final embedding beat embedding_written has just become true,
            // so use the not-yet-observed response to preserve request identity.
            if (full_chain_mode && embedding_true3_mode &&
                !embedding_response_seen &&
                (embedding_write_accept_count == VEC1024) &&
                in_range(addr, full_chain_embedding_output_base_addr,
                         VEC1024 * MEM_DATA_BYTES)) begin
                classify_completed_write_addr = WRITE_EMBEDDING;
            end
`endif
        end
    endfunction

    function automatic integer cache_layer_for_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        integer element_index;
        logic [ADDR_WIDTH-1 : 0] selected_base;
        begin
            selected_base = full_chain_mode ? full_chain_kv_cache_base_addr : CACHE_BASE_ADDR;
            element_index = (addr - selected_base) >> 2;
            cache_layer_for_addr = element_index / CACHE_WORDS_FULL;
        end
    endfunction

    function automatic integer write_layer_for_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                if (is_cache_addr(addr)) begin
                    write_layer_for_addr = cache_layer_for_addr(addr);
                end
                else begin
                    write_layer_for_addr = active_layer_index;
                end
            end
            else begin
`endif
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
                     in_range(addr, layer_output_base_l1, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, input_norm_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
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
                     in_range(addr, layer_output_base_l2, VEC1024 * MEM_DATA_BYTES) ||
                     in_range(addr, input_norm_output_base_l2, VEC1024 * MEM_DATA_BYTES)) begin
                write_layer_for_addr = 2;
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] expected_write_base(
        input integer kind,
        input integer layer
    );
        begin
            expected_write_base = '0;
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                case (kind)
                    WRITE_QKV_Q: expected_write_base = full_q_base[layer];
                    WRITE_QKV_K: expected_write_base = full_k_base[layer];
                    WRITE_QKV_V: expected_write_base = full_v_base[layer];
                    WRITE_Q_ROPE: expected_write_base = full_q_rope_base[layer];
                    WRITE_ATTN_OUT: expected_write_base = full_attn_out_base[layer];
                    WRITE_O_PROJ: expected_write_base = full_o_proj_output_base[layer];
                    WRITE_POST_HIDDEN: expected_write_base = full_post_hidden_base[layer];
                    WRITE_POST_NORM: expected_write_base = full_post_norm_base[layer];
                    WRITE_GATE: expected_write_base = full_gate_output_base[layer];
                    WRITE_UP: expected_write_base = full_up_output_base[layer];
                    WRITE_SILU_HIDDEN: expected_write_base = full_silu_output_base[layer];
                    WRITE_DOWN: expected_write_base = full_down_output_base[layer];
                    WRITE_LAYER: expected_write_base =
                        full_chain_effective_layer_output_base(layer);
                    WRITE_INPUT_NORM: expected_write_base = full_input_norm_output_base[layer];
                    WRITE_EMBEDDING: expected_write_base =
                        full_chain_embedding_output_base_addr;
                    default: expected_write_base = '0;
                endcase
            end
            else begin
`endif
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
                WRITE_INPUT_NORM: expected_write_base = (layer == 0) ? input_norm_output_base : ((layer == 1) ? input_norm_output_base_l1 : input_norm_output_base_l2);
                WRITE_EMBEDDING: expected_write_base = layer_output_base;
                default: expected_write_base = '0;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
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
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_qkv_word = full_expected_qkv[(layer * (VEC2048 + (2 * VEC1024))) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_qkv_word = use_input_norm_qkv_expected ? expected_qkv_input_norm[index] : expected_qkv[index];
                1: expected_qkv_word = expected_qkv_l1[index];
                2: expected_qkv_word = expected_qkv_l2[index];
                default: expected_qkv_word = 32'hDEAD_E000;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_input_norm_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_input_norm_word = full_expected_input_norm[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_input_norm_word = expected_input_norm[index];
                1: expected_input_norm_word = expected_input_norm_l1[index];
                2: expected_input_norm_word = expected_input_norm_l2[index];
                default: expected_input_norm_word = 32'hDEAD_E0D0;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [63 : 0] expected_cache_addr_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_cache_addr_word = full_expected_cache_addr[(layer * TOTAL_CACHE_WRITES) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_cache_addr_word = expected_cache_addr[index];
                1: expected_cache_addr_word = expected_cache_addr_l1[index];
                2: expected_cache_addr_word = expected_cache_addr_l2[index];
                default: expected_cache_addr_word = 64'hDEAD_E001_DEAD_E001;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_cache_data_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_cache_data_word = full_expected_cache_data[(layer * TOTAL_CACHE_WRITES) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_cache_data_word = expected_cache_data[index];
                1: expected_cache_data_word = expected_cache_data_l1[index];
                2: expected_cache_data_word = expected_cache_data_l2[index];
                default: expected_cache_data_word = 32'hDEAD_E002;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_q_rope_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_q_rope_word = full_expected_q_rope[(layer * VEC2048) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_q_rope_word = expected_q_rope[index];
                1: expected_q_rope_word = expected_q_rope_l1[index];
                2: expected_q_rope_word = expected_q_rope_l2[index];
                default: expected_q_rope_word = 32'hDEAD_E003;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_attn_out_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_attn_out_word = full_expected_attn_out[(layer * VEC2048) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_attn_out_word = expected_attn_out[index];
                1: expected_attn_out_word = expected_attn_out_l1[index];
                2: expected_attn_out_word = expected_attn_out_l2[index];
                default: expected_attn_out_word = 32'hDEAD_E004;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_o_proj_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_o_proj_word = full_expected_o_proj[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_o_proj_word = expected_o_proj[index];
                1: expected_o_proj_word = expected_o_proj_l1[index];
                2: expected_o_proj_word = expected_o_proj_l2[index];
                default: expected_o_proj_word = 32'hDEAD_E005;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_post_hidden_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_post_hidden_word = full_expected_post_hidden[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_post_hidden_word = expected_post_hidden[index];
                1: expected_post_hidden_word = expected_post_hidden_l1[index];
                2: expected_post_hidden_word = expected_post_hidden_l2[index];
                default: expected_post_hidden_word = 32'hDEAD_E006;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_post_norm_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_post_norm_word = full_expected_post_norm[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_post_norm_word = expected_post_norm[index];
                1: expected_post_norm_word = expected_post_norm_l1[index];
                2: expected_post_norm_word = expected_post_norm_l2[index];
                default: expected_post_norm_word = 32'hDEAD_E007;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_gate_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_gate_word = full_expected_gate[(layer * VEC3072) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_gate_word = expected_gate[index];
                1: expected_gate_word = expected_gate_l1[index];
                2: expected_gate_word = expected_gate_l2[index];
                default: expected_gate_word = 32'hDEAD_E008;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_up_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_up_word = full_expected_up[(layer * VEC3072) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_up_word = expected_up[index];
                1: expected_up_word = expected_up_l1[index];
                2: expected_up_word = expected_up_l2[index];
                default: expected_up_word = 32'hDEAD_E009;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_silu_hidden_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_silu_hidden_word = full_expected_silu_hidden[(layer * VEC3072) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_silu_hidden_word = expected_silu_hidden[index];
                1: expected_silu_hidden_word = expected_silu_hidden_l1[index];
                2: expected_silu_hidden_word = expected_silu_hidden_l2[index];
                default: expected_silu_hidden_word = 32'hDEAD_E00A;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_down_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_down_word = full_expected_down[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_down_word = expected_down[index];
                1: expected_down_word = expected_down_l1[index];
                2: expected_down_word = expected_down_l2[index];
                default: expected_down_word = 32'hDEAD_E00B;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    function automatic logic [31 : 0] expected_layer_word(input integer layer, input integer index);
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                expected_layer_word = full_expected_layer[(layer * VEC1024) + index];
            end
            else begin
`endif
            case (layer)
                0: expected_layer_word = expected_layer[index];
                1: expected_layer_word = expected_layer_l1[index];
                2: expected_layer_word = expected_layer_l2[index];
                default: expected_layer_word = 32'hDEAD_E00C;
            endcase
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endfunction

    task tail_fail(input string message);
        begin
            tail_mismatch_count = tail_mismatch_count + 1;
            fail_once(message);
        end
    endtask

    task calculate_tail_expected;
        integer i;
        begin
            if (persistent_two_token_mode) begin
                tail_expected_token = tail_expected_words[0];
                tail_expected_score_q26 = $signed({
                    tail_expected_words[2],
                    tail_expected_words[1]
                });
            end
            else begin
                tail_expected_token = tail_scan_base_token_mem[0];
                tail_expected_score_q26 = tail_expected_logits[0];
                for (i = 1; i < TAIL_SCAN_ROWS; i = i + 1) begin
                    if ($signed(tail_expected_logits[i]) > tail_expected_score_q26) begin
                        tail_expected_score_q26 = tail_expected_logits[i];
                        tail_expected_token = tail_scan_base_token_mem[0] + i[31 : 0];
                    end
                end
            end
        end
    endtask

    task clear_tail_scoreboard;
        integer i;
        integer norm_index;
        integer output_index;
        begin
            tail_mismatch_count = 0;
            tail_write_mismatch_count = 0;
            tail_read_req_count = 0;
            tail_hidden_req_count = 0;
            tail_gamma_req_count = 0;
            tail_norm_read_req_count = 0;
            tail_weight_req_count = 0;
            tail_scale_req_count = 0;
            tail_lm_req_count = 0;
            tail_norm_write_word_count = 0;
            tail_output_write_word_count = 0;
            tail_done_seen_count = 0;
            tail_first_hidden_read_cycle = -1;
            tail_first_weight_read_cycle = -1;
            tail_first_output_write_cycle = -1;
            tail_scheduler_done_cycle = -1;
            tail_start_cycle = -1;
            tail_done_cycle = -1;
            calculate_tail_expected();

            norm_index = (tail_norm_output_base - tail_qmap_base_addr) >> 2;
            output_index = (tail_output_base - tail_qmap_base_addr) >> 2;
            for (i = 0; i < TAIL_INPUT_SIZE; i = i + 1) begin
                tail_qmap[norm_index + i] = 32'hA5A5_0000 | i[15 : 0];
            end
            tail_qmap[output_index + 0] = 32'hFFFF_FFFF;
            tail_qmap[output_index + 1] = 32'hFFFF_FFFF;
            tail_qmap[output_index + 2] = 32'hFFFF_FFFF;

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if ((full_chain_mode &&
                 (tail_final_hidden_base !==
                  full_actual_layer_output_base[full_chain_layer_count - 1])) ||
                (!full_chain_mode && (tail_final_hidden_base !== layer_output_base_l2))) begin
`else
            if (tail_final_hidden_base !== layer_output_base_l2) begin
`endif
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
                tail_fail("FAIL: top expected final hidden source does not point to final scheduler output");
`else
                tail_fail("FAIL: tail final hidden descriptor was not patched to final scheduler output");
`endif
            end
            if (tail_desc_base(TAIL_SLOT_WEIGHT) !== tail_weight_base_addr) begin
                tail_fail("FAIL: tail weight descriptor base does not match LM-head weight file base");
            end
            if (tail_desc_base(TAIL_SLOT_SCALE) !== tail_scale_base_addr) begin
                tail_fail("FAIL: tail scale descriptor base does not match LM-head scale file base");
            end
        end
    endtask

    task check_tail_read_request;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer len_bytes;
        integer tile_id;
        integer row_id;
        integer burst_id;
        integer tile_token;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        integer expected_len;
        begin
            tail_read_req_count = tail_read_req_count + 1;
            if (in_range(addr, tail_final_hidden_base, TAIL_INPUT_SIZE * MEM_DATA_BYTES)) begin
                tail_hidden_req_count = tail_hidden_req_count + 1;
                if (!layer_written) begin
                    tail_fail("FAIL: final-token tail read Layer2 output before scheduler write completed");
                end
                if (embedding_true3_mode && !layer_response_seen[2]) begin
                    tail_fail("FAIL: final-token tail read Layer2 output before write response");
                end
                if (tail_scheduler_done_cycle < 0) begin
                    tail_fail("FAIL: final-token tail read hidden before scheduler done was recorded");
                end
                if (tail_first_hidden_read_cycle < 0) begin
                    tail_first_hidden_read_cycle = cycle_count;
                end
            end
            else if (in_range(addr, tail_qmap_base_addr, TAIL_QMAP_IMAGE_BYTES)) begin
                if (in_range(addr, tail_final_gamma_base, TAIL_INPUT_SIZE * MEM_DATA_BYTES)) begin
                    tail_gamma_req_count = tail_gamma_req_count + 1;
                end
                if (in_range(addr, tail_norm_output_base, TAIL_INPUT_SIZE * MEM_DATA_BYTES)) begin
                    tail_norm_read_req_count = tail_norm_read_req_count + 1;
                end
            end
            else if (in_range(addr, tail_weight_base_addr, TAIL_WEIGHT_WORDS * MEM_DATA_BYTES) ||
                     in_range(addr, tail_scale_base_addr, TAIL_SCALE_WORDS * MEM_DATA_BYTES)) begin
                row_id = tail_lm_req_count / 2;
                tile_id = row_id / TAIL_TILE_ROWS;
                burst_id = tail_lm_req_count % 2;
                tile_token = tail_scan_base_token_mem[0] + row_id;

                if (tile_id >= TAIL_MAX_TILES) begin
                    tail_fail("FAIL: final-token tail issued extra LM-head tile request");
                end
                if (burst_id == 0) begin
                    expected_addr = tail_weight_base_addr +
                                    (tile_token * TAIL_WEIGHT_ROW_BYTES);
                    expected_len = TAIL_WEIGHT_ROW_BYTES;
                    tail_weight_req_count = tail_weight_req_count + 1;
                    if (tail_first_weight_read_cycle < 0) begin
                        tail_first_weight_read_cycle = cycle_count;
                    end
                end
                else begin
                    expected_addr = tail_scale_base_addr + (tile_token * TAIL_SCALE_ROW_BYTES);
                    expected_len = TAIL_SCALE_ROW_BYTES;
                    tail_scale_req_count = tail_scale_req_count + 1;
                end

                if ((addr !== expected_addr) || (len_bytes != expected_len)) begin
                    tail_fail("FAIL: final-token tail LM-head read request mismatch");
                end
                tail_lm_req_count = tail_lm_req_count + 1;
            end
            else begin
                tail_fail("FAIL: final-token tail read outside known regions");
            end
        end
    endtask

    function automatic integer classify_tail_write_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            classify_tail_write_addr = TAIL_WRITE_NONE;
            if (in_range(addr, tail_norm_output_base, TAIL_INPUT_SIZE * MEM_DATA_BYTES)) begin
                classify_tail_write_addr = TAIL_WRITE_NORM;
            end
            else if (in_range(addr, tail_output_base, 3 * MEM_DATA_BYTES)) begin
                classify_tail_write_addr = TAIL_WRITE_OUTPUT;
            end
        end
    endfunction

    task check_tail_write_request;
        input integer kind;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer words;
        begin
            case (kind)
                TAIL_WRITE_NORM: begin
                    if ((addr !== tail_norm_output_base) || (words != TAIL_INPUT_SIZE)) begin
                        tail_fail("FAIL: final-token tail norm write request mismatch");
                    end
                end
                TAIL_WRITE_OUTPUT: begin
                    if ((addr !== tail_output_base) || (words != 3)) begin
                        tail_fail("FAIL: final-token tail output write request mismatch");
                    end
                    if (tail_first_output_write_cycle < 0) begin
                        tail_first_output_write_cycle = cycle_count;
                    end
                end
                default: begin
                    tail_fail("FAIL: final-token tail write request outside known targets");
                end
            endcase
        end
    endtask

    task check_tail_write_word;
        input integer kind;
        input integer index;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        input logic is_last;
        logic [31 : 0] expected;
        begin
            expected = 32'h0;
            case (kind)
                TAIL_WRITE_NORM: begin
                    expected = tail_norm_expected_word(index);
                    if ((index >= TAIL_INPUT_SIZE) || (data !== expected)) begin
                        tail_fail("FAIL: final-token tail norm write data mismatch");
                        tail_write_mismatch_count = tail_write_mismatch_count + 1;
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    if (is_last !== (index == (TAIL_INPUT_SIZE - 1))) begin
                        tail_fail("FAIL: final-token tail norm write last mismatch");
                    end
                    write_qmap_word(addr, data);
                    tail_norm_write_word_count = tail_norm_write_word_count + 1;
                end
                TAIL_WRITE_OUTPUT: begin
                    expected = tail_expected_words[index];
                    if ((index >= 3) || (data !== expected)) begin
                        tail_fail("FAIL: final-token tail output write data mismatch");
                        tail_write_mismatch_count = tail_write_mismatch_count + 1;
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    if (is_last !== (index == 2)) begin
                        tail_fail("FAIL: final-token tail output write last mismatch");
                    end
                    write_qmap_word(addr, data);
                    tail_output_write_word_count = tail_output_write_word_count + 1;
                end
                default: begin
                    tail_fail("FAIL: final-token tail write data for unknown target");
                end
            endcase
        end
    endtask

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
                WRITE_EMBEDDING: begin
                    base_addr = expected_write_base(kind, 0);
                    if ((addr !== base_addr) || (words != VEC1024) ||
                        (embedding_write_accept_count != 0)) begin
                        fail_once("FAIL: embedding hidden write request mismatch");
                    end
                    else begin
                        embedding_written = 1'b0;
                        embedding_response_seen = 1'b0;
                    end
                end
                WRITE_INPUT_NORM: begin
                    base_addr = expected_write_base(kind, write_layer);
                    if ((addr !== base_addr) || (words != VEC1024)) begin
                        fail_once("FAIL: input RMSNorm output write request mismatch");
                    end
                    else begin
                        if (full_chain_mode || (write_layer == 0)) begin
                            input_norm_written = 1'b0;
                        end
                        else if (write_layer == 1) begin
                            input_norm_written_l1 = 1'b0;
                        end
                        else begin
                            input_norm_written_l2 = 1'b0;
                        end
                    end
                end
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
                        if (full_chain_mode && runtime_context_mode) begin
                            runtime_layer_output_write_req_count[write_layer] =
                                runtime_layer_output_write_req_count[write_layer] + 1;
                        end
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
                WRITE_EMBEDDING: begin
                    expected = embedding_expected[index];
                    if (data !== expected) begin
                        fail_once("FAIL: embedding hidden write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    if (is_last !== (index == (VEC1024 - 1))) begin
                        fail_once("FAIL: embedding hidden write last mismatch");
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    embedding_write_accept_count = embedding_write_accept_count + 1;
                    if (is_last) begin
                        embedding_written = 1'b1;
                    end
                end
                WRITE_INPUT_NORM: begin
                    expected = expected_input_norm_word(write_layer, index);
                    if (data !== expected) begin
                        fail_once("FAIL: input RMSNorm output write data mismatch");
                        write_mismatch_count = write_mismatch_count + 1;
                    end
                    write_qmap_word(addr, data);
                    track_diff(data, expected);
                    input_norm_write_accept_count = input_norm_write_accept_count + 1;
                    if (is_last) begin
                        if (full_chain_mode || (write_layer == 0)) begin
                            input_norm_written = 1'b1;
                        end
                        else if (write_layer == 1) begin
                            input_norm_written_l1 = 1'b1;
                        end
                        else begin
                            input_norm_written_l2 = 1'b1;
                        end
                    end
                end
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
                        if (full_chain_mode && runtime_context_mode) begin
                            runtime_cache_write_word_count[write_layer] =
                                runtime_cache_write_word_count[write_layer] + 1;
                        end
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
                    if (print_count < 64) begin
                        $display("FAIL: unknown write kind=%0d layer=%0d index=%0d addr=0x%016h",
                                 kind, write_layer, index, addr);
                    end
                    fail_once("FAIL: write data for unknown write kind");
                end
            endcase
        end
    endtask

`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
    task check_full_chain_read_ready;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer len_bytes;
        integer layer;
        integer cache_start_element;
        integer cache_end_element;
        integer cache_start_local;
        integer cache_end_local;
        integer cache_start_plane;
        integer cache_end_plane;
        integer cache_start_position;
        integer cache_end_position;
        logic [ADDR_WIDTH-1 : 0] request_end_addr;
        logic [ADDR_WIDTH-1 : 0] runtime_cos_row_base;
        logic [ADDR_WIDTH-1 : 0] runtime_sin_row_base;
        begin
            layer = active_layer_index;
            request_end_addr = addr + len_bytes - 1;
            runtime_cos_row_base = full_chain_runtime_rope_cos_base_addr +
                (full_chain_runtime_position * HEAD_DIM * MEM_DATA_BYTES);
            runtime_sin_row_base = full_chain_runtime_rope_sin_base_addr +
                (full_chain_runtime_position * HEAD_DIM * MEM_DATA_BYTES);
            if ((layer < 0) || (layer >= full_chain_layer_count)) begin
                fail_once("FAIL: full-chain read resolved to an invalid active layer");
            end
            else begin
                if (in_range(addr, full_chain_effective_input_hidden_base(layer),
                             VEC1024 * MEM_DATA_BYTES)) begin
                    if (runtime_context_mode) begin
                        runtime_hidden_read_req_count[layer] =
                            runtime_hidden_read_req_count[layer] + 1;
                    end
                    if (first_layer_hidden_read_cycle[layer] < 0) begin
                        first_layer_hidden_read_cycle[layer] = cycle_count;
                    end
                    if (layer == 0) begin
                        if (first_embedding_hidden_read_cycle < 0) begin
                            first_embedding_hidden_read_cycle = cycle_count;
                        end
                        if (!embedding_response_seen) begin
                            fail_once("FAIL: Layer 0 read embedding hidden before write response");
                        end
                    end
                    else begin
                        if ((layer == 1) && (first_layer1_hidden_read_cycle < 0)) begin
                            first_layer1_hidden_read_cycle = cycle_count;
                        end
                        if ((layer == 2) && (first_layer2_hidden_read_cycle < 0)) begin
                            first_layer2_hidden_read_cycle = cycle_count;
                        end
                        if (!layer_response_seen[layer - 1]) begin
                            fail_once("FAIL: full-chain layer input read before previous layer write response");
                        end
                    end
                end
                if (in_range(addr, full_input_norm_output_base[layer], VEC1024 * MEM_DATA_BYTES) &&
                    !input_norm_written) begin
                    fail_once("FAIL: full-chain input RMSNorm output read before producer write completed");
                end
                if (in_range(addr, full_q_base[layer], VEC2048 * MEM_DATA_BYTES) && !qkv_q_written) begin
                    fail_once("FAIL: full-chain QKV Q output read before producer write completed");
                end
                if (in_range(addr, full_k_base[layer], VEC1024 * MEM_DATA_BYTES) && !qkv_k_written) begin
                    fail_once("FAIL: full-chain QKV K output read before producer write completed");
                end
                if (in_range(addr, full_v_base[layer], VEC1024 * MEM_DATA_BYTES) && !qkv_v_written) begin
                    fail_once("FAIL: full-chain QKV V output read before producer write completed");
                end
                if (is_cache_addr(addr) && !cache_written && (active_stage_debug > 0)) begin
                    fail_once("FAIL: full-chain K/V cache read before frontend cache writes completed");
                end
                if (runtime_context_mode && is_cache_addr(addr)) begin
                    cache_start_element = (addr - full_chain_kv_cache_base_addr) >> 2;
                    cache_end_element = (request_end_addr - full_chain_kv_cache_base_addr) >> 2;
                    cache_start_local = cache_start_element % CACHE_WORDS_FULL;
                    cache_end_local = cache_end_element % CACHE_WORDS_FULL;
                    cache_start_plane = cache_start_local / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
                    cache_end_plane = cache_end_local / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
                    cache_start_position =
                        (cache_start_local % (MAX_CONTEXT * HEAD_DIM)) / HEAD_DIM;
                    cache_end_position =
                        (cache_end_local % (MAX_CONTEXT * HEAD_DIM)) / HEAD_DIM;
                    if ((cache_layer_for_addr(addr) != layer) ||
                        (cache_layer_for_addr(request_end_addr) != layer) ||
                        (cache_start_plane > 1) || (cache_end_plane > 1) ||
                        (cache_start_position > full_chain_runtime_position) ||
                        (cache_end_position > full_chain_runtime_position)) begin
                        fail_once("FAIL: runtime cache read escaped active layer/position context");
                    end
                    runtime_cache_read_req_count[layer] =
                        runtime_cache_read_req_count[layer] + 1;
                    if (persistent_two_token_mode &&
                        (persistent_active_step > 0) &&
                        (cache_start_position < persistent_active_step)) begin
                        persistent_prior_cache_read_req_count[layer] =
                            persistent_prior_cache_read_req_count[layer] + 1;
                    end
                end
                if (runtime_context_mode &&
                    in_range(addr, full_chain_runtime_rope_cos_base_addr,
                             ROPE_TABLE_WORDS * MEM_DATA_BYTES)) begin
                    if (!((in_range(addr, runtime_cos_row_base,
                                    HEAD_DIM * MEM_DATA_BYTES) &&
                           in_range(request_end_addr, runtime_cos_row_base,
                                    HEAD_DIM * MEM_DATA_BYTES)) ||
                          (in_range(addr, runtime_sin_row_base,
                                    HEAD_DIM * MEM_DATA_BYTES) &&
                           in_range(request_end_addr, runtime_sin_row_base,
                                    HEAD_DIM * MEM_DATA_BYTES)))) begin
                        fail_once("FAIL: runtime RoPE read did not select the manifest token row");
                    end
                    runtime_rope_read_req_count[layer] =
                        runtime_rope_read_req_count[layer] + 1;
                end
                if (in_range(addr, full_q_rope_base[layer], VEC2048 * MEM_DATA_BYTES) && !q_rope_written) begin
                    fail_once("FAIL: full-chain Q RoPE read before producer write completed");
                end
                if (in_range(addr, full_attn_out_base[layer], VEC2048 * MEM_DATA_BYTES) && !attn_out_written) begin
                    fail_once("FAIL: full-chain attention output read before producer write completed");
                end
                if (in_range(addr, full_o_proj_output_base[layer], VEC1024 * MEM_DATA_BYTES) && !o_proj_written) begin
                    fail_once("FAIL: full-chain o_proj output read before producer write completed");
                end
                if (in_range(addr, full_post_hidden_base[layer], VEC1024 * MEM_DATA_BYTES) && !post_hidden_written) begin
                    fail_once("FAIL: full-chain post-hidden read before producer write completed");
                end
                if (in_range(addr, full_post_norm_base[layer], VEC1024 * MEM_DATA_BYTES) && !post_norm_written) begin
                    fail_once("FAIL: full-chain post-norm read before producer write completed");
                end
                if (in_range(addr, full_gate_output_base[layer], VEC3072 * MEM_DATA_BYTES) && !gate_written) begin
                    fail_once("FAIL: full-chain gate output read before producer write completed");
                end
                if (in_range(addr, full_up_output_base[layer], VEC3072 * MEM_DATA_BYTES) && !up_written) begin
                    fail_once("FAIL: full-chain up output read before producer write completed");
                end
                if (in_range(addr, full_silu_output_base[layer], VEC3072 * MEM_DATA_BYTES) && !silu_hidden_written) begin
                    fail_once("FAIL: full-chain SiLU output read before producer write completed");
                end
                if (in_range(addr, full_down_output_base[layer], VEC1024 * MEM_DATA_BYTES) && !down_written) begin
                    fail_once("FAIL: full-chain down output read before producer write completed");
                end
            end
        end
    endtask
`endif

    task check_chained_read_ready;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input integer len_bytes;
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                check_full_chain_read_ready(addr, len_bytes);
            end
            else begin
`endif
            if (in_range(addr, input_norm_output_base, VEC1024 * MEM_DATA_BYTES) && !input_norm_written) begin
                fail_once("FAIL: input RMSNorm output read before producer write completed");
            end
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
            if (in_range(addr, layer_output_base, VEC1024 * MEM_DATA_BYTES)) begin
                if (embedding_true3_mode && (layers_completed == 0)) begin
                    if (first_embedding_hidden_read_cycle < 0) begin
                        first_embedding_hidden_read_cycle = cycle_count;
                    end
                    if (!embedding_response_seen) begin
                        fail_once("FAIL: Layer 0 read embedding hidden before write response");
                    end
                end
                else begin
                    if (embedding_true3_mode && (first_layer1_hidden_read_cycle < 0)) begin
                        first_layer1_hidden_read_cycle = cycle_count;
                    end
                    if ((embedding_true3_mode && !layer_response_seen[0]) ||
                        (!embedding_true3_mode && !layer_written)) begin
                        fail_once("FAIL: layer output read before producer write response");
                    end
                end
            end
            if (in_range(addr, input_norm_output_base_l1, VEC1024 * MEM_DATA_BYTES) && !input_norm_written_l1) begin
                fail_once("FAIL: Layer 1 input RMSNorm output read before producer write completed");
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
            if (in_range(addr, layer_output_base_l1, VEC1024 * MEM_DATA_BYTES)) begin
                if (embedding_true3_mode && (first_layer2_hidden_read_cycle < 0)) begin
                    first_layer2_hidden_read_cycle = cycle_count;
                end
                if ((embedding_true3_mode && !layer_response_seen[1]) ||
                    (!embedding_true3_mode && !layer_written)) begin
                    fail_once("FAIL: Layer 1 output read before producer write response");
                end
            end
            if (in_range(addr, input_norm_output_base_l2, VEC1024 * MEM_DATA_BYTES) && !input_norm_written_l2) begin
                fail_once("FAIL: Layer 2 input RMSNorm output read before producer write completed");
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
            if (in_range(addr, layer_output_base_l2, VEC1024 * MEM_DATA_BYTES) &&
                ((embedding_true3_mode && !layer_response_seen[2]) ||
                 (!embedding_true3_mode && !layer_written))) begin
                fail_once("FAIL: Layer 2 output read before producer write response");
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endtask

    task clear_scoreboard;
        integer i;
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
            input_norm_write_accept_count = 0;
            embedding_write_accept_count = 0;
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
            pending_write_kind = 0;
            pending_write_layer = 0;
            pending_write_addr = '0;
            use_tail_mem_manual = 1'b0;
            tail_start = 1'b0;
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
            input_norm_written = 1'b0;
            input_norm_written_l1 = 1'b0;
            input_norm_written_l2 = 1'b0;
            use_input_norm_qkv_expected = 1'b0;
            embedding_written = 1'b0;
            embedding_response_seen = 1'b0;
            layer_response_seen = '0;
            embedding_response_cycle = -1;
            first_embedding_hidden_read_cycle = -1;
            first_layer1_hidden_read_cycle = -1;
            first_layer2_hidden_read_cycle = -1;
            for (i = 0; i < MAX_LAYERS; i = i + 1) begin
                layer_response_cycle[i] = -1;
                first_layer_hidden_read_cycle[i] = -1;
                runtime_hidden_read_req_count[i] = 0;
                runtime_layer_output_write_req_count[i] = 0;
                runtime_rope_read_req_count[i] = 0;
                runtime_cache_read_req_count[i] = 0;
                runtime_cache_write_word_count[i] = 0;
                persistent_prior_cache_read_req_count[i] = 0;
            end
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
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
            top_scheduler_done_seen_count = 0;
            top_tail_start_seen_count = 0;
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
            axil_awaddr = 12'd0;
            axil_awprot = 3'b000;
            axil_awvalid = 1'b0;
            axil_wdata = 32'd0;
            axil_wstrb = 4'hF;
            axil_wvalid = 1'b0;
            axil_bready = 1'b0;
            axil_araddr = 12'd0;
            axil_arprot = 3'b000;
            axil_arvalid = 1'b0;
            axil_rready = 1'b0;
`endif
`endif
        end
    endtask

`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
    task clear_scoreboard_for_next_token;
        integer saved_cycle_count;
        begin
            saved_cycle_count = cycle_count;
            clear_scoreboard();
            cycle_count = saved_cycle_count;
            if ($test$plusargs("fastmem")) begin
                fastmem = 1'b1;
            end
        end
    endtask

    task clear_persistent_cache;
        integer i;
        begin
            for (i = 0;
                 i < (full_chain_layer_count * CACHE_LENGTH * KV_COUNT);
                 i = i + 1) begin
                full_k_cache_mem[i] = '0;
                full_v_cache_mem[i] = '0;
            end
        end
    endtask

    task check_and_capture_persistent_cache;
        input integer step;
        integer layer;
        integer element;
        integer cache_index;
        integer snapshot_index;
        logic [31 : 0] expected_k_word;
        logic [31 : 0] expected_v_word;
        begin
            for (layer = 0; layer < full_chain_layer_count; layer = layer + 1) begin
                for (element = 0; element < KV_COUNT; element = element + 1) begin
                    cache_index = (layer * CACHE_LENGTH * KV_COUNT) +
                        (step * KV_COUNT) + element;
                    expected_k_word =
                        full_expected_cache_data[(layer * TOTAL_CACHE_WRITES) + element];
                    expected_v_word =
                        full_expected_cache_data[(layer * TOTAL_CACHE_WRITES) +
                                                 KV_COUNT + element];
                    if ((full_k_cache_mem[cache_index] !== expected_k_word[IN_WIDTH-1 : 0]) ||
                        (full_v_cache_mem[cache_index] !== expected_v_word[IN_WIDTH-1 : 0])) begin
                        fail_once("FAIL: persistent K/V cache contents do not match current-step golden");
                    end

                    snapshot_index = (layer * KV_COUNT) + element;
                    if (step == 0) begin
                        persistent_step0_k_cache[snapshot_index] =
                            full_k_cache_mem[cache_index];
                        persistent_step0_v_cache[snapshot_index] =
                            full_v_cache_mem[cache_index];
                    end
                    else begin
                        cache_index = (layer * CACHE_LENGTH * KV_COUNT) + element;
                        if ((full_k_cache_mem[cache_index] !==
                             persistent_step0_k_cache[snapshot_index]) ||
                            (full_v_cache_mem[cache_index] !==
                             persistent_step0_v_cache[snapshot_index])) begin
                            fail_once("FAIL: step 1 modified retained step-0 K/V cache data");
                        end
                    end
                end
                if ((step > 0) &&
                    (persistent_prior_cache_read_req_count[layer] <= 0)) begin
                    fail_once("FAIL: step 1 did not read retained step-0 K/V cache data");
                end
            end
        end
    endtask

    task check_persistent_step_result;
        input integer step;
        integer expected_cache_length;
        integer expected_layer_read_reqs;
        integer expected_layer_read_words;
        logic [27 : 0] expected_mask;
        logic signed [63 : 0] expected_score_ext;
        begin
            expected_cache_length = step + 1;
            expected_layer_read_reqs = EXPECTED_INPUT_NORM_FULL_RD_REQS -
                ((CACHE_LENGTH - expected_cache_length) *
                 CACHE_READ_WORDS_PER_POSITION);
            expected_layer_read_words = EXPECTED_INPUT_NORM_FULL_RD_WORDS -
                ((CACHE_LENGTH - expected_cache_length) *
                 CACHE_READ_WORDS_PER_POSITION);
            expected_mask = expected_layer_mask(0, full_chain_layer_count);
            expected_score_ext = {
                {(64-TAIL_ROW_ACC_WIDTH){
                    persistent_expected_output_score[step][TAIL_ROW_ACC_WIDTH-1]
                }},
                persistent_expected_output_score[step]
            };

            if ((layers_started != full_chain_layer_count) ||
                (layers_completed != full_chain_layer_count) ||
                (layer_done_mask != expected_mask) ||
                (layer_error_mask != 28'd0) ||
                error || tail_error || tail_norm_saturation) begin
                fail_once("FAIL: persistent full-chain layer/tail status mismatch");
            end
            if ((done_seen_count != 1) ||
                (top_scheduler_done_seen_count != 1) ||
                (top_tail_start_seen_count != 1) ||
                (tail_done_seen_count != 1)) begin
                fail_once("FAIL: persistent full-chain pulse count mismatch");
            end
            if ((tail_best_token_id !== persistent_expected_output_token[step]) ||
                (tail_best_score_q26 !== persistent_expected_output_score[step]) ||
                (tail_expected_token !== persistent_expected_output_token[step]) ||
                (tail_expected_score_q26 !== persistent_expected_output_score[step])) begin
                fail_once("FAIL: persistent full-chain token/score mismatch");
            end
`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
            mmio_read_reg(MMIO_REG_OUT_TOKEN, mmio_read_data);
            if (mmio_read_data !== persistent_expected_output_token[step]) begin
                fail_once("FAIL: persistent MMIO output token register mismatch");
            end
            mmio_read_reg(MMIO_REG_OUT_SCORE_LO, mmio_read_data);
            if (mmio_read_data !==
                persistent_expected_output_score[step][31 : 0]) begin
                fail_once("FAIL: persistent MMIO output score low register mismatch");
            end
            mmio_read_reg(MMIO_REG_OUT_SCORE_HI, mmio_read_data);
            if (mmio_read_data !== expected_score_ext[63 : 32]) begin
                fail_once("FAIL: persistent MMIO output score high register mismatch");
            end
`endif
            if ((step == 0) &&
                (tail_best_token_id !== persistent_input_token[1])) begin
                fail_once("FAIL: step-1 input token is not the exact step-0 argmax");
            end
            if ((dut_read_burst_count !=
                 (full_chain_layer_count * expected_layer_read_reqs)) ||
                (dut_read_word_count !=
                 (full_chain_layer_count * expected_layer_read_words)) ||
                (dut_write_req_count !=
                 (full_chain_layer_count * EXPECTED_INPUT_NORM_FULL_WR_REQS)) ||
                (dut_write_word_count !=
                 (full_chain_layer_count * EXPECTED_INPUT_NORM_FULL_WR_WORDS))) begin
                fail_once("FAIL: persistent scheduler memory counters mismatch");
            end
            if ((top_mem_read_burst_count !=
                 ((full_chain_layer_count * expected_layer_read_reqs) +
                  EXPECTED_EMBEDDING_TAIL_EXTRA_RD_REQS)) ||
                (top_mem_read_word_count !=
                 ((full_chain_layer_count * expected_layer_read_words) +
                  EXPECTED_EMBEDDING_TAIL_EXTRA_RD_WORDS)) ||
                (top_mem_write_req_count !=
                 ((full_chain_layer_count * EXPECTED_INPUT_NORM_FULL_WR_REQS) +
                  EXPECTED_EMBEDDING_TAIL_EXTRA_WR_REQS)) ||
                (top_mem_write_word_count !=
                 ((full_chain_layer_count * EXPECTED_INPUT_NORM_FULL_WR_WORDS) +
                  EXPECTED_EMBEDDING_TAIL_EXTRA_WR_WORDS))) begin
                fail_once("FAIL: persistent top-level memory counters mismatch");
            end
            if ((embedding_write_accept_count != VEC1024) ||
                (input_norm_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (qkv_q_write_accept_count !=
                 (full_chain_layer_count * VEC2048)) ||
                (qkv_k_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (qkv_v_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (cache_write_accept_count !=
                 (full_chain_layer_count * TOTAL_CACHE_WRITES)) ||
                (q_rope_write_accept_count !=
                 (full_chain_layer_count * VEC2048)) ||
                (attn_out_write_accept_count !=
                 (full_chain_layer_count * VEC2048)) ||
                (o_proj_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (post_hidden_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (post_norm_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (gate_write_accept_count !=
                 (full_chain_layer_count * VEC3072)) ||
                (up_write_accept_count !=
                 (full_chain_layer_count * VEC3072)) ||
                (silu_write_accept_count !=
                 (full_chain_layer_count * VEC3072)) ||
                (down_write_accept_count !=
                 (full_chain_layer_count * VEC1024)) ||
                (layer_write_accept_count !=
                 (full_chain_layer_count * VEC1024))) begin
                fail_once("FAIL: persistent per-stage write word counts mismatch");
            end
            if ((tail_tiles_started != TAIL_MAX_TILES) ||
                (tail_tiles_completed != TAIL_MAX_TILES) ||
                (tail_hidden_req_count != 4) ||
                (tail_gamma_req_count != 4) ||
                (tail_norm_read_req_count != 4) ||
                (tail_weight_req_count !=
                 (TAIL_MAX_TILES * TAIL_WEIGHT_BURSTS_PER_TILE)) ||
                (tail_scale_req_count !=
                 (TAIL_MAX_TILES * TAIL_SCALE_BURSTS_PER_TILE)) ||
                (tail_norm_write_word_count != TAIL_INPUT_SIZE) ||
                (tail_output_write_word_count != 3)) begin
                $display(
                    "PERSISTENT_TAIL_COUNT_MISMATCH tiles=%0d/%0d hidden=%0d gamma=%0d norm=%0d weight=%0d/%0d scale=%0d/%0d norm_words=%0d output_words=%0d",
                    tail_tiles_started, tail_tiles_completed,
                    tail_hidden_req_count, tail_gamma_req_count,
                    tail_norm_read_req_count,
                    tail_weight_req_count,
                    TAIL_MAX_TILES * TAIL_WEIGHT_BURSTS_PER_TILE,
                    tail_scale_req_count,
                    TAIL_MAX_TILES * TAIL_SCALE_BURSTS_PER_TILE,
                    tail_norm_write_word_count,
                    tail_output_write_word_count
                );
                fail_once("FAIL: persistent final-tail exact counts mismatch");
            end
            if ((embedding_response_cycle < 0) ||
                (first_layer_hidden_read_cycle[0] <= embedding_response_cycle) ||
                (layer_response_cycle[full_chain_layer_count - 1] < 0) ||
                (tail_scheduler_done_cycle <=
                 layer_response_cycle[full_chain_layer_count - 1]) ||
                (tail_start_cycle <= tail_scheduler_done_cycle) ||
                (tail_first_hidden_read_cycle <= tail_start_cycle) ||
                (tail_first_weight_read_cycle <= tail_first_hidden_read_cycle) ||
                (tail_first_output_write_cycle <= tail_first_weight_read_cycle)) begin
                fail_once("FAIL: persistent producer/consumer response ordering mismatch");
            end
            if ((mismatch_count != 0) ||
                (write_mismatch_count != 0) ||
                (tail_mismatch_count != 0) ||
                (tail_write_mismatch_count != 0) ||
                (max_abs_diff != 0) ||
                (total_fail_count != 0)) begin
                $display(
                    "FAIL: persistent step %0d mismatches scheduler=%0d writes=%0d tail=%0d tail_writes=%0d max_abs=%0d total=%0d",
                    step, mismatch_count, write_mismatch_count, tail_mismatch_count,
                    tail_write_mismatch_count, max_abs_diff, total_fail_count
                );
                $finish(1);
            end

            check_and_capture_persistent_cache(step);
            if (total_fail_count != 0) begin
                $finish(1);
            end
        end
    endtask
`endif

    task set_layer_packet_bases;
        input integer layer_index;
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_qkv_qmap_base[layer_index];
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_frontend_qmap_base[layer_index];
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_score_qmap_base[layer_index];
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_oproj_qmap_base[layer_index];
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_post_qmap_base[layer_index];
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_gate_qmap_base[layer_index];
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_silu_qmap_base[layer_index];
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_down_qmap_base[layer_index];
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = full_residual_qmap_base[layer_index];
            end
            else begin
`endif
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
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = layer0_qkv_packet_base_addr;
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_FRONTEND_BASE_ADDR;
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_SCORE_VALUE_BASE_ADDR;
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_O_PROJ_BASE_ADDR;
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_POST_ATTN_NORM_BASE_ADDR;
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_GATE_UP_BASE_ADDR;
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_SILU_MUL_BASE_ADDR;
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_DOWN_BASE_ADDR;
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR;
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endtask

    task set_valid_loop_contract;
        integer i;
        begin
            layer_start_index = 5'd0;
            token_position = CACHE_LENGTH - 1;
            qkv_qmap_base_addr_table = '0;
            input_norm_qmap_base_addr_table = '0;
            attn_frontend_qmap_base_addr_table = '0;
            attn_score_value_qmap_base_addr_table = '0;
            o_proj_qmap_base_addr_table = '0;
            post_attn_norm_qmap_base_addr_table = '0;
            mlp_gate_up_qmap_base_addr_table = '0;
            mlp_silu_mul_qmap_base_addr_table = '0;
            mlp_down_qmap_base_addr_table = '0;
            mlp_residual_add_qmap_base_addr_table = '0;
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                layer_count = full_chain_layer_count;
                if (runtime_context_mode) begin
                    token_position = full_chain_runtime_position[POSITION_WIDTH-1 : 0];
                    input_hidden_base_addr = full_chain_runtime_hidden_a_base_addr;
                    output_hidden_base_addr = full_chain_runtime_hidden_b_base_addr;
                end
                else begin
                    input_hidden_base_addr = full_input_hidden_base[0];
                    output_hidden_base_addr = full_layer_output_base[full_chain_layer_count - 1];
                end
                kv_cache_base_addr = full_chain_kv_cache_base_addr;
                for (i = 0; i < full_chain_layer_count; i = i + 1) begin
                    set_layer_packet_bases(i);
                    enable_layer_input_norm(i);
                end
            end
            else begin
`endif
                layer_count = 5'd1;
                input_hidden_base_addr = qkv_desc_base(QKV_SLOT_ACTIVATION);
                output_hidden_base_addr = layer_output_base;
                kv_cache_base_addr = CACHE_BASE_ADDR;
                set_layer_packet_bases(0);
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
        end
    endtask

    task enable_layer_input_norm;
        input integer layer_index;
        begin
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode) begin
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] =
                    full_input_norm_qmap_base[layer_index];
            end
            else begin
`endif
            if (layer_index == 0) begin
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_INPUT_NORM_BASE_ADDR;
            end
            else if (layer_index == 1) begin
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER1_INPUT_NORM_BASE_ADDR;
            end
            else if (layer_index == 2) begin
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = QMAP_LAYER2_INPUT_NORM_BASE_ADDR;
            end
            else begin
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = '0;
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            end
`endif
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

`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
    task mmio_write_reg;
        input logic [11 : 0] addr;
        input logic [31 : 0] data;
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
        integer wait_cycles;
`endif
        begin
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
            wait_cycles = 0;
            @(negedge clk);
            axil_awaddr = addr;
            axil_awvalid = 1'b1;
            axil_wdata = data;
            axil_wstrb = 4'hF;
            axil_wvalid = 1'b1;
            #1;
            while (!(axil_awready && axil_wready)) begin
                @(negedge clk);
                #1;
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 1000) begin
                    fail_once("FAIL: AXI-Lite MMIO write address/data handshake timeout");
                    $finish(1);
                end
            end
            @(posedge clk);
            @(negedge clk);
            axil_awvalid = 1'b0;
            axil_wvalid = 1'b0;
            axil_awaddr = 12'd0;
            axil_wdata = 32'd0;
            wait_cycles = 0;
            while (!axil_bvalid) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 1000) begin
                    fail_once("FAIL: AXI-Lite MMIO write response timeout");
                    $finish(1);
                end
            end
            if (axil_bresp != AXIL_RESP_OKAY) begin
                fail_once("FAIL: AXI-Lite MMIO write returned non-OKAY BRESP");
                $finish(1);
            end
            @(negedge clk);
            axil_bready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            axil_bready = 1'b0;
`else
            @(negedge clk);
            mmio_reg_addr = addr;
            mmio_reg_wdata = data;
            mmio_reg_wr_valid = 1'b1;
            @(negedge clk);
            mmio_reg_wr_valid = 1'b0;
`endif
        end
    endtask

    task mmio_read_reg;
        input logic [11 : 0] addr;
        output logic [31 : 0] data;
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
        integer wait_cycles;
`endif
        begin
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
            wait_cycles = 0;
            @(negedge clk);
            axil_araddr = addr;
            axil_arvalid = 1'b1;
            #1;
            while (!axil_arready) begin
                @(negedge clk);
                #1;
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 1000) begin
                    fail_once("FAIL: AXI-Lite MMIO read address handshake timeout");
                    $finish(1);
                end
            end
            @(posedge clk);
            @(negedge clk);
            axil_arvalid = 1'b0;
            axil_araddr = 12'd0;
            wait_cycles = 0;
            while (!axil_rvalid) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if (wait_cycles > 1000) begin
                    fail_once("FAIL: AXI-Lite MMIO read response timeout");
                    $finish(1);
                end
            end
            data = axil_rdata;
            if (axil_rresp != AXIL_RESP_OKAY) begin
                fail_once("FAIL: AXI-Lite MMIO read returned non-OKAY RRESP");
                $finish(1);
            end
            @(negedge clk);
            axil_rready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            axil_rready = 1'b0;
`else
            @(negedge clk);
            mmio_reg_addr = addr;
            mmio_reg_rd_valid = 1'b1;
            #1;
            data = mmio_reg_rdata;
            @(negedge clk);
            mmio_reg_rd_valid = 1'b0;
`endif
        end
    endtask

    task mmio_write_addr64;
        input logic [11 : 0] lo_addr;
        input logic [11 : 0] hi_addr;
        input logic [ADDR_WIDTH-1 : 0] value;
        begin
            mmio_write_reg(lo_addr, value[31 : 0]);
            mmio_write_reg(hi_addr, value[63 : 32]);
        end
    endtask

    task mmio_commit_table_addr;
        input logic [7 : 0] table_id;
        input integer layer_index;
        input logic [ADDR_WIDTH-1 : 0] value;
        begin
            if (value != '0) begin
                mmio_write_reg(MMIO_REG_TABLE_SELECT, {16'd0, layer_index[7 : 0], table_id});
                mmio_write_addr64(MMIO_REG_TABLE_DATA_LO, MMIO_REG_TABLE_DATA_HI, value);
                mmio_write_reg(MMIO_REG_TABLE_COMMIT, 32'd1);
            end
        end
    endtask

    task mmio_load_current_loop_contract;
        integer li;
        logic [ADDR_WIDTH-1 : 0] table_addr;
        begin
            mmio_write_reg(MMIO_REG_CTRL, 32'd2);
            mmio_write_reg(MMIO_REG_LAYER_START, {{(32-LAYER_INDEX_WIDTH){1'b0}}, layer_start_index});
            mmio_write_reg(MMIO_REG_LAYER_COUNT, {{(32-LAYER_COUNT_WIDTH){1'b0}}, layer_count});
            mmio_write_reg(MMIO_REG_POSITION, {{(32-POSITION_WIDTH){1'b0}}, token_position});
            mmio_write_reg(MMIO_REG_RUNTIME_CTRL, runtime_context_mode ? 32'd1 : 32'd0);
            mmio_write_reg(MMIO_REG_INPUT_TOKEN,
                           embedding_true3_mode ? embedding_token_mem[0] : 32'd0);
            mmio_write_addr64(MMIO_REG_INPUT_HIDDEN_LO, MMIO_REG_INPUT_HIDDEN_HI, input_hidden_base_addr);
            mmio_write_addr64(MMIO_REG_OUTPUT_HIDDEN_LO, MMIO_REG_OUTPUT_HIDDEN_HI, output_hidden_base_addr);
            mmio_write_addr64(MMIO_REG_KV_CACHE_LO, MMIO_REG_KV_CACHE_HI, kv_cache_base_addr);
            mmio_write_addr64(MMIO_REG_FINAL_TAIL_QMAP_LO, MMIO_REG_FINAL_TAIL_QMAP_HI, tail_qmap_base_addr);
            mmio_write_reg(MMIO_REG_FINAL_OVERRIDE_CTRL, 32'd0);
            mmio_write_addr64(MMIO_REG_EMBED_WEIGHT_LO, MMIO_REG_EMBED_WEIGHT_HI,
                              embedding_true3_mode ? EMBEDDING_WEIGHT_BASE_ADDR : '0);
            mmio_write_addr64(MMIO_REG_EMBED_SCALE_LO, MMIO_REG_EMBED_SCALE_HI,
                              embedding_true3_mode ? EMBEDDING_SCALE_BASE_ADDR : '0);
            mmio_write_reg(MMIO_REG_EMBEDDING_CTRL,
                           embedding_true3_mode ? 32'd1 : 32'd0);

            for (li = 0; li < MAX_LAYERS; li = li + 1) begin
                table_addr = qkv_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_QKV, li, table_addr);
                table_addr = input_norm_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_INPUT_NORM, li, table_addr);
                table_addr = attn_frontend_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_ATTN_FRONTEND, li, table_addr);
                table_addr = attn_score_value_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_ATTN_SCORE, li, table_addr);
                table_addr = o_proj_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_O_PROJ, li, table_addr);
                table_addr = post_attn_norm_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_POST_ATTN_NORM, li, table_addr);
                table_addr = mlp_gate_up_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_MLP_GATE_UP, li, table_addr);
                table_addr = mlp_silu_mul_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_MLP_SILU_MUL, li, table_addr);
                table_addr = mlp_down_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_MLP_DOWN, li, table_addr);
                table_addr = mlp_residual_add_qmap_base_addr_table[li*ADDR_WIDTH +: ADDR_WIDTH];
                mmio_commit_table_addr(MMIO_TABLE_MLP_RESIDUAL, li, table_addr);
            end
        end
    endtask

    task mmio_expect_reg32;
        input string reg_name;
        input logic [11 : 0] addr;
        input logic [31 : 0] expected;
        logic [31 : 0] observed;
        begin
            mmio_read_reg(addr, observed);
            if (observed !== expected) begin
                $display("FAIL: qmap_one_token_top MMIO register mismatch reg=%s expected=0x%08h observed=0x%08h",
                         reg_name, expected, observed);
                fail_once("FAIL: qmap_one_token_top MMIO scalar register load mismatch before start");
            end
        end
    endtask

    task mmio_expect_addr64;
        input string reg_name;
        input logic [11 : 0] lo_addr;
        input logic [11 : 0] hi_addr;
        input logic [ADDR_WIDTH-1 : 0] expected;
        logic [31 : 0] lo;
        logic [31 : 0] hi;
        logic [ADDR_WIDTH-1 : 0] observed;
        begin
            mmio_read_reg(lo_addr, lo);
            mmio_read_reg(hi_addr, hi);
            observed = {hi, lo};
            if (observed !== expected) begin
                $display("FAIL: qmap_one_token_top MMIO address register mismatch reg=%s expected=0x%016h observed=0x%016h",
                         reg_name, expected, observed);
                fail_once("FAIL: qmap_one_token_top MMIO scalar address load mismatch before start");
            end
        end
    endtask

    task mmio_check_scalar_registers;
        begin
            mmio_expect_reg32("layer_start", MMIO_REG_LAYER_START,
                              {{(32-LAYER_INDEX_WIDTH){1'b0}}, layer_start_index});
            mmio_expect_reg32("layer_count", MMIO_REG_LAYER_COUNT,
                              {{(32-LAYER_COUNT_WIDTH){1'b0}}, layer_count});
            mmio_expect_reg32("position", MMIO_REG_POSITION,
                              {{(32-POSITION_WIDTH){1'b0}}, token_position});
            mmio_expect_reg32("runtime_ctrl", MMIO_REG_RUNTIME_CTRL,
                              runtime_context_mode ? 32'd1 : 32'd0);
            mmio_expect_reg32("input_token", MMIO_REG_INPUT_TOKEN,
                              embedding_true3_mode ? embedding_token_mem[0] : 32'd0);
            mmio_expect_reg32("embedding_ctrl", MMIO_REG_EMBEDDING_CTRL,
                              embedding_true3_mode ? 32'd1 : 32'd0);
            mmio_expect_addr64("input_hidden", MMIO_REG_INPUT_HIDDEN_LO, MMIO_REG_INPUT_HIDDEN_HI,
                               input_hidden_base_addr);
            mmio_expect_addr64("output_hidden", MMIO_REG_OUTPUT_HIDDEN_LO, MMIO_REG_OUTPUT_HIDDEN_HI,
                               output_hidden_base_addr);
            mmio_expect_addr64("kv_cache", MMIO_REG_KV_CACHE_LO, MMIO_REG_KV_CACHE_HI,
                               kv_cache_base_addr);
            mmio_expect_addr64("final_tail_qmap", MMIO_REG_FINAL_TAIL_QMAP_LO, MMIO_REG_FINAL_TAIL_QMAP_HI,
                            tail_qmap_base_addr);
            mmio_expect_addr64("embedding_weight", MMIO_REG_EMBED_WEIGHT_LO, MMIO_REG_EMBED_WEIGHT_HI,
                               embedding_true3_mode ? EMBEDDING_WEIGHT_BASE_ADDR : '0);
            mmio_expect_addr64("embedding_scale", MMIO_REG_EMBED_SCALE_LO, MMIO_REG_EMBED_SCALE_HI,
                               embedding_true3_mode ? EMBEDDING_SCALE_BASE_ADDR : '0);
        end
    endtask

    task mmio_read_table_addr;
        input logic [7 : 0] table_id;
        input integer layer_index;
        output logic [ADDR_WIDTH-1 : 0] value;
        logic [31 : 0] lo;
        logic [31 : 0] hi;
        begin
            mmio_write_reg(MMIO_REG_TABLE_SELECT, {16'd0, layer_index[7 : 0], table_id});
            mmio_read_reg(MMIO_REG_TABLE_DATA_LO, lo);
            mmio_read_reg(MMIO_REG_TABLE_DATA_HI, hi);
            value = {hi, lo};
        end
    endtask

    task mmio_expect_table_addr;
        input string table_name;
        input logic [7 : 0] table_id;
        input integer layer_index;
        input logic [ADDR_WIDTH-1 : 0] expected;
        logic [ADDR_WIDTH-1 : 0] observed;
        begin
            mmio_read_table_addr(table_id, layer_index, observed);
            if (observed !== expected) begin
                $display("FAIL: qmap_one_token_top MMIO table mismatch table=%s layer=%0d expected=0x%016h observed=0x%016h",
                         table_name, layer_index, expected, observed);
                fail_once("FAIL: qmap_one_token_top MMIO table register load mismatch before start");
            end
        end
    endtask

    task mmio_check_layer_tables;
        input integer layer_index;
        begin
            mmio_expect_table_addr("qkv", MMIO_TABLE_QKV, layer_index,
                qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("input_norm", MMIO_TABLE_INPUT_NORM, layer_index,
                input_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("attn_frontend", MMIO_TABLE_ATTN_FRONTEND, layer_index,
                attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("attn_score_value", MMIO_TABLE_ATTN_SCORE, layer_index,
                attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("o_proj", MMIO_TABLE_O_PROJ, layer_index,
                o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("post_attn_norm", MMIO_TABLE_POST_ATTN_NORM, layer_index,
                post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("mlp_gate_up", MMIO_TABLE_MLP_GATE_UP, layer_index,
                mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("mlp_silu_mul", MMIO_TABLE_MLP_SILU_MUL, layer_index,
                mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("mlp_down", MMIO_TABLE_MLP_DOWN, layer_index,
                mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
            mmio_expect_table_addr("mlp_residual_add", MMIO_TABLE_MLP_RESIDUAL, layer_index,
                mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH]);
        end
    endtask

    task mmio_run_until_done;
        input integer timeout_cycles;
        integer wait_cycles;
        integer next_progress_cycle;
        begin
            wait_cycles = 0;
            next_progress_cycle = 1000000;
            mmio_write_reg(MMIO_REG_CTRL, 32'd1);
            while ((done != 1'b1) && (wait_cycles < timeout_cycles)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if ($test$plusargs("progress") && (wait_cycles >= next_progress_cycle)) begin
                    $display(
                        "MMIO_PROGRESS: wait=%0d cycle=%0d state=0x%0h phase=0x%0h active_layer=%0d started=%0d completed=%0d rd=%0d/%0d wr=%0d/%0d",
                        wait_cycles,
                        cycle_count,
                        state_debug,
                        top_phase_debug,
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
                $display("FAIL: timed out waiting for qmap_one_token_top MMIO-launched done");
                $finish(1);
            end
            @(posedge clk);
        end
    endtask
`endif

`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
    task run_tail_until_done;
        input integer timeout_cycles;
        integer wait_cycles;
        integer next_progress_cycle;
        begin
            wait_cycles = 0;
            next_progress_cycle = 10000000;
            tail_start <= 1'b1;
            @(posedge clk);
            tail_start <= 1'b0;
            while ((tail_done != 1'b1) && (wait_cycles < timeout_cycles)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if ($test$plusargs("progress") && (wait_cycles >= next_progress_cycle)) begin
                    $display(
                        "TAIL_PROGRESS: wait=%0d cycle=%0d tiles=%0d/%0d rd=%0d/%0d wr_words=%0d best=%0d score=%0d",
                        wait_cycles,
                        cycle_count,
                        tail_tiles_started,
                        tail_tiles_completed,
                        tail_mem_read_burst_count,
                        tail_mem_read_word_count,
                        tail_mem_write_word_count,
                        tail_best_token_id,
                        tail_best_score_q26
                    );
                    $fflush();
                    next_progress_cycle = next_progress_cycle + 10000000;
                end
            end
            if (tail_done != 1'b1) begin
                $display("FAIL: timed out waiting for qmap_final_token_tail_compute_path done");
                $finish(1);
            end
            tail_done_cycle = cycle_count;
            @(posedge clk);
        end
    endtask
`endif

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

    task corrupt_frontend_cos_dtype_l1;
        begin
            frontend_qmap_l1[desc_idx(6, DESC_DTYPE_WORD)] = 32'd5;
        end
    endtask

    task corrupt_frontend_cos_dtype_l2;
        begin
            frontend_qmap_l2[desc_idx(6, DESC_DTYPE_WORD)] = 32'd5;
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
            pending_write_kind <= 0;
            pending_write_layer <= 0;
            pending_write_addr <= '0;
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
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                if (use_tail_mem) begin
                    check_tail_read_request(mem_rd_req_addr, mem_rd_req_len_bytes);
                end
                else begin
                    check_chained_read_ready(mem_rd_req_addr, mem_rd_req_len_bytes);
                end
`else
                check_chained_read_ready(mem_rd_req_addr, mem_rd_req_len_bytes);
`endif
                read_active <= 1'b1;
                active_read_addr <= mem_rd_req_addr;
                active_read_index <= 0;
                active_words_left <= mem_rd_req_len_bytes / MEM_DATA_BYTES;
                active_total_words <= mem_rd_req_len_bytes / MEM_DATA_BYTES;
                read_gap_count <= fastmem ? 0 : (rd_req_accept_count % 5);
                rd_req_accept_count = rd_req_accept_count + 1;
                if (event_trace_fd != 0) begin
                    $fwrite(event_trace_fd, "%0d,read_req,0x%016h,%0d,0x%08h,0\n",
                            cycle_count, mem_rd_req_addr,
                            mem_rd_req_len_bytes / MEM_DATA_BYTES,
                            mem_rd_req_len_bytes);
                end
            end

            if (read_active) begin
                if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                end
                else if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                    rd_rsp_accept_count = rd_rsp_accept_count + 1;
                    if (mem_rd_rsp_last) begin
                        if (event_trace_fd != 0) begin
                            $fwrite(event_trace_fd, "%0d,read_last,0x%016h,%0d,0x%08h,1\n",
                                    cycle_count,
                                    active_read_addr + (active_read_index * MEM_DATA_BYTES),
                                    active_total_words, active_total_words * MEM_DATA_BYTES);
                        end
                        mem_rd_rsp_valid <= 1'b0;
                        mem_rd_rsp_last <= 1'b0;
                        read_active <= 1'b0;
                    end
                    else begin
                        active_read_index <= active_read_index + 1;
                        active_words_left <= active_words_left - 1;
                        if (fastmem) begin
                            // Model a legal zero-bubble memory response in the
                            // fast functional regression.  The non-fast modes
                            // below retain their response backpressure gaps.
                            mem_rd_rsp_valid <= 1'b1;
                            mem_rd_rsp_data <= memory_word(
                                active_read_addr +
                                ((active_read_index + 1) * MEM_DATA_BYTES)
                            );
                            mem_rd_rsp_last <= (active_words_left == 2);
                            read_gap_count <= 0;
                        end
                        else begin
                            mem_rd_rsp_valid <= 1'b0;
                            mem_rd_rsp_last <= 1'b0;
                            read_gap_count <=
                                ((active_read_index % 17) != 3) ? 0 : 2;
                        end
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
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                if (use_tail_mem) begin
                    active_write_kind = classify_tail_write_addr(mem_wr_req_addr);
                    check_tail_write_request(
                        classify_tail_write_addr(mem_wr_req_addr),
                        mem_wr_req_addr,
                        mem_wr_req_len_bytes / MEM_DATA_BYTES
                    );
                end
                else begin
                    active_write_kind = classify_write_addr(mem_wr_req_addr);
                    check_write_request(classify_write_addr(mem_wr_req_addr), mem_wr_req_addr, mem_wr_req_len_bytes / MEM_DATA_BYTES);
                end
`else
                active_write_kind = classify_write_addr(mem_wr_req_addr);
                check_write_request(classify_write_addr(mem_wr_req_addr), mem_wr_req_addr, mem_wr_req_len_bytes / MEM_DATA_BYTES);
`endif
                wr_req_accept_count = wr_req_accept_count + 1;
                if (event_trace_fd != 0) begin
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                    if (use_tail_mem) begin
                        $fwrite(event_trace_fd, "%0d,write_req,0x%016h,%0d,0x%08h,0\n",
                                cycle_count, mem_wr_req_addr,
                                classify_tail_write_addr(mem_wr_req_addr), mem_wr_req_len_bytes);
                    end
                    else begin
                        $fwrite(event_trace_fd, "%0d,write_req,0x%016h,%0d,0x%08h,0\n",
                                cycle_count, mem_wr_req_addr,
                                classify_write_addr(mem_wr_req_addr), mem_wr_req_len_bytes);
                    end
`else
                    $fwrite(event_trace_fd, "%0d,write_req,0x%016h,%0d,0x%08h,0\n",
                            cycle_count, mem_wr_req_addr,
                            classify_write_addr(mem_wr_req_addr), mem_wr_req_len_bytes);
`endif
                end
            end

            if (write_active && mem_wr_data_valid && mem_wr_data_ready) begin
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                if (use_tail_mem) begin
                    check_tail_write_word(
                        classify_tail_write_addr(active_write_addr),
                        active_write_index,
                        active_write_addr + (active_write_index * MEM_DATA_BYTES),
                        mem_wr_data,
                        mem_wr_data_last
                    );
                end
                else begin
                    check_write_word(
                        classify_write_addr(active_write_addr),
                        active_write_index,
                        active_write_addr + (active_write_index * MEM_DATA_BYTES),
                        mem_wr_data,
                        mem_wr_data_last
                    );
                end
`else
                check_write_word(
                    classify_write_addr(active_write_addr),
                    active_write_index,
                    active_write_addr + (active_write_index * MEM_DATA_BYTES),
                    mem_wr_data,
                    mem_wr_data_last
                );
`endif
                wr_data_accept_count = wr_data_accept_count + 1;
                if (active_write_words_left == 1) begin
                    if (!mem_wr_data_last) begin
                        fail_once("FAIL: final write word missing last");
                    end
                    write_active <= 1'b0;
                    active_write_words_left <= 0;
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                    pending_write_kind <= use_tail_mem ?
                        classify_tail_write_addr(active_write_addr) :
                        classify_completed_write_addr(active_write_addr);
`else
                    pending_write_kind <= classify_completed_write_addr(active_write_addr);
`endif
                    pending_write_layer <= active_write_layer;
                    pending_write_addr <= active_write_addr;
                    write_done_delay <= 3;
                    if (event_trace_fd != 0) begin
                        $fwrite(event_trace_fd, "%0d,write_last,0x%016h,%0d,0x%08h,1\n",
                                cycle_count,
                                active_write_addr + (active_write_index * MEM_DATA_BYTES),
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
                                use_tail_mem ? classify_tail_write_addr(active_write_addr) :
                                               classify_completed_write_addr(active_write_addr),
`else
                                classify_completed_write_addr(active_write_addr),
`endif
                                mem_wr_data);
                    end
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
                    if (pending_write_kind == WRITE_EMBEDDING) begin
                        embedding_response_seen = 1'b1;
                        embedding_response_cycle = cycle_count;
                    end
                    else if ((pending_write_kind == WRITE_LAYER) &&
                             (pending_write_layer >= 0) && (pending_write_layer < MAX_LAYERS)) begin
                        layer_response_seen[pending_write_layer] = 1'b1;
                        layer_response_cycle[pending_write_layer] = cycle_count;
                    end
                    if (event_trace_fd != 0) begin
                        $fwrite(event_trace_fd, "%0d,write_done,0x%016h,%0d,0x00000000,1\n",
                                cycle_count, pending_write_addr, pending_write_kind);
                    end
                end
            end

            if (done) begin
                done_seen_count = done_seen_count + 1;
            end
`ifdef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
            if (top_scheduler_done_pulse && (tail_scheduler_done_cycle < 0)) begin
                tail_scheduler_done_cycle = cycle_count;
            end
            if (top_scheduler_done_pulse) begin
                top_scheduler_done_seen_count = top_scheduler_done_seen_count + 1;
                if (event_trace_fd != 0) begin
                    $fwrite(event_trace_fd, "%0d,scheduler_done,0x0000000000000000,0,0x00000000,1\n",
                            cycle_count);
                end
            end
            if (top_tail_start_pulse) begin
                top_tail_start_seen_count = top_tail_start_seen_count + 1;
                if (tail_start_cycle < 0) begin
                    tail_start_cycle = cycle_count;
                end
                if (event_trace_fd != 0) begin
                    $fwrite(event_trace_fd, "%0d,tail_start,0x0000000000000000,0,0x00000000,1\n",
                            cycle_count);
                end
            end
            if (top_tail_done_pulse && (tail_done_cycle < 0)) begin
                tail_done_cycle = cycle_count;
            end
            if (top_tail_done_pulse) begin
                tail_done_seen_count = tail_done_seen_count + 1;
            end
`else
            if (use_tail_mem && tail_done) begin
                tail_done_seen_count = tail_done_seen_count + 1;
            end
`endif
`endif

            if (done && (event_trace_fd != 0)) begin
                $fwrite(event_trace_fd, "%0d,top_done,0x0000000000000000,0,0x00000000,1\n",
                        cycle_count);
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
        if ($test$plusargs("l2_tail_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_layer2_input_norm_to_final_tail_trace.csv";
        end
        if ($test$plusargs("l2_top_tail_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_top_layer2_input_norm_to_final_tail_trace.csv";
        end
        /*
        if ($test$plusargs("l1_l2_mmio_top_tail_only")) begin
`ifndef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $display("FAIL: +l1_l2_mmio_top_tail_only requires compile define QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_TOP
            $display("FAIL: +l1_l2_mmio_top_tail_only requires compile define QMAP_ONE_TOKEN_TB_USE_TOP");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
            $display("FAIL: +l1_l2_mmio_top_tail_only requires compile define QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL");
            $finish(1);
`else
            set_layer_packet_bases(1);
            set_layer_packet_bases(2);
            enable_layer_input_norm(1);
            enable_layer_input_norm(2);
            layer_start_index = 5'd1;
            layer_count = 5'd2;
            scoreboard_layer_iterations = 2;
            prefill_layer0_output_for_l1_only();
            clear_tail_scoreboard();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
            mmio_load_current_loop_contract();

            $display("SCENARIO: qmap_one_token_top MMIO Layer1 -> Layer2 scheduler -> final-token tail start");
            $display("  descriptor final hidden base = 0x%016h", tail_descriptor_final_hidden_base);
            $display("  mmio expected hidden base    = 0x%016h", tail_final_hidden_base);
            $display("  layer1 output base           = 0x%016h", layer_output_base_l1);
            $display("  layer2 output base           = 0x%016h", layer_output_base_l2);
            $fflush();
            mmio_run_until_done(120000000);

            if ((layers_started != 5'd2) ||
                (layers_completed != 5'd2) ||
                (layer_done_mask != 28'h0000006) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: qmap_one_token_top MMIO Layer1->Layer2 scheduler layer masks mismatch");
            end
            if ((top_scheduler_done_seen_count != 1) ||
                (top_tail_start_seen_count != 1) ||
                (tail_done_seen_count != 1) ||
                (done_seen_count != 1)) begin
                fail_once("FAIL: qmap_one_token_top MMIO Layer1->Layer2 control pulse count mismatch");
            end
            if (top_tail_effective_final_hidden_base !== layer_output_base_l2) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 tail effective hidden base mismatch");
            end
            if (scheduler_last_layer_output_base !== layer_output_base_l2) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 scheduler last layer output base mismatch");
            end
            if (scheduler_last_layer_output_base === layer_output_base_l1) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 selected the previous layer output");
            end
            if ((tail_descriptor_final_hidden_base === layer_output_base_l1) ||
                (tail_descriptor_final_hidden_base === layer_output_base_l2)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 test accidentally pre-patched the tail descriptor");
            end
            if ((input_norm_write_accept_count != (2 * VEC1024)) ||
                (qkv_q_write_accept_count != (2 * VEC2048)) ||
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
                fail_once("FAIL: qmap_one_token_top MMIO Layer1->Layer2 producer write counts mismatch");
            end
            if ((dut_read_burst_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_WORDS)) begin
                fail_once("FAIL: qmap_one_token_top MMIO Layer1->Layer2 scheduler counters mismatch");
            end
            if ((tail_best_token_id !== tail_expected_token) ||
                (tail_best_score_q26 !== tail_expected_score_q26)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 token/score mismatch");
            end
            if ((tail_norm_write_word_count != TAIL_INPUT_SIZE) ||
                (tail_output_write_word_count != 3)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 final-token tail write word count mismatch");
            end
            if ((tail_scheduler_done_cycle < 0) ||
                (tail_first_hidden_read_cycle <= tail_scheduler_done_cycle) ||
                (tail_first_hidden_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 tail hidden read did not occur after scheduler done");
            end
            if ((tail_first_weight_read_cycle <= tail_first_hidden_read_cycle) ||
                (tail_first_weight_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 tail LM-head read did not occur after hidden read");
            end
            if ((tail_first_output_write_cycle <= tail_first_weight_read_cycle) ||
                (tail_first_output_write_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 tail output write did not occur after LM-head reads");
            end
            if ((top_mem_read_burst_count != (dut_read_burst_count + tail_mem_read_burst_count)) ||
                (top_mem_read_word_count != (dut_read_word_count + tail_mem_read_word_count)) ||
                (top_mem_write_req_count != (dut_write_req_count + 32'd2)) ||
                (top_mem_write_word_count != (dut_write_word_count + tail_mem_write_word_count))) begin
                tail_fail("FAIL: qmap_one_token_top MMIO Layer1->Layer2 aggregate memory counters mismatch");
            end

            mmio_read_reg(MMIO_REG_STATUS, mmio_read_data);
            if ((mmio_read_data[1] != 1'b1) || (mmio_read_data[2] != 1'b0) ||
                (mmio_read_data[3] != 1'b0)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO status register mismatch after done");
            end
            mmio_read_reg(MMIO_REG_OUT_TOKEN, mmio_read_data);
            if (mmio_read_data != tail_expected_token) begin
                tail_fail("FAIL: qmap_one_token_top MMIO output token register mismatch");
            end
            mmio_read_reg(MMIO_REG_OUT_SCORE_LO, mmio_read_data);
            if (mmio_read_data != tail_expected_score_q26[31 : 0]) begin
                tail_fail("FAIL: qmap_one_token_top MMIO output score low register mismatch");
            end
            mmio_read_reg(MMIO_REG_OUT_SCORE_HI, mmio_read_data);
            if (mmio_read_data != 32'd0) begin
                tail_fail("FAIL: qmap_one_token_top MMIO output score high register mismatch");
            end
            mmio_read_reg(MMIO_REG_LAYERS, mmio_read_data);
            if ((mmio_read_data[4 : 0] != 5'd2) || (mmio_read_data[20 : 16] != 5'd2)) begin
                tail_fail("FAIL: qmap_one_token_top MMIO layers register mismatch");
            end
            mmio_read_reg(MMIO_REG_LAYER_DONE_MASK, mmio_read_data);
            if (mmio_read_data[27 : 0] != 28'h0000006) begin
                tail_fail("FAIL: qmap_one_token_top MMIO layer done mask register mismatch");
            end
            mmio_read_reg(MMIO_REG_LAST_OUTPUT_LO, mmio_read_data);
            if (mmio_read_data != layer_output_base_l2[31 : 0]) begin
                tail_fail("FAIL: qmap_one_token_top MMIO last output base register mismatch");
            end
            mmio_read_reg(MMIO_REG_TAIL_HIDDEN_LO, mmio_read_data);
            if (mmio_read_data != layer_output_base_l2[31 : 0]) begin
                tail_fail("FAIL: qmap_one_token_top MMIO tail hidden base register mismatch");
            end
            mmio_read_reg(MMIO_REG_MEM_RD_REQS, mmio_read_data);
            if (mmio_read_data != top_mem_read_burst_count) begin
                tail_fail("FAIL: qmap_one_token_top MMIO read request counter register mismatch");
            end
            mmio_read_reg(MMIO_REG_MEM_RD_WORDS, mmio_read_data);
            if (mmio_read_data != top_mem_read_word_count) begin
                tail_fail("FAIL: qmap_one_token_top MMIO read word counter register mismatch");
            end
            mmio_read_reg(MMIO_REG_MEM_WR_REQS, mmio_read_data);
            if (mmio_read_data != top_mem_write_req_count) begin
                tail_fail("FAIL: qmap_one_token_top MMIO write request counter register mismatch");
            end
            mmio_read_reg(MMIO_REG_MEM_WR_WORDS, mmio_read_data);
            if (mmio_read_data != top_mem_write_word_count) begin
                tail_fail("FAIL: qmap_one_token_top MMIO write word counter register mismatch");
            end

            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (tail_mismatch_count != 0) || (tail_write_mismatch_count != 0) ||
                (max_abs_diff != 0) || error || (total_fail_count != 0)) begin
                $display("FAIL: qmap_one_token_top MMIO Layer1->Layer2 found mismatch=%0d write_mismatch=%0d tail_mismatch=%0d tail_write_mismatch=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, tail_mismatch_count,
                         tail_write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end

            $display("qmap_one_token_top MMIO Layer1 -> Layer2 input RMSNorm scheduler -> final-token tail test");
            $display("  scheduler done cycle   = %0d", tail_scheduler_done_cycle);
            $display("  tail done cycle        = %0d", tail_done_cycle);
            $display("  first hidden read      = %0d", tail_first_hidden_read_cycle);
            $display("  first LM-head read     = %0d", tail_first_weight_read_cycle);
            $display("  first output write     = %0d", tail_first_output_write_cycle);
            $display("  expected token/score   = %0d / %0d", tail_expected_token, tail_expected_score_q26);
            $display("  observed token/score   = %0d / %0d", tail_best_token_id, tail_best_score_q26);
            $display("  top pulses             = scheduler_done %0d tail_start %0d tail_done %0d top_done %0d",
                     top_scheduler_done_seen_count, top_tail_start_seen_count, tail_done_seen_count, done_seen_count);
            $display("  scheduler rd/wr        = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count,
                     layer_done_mask);
            $display("  top rd/wr counters     = %0d/%0d reads, %0d/%0d writes",
                     top_mem_read_burst_count, top_mem_read_word_count,
                     top_mem_write_req_count, top_mem_write_word_count);
            $display("  trace                  = %s", tracefile);
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            $display("PASS: qmap_one_token_top MMIO registers launched Layer1 and Layer2 before final-token tail.");
            $finish;
`endif
`endif
`endif
`endif
        end

        */
        if ($test$plusargs("l1_l2_top_tail_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_top_layer1_layer2_input_norm_to_final_tail_trace.csv";
        end
        if ($test$plusargs("l1_l2_mmio_top_tail_only")) begin
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
            tracefile = "FPGA_Project/sim/qmap_one_token_axil_layer1_layer2_input_norm_to_final_tail_trace.csv";
`else
            tracefile = "FPGA_Project/sim/qmap_one_token_mmio_layer1_layer2_input_norm_to_final_tail_trace.csv";
`endif
        end
        if ($test$plusargs("true3_top_tail_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_top_true3_input_norm_to_final_tail_trace.csv";
        end
        if ($test$plusargs("true3_mmio_top_tail_only")) begin
`ifdef QMAP_ONE_TOKEN_TB_USE_AXIL_TOP
            tracefile = "FPGA_Project/sim/qmap_one_token_axil_true3_input_norm_to_final_tail_trace.csv";
`else
            tracefile = "FPGA_Project/sim/qmap_one_token_mmio_true3_input_norm_to_final_tail_trace.csv";
`endif
        end
        if ($test$plusargs("input_norm_qkv_only")) begin
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
            tracefile = "FPGA_Project/sim/qmap_one_token_top_input_norm_to_qkv_trace.csv";
`else
            tracefile = "FPGA_Project/sim/qmap_one_token_input_norm_to_qkv_trace.csv";
`endif
        end
        if ($test$plusargs("l1_input_norm_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_layer1_input_norm_trace.csv";
        end
        if ($test$plusargs("l2_input_norm_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_layer2_input_norm_trace.csv";
        end
        if ($test$plusargs("l1_only") || $test$plusargs("l1_input_norm_full_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_layer1_input_norm_full_trace.csv";
        end
        if ($test$plusargs("l2_only") || $test$plusargs("l2_input_norm_full_only")) begin
            tracefile = "FPGA_Project/sim/qmap_one_token_layer2_input_norm_full_trace.csv";
        end
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        if ($test$plusargs("trace_to_cwd")) begin
            if ($test$plusargs("true3_mmio_top_tail_only")) begin
                tracefile = "true3/timing_trace.csv";
            end
            else if ($test$plusargs("l1_l2_mmio_top_tail_only")) begin
                tracefile = "l1_l2/timing_trace.csv";
            end
            else begin
                tracefile = "timing_trace.csv";
            end
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
            $display("Using tracefile override: %s", tracefile);
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
        event_trace_fd = 0;
        if ($test$plusargs("embedding_true3") || $test$plusargs("embedding_full_chain")) begin
            if ($test$plusargs("persistent_two_token")) begin
                event_tracefile = "persistent_two_token_events.csv";
            end
            else if ($test$plusargs("embedding_full_chain")) begin
                event_tracefile = "embedding_full_chain_events.csv";
            end
            else begin
                event_tracefile = "embedding_true3_events.csv";
            end
            if ($value$plusargs("event_tracefile=%s", event_tracefile)) begin
                $display("Using event tracefile override: %s", event_tracefile);
            end
            event_trace_fd = $fopen(event_tracefile, "w");
            if (event_trace_fd == 0) begin
                $display("FAIL: could not open event trace file %s", event_tracefile);
                $finish(1);
            end
            $fwrite(event_trace_fd,
                    "cycle,event,address,index_or_kind,data_or_length,last\n");
        end

        persistent_reset_release_count = 0;
        rst_n = 1'b0;
        start = 1'b0;
`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
        mmio_reg_wr_valid = 1'b0;
        mmio_reg_rd_valid = 1'b0;
        mmio_reg_addr = 12'd0;
        mmio_reg_wdata = 32'd0;
`endif
        total_fail_count = 0;
        load_vectors();
`ifdef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
        if (persistent_two_token_mode) begin
            load_persistent_step_vectors(0);
        end
`endif
        clear_scoreboard();
        set_valid_loop_contract();
        if ($test$plusargs("fastmem")) begin
            fastmem = 1'b1;
        end
        if ($test$plusargs("l2_top_tail_only")) begin
`ifndef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $display("FAIL: +l2_top_tail_only requires compile define QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_TOP
            $display("FAIL: +l2_top_tail_only requires compile define QMAP_ONE_TOKEN_TB_USE_TOP");
            $finish(1);
`else
            set_layer_packet_bases(2);
            enable_layer_input_norm(2);
            layer_start_index = 5'd2;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer1_output_for_l2_only();
            clear_tail_scoreboard();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: qmap_one_token_top focused Layer2 scheduler -> final-token tail start");
            $display("  descriptor final hidden base = 0x%016h", tail_descriptor_final_hidden_base);
            $display("  top selected hidden base     = 0x%016h", tail_final_hidden_base);
            $display("  layer2 output base           = 0x%016h", layer_output_base_l2);
            $fflush();
            run_until_done(100000000);

            if ((layers_started != 5'd1) ||
                (layers_completed != 5'd1) ||
                (layer_done_mask != 28'h0000004) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: qmap_one_token_top Layer2 scheduler layer masks mismatch");
            end
            if ((top_scheduler_done_seen_count != 1) ||
                (top_tail_start_seen_count != 1) ||
                (tail_done_seen_count != 1) ||
                (done_seen_count != 1)) begin
                fail_once("FAIL: qmap_one_token_top control pulse count mismatch");
            end
            if (top_tail_effective_final_hidden_base !== layer_output_base_l2) begin
                tail_fail("FAIL: qmap_one_token_top tail effective hidden base mismatch");
            end
            if (scheduler_last_layer_output_base !== layer_output_base_l2) begin
                tail_fail("FAIL: qmap_one_token_top scheduler last layer output base mismatch");
            end
            if (tail_descriptor_final_hidden_base === layer_output_base_l2) begin
                tail_fail("FAIL: qmap_one_token_top test accidentally pre-patched the tail descriptor");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || error || (total_fail_count != 0)) begin
                $display("FAIL: qmap_one_token_top source failed mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end
            if (!layer_written) begin
                fail_once("FAIL: qmap_one_token_top scheduler did not mark final layer output written");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (dut_read_burst_count != EXPECTED_INPUT_NORM_FULL_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_FULL_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_FULL_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_FULL_WR_WORDS)) begin
                fail_once("FAIL: qmap_one_token_top Layer2 input RMSNorm scheduler counters mismatch");
            end
            if (tail_error) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail asserted error");
            end
            if (tail_norm_saturation) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail asserted norm saturation");
            end
            if ((tail_best_token_id !== tail_expected_token) ||
                (tail_best_score_q26 !== tail_expected_score_q26)) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail token/score mismatch");
            end
            if ((tail_tiles_started != TAIL_MAX_TILES) ||
                (tail_tiles_completed != TAIL_MAX_TILES)) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail tile count mismatch");
            end
            if ((tail_hidden_req_count != 4) ||
                (tail_gamma_req_count != 4) ||
                (tail_norm_read_req_count != 4)) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail hidden/gamma/norm read count mismatch");
            end
            if ((tail_weight_req_count != (TAIL_MAX_TILES * TAIL_WEIGHT_BURSTS_PER_TILE)) ||
                (tail_scale_req_count != (TAIL_MAX_TILES * TAIL_SCALE_BURSTS_PER_TILE))) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail LM-head weight/scale request count mismatch");
            end
            if ((tail_norm_write_word_count != TAIL_INPUT_SIZE) ||
                (tail_output_write_word_count != 3)) begin
                tail_fail("FAIL: qmap_one_token_top final-token tail write word count mismatch");
            end
            if ((tail_scheduler_done_cycle < 0) ||
                (tail_first_hidden_read_cycle <= tail_scheduler_done_cycle) ||
                (tail_first_hidden_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top tail hidden read did not occur after scheduler done");
            end
            if ((tail_first_weight_read_cycle <= tail_first_hidden_read_cycle) ||
                (tail_first_weight_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top tail LM-head read did not occur after hidden read");
            end
            if ((tail_first_output_write_cycle <= tail_first_weight_read_cycle) ||
                (tail_first_output_write_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top tail output write did not occur after LM-head reads");
            end
            if ((top_mem_read_burst_count != (dut_read_burst_count + tail_mem_read_burst_count)) ||
                (top_mem_read_word_count != (dut_read_word_count + tail_mem_read_word_count)) ||
                (top_mem_write_req_count != (dut_write_req_count + 32'd2)) ||
                (top_mem_write_word_count != (dut_write_word_count + tail_mem_write_word_count))) begin
                tail_fail("FAIL: qmap_one_token_top aggregate memory counters mismatch");
            end

            $display("qmap_one_token_top Layer2 input RMSNorm scheduler -> final-token tail test");
            $display("  scheduler done cycle   = %0d", tail_scheduler_done_cycle);
            $display("  tail done cycle        = %0d", tail_done_cycle);
            $display("  first hidden read      = %0d", tail_first_hidden_read_cycle);
            $display("  first LM-head read     = %0d", tail_first_weight_read_cycle);
            $display("  first output write     = %0d", tail_first_output_write_cycle);
            $display("  expected token/score   = %0d / %0d", tail_expected_token, tail_expected_score_q26);
            $display("  observed token/score   = %0d / %0d", tail_best_token_id, tail_best_score_q26);
            $display("  top pulses             = scheduler_done %0d tail_start %0d tail_done %0d top_done %0d",
                     top_scheduler_done_seen_count, top_tail_start_seen_count, tail_done_seen_count, done_seen_count);
            $display("  top rd/wr counters     = %0d/%0d reads, %0d/%0d writes",
                     top_mem_read_burst_count, top_mem_read_word_count,
                     top_mem_write_req_count, top_mem_write_word_count);
            $display("  tail read reqs         = hidden %0d gamma %0d norm %0d weight %0d scale %0d total %0d",
                     tail_hidden_req_count, tail_gamma_req_count, tail_norm_read_req_count,
                     tail_weight_req_count, tail_scale_req_count, tail_read_req_count);
            $display("  tail writes            = norm %0d output %0d write_mismatches %0d",
                     tail_norm_write_word_count, tail_output_write_word_count, tail_write_mismatch_count);
            $display("  trace                  = %s", tracefile);

            if ((tail_mismatch_count != 0) || (tail_write_mismatch_count != 0) ||
                (mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || (total_fail_count != 0)) begin
                $display("FAIL: qmap_one_token_top handoff found tail_mismatches=%0d tail_write_mismatches=%0d scheduler_mismatches=%0d scheduler_write_mismatches=%0d max_abs=%0d total_fail=%0d",
                         tail_mismatch_count, tail_write_mismatch_count,
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count);
                $finish(1);
            end
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            $display("PASS: qmap_one_token_top ran Layer2 scheduler output directly into final-token tail.");
            $finish;
`endif
`endif
        end
        if ($test$plusargs("persistent_two_token")) begin
`ifndef QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN
            $display("FAIL: +persistent_two_token requires compile define QMAP_ONE_TOKEN_TB_PERSISTENT_TWO_TOKEN");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $display("FAIL: +persistent_two_token requires compile define QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_TOP
            $display("FAIL: +persistent_two_token requires compile define QMAP_ONE_TOKEN_TB_USE_TOP");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
            $display("FAIL: +persistent_two_token requires compile define QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL");
            $finish(1);
`else
            if (!full_chain_mode || !runtime_context_mode ||
                (full_chain_layer_count < 1)) begin
                $display("FAIL: +persistent_two_token requires full-chain RuntimeContext vectors");
                $finish(1);
            end
            if ((tail_weight_base_addr !== EMBEDDING_WEIGHT_BASE_ADDR) ||
                (tail_scale_base_addr !== EMBEDDING_SCALE_BASE_ADDR)) begin
                $display(
                    "FAIL: persistent embedding and tied LM-head bases differ weight=0x%016h/0x%016h scale=0x%016h/0x%016h",
                    EMBEDDING_WEIGHT_BASE_ADDR, tail_weight_base_addr,
                    EMBEDDING_SCALE_BASE_ADDR, tail_scale_base_addr
                );
                $finish(1);
            end

            layer_start_index = 5'd0;
            layer_count = full_chain_layer_count;
            scoreboard_layer_iterations = full_chain_layer_count;
            use_input_norm_qkv_expected = 1'b1;
            clear_persistent_cache();
            clear_tail_scoreboard();

            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            mmio_load_current_loop_contract();
            mmio_check_scalar_registers();
            for (full_chain_check_index = 0;
                 full_chain_check_index < full_chain_layer_count;
                 full_chain_check_index = full_chain_check_index + 1) begin
                mmio_check_layer_tables(full_chain_check_index);
            end
            if (total_fail_count != 0) begin
                $finish(1);
            end

            if (event_trace_fd != 0) begin
                $fwrite(
                    event_trace_fd,
                    "%0d,persistent_step_start,0x0000000000000000,0,0x%08h,0\n",
                    cycle_count, persistent_input_token[0]
                );
            end
            $display(
                "PERSISTENT_STEP_START step=0 input_token=%0d position=0 layers=%0d",
                persistent_input_token[0], full_chain_layer_count
            );
            $fflush();
            // The routed BRAM-backed datapath takes about 6.8e8 cycles per
            // token: roughly 4.77e8 through all 28 layers plus 1.98e8 for the
            // row-streamed full-vocabulary tail. Keep enough watchdog margin
            // to catch a real stall without rejecting a healthy run.
            mmio_run_until_done(900000000);
            check_persistent_step_result(0);
            $display(
                "PERSISTENT_STEP_RESULT step=0 input_token=%0d position=0 output_token=%0d output_score=%0d",
                persistent_input_token[0], tail_best_token_id, tail_best_score_q26
            );
            if (event_trace_fd != 0) begin
                $fwrite(
                    event_trace_fd,
                    "%0d,persistent_step_done,0x0000000000000000,0,0x%08h,1\n",
                    cycle_count, tail_best_token_id
                );
            end

            mmio_read_reg(MMIO_REG_STATUS, mmio_read_data);
            if ((mmio_read_data & 32'h0000_000E) !== 32'h0000_0002) begin
                fail_once("FAIL: step 0 sticky done/error status mismatch");
                $finish(1);
            end
            mmio_write_reg(MMIO_REG_CTRL, 32'd2);
            mmio_read_reg(MMIO_REG_STATUS, mmio_read_data);
            if ((mmio_read_data & 32'h0000_000E) !== 32'd0) begin
                fail_once("FAIL: sticky clear changed or failed persistent status contract");
                $finish(1);
            end

            clear_scoreboard_for_next_token();
            load_persistent_step_vectors(1);
            token_position = 1;
            scoreboard_layer_iterations = full_chain_layer_count;
            use_input_norm_qkv_expected = 1'b1;
            clear_tail_scoreboard();

            mmio_write_reg(MMIO_REG_POSITION, 32'd1);
            mmio_write_reg(MMIO_REG_INPUT_TOKEN, persistent_input_token[1]);
            mmio_check_scalar_registers();
            for (full_chain_check_index = 0;
                 full_chain_check_index < full_chain_layer_count;
                 full_chain_check_index = full_chain_check_index + 1) begin
                mmio_check_layer_tables(full_chain_check_index);
            end
            if (total_fail_count != 0) begin
                $finish(1);
            end

            if (event_trace_fd != 0) begin
                $fwrite(
                    event_trace_fd,
                    "%0d,persistent_step_start,0x0000000000000000,1,0x%08h,0\n",
                    cycle_count, persistent_input_token[1]
                );
            end
            $display(
                "PERSISTENT_STEP_START step=1 input_token=%0d position=1 layers=%0d",
                persistent_input_token[1], full_chain_layer_count
            );
            $fflush();
            mmio_run_until_done(900000000);
            check_persistent_step_result(1);
            $display(
                "PERSISTENT_STEP_RESULT step=1 input_token=%0d position=1 output_token=%0d output_score=%0d",
                persistent_input_token[1], tail_best_token_id, tail_best_score_q26
            );
            if (event_trace_fd != 0) begin
                $fwrite(
                    event_trace_fd,
                    "%0d,persistent_step_done,0x0000000000000000,1,0x%08h,1\n",
                    cycle_count, tail_best_token_id
                );
            end

            if (persistent_reset_release_count != 1) begin
                $display(
                    "FAIL: persistent regression observed %0d reset releases; expected exactly one",
                    persistent_reset_release_count
                );
                $finish(1);
            end
            $display(
                "PERSISTENT_NO_RESET reset_release_count=%0d retained_layers=%0d retained_words_per_kind=%0d",
                persistent_reset_release_count, full_chain_layer_count, KV_COUNT
            );
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            if (event_trace_fd != 0) begin
                $fclose(event_trace_fd);
            end
            $display(
                "PASS: AXI-Lite RuntimeContext persistent two-token decode ran through %0d complete layers without reset.",
                full_chain_layer_count
            );
            $finish;
`endif
`endif
`endif
`endif
        end
        if ($test$plusargs("l1_l2_top_tail_only") ||
            $test$plusargs("l1_l2_mmio_top_tail_only") ||
            $test$plusargs("true3_top_tail_only") ||
            $test$plusargs("true3_mmio_top_tail_only")) begin
`ifndef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $display("FAIL: top-tail scenarios require compile define QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_TOP
            $display("FAIL: top-tail scenarios require compile define QMAP_ONE_TOKEN_TB_USE_TOP");
            $finish(1);
`else
`ifndef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
            if ($test$plusargs("l1_l2_mmio_top_tail_only") ||
                $test$plusargs("true3_mmio_top_tail_only")) begin
                $display("FAIL: MMIO top-tail scenarios require compile define QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL");
                $finish(1);
            end
`endif
            if (full_chain_mode) begin
                layer_start_index = 5'd0;
                layer_count = full_chain_layer_count;
                scoreboard_layer_iterations = full_chain_layer_count;
                use_input_norm_qkv_expected = 1'b1;
            end
            else begin
                set_layer_packet_bases(1);
                set_layer_packet_bases(2);
                enable_layer_input_norm(1);
                enable_layer_input_norm(2);

                if ($test$plusargs("true3_top_tail_only") ||
                    $test$plusargs("true3_mmio_top_tail_only")) begin
                    if (embedding_true3_mode) begin
                        enable_layer_input_norm(0);
                        use_input_norm_qkv_expected = 1'b1;
                        patch_qkv_base(QKV_SLOT_ACTIVATION, input_norm_output_base);
                        patch_post_base(POST_SLOT_RESIDUAL, layer_output_base);
                        input_hidden_base_addr = layer_output_base;
                    end
                    patch_qkv_base_l1(QKV_SLOT_ACTIVATION, input_norm_output_base_l1);
                    patch_qkv_base_l2(QKV_SLOT_ACTIVATION, input_norm_output_base_l2);
                    layer_start_index = 5'd0;
                    layer_count = 5'd3;
                    scoreboard_layer_iterations = 3;
                end
                else begin
                    layer_start_index = 5'd1;
                    layer_count = 5'd2;
                    scoreboard_layer_iterations = 2;
                    prefill_layer0_output_for_l1_only();
                end
            end

            clear_tail_scoreboard();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            if ($test$plusargs("l1_l2_mmio_top_tail_only") ||
                $test$plusargs("true3_mmio_top_tail_only")) begin
`ifdef QMAP_ONE_TOKEN_TB_USE_MMIO_CONTROL
                mmio_load_current_loop_contract();
                mmio_check_scalar_registers();
                if (full_chain_mode) begin
                    for (full_chain_check_index = 0;
                         full_chain_check_index < full_chain_layer_count;
                         full_chain_check_index = full_chain_check_index + 1) begin
                        mmio_check_layer_tables(full_chain_check_index);
                    end
                end
                else begin
                    if ($test$plusargs("true3_mmio_top_tail_only")) begin
                        mmio_check_layer_tables(0);
                    end
                    mmio_check_layer_tables(1);
                    mmio_check_layer_tables(2);
                end
                if (total_fail_count != 0) begin
                    $finish(1);
                end
                if ($test$plusargs("true3_mmio_top_tail_only")) begin
                    if (full_chain_mode) begin
                        $display("SCENARIO: qmap_one_token AXI-Lite tied-Q4 embedding -> %0d complete layers -> final-token tail",
                                 full_chain_layer_count);
                    end
                    else if (embedding_true3_mode) begin
                        $display("SCENARIO: qmap_one_token AXI-Lite tied-Q4 embedding -> Layer0 -> Layer1 -> Layer2 -> final-token tail");
                    end
                    else begin
                        $display("SCENARIO: qmap_one_token_top MMIO Layer0(QKV-first) -> Layer1 -> Layer2 input-RMSNorm scheduler -> final-token tail start");
                    end
                end
                else begin
                    $display("SCENARIO: qmap_one_token_top MMIO Layer1 -> Layer2 scheduler -> final-token tail start");
                end
                $display("  descriptor final hidden base = 0x%016h", tail_descriptor_final_hidden_base);
                $display("  mmio expected hidden base    = 0x%016h", tail_final_hidden_base);
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
                if (full_chain_mode) begin
                    $display("  first layer output base      = 0x%016h",
                             full_actual_layer_output_base[0]);
                    $display("  last layer output base       = 0x%016h", tail_final_hidden_base);
                end
                else
`endif
                begin
                    $display("  layer0 output base           = 0x%016h", layer_output_base);
                    $display("  layer1 output base           = 0x%016h", layer_output_base_l1);
                    $display("  layer2 output base           = 0x%016h", layer_output_base_l2);
                end
                $fflush();
            mmio_run_until_done(full_chain_mode ? 700000000 : 180000000);
`endif
            end
            else begin
                if ($test$plusargs("true3_top_tail_only")) begin
                    $display("SCENARIO: qmap_one_token_top Layer0(QKV-first) -> Layer1 -> Layer2 input-RMSNorm scheduler -> final-token tail start");
                end
                else begin
                    $display("SCENARIO: qmap_one_token_top Layer1 -> Layer2 scheduler -> final-token tail start");
                end
                $display("  descriptor final hidden base = 0x%016h", tail_descriptor_final_hidden_base);
                $display("  top expected hidden base     = 0x%016h", tail_final_hidden_base);
                $display("  layer0 output base           = 0x%016h", layer_output_base);
                $display("  layer1 output base           = 0x%016h", layer_output_base_l1);
                $display("  layer2 output base           = 0x%016h", layer_output_base_l2);
                $fflush();
                run_until_done(180000000);
            end

            if ($test$plusargs("l1_l2_mmio_top_tail_only") ||
                $test$plusargs("true3_mmio_top_tail_only")) begin
                $display("  mmio post-run debug: state=0x%0h phase=0x%0h error=%0d layers=%0d/%0d mask=0x%0h layer_error=0x%0h rd=%0d/%0d wr=%0d/%0d tail_hidden=0x%016h last_layer=0x%016h",
                         state_debug, top_phase_debug, error, layers_started, layers_completed,
                         layer_done_mask, layer_error_mask, dut_read_burst_count,
                         dut_read_word_count, dut_write_req_count, dut_write_word_count,
                         top_tail_effective_final_hidden_base, scheduler_last_layer_output_base);
            end

            if ((layers_started != layer_count) ||
                (layers_completed != layer_count) ||
                (layer_done_mask != expected_layer_mask(layer_start_index, layer_count)) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: qmap_one_token_top true top-tail scheduler layer masks mismatch");
            end
            if ((stage_done_mask != 2'b11) ||
                (stage_error_mask != 2'b00) ||
                (layer0_full_stage_done_mask != 4'hf) ||
                (layer0_full_stage_error_mask != 4'h0) ||
                (body_stage_done_mask != 5'h1f) ||
                (body_stage_error_mask != 5'h00)) begin
                fail_once("FAIL: qmap_one_token_top true top-tail scheduler stage masks mismatch");
            end
            if ((top_scheduler_done_seen_count != 1) ||
                (top_tail_start_seen_count != 1) ||
                (tail_done_seen_count != 1) ||
                (done_seen_count != 1)) begin
                fail_once("FAIL: qmap_one_token_top true top-tail control pulse count mismatch");
            end
            if (top_tail_effective_final_hidden_base !== tail_final_hidden_base) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail tail effective hidden base mismatch");
            end
            if (scheduler_last_layer_output_base !== tail_final_hidden_base) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail scheduler last layer output base mismatch");
            end
`ifdef QMAP_ONE_TOKEN_TB_FULL_CHAIN
            if (full_chain_mode && (full_chain_layer_count > 1) &&
                (scheduler_last_layer_output_base ===
                 full_actual_layer_output_base[full_chain_layer_count - 2])) begin
                tail_fail("FAIL: qmap_one_token_top full-chain selected the previous layer output");
            end
            else if (!full_chain_mode &&
                     (scheduler_last_layer_output_base === layer_output_base_l1)) begin
`else
            if (scheduler_last_layer_output_base === layer_output_base_l1) begin
`endif
                tail_fail("FAIL: qmap_one_token_top true top-tail selected the previous layer output");
            end
            if (!full_chain_mode &&
                ((tail_descriptor_final_hidden_base === layer_output_base_l1) ||
                 (tail_descriptor_final_hidden_base === layer_output_base_l2))) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail test accidentally pre-patched the tail descriptor");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || error || (total_fail_count != 0)) begin
                $display("FAIL: qmap_one_token_top true top-tail source failed mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end
            if (!layer_written) begin
                fail_once("FAIL: qmap_one_token_top true top-tail scheduler did not mark final layer output written");
            end
            if ((input_norm_write_accept_count != (layer_count * VEC1024)) ||
                (qkv_q_write_accept_count != (layer_count * VEC2048)) ||
                (qkv_k_write_accept_count != (layer_count * VEC1024)) ||
                (qkv_v_write_accept_count != (layer_count * VEC1024)) ||
                (cache_write_accept_count != (layer_count * TOTAL_CACHE_WRITES)) ||
                (q_rope_write_accept_count != (layer_count * VEC2048)) ||
                (attn_out_write_accept_count != (layer_count * VEC2048)) ||
                (o_proj_write_accept_count != (layer_count * VEC1024)) ||
                (post_hidden_write_accept_count != (layer_count * VEC1024)) ||
                (post_norm_write_accept_count != (layer_count * VEC1024)) ||
                (gate_write_accept_count != (layer_count * VEC3072)) ||
                (up_write_accept_count != (layer_count * VEC3072)) ||
                (silu_write_accept_count != (layer_count * VEC3072)) ||
                (down_write_accept_count != (layer_count * VEC1024)) ||
                (layer_write_accept_count != (layer_count * VEC1024))) begin
                fail_once("FAIL: qmap_one_token_top true top-tail producer write counts mismatch");
            end
            if (full_chain_mode) begin
                if ((dut_read_burst_count != (layer_count * EXPECTED_INPUT_NORM_FULL_RD_REQS)) ||
                    (dut_read_word_count != (layer_count * EXPECTED_INPUT_NORM_FULL_RD_WORDS)) ||
                    (dut_write_req_count != (layer_count * EXPECTED_INPUT_NORM_FULL_WR_REQS)) ||
                    (dut_write_word_count != (layer_count * EXPECTED_INPUT_NORM_FULL_WR_WORDS))) begin
                    fail_once("FAIL: embedding full-chain scheduler counters mismatch");
                end
            end
            else if (layer_count == 5'd3) begin
                if (embedding_true3_mode) begin
                    if ((dut_read_burst_count != EXPECTED_INPUT_NORM_THREE_LAYER_FULL_RD_REQS) ||
                        (dut_read_word_count != EXPECTED_INPUT_NORM_THREE_LAYER_FULL_RD_WORDS) ||
                        (dut_write_req_count != EXPECTED_INPUT_NORM_THREE_LAYER_FULL_WR_REQS) ||
                        (dut_write_word_count != EXPECTED_INPUT_NORM_THREE_LAYER_FULL_WR_WORDS)) begin
                        fail_once("FAIL: embedding true3 scheduler counters mismatch");
                    end
                end
                else if ((dut_read_burst_count != EXPECTED_MIXED_THREE_LAYER_FULL_RD_REQS) ||
                         (dut_read_word_count != EXPECTED_MIXED_THREE_LAYER_FULL_RD_WORDS) ||
                         (dut_write_req_count != EXPECTED_MIXED_THREE_LAYER_FULL_WR_REQS) ||
                         (dut_write_word_count != EXPECTED_MIXED_THREE_LAYER_FULL_WR_WORDS)) begin
                    fail_once("FAIL: qmap_one_token_top mixed true three-layer scheduler counters mismatch");
                end
            end
            else begin
                if ((dut_read_burst_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_REQS) ||
                    (dut_read_word_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_RD_WORDS) ||
                    (dut_write_req_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_REQS) ||
                    (dut_write_word_count != EXPECTED_INPUT_NORM_TWO_LAYER_FULL_WR_WORDS)) begin
                    fail_once("FAIL: qmap_one_token_top Layer1->Layer2 scheduler counters mismatch");
                end
            end
            if (tail_error) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail asserted error");
            end
            if (tail_norm_saturation) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail asserted norm saturation");
            end
            if ((tail_best_token_id !== tail_expected_token) ||
                (tail_best_score_q26 !== tail_expected_score_q26)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail token/score mismatch");
            end
            if ((tail_tiles_started != TAIL_MAX_TILES) ||
                (tail_tiles_completed != TAIL_MAX_TILES)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail tile count mismatch");
            end
            if ((tail_hidden_req_count != 4) ||
                (tail_gamma_req_count != 4) ||
                (tail_norm_read_req_count != 4)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail hidden/gamma/norm read count mismatch");
            end
            if ((tail_weight_req_count != (TAIL_MAX_TILES * TAIL_WEIGHT_BURSTS_PER_TILE)) ||
                (tail_scale_req_count != (TAIL_MAX_TILES * TAIL_SCALE_BURSTS_PER_TILE))) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail LM-head weight/scale request count mismatch");
            end
            if ((tail_norm_write_word_count != TAIL_INPUT_SIZE) ||
                (tail_output_write_word_count != 3)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail final-token tail write word count mismatch");
            end
            if ((tail_scheduler_done_cycle < 0) ||
                (tail_first_hidden_read_cycle <= tail_scheduler_done_cycle) ||
                (tail_first_hidden_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail tail hidden read did not occur after scheduler done");
            end
            if ((tail_first_weight_read_cycle <= tail_first_hidden_read_cycle) ||
                (tail_first_weight_read_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail tail LM-head read did not occur after hidden read");
            end
            if ((tail_first_output_write_cycle <= tail_first_weight_read_cycle) ||
                (tail_first_output_write_cycle < 0)) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail tail output write did not occur after LM-head reads");
            end
            if (full_chain_mode) begin
                if ((embedding_write_accept_count != VEC1024) || !embedding_written ||
                    !embedding_response_seen ||
                    (layer_response_seen != expected_layer_mask(0, full_chain_layer_count))) begin
                    fail_once("FAIL: embedding full-chain producer completion flags mismatch");
                end
                if ((embedding_response_cycle < 0) ||
                    (first_layer_hidden_read_cycle[0] <= embedding_response_cycle)) begin
                    fail_once("FAIL: embedding response did not precede Layer 0 hidden read");
                end
                for (full_chain_check_index = 0;
                     full_chain_check_index < full_chain_layer_count;
                     full_chain_check_index = full_chain_check_index + 1) begin
                    if (layer_response_cycle[full_chain_check_index] < 0) begin
                        fail_once("FAIL: full-chain layer write response was not observed");
                    end
                    if ((full_chain_check_index > 0) &&
                        (first_layer_hidden_read_cycle[full_chain_check_index] <=
                         layer_response_cycle[full_chain_check_index - 1])) begin
                        fail_once("FAIL: full-chain layer consumed hidden data before previous response");
                    end
                    if (runtime_context_mode) begin
                        if ((full_actual_input_hidden_base[full_chain_check_index] !==
                             ((full_chain_check_index[0] == 1'b0) ?
                              full_chain_runtime_hidden_a_base_addr :
                              full_chain_runtime_hidden_b_base_addr)) ||
                            (full_actual_layer_output_base[full_chain_check_index] !==
                             ((full_chain_check_index[0] == 1'b0) ?
                              full_chain_runtime_hidden_b_base_addr :
                              full_chain_runtime_hidden_a_base_addr))) begin
                            fail_once("FAIL: runtime hidden A/B effective address parity mismatch");
                        end
                        if ((runtime_hidden_read_req_count[full_chain_check_index] <= 0) ||
                            (runtime_layer_output_write_req_count[full_chain_check_index] != 1)) begin
                            fail_once("FAIL: runtime hidden effective read/write address coverage mismatch");
                        end
                        if (runtime_rope_read_req_count[full_chain_check_index] != 2) begin
                            fail_once("FAIL: runtime RoPE did not issue exactly one cos and one sin row read");
                        end
                        if ((runtime_cache_read_req_count[full_chain_check_index] <= 0) ||
                            (runtime_cache_write_word_count[full_chain_check_index] !=
                             TOTAL_CACHE_WRITES)) begin
                            fail_once("FAIL: runtime K/V cache read/write coverage mismatch");
                        end
                    end
                end
                if (runtime_context_mode &&
                    ((full_chain_embedding_output_base_addr !==
                      full_chain_runtime_hidden_a_base_addr) ||
                     (full_chain_runtime_last_hidden_base_addr !==
                      full_actual_layer_output_base[full_chain_layer_count - 1]) ||
                     (tail_final_hidden_base !== full_chain_runtime_last_hidden_base_addr))) begin
                    fail_once("FAIL: runtime embedding/layer/tail effective hidden handoff mismatch");
                end
                if ((tail_scheduler_done_cycle <=
                     layer_response_cycle[full_chain_layer_count - 1]) ||
                    (tail_start_cycle <= tail_scheduler_done_cycle) ||
                    (tail_first_hidden_read_cycle <= tail_start_cycle)) begin
                    fail_once("FAIL: full-chain scheduler/tail response ordering mismatch");
                end
                if ((top_mem_read_burst_count !=
                     ((layer_count * EXPECTED_INPUT_NORM_FULL_RD_REQS) +
                      EXPECTED_EMBEDDING_TAIL_EXTRA_RD_REQS)) ||
                    (top_mem_read_word_count !=
                     ((layer_count * EXPECTED_INPUT_NORM_FULL_RD_WORDS) +
                      EXPECTED_EMBEDDING_TAIL_EXTRA_RD_WORDS)) ||
                    (top_mem_write_req_count !=
                     ((layer_count * EXPECTED_INPUT_NORM_FULL_WR_REQS) +
                      EXPECTED_EMBEDDING_TAIL_EXTRA_WR_REQS)) ||
                    (top_mem_write_word_count !=
                     ((layer_count * EXPECTED_INPUT_NORM_FULL_WR_WORDS) +
                      EXPECTED_EMBEDDING_TAIL_EXTRA_WR_WORDS))) begin
                    tail_fail("FAIL: embedding full-chain aggregate memory counters mismatch");
                end
            end
            else if (embedding_true3_mode) begin
                if ((embedding_write_accept_count != VEC1024) || !embedding_written ||
                    !embedding_response_seen || (layer_response_seen != 3'b111)) begin
                    fail_once("FAIL: embedding true3 producer completion flags mismatch");
                end
                if ((embedding_response_cycle < 0) ||
                    (first_embedding_hidden_read_cycle <= embedding_response_cycle) ||
                    (layer_response_cycle[0] < 0) ||
                    (first_layer1_hidden_read_cycle <= layer_response_cycle[0]) ||
                    (layer_response_cycle[1] < 0) ||
                    (first_layer2_hidden_read_cycle <= layer_response_cycle[1]) ||
                    (layer_response_cycle[2] < 0) ||
                    (tail_scheduler_done_cycle <= layer_response_cycle[2]) ||
                    (tail_start_cycle <= tail_scheduler_done_cycle) ||
                    (tail_first_hidden_read_cycle <= tail_start_cycle)) begin
                    fail_once("FAIL: embedding true3 producer/consumer response ordering mismatch");
                end
                if ((top_mem_read_burst_count != EXPECTED_EMBEDDING_TRUE3_TOP_RD_REQS) ||
                    (top_mem_read_word_count != EXPECTED_EMBEDDING_TRUE3_TOP_RD_WORDS) ||
                    (top_mem_write_req_count != EXPECTED_EMBEDDING_TRUE3_TOP_WR_REQS) ||
                    (top_mem_write_word_count != EXPECTED_EMBEDDING_TRUE3_TOP_WR_WORDS)) begin
                    tail_fail("FAIL: embedding true3 aggregate memory counters mismatch");
                end
            end
            else if ((top_mem_read_burst_count != (dut_read_burst_count + tail_mem_read_burst_count)) ||
                     (top_mem_read_word_count != (dut_read_word_count + tail_mem_read_word_count)) ||
                     (top_mem_write_req_count != (dut_write_req_count + 32'd2)) ||
                     (top_mem_write_word_count != (dut_write_word_count + tail_mem_write_word_count))) begin
                tail_fail("FAIL: qmap_one_token_top true top-tail aggregate memory counters mismatch");
            end

            $display("qmap_one_token_top true input RMSNorm scheduler -> final-token tail test");
            $display("  layer start/count      = %0d / %0d", layer_start_index, layer_count);
            $display("  scheduler done cycle   = %0d", tail_scheduler_done_cycle);
            $display("  tail done cycle        = %0d", tail_done_cycle);
            $display("  first hidden read      = %0d", tail_first_hidden_read_cycle);
            $display("  first LM-head read     = %0d", tail_first_weight_read_cycle);
            $display("  first output write     = %0d", tail_first_output_write_cycle);
            if (full_chain_mode) begin
                $display("  embedding response     = %0d, Layer0 first hidden read = %0d",
                         embedding_response_cycle, first_layer_hidden_read_cycle[0]);
                $display("  first/last layer resp  = %0d / %0d",
                         layer_response_cycle[0],
                         layer_response_cycle[full_chain_layer_count - 1]);
                $display("  last layer first read  = %0d",
                         first_layer_hidden_read_cycle[full_chain_layer_count - 1]);
                if (runtime_context_mode) begin
                    $display("  runtime position       = %0d", full_chain_runtime_position);
                    $display("  runtime hidden A/B     = 0x%016h / 0x%016h",
                             full_chain_runtime_hidden_a_base_addr,
                             full_chain_runtime_hidden_b_base_addr);
                    $display("  runtime final hidden   = 0x%016h",
                             full_chain_runtime_last_hidden_base_addr);
                end
            end
            else if (embedding_true3_mode) begin
                $display("  embedding response     = %0d, Layer0 first hidden read = %0d",
                         embedding_response_cycle, first_embedding_hidden_read_cycle);
                $display("  layer responses        = %0d / %0d / %0d",
                         layer_response_cycle[0], layer_response_cycle[1], layer_response_cycle[2]);
                $display("  next-layer first reads = %0d / %0d",
                         first_layer1_hidden_read_cycle, first_layer2_hidden_read_cycle);
            end
            $display("  expected token/score   = %0d / %0d", tail_expected_token, tail_expected_score_q26);
            $display("  observed token/score   = %0d / %0d", tail_best_token_id, tail_best_score_q26);
            $display("  top pulses             = scheduler_done %0d tail_start %0d tail_done %0d top_done %0d",
                     top_scheduler_done_seen_count, top_tail_start_seen_count, tail_done_seen_count, done_seen_count);
            $display("  scheduler rd/wr        = %0d/%0d reads, %0d/%0d writes mask 0x%0h",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count,
                     layer_done_mask);
            $display("  top rd/wr counters     = %0d/%0d reads, %0d/%0d writes",
                     top_mem_read_burst_count, top_mem_read_word_count,
                     top_mem_write_req_count, top_mem_write_word_count);
            $display("  tail read reqs         = hidden %0d gamma %0d norm %0d weight %0d scale %0d total %0d",
                     tail_hidden_req_count, tail_gamma_req_count, tail_norm_read_req_count,
                     tail_weight_req_count, tail_scale_req_count, tail_read_req_count);
            $display("  tail writes            = norm %0d output %0d write_mismatches %0d",
                     tail_norm_write_word_count, tail_output_write_word_count, tail_write_mismatch_count);
            $display("  trace                  = %s", tracefile);

            if ((tail_mismatch_count != 0) || (tail_write_mismatch_count != 0) ||
                (mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || (total_fail_count != 0)) begin
                $display("FAIL: qmap_one_token_top true top-tail handoff found tail_mismatches=%0d tail_write_mismatches=%0d scheduler_mismatches=%0d scheduler_write_mismatches=%0d max_abs=%0d total_fail=%0d",
                         tail_mismatch_count, tail_write_mismatch_count,
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count);
                $finish(1);
            end
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            if (event_trace_fd != 0) begin
                $fclose(event_trace_fd);
            end
            if (full_chain_mode) begin
                $display("PASS: AXI-Lite tied-Q4 embedding ran through %0d complete layers and the full-vocabulary final-token tail exactly.",
                         full_chain_layer_count);
            end
            else if (layer_count == 5'd3) begin
                if (embedding_true3_mode) begin
                    $display("PASS: AXI-Lite tied-Q4 embedding ran through three complete layers and the full-vocabulary final-token tail exactly.");
                end
                else begin
                    $display("PASS: qmap_one_token_top ran Layer0(QKV-first), Layer1, and Layer2 before automatically feeding Layer2 output into final-token tail.");
                end
            end
            else begin
                $display("PASS: qmap_one_token_top ran Layer1 and Layer2 before automatically feeding Layer2 output into final-token tail.");
            end
            $finish;
`endif
`endif
        end
        if ($test$plusargs("input_norm_qkv_only")) begin
            enable_layer_input_norm(0);
            use_input_norm_qkv_expected = 1'b1;
            patch_qkv_base(QKV_SLOT_ACTIVATION, input_norm_output_base);
            corrupt_frontend_cos_dtype();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused input RMSNorm -> QKV precheck start");
            $display("  input norm output base = 0x%016h", input_norm_output_base);
            $fflush();
            run_until_done(12000000);
            input_norm_qkv_done_cycle = cycle_count;
            input_norm_qkv_read_bursts = dut_read_burst_count;
            input_norm_qkv_read_words = dut_read_word_count;
            input_norm_qkv_write_reqs = dut_write_req_count;
            input_norm_qkv_write_words = dut_write_word_count;
            input_norm_qkv_stage_done_mask = stage_done_mask;
            input_norm_qkv_stage_error_mask = stage_error_mask;
            input_norm_qkv_layer0_full_stage_done_mask = layer0_full_stage_done_mask;
            input_norm_qkv_layer0_full_stage_error_mask = layer0_full_stage_error_mask;
            input_norm_qkv_error = error;

            if (!input_norm_qkv_error) begin
                fail_once("FAIL: input RMSNorm -> QKV precheck did not stop at invalid frontend descriptor");
            end
            if ((input_norm_qkv_stage_done_mask != 2'b01) ||
                (input_norm_qkv_stage_error_mask != 2'b10) ||
                (input_norm_qkv_layer0_full_stage_done_mask != 4'h0) ||
                (input_norm_qkv_layer0_full_stage_error_mask != 4'h1)) begin
                fail_once("FAIL: input RMSNorm -> QKV precheck stage masks mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (qkv_q_write_accept_count != VEC2048) ||
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
                fail_once("FAIL: input RMSNorm -> QKV precheck producer write counts mismatch");
            end
            if ((input_norm_qkv_write_reqs != EXPECTED_INPUT_NORM_QKV_WR_REQS) ||
                (input_norm_qkv_write_words != EXPECTED_INPUT_NORM_QKV_WR_WORDS) ||
                (input_norm_qkv_read_bursts != EXPECTED_INPUT_NORM_QKV_RD_REQS) ||
                (input_norm_qkv_read_words != EXPECTED_INPUT_NORM_QKV_RD_WORDS)) begin
                fail_once("FAIL: input RMSNorm -> QKV precheck memory counters mismatch");
            end
`ifdef QMAP_ONE_TOKEN_TB_USE_TOP
            if ((top_tail_start_seen_count != 0) ||
                (tail_done_seen_count != 0)) begin
                fail_once("FAIL: input RMSNorm -> QKV precheck unexpectedly started final tail");
            end
`endif
            if ((mismatch_count != 0) ||
                (write_mismatch_count != 0) ||
                (max_abs_diff != 0) ||
                (total_fail_count != 0)) begin
                $display("FAIL: input RMSNorm -> QKV precheck found mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count);
                $finish(1);
            end

            $display("qmap_one_token_layer_scheduler input RMSNorm -> QKV precheck");
            $display("  done cycle        = %0d", input_norm_qkv_done_cycle);
            $display("  input_norm writes = %0d", input_norm_write_accept_count);
            $display("  qkv writes        = q %0d k %0d v %0d",
                     qkv_q_write_accept_count,
                     qkv_k_write_accept_count,
                     qkv_v_write_accept_count);
            $display("  rd/wr             = %0d/%0d reads, %0d/%0d writes",
                     input_norm_qkv_read_bursts,
                     input_norm_qkv_read_words,
                     input_norm_qkv_write_reqs,
                     input_norm_qkv_write_words);
            $display("  stage masks       = done 0x%0h error 0x%0h layer0_full done 0x%0h error 0x%0h",
                     input_norm_qkv_stage_done_mask,
                     input_norm_qkv_stage_error_mask,
                     input_norm_qkv_layer0_full_stage_done_mask,
                     input_norm_qkv_layer0_full_stage_error_mask);
            if (trace_fd == 0) begin
                $display("  trace             = disabled by +notrace");
            end
            else begin
                $display("  trace             = %s", tracefile);
            end
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            $display("PASS: qmap_one_token_layer_scheduler chained input RMSNorm output into QKV.");
            $finish;
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
        if ($test$plusargs("l1_input_norm_only")) begin
            set_layer_packet_bases(1);
            enable_layer_input_norm(1);
            layer_start_index = 5'd1;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer0_output_for_l1_only();
            corrupt_frontend_cos_dtype_l1();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer1 input RMSNorm -> QKV precheck start");
            $display("  input norm output base = 0x%016h", input_norm_output_base_l1);
            $fflush();
            run_until_done(12000000);

            if (!error) begin
                fail_once("FAIL: focused Layer1 input RMSNorm precheck did not stop at invalid frontend descriptor");
            end
            if ((stage_done_mask != 2'b01) ||
                (stage_error_mask != 2'b10)) begin
                fail_once("FAIL: focused Layer1 input RMSNorm precheck stage masks mismatch");
            end
            if ((dut_read_burst_count != EXPECTED_INPUT_NORM_QKV_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_QKV_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_QKV_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_QKV_WR_WORDS)) begin
                fail_once("FAIL: focused Layer1 input RMSNorm precheck memory counters mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (qkv_q_write_accept_count != VEC2048) ||
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
                fail_once("FAIL: focused Layer1 input RMSNorm producer write counts mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || (total_fail_count != 0)) begin
                $display("FAIL: focused Layer1 input RMSNorm found mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end

            $display("qmap_one_token_layer_scheduler focused Layer1 input RMSNorm -> QKV precheck");
            $display("  done cycle        = %0d", cycle_count);
            $display("  input_norm writes = %0d", input_norm_write_accept_count);
            $display("  qkv writes        = q %0d k %0d v %0d",
                     qkv_q_write_accept_count,
                     qkv_k_write_accept_count,
                     qkv_v_write_accept_count);
            $display("  rd/wr             = %0d/%0d reads, %0d/%0d writes",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count);
            if (trace_fd != 0) begin
                $display("  trace             = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace             = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran Layer1 input RMSNorm output into Layer1 QKV.");
            $finish;
        end
        if ($test$plusargs("l2_input_norm_only")) begin
            set_layer_packet_bases(2);
            enable_layer_input_norm(2);
            layer_start_index = 5'd2;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer1_output_for_l2_only();
            corrupt_frontend_cos_dtype_l2();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer2 input RMSNorm -> QKV precheck start");
            $display("  input norm output base = 0x%016h", input_norm_output_base_l2);
            $fflush();
            run_until_done(12000000);

            if (!error) begin
                fail_once("FAIL: focused Layer2 input RMSNorm precheck did not stop at invalid frontend descriptor");
            end
            if ((stage_done_mask != 2'b01) ||
                (stage_error_mask != 2'b10)) begin
                fail_once("FAIL: focused Layer2 input RMSNorm precheck stage masks mismatch");
            end
            if ((dut_read_burst_count != EXPECTED_INPUT_NORM_QKV_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_QKV_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_QKV_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_QKV_WR_WORDS)) begin
                fail_once("FAIL: focused Layer2 input RMSNorm precheck memory counters mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (qkv_q_write_accept_count != VEC2048) ||
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
                fail_once("FAIL: focused Layer2 input RMSNorm producer write counts mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || (total_fail_count != 0)) begin
                $display("FAIL: focused Layer2 input RMSNorm found mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end

            $display("qmap_one_token_layer_scheduler focused Layer2 input RMSNorm -> QKV precheck");
            $display("  done cycle        = %0d", cycle_count);
            $display("  input_norm writes = %0d", input_norm_write_accept_count);
            $display("  qkv writes        = q %0d k %0d v %0d",
                     qkv_q_write_accept_count,
                     qkv_k_write_accept_count,
                     qkv_v_write_accept_count);
            $display("  rd/wr             = %0d/%0d reads, %0d/%0d writes",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count);
            if (trace_fd != 0) begin
                $display("  trace             = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace             = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran Layer2 input RMSNorm output into Layer2 QKV.");
            $finish;
        end
        if ($test$plusargs("l1_only") || $test$plusargs("l1_input_norm_full_only")) begin
            set_layer_packet_bases(1);
            enable_layer_input_norm(1);
            layer_start_index = 5'd1;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer0_output_for_l1_only();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer1 input RMSNorm -> full layer start");
            $fflush();
            run_until_done(25000000);
            if ((layers_started != 5'd1) ||
                (layers_completed != 5'd1) ||
                (layer_done_mask != 28'h0000002) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: focused Layer1 input RMSNorm full-layer masks mismatch");
            end
            if ((stage_done_mask != 2'b11) ||
                (stage_error_mask != 2'b00) ||
                (layer0_full_stage_done_mask != 4'hf) ||
                (layer0_full_stage_error_mask != 4'h0) ||
                (body_stage_done_mask != 5'h1f) ||
                (body_stage_error_mask != 5'h00)) begin
                fail_once("FAIL: focused Layer1 input RMSNorm full-layer stage masks mismatch");
            end
            if ((dut_read_burst_count != EXPECTED_INPUT_NORM_FULL_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_FULL_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_FULL_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_FULL_WR_WORDS)) begin
                fail_once("FAIL: focused Layer1 input RMSNorm full-layer memory counters mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (qkv_q_write_accept_count != VEC2048) ||
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
                fail_once("FAIL: focused Layer1 input RMSNorm full-layer producer write counts mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) || (max_abs_diff != 0) ||
                error || (total_fail_count != 0)) begin
                $display("FAIL: focused Layer1 input RMSNorm full-layer found mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end
            $display("qmap_one_token_layer_scheduler focused Layer1 input RMSNorm -> full layer test");
            $display("  done cycle        = %0d", cycle_count);
            $display("  input_norm writes = %0d", input_norm_write_accept_count);
            $display("  rd/wr             = %0d/%0d reads, %0d/%0d writes",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count);
            $display("  layer mask        = done 0x%0h error 0x%0h", layer_done_mask, layer_error_mask);
            if (trace_fd != 0) begin
                $display("  trace             = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace             = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran Layer1 input RMSNorm through full layer write-back.");
            $finish;
        end
        if ($test$plusargs("l2_only") || $test$plusargs("l2_input_norm_full_only")) begin
            set_layer_packet_bases(2);
            enable_layer_input_norm(2);
            layer_start_index = 5'd2;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer1_output_for_l2_only();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer2 input RMSNorm -> full layer start");
            $fflush();
            run_until_done(25000000);
            if ((layers_started != 5'd1) ||
                (layers_completed != 5'd1) ||
                (layer_done_mask != 28'h0000004) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: focused Layer2 input RMSNorm full-layer masks mismatch");
            end
            if ((stage_done_mask != 2'b11) ||
                (stage_error_mask != 2'b00) ||
                (layer0_full_stage_done_mask != 4'hf) ||
                (layer0_full_stage_error_mask != 4'h0) ||
                (body_stage_done_mask != 5'h1f) ||
                (body_stage_error_mask != 5'h00)) begin
                fail_once("FAIL: focused Layer2 input RMSNorm full-layer stage masks mismatch");
            end
            if ((dut_read_burst_count != EXPECTED_INPUT_NORM_FULL_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_FULL_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_FULL_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_FULL_WR_WORDS)) begin
                fail_once("FAIL: focused Layer2 input RMSNorm full-layer memory counters mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (qkv_q_write_accept_count != VEC2048) ||
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
                fail_once("FAIL: focused Layer2 input RMSNorm full-layer producer write counts mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) || (max_abs_diff != 0) ||
                error || (total_fail_count != 0)) begin
                $display("FAIL: focused Layer2 input RMSNorm full-layer found mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end
            $display("qmap_one_token_layer_scheduler focused Layer2 input RMSNorm -> full layer test");
            $display("  done cycle        = %0d", cycle_count);
            $display("  input_norm writes = %0d", input_norm_write_accept_count);
            $display("  rd/wr             = %0d/%0d reads, %0d/%0d writes",
                     dut_read_burst_count, dut_read_word_count,
                     dut_write_req_count, dut_write_word_count);
            $display("  layer mask        = done 0x%0h error 0x%0h", layer_done_mask, layer_error_mask);
            if (trace_fd != 0) begin
                $display("  trace             = %s", tracefile);
                $fclose(trace_fd);
            end
            else begin
                $display("  trace             = disabled by +notrace");
            end
            $display("PASS: qmap_one_token_layer_scheduler ran Layer2 input RMSNorm through full layer write-back.");
            $finish;
        end
        if ($test$plusargs("l2_tail_only")) begin
`ifndef QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL
            $display("FAIL: +l2_tail_only requires compile define QMAP_ONE_TOKEN_TB_WITH_FINAL_TAIL");
            $finish(1);
`else
            set_layer_packet_bases(2);
            enable_layer_input_norm(2);
            layer_start_index = 5'd2;
            layer_count = 5'd1;
            scoreboard_layer_iterations = 1;
            prefill_layer1_output_for_l2_only();
            repeat (10) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: focused Layer2 scheduler -> final-token tail start");
            $display("  tail final hidden base = 0x%016h", tail_final_hidden_base);
            $display("  layer2 output base     = 0x%016h", layer_output_base_l2);
            $fflush();
            run_until_done(25000000);
            tail_scheduler_done_cycle = cycle_count;
            if ((layers_started != 5'd1) ||
                (layers_completed != 5'd1) ||
                (layer_done_mask != 28'h0000004) ||
                (layer_error_mask != 28'h0000000)) begin
                fail_once("FAIL: Layer2 scheduler handoff layer masks mismatch");
            end
            if ((mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || error || (total_fail_count != 0)) begin
                $display("FAIL: Layer2 scheduler handoff source failed mismatch_count=%0d write_mismatches=%0d max_abs=%0d total_fail=%0d error=%0d",
                         mismatch_count, write_mismatch_count, max_abs_diff, total_fail_count, error);
                $finish(1);
            end
            if (!layer_written) begin
                fail_once("FAIL: Layer2 scheduler did not mark final layer output written");
            end
            if (scheduler_last_layer_output_base !== layer_output_base_l2) begin
                fail_once("FAIL: Layer2 scheduler last layer output base mismatch");
            end
            if ((input_norm_write_accept_count != VEC1024) ||
                (dut_read_burst_count != EXPECTED_INPUT_NORM_FULL_RD_REQS) ||
                (dut_read_word_count != EXPECTED_INPUT_NORM_FULL_RD_WORDS) ||
                (dut_write_req_count != EXPECTED_INPUT_NORM_FULL_WR_REQS) ||
                (dut_write_word_count != EXPECTED_INPUT_NORM_FULL_WR_WORDS)) begin
                fail_once("FAIL: Layer2 input RMSNorm scheduler handoff counters mismatch");
            end
            if (read_active || write_active || mem_rd_rsp_valid) begin
                fail_once("FAIL: memory model still had active traffic before tail handoff");
            end
            clear_tail_scoreboard();
            tail_scheduler_done_cycle = cycle_count;
            use_tail_mem_manual = 1'b1;
            repeat (4) @(posedge clk);

            $display("SCENARIO: final-token tail consuming scheduler-written Layer2 output");
            $fflush();
            run_tail_until_done(90000000);

            if (tail_error) begin
                tail_fail("FAIL: final-token tail asserted error");
            end
            if (tail_norm_saturation) begin
                tail_fail("FAIL: final-token tail asserted norm saturation");
            end
            if ((tail_best_token_id !== tail_expected_token) ||
                (tail_best_score_q26 !== tail_expected_score_q26)) begin
                tail_fail("FAIL: final-token tail token/score mismatch");
            end
            if ((tail_tiles_started != TAIL_MAX_TILES) ||
                (tail_tiles_completed != TAIL_MAX_TILES)) begin
                tail_fail("FAIL: final-token tail tile count mismatch");
            end
            if ((tail_hidden_req_count != 4) ||
                (tail_gamma_req_count != 4) ||
                (tail_norm_read_req_count != 4)) begin
                tail_fail("FAIL: final-token tail hidden/gamma/norm read count mismatch");
            end
            if ((tail_weight_req_count != (TAIL_MAX_TILES * TAIL_WEIGHT_BURSTS_PER_TILE)) ||
                (tail_scale_req_count != (TAIL_MAX_TILES * TAIL_SCALE_BURSTS_PER_TILE))) begin
                tail_fail("FAIL: final-token tail LM-head weight/scale request count mismatch");
            end
            if ((tail_norm_write_word_count != TAIL_INPUT_SIZE) ||
                (tail_output_write_word_count != 3)) begin
                tail_fail("FAIL: final-token tail write word count mismatch");
            end
            if ((tail_first_hidden_read_cycle <= tail_scheduler_done_cycle) ||
                (tail_first_hidden_read_cycle < 0)) begin
                tail_fail("FAIL: final-token tail hidden read did not occur after scheduler done");
            end
            if ((tail_first_weight_read_cycle <= tail_first_hidden_read_cycle) ||
                (tail_first_weight_read_cycle < 0)) begin
                tail_fail("FAIL: final-token tail LM-head read did not occur after hidden read");
            end
            if ((tail_first_output_write_cycle <= tail_first_weight_read_cycle) ||
                (tail_first_output_write_cycle < 0)) begin
                tail_fail("FAIL: final-token tail output write did not occur after LM-head reads");
            end
            if (tail_done_seen_count != 1) begin
                tail_fail("FAIL: final-token tail done pulse count mismatch");
            end

            $display("qmap_one_token_layer_scheduler Layer2 input RMSNorm -> qmap_final_token_tail_compute_path handoff test");
            $display("  scheduler done cycle   = %0d", tail_scheduler_done_cycle);
            $display("  tail done cycle        = %0d", tail_done_cycle);
            $display("  first hidden read      = %0d", tail_first_hidden_read_cycle);
            $display("  first LM-head read     = %0d", tail_first_weight_read_cycle);
            $display("  first output write     = %0d", tail_first_output_write_cycle);
            $display("  expected token/score   = %0d / %0d", tail_expected_token, tail_expected_score_q26);
            $display("  observed token/score   = %0d / %0d", tail_best_token_id, tail_best_score_q26);
            $display("  tail read reqs         = hidden %0d gamma %0d norm %0d weight %0d scale %0d total %0d",
                     tail_hidden_req_count, tail_gamma_req_count, tail_norm_read_req_count,
                     tail_weight_req_count, tail_scale_req_count, tail_read_req_count);
            $display("  tail writes            = norm %0d output %0d write_mismatches %0d",
                     tail_norm_write_word_count, tail_output_write_word_count, tail_write_mismatch_count);
            $display("  trace                  = %s", tracefile);

            if ((tail_mismatch_count != 0) || (tail_write_mismatch_count != 0) ||
                (mismatch_count != 0) || (write_mismatch_count != 0) ||
                (max_abs_diff != 0) || (total_fail_count != 0)) begin
                $display("FAIL: scheduler-to-tail handoff found tail_mismatch=%0d tail_write_mismatch=%0d mismatch=%0d write_mismatch=%0d max_abs=%0d total_fail=%0d",
                         tail_mismatch_count, tail_write_mismatch_count, mismatch_count,
                         write_mismatch_count, max_abs_diff, total_fail_count);
                $finish(1);
            end
            if (trace_fd != 0) begin
                $fclose(trace_fd);
            end
            $display("PASS: Layer2 scheduler output was consumed directly by final-token tail in one shared-memory RTL simulation.");
            $finish;
`endif
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
