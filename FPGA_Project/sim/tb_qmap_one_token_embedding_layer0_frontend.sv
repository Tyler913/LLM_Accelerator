`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_one_token_embedding_layer0_frontend;

    localparam int ADDR_WIDTH = 64;
    localparam int MAX_LAYERS = 28;
    localparam int BASE_TABLE_BITS = MAX_LAYERS * ADDR_WIDTH;
    localparam int LAYER_INDEX_WIDTH = $clog2(MAX_LAYERS);
    localparam int LAYER_COUNT_WIDTH = $clog2(MAX_LAYERS + 1);

    localparam int EMBED_WEIGHT_WORDS = 128;
    localparam int EMBED_SCALE_WORDS = 8;
    localparam int VEC1024 = 1024;
    localparam int VEC2048 = 2048;
    localparam int VEC3072 = 3072;
    localparam int QKV_EXPECTED_WORDS = 4096;
    localparam int QKV_IMAGE_BYTES = 32'h0022_B000;
    localparam int QKV_IMAGE_WORDS = QKV_IMAGE_BYTES / 4;
    localparam int INPUT_NORM_IMAGE_BYTES = 32'h0000_5000;
    localparam int INPUT_NORM_IMAGE_WORDS = INPUT_NORM_IMAGE_BYTES / 4;
    localparam int FRONT_IMAGE_BYTES = 32'h0000_8000;
    localparam int FRONT_IMAGE_WORDS = FRONT_IMAGE_BYTES / 4;
    localparam int SCORE_IMAGE_BYTES = 32'h0000_5000;
    localparam int SCORE_IMAGE_WORDS = SCORE_IMAGE_BYTES / 4;
    localparam int OPROJ_IMAGE_BYTES = 32'h0000_5000;
    localparam int OPROJ_IMAGE_WORDS = OPROJ_IMAGE_BYTES / 4;
    localparam int POST_IMAGE_BYTES = 32'h0000_8000;
    localparam int POST_IMAGE_WORDS = POST_IMAGE_BYTES / 4;
    localparam int GATE_IMAGE_BYTES = 32'h0000_E000;
    localparam int GATE_IMAGE_WORDS = GATE_IMAGE_BYTES / 4;
    localparam int SILU_IMAGE_BYTES = 32'h0000_E000;
    localparam int SILU_IMAGE_WORDS = SILU_IMAGE_BYTES / 4;
    localparam int DOWN_IMAGE_BYTES = 32'h0000_6000;
    localparam int DOWN_IMAGE_WORDS = DOWN_IMAGE_BYTES / 4;
    localparam int RESIDUAL_IMAGE_BYTES = 32'h0000_5000;
    localparam int RESIDUAL_IMAGE_WORDS = RESIDUAL_IMAGE_BYTES / 4;
    localparam int TAIL_IMAGE_BYTES = 32'h0000_4000;
    localparam int TAIL_IMAGE_WORDS = TAIL_IMAGE_BYTES / 4;

    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int QKV_SLOT_Q_OUT = 8;
    localparam int QKV_SLOT_K_OUT = 9;
    localparam int QKV_SLOT_V_OUT = 10;
    localparam int INPUT_NORM_SLOT_OUTPUT = 3;
    localparam int FRONT_SLOT_COS = 6;
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
    localparam int TAIL_SLOT_FINAL_GAMMA = 7;

    localparam int NUM_KV_HEADS = 8;
    localparam int HEAD_DIM = 128;
    localparam int MAX_CONTEXT = 256;
    localparam int CACHE_LENGTH = 5;
    localparam int CACHE_VECTOR_WORDS = CACHE_LENGTH * NUM_KV_HEADS * HEAD_DIM;
    localparam int CACHE_KIND_WORDS = NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM;
    localparam int CACHE_TOTAL_BYTES = 2 * CACHE_KIND_WORDS * 4;
    localparam int OPROJ_WEIGHT_WORDS = (1024 * 1024) / 4;
    localparam int OPROJ_SCALE_WORDS = (1024 * 64) / 4;
    localparam int GATE_WEIGHT_WORDS = (3072 * 512) / 4;
    localparam int GATE_SCALE_WORDS = (3072 * 32) / 4;
    localparam int DOWN_WEIGHT_WORDS = (1024 * 1536) / 4;
    localparam int DOWN_SCALE_WORDS = (1024 * 96) / 4;

    localparam logic [63 : 0] EMBED_WEIGHT_BASE = 64'h0000_0004_0010_0000;
    localparam logic [63 : 0] EMBED_SCALE_BASE = 64'h0000_0004_04B3_0000;
    localparam logic [63 : 0] INPUT_HIDDEN_BASE = 64'h0000_0004_0509_2540;
    localparam logic [63 : 0] OUTPUT_HIDDEN_BASE = 64'h0000_0004_1509_2540;
    localparam logic [63 : 0] KV_CACHE_BASE = 64'h0000_0004_1410_0000;
    localparam logic [63 : 0] QKV_QMAP_BASE = 64'h0000_0004_1B40_0000;

    localparam int WRITE_EMBEDDING = 1;
    localparam int WRITE_INPUT_NORM = 2;
    localparam int WRITE_Q = 3;
    localparam int WRITE_K = 4;
    localparam int WRITE_V = 5;
    localparam int WRITE_CACHE = 6;
    localparam int WRITE_Q_ROPE = 7;
    localparam int WRITE_ATTN_OUT = 8;
    localparam int WRITE_O_PROJ = 9;
    localparam int WRITE_POST_HIDDEN = 10;
    localparam int WRITE_POST_NORM = 11;
    localparam int WRITE_GATE = 12;
    localparam int WRITE_UP = 13;
    localparam int WRITE_SILU_HIDDEN = 14;
    localparam int WRITE_DOWN = 15;
    localparam int WRITE_LAYER = 16;

    localparam int EXPECTED_SCHED_RD_REQS = 8234;
    localparam int EXPECTED_SCHED_RD_WORDS = 561040;
    localparam int EXPECTED_SCHED_WR_REQS = 4097;
    localparam int EXPECTED_SCHED_WR_WORDS = 5120;
    localparam int EXPECTED_TOP_RD_REQS = EXPECTED_SCHED_RD_REQS + 2;
    localparam int EXPECTED_TOP_RD_WORDS = EXPECTED_SCHED_RD_WORDS + 136;
    localparam int EXPECTED_TOP_WR_REQS = EXPECTED_SCHED_WR_REQS + 1;
    localparam int EXPECTED_TOP_WR_WORDS = EXPECTED_SCHED_WR_WORDS + VEC1024;
    localparam int EXPECTED_FULL_SCHED_RD_REQS = 46278;
    localparam int EXPECTED_FULL_SCHED_RD_WORDS = 2140354;
    localparam int EXPECTED_FULL_SCHED_WR_REQS = 6155;
    localparam int EXPECTED_FULL_SCHED_WR_WORDS = 25600;

    logic clk;
    logic rst_n;
    logic start;

    logic busy;
    logic done;
    logic error;
    logic [7 : 0] state_debug;
    logic [7 : 0] phase_debug;
    logic scheduler_done_pulse;
    logic tail_start_pulse;
    logic [LAYER_COUNT_WIDTH-1 : 0] layers_started;
    logic [LAYER_COUNT_WIDTH-1 : 0] layers_completed;
    logic [MAX_LAYERS-1 : 0] layer_done_mask;
    logic [MAX_LAYERS-1 : 0] layer_error_mask;
    logic [1 : 0] stage_done_mask;
    logic [1 : 0] stage_error_mask;
    logic [3 : 0] full_stage_done_mask;
    logic [3 : 0] full_stage_error_mask;
    logic [4 : 0] body_stage_done_mask;
    logic [4 : 0] body_stage_error_mask;
    logic [31 : 0] scheduler_rd_reqs;
    logic [31 : 0] scheduler_rd_words;
    logic [31 : 0] scheduler_wr_reqs;
    logic [31 : 0] scheduler_wr_words;
    logic [31 : 0] top_rd_reqs;
    logic [31 : 0] top_rd_words;
    logic [31 : 0] top_wr_reqs;
    logic [31 : 0] top_wr_words;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [63 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [31 : 0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;
    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [63 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [31 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    logic [BASE_TABLE_BITS-1 : 0] qkv_table;
    logic [BASE_TABLE_BITS-1 : 0] input_norm_table;
    logic [BASE_TABLE_BITS-1 : 0] frontend_table;
    logic [BASE_TABLE_BITS-1 : 0] score_table;
    logic [BASE_TABLE_BITS-1 : 0] oproj_table;
    logic [BASE_TABLE_BITS-1 : 0] post_table;
    logic [BASE_TABLE_BITS-1 : 0] gate_table;
    logic [BASE_TABLE_BITS-1 : 0] silu_table;
    logic [BASE_TABLE_BITS-1 : 0] down_table;
    logic [BASE_TABLE_BITS-1 : 0] residual_table;

    logic [31 : 0] embedding_weight [0 : EMBED_WEIGHT_WORDS-1];
    logic [31 : 0] embedding_scale [0 : EMBED_SCALE_WORDS-1];
    logic [31 : 0] embedding_expected [0 : VEC1024-1];
    logic [31 : 0] token_file [0 : 0];
    logic [31 : 0] input_hidden_mem [0 : VEC1024-1];
    logic [31 : 0] input_norm_image [0 : INPUT_NORM_IMAGE_WORDS-1];
    logic [31 : 0] input_norm_expected [0 : VEC1024-1];
    logic [31 : 0] qkv_image [0 : QKV_IMAGE_WORDS-1];
    logic [31 : 0] qkv_expected [0 : QKV_EXPECTED_WORDS-1];
    logic [31 : 0] frontend_image [0 : FRONT_IMAGE_WORDS-1];
    logic [31 : 0] score_image [0 : SCORE_IMAGE_WORDS-1];
    logic [31 : 0] oproj_image [0 : OPROJ_IMAGE_WORDS-1];
    logic [31 : 0] post_image [0 : POST_IMAGE_WORDS-1];
    logic [31 : 0] gate_image [0 : GATE_IMAGE_WORDS-1];
    logic [31 : 0] silu_image [0 : SILU_IMAGE_WORDS-1];
    logic [31 : 0] down_image [0 : DOWN_IMAGE_WORDS-1];
    logic [31 : 0] residual_image [0 : RESIDUAL_IMAGE_WORDS-1];
    logic [31 : 0] tail_image [0 : TAIL_IMAGE_WORDS-1];

    logic [23 : 0] k_cache_mem [0 : CACHE_VECTOR_WORDS-1];
    logic [23 : 0] v_cache_mem [0 : CACHE_VECTOR_WORDS-1];
    logic [31 : 0] oproj_weight_mem [0 : OPROJ_WEIGHT_WORDS-1];
    logic [31 : 0] oproj_scale_mem [0 : OPROJ_SCALE_WORDS-1];
    logic [31 : 0] gate_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] gate_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] up_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] up_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] down_weight_mem [0 : DOWN_WEIGHT_WORDS-1];
    logic [31 : 0] down_scale_mem [0 : DOWN_SCALE_WORDS-1];

    logic [63 : 0] cache_expected_addr [0 : (2*VEC1024)-1];
    logic [31 : 0] cache_expected_data [0 : (2*VEC1024)-1];
    logic [31 : 0] q_rope_expected [0 : VEC2048-1];
    logic [31 : 0] attn_out_expected [0 : VEC2048-1];
    logic [31 : 0] oproj_expected [0 : VEC1024-1];
    logic [31 : 0] post_hidden_expected [0 : VEC1024-1];
    logic [31 : 0] post_norm_expected [0 : VEC1024-1];
    logic [31 : 0] gate_expected [0 : VEC3072-1];
    logic [31 : 0] up_expected [0 : VEC3072-1];
    logic [31 : 0] silu_hidden_expected [0 : VEC3072-1];
    logic [31 : 0] down_expected [0 : VEC1024-1];
    logic [31 : 0] layer_expected [0 : VEC1024-1];

    string vector_dir;
    string vector_path;
    string frontend_hex;
    string tail_hex;
    logic [31 : 0] token_id;
    logic [63 : 0] embedding_weight_row_addr;
    logic [63 : 0] embedding_scale_row_addr;
    logic [63 : 0] input_norm_output_base;
    logic [63 : 0] q_base;
    logic [63 : 0] k_base;
    logic [63 : 0] v_base;
    logic [63 : 0] q_rope_base;
    logic [63 : 0] attn_out_base;
    logic [63 : 0] oproj_base;
    logic [63 : 0] post_hidden_base;
    logic [63 : 0] post_norm_base;
    logic [63 : 0] gate_base;
    logic [63 : 0] up_base;
    logic [63 : 0] silu_hidden_base;
    logic [63 : 0] down_base;
    logic [63 : 0] layer_base;

    logic rd_active;
    logic [63 : 0] rd_base;
    integer rd_words;
    integer rd_index;
    logic wr_active;
    logic [63 : 0] wr_base;
    integer wr_words;
    integer wr_index;
    integer wr_kind;
    logic wr_done_pending;
    integer wr_done_kind;

    integer cycle_count;
    integer fail_count;
    integer mismatch_count;
    integer unknown_read_count;
    integer embedding_write_reqs;
    integer embedding_write_words;
    integer input_norm_write_reqs;
    integer input_norm_write_words;
    integer q_write_reqs;
    integer q_write_words;
    integer k_write_reqs;
    integer k_write_words;
    integer v_write_reqs;
    integer v_write_words;
    integer embedding_done_cycle;
    integer first_input_hidden_read_cycle;
    integer input_norm_done_cycle;
    integer first_input_norm_read_cycle;
    integer last_qkv_done_cycle;
    integer scheduler_done_cycle;
    integer top_done_cycle;
    integer tail_start_count;
    integer tail_start_cycle;
    integer write_req_count [1 : WRITE_LAYER];
    integer write_word_count [1 : WRITE_LAYER];
    integer write_done_cycle [1 : WRITE_LAYER];
    integer first_frontend_read_cycle;
    integer first_score_q_rope_read_cycle;
    integer first_score_cache_read_cycle;
    integer first_oproj_read_cycle;
    integer first_post_read_cycle;
    integer first_gate_read_cycle;
    integer first_silu_read_cycle;
    integer first_down_read_cycle;
    integer first_residual_post_read_cycle;
    integer first_residual_down_read_cycle;
    integer init_kind;
    integer trace_fd;
    logic stall_enable;
    logic full_layer;
    logic tail_error;
    logic [31 : 0] tail_rd_reqs;
    logic [31 : 0] tail_rd_words;
    logic [31 : 0] tail_wr_words;

    function automatic integer desc_index(input integer slot, input integer word);
        desc_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word;
    endfunction

    function automatic logic [63 : 0] qkv_desc_base(input integer slot);
        qkv_desc_base = {
            qkv_image[desc_index(slot, DESC_BASE_HI_WORD)],
            qkv_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] input_norm_desc_base(input integer slot);
        input_norm_desc_base = {
            input_norm_image[desc_index(slot, DESC_BASE_HI_WORD)],
            input_norm_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] frontend_desc_base(input integer slot);
        frontend_desc_base = {
            frontend_image[desc_index(slot, DESC_BASE_HI_WORD)],
            frontend_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] score_desc_base(input integer slot);
        score_desc_base = {
            score_image[desc_index(slot, DESC_BASE_HI_WORD)],
            score_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] oproj_desc_base(input integer slot);
        oproj_desc_base = {
            oproj_image[desc_index(slot, DESC_BASE_HI_WORD)],
            oproj_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] post_desc_base(input integer slot);
        post_desc_base = {
            post_image[desc_index(slot, DESC_BASE_HI_WORD)],
            post_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] gate_desc_base(input integer slot);
        gate_desc_base = {
            gate_image[desc_index(slot, DESC_BASE_HI_WORD)],
            gate_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] silu_desc_base(input integer slot);
        silu_desc_base = {
            silu_image[desc_index(slot, DESC_BASE_HI_WORD)],
            silu_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] down_desc_base(input integer slot);
        down_desc_base = {
            down_image[desc_index(slot, DESC_BASE_HI_WORD)],
            down_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic [63 : 0] residual_desc_base(input integer slot);
        residual_desc_base = {
            residual_image[desc_index(slot, DESC_BASE_HI_WORD)],
            residual_image[desc_index(slot, DESC_BASE_LO_WORD)]
        };
    endfunction

    function automatic logic in_range(
        input logic [63 : 0] addr,
        input logic [63 : 0] base,
        input integer bytes
    );
        in_range = (addr >= base) && (addr < (base + bytes));
    endfunction

    function automatic logic is_cache_addr(input logic [63 : 0] addr);
        is_cache_addr = in_range(addr, KV_CACHE_BASE, CACHE_TOTAL_BYTES);
    endfunction

    function automatic logic [31 : 0] cache_word(input logic [63 : 0] addr);
        integer element_index;
        integer kind;
        integer kind_index;
        integer head;
        integer position;
        integer dim;
        integer vector_index;
        logic signed [23 : 0] value;
        begin
            element_index = (addr - KV_CACHE_BASE) >> 2;
            kind = element_index / CACHE_KIND_WORDS;
            kind_index = element_index % CACHE_KIND_WORDS;
            head = kind_index / (MAX_CONTEXT * HEAD_DIM);
            position = (kind_index % (MAX_CONTEXT * HEAD_DIM)) / HEAD_DIM;
            dim = kind_index % HEAD_DIM;
            vector_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;
            if ((kind > 1) || (head >= NUM_KV_HEADS) ||
                (position >= CACHE_LENGTH) || (dim >= HEAD_DIM)) begin
                cache_word = 32'hCAFE_BAD0;
            end else begin
                value = (kind == 0) ? k_cache_mem[vector_index] : v_cache_mem[vector_index];
                cache_word = {{8{value[23]}}, value};
            end
        end
    endfunction

    task automatic patch_frontend_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            frontend_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            frontend_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_score_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            score_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            score_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_oproj_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            oproj_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            oproj_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_post_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            post_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            post_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_gate_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            gate_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            gate_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_silu_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            silu_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            silu_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_down_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            down_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            down_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task automatic patch_residual_base(input integer slot, input logic [63 : 0] base_addr);
        begin
            residual_image[desc_index(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            residual_image[desc_index(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    function automatic integer classify_write(input logic [63 : 0] addr);
        begin
            if ((addr == INPUT_HIDDEN_BASE) &&
                (!full_layer || (embedding_done_cycle < 0))) classify_write = WRITE_EMBEDDING;
            else if (addr == input_norm_output_base) classify_write = WRITE_INPUT_NORM;
            else if (in_range(addr, q_base, VEC2048 * 4)) classify_write = WRITE_Q;
            else if (in_range(addr, k_base, VEC1024 * 4)) classify_write = WRITE_K;
            else if (in_range(addr, v_base, VEC1024 * 4)) classify_write = WRITE_V;
            else if (full_layer && is_cache_addr(addr)) classify_write = WRITE_CACHE;
            else if (full_layer && in_range(addr, q_rope_base, VEC2048 * 4)) classify_write = WRITE_Q_ROPE;
            else if (full_layer && in_range(addr, attn_out_base, VEC2048 * 4)) classify_write = WRITE_ATTN_OUT;
            else if (full_layer && in_range(addr, oproj_base, VEC1024 * 4)) classify_write = WRITE_O_PROJ;
            else if (full_layer && in_range(addr, post_hidden_base, VEC1024 * 4)) classify_write = WRITE_POST_HIDDEN;
            else if (full_layer && in_range(addr, post_norm_base, VEC1024 * 4)) classify_write = WRITE_POST_NORM;
            else if (full_layer && in_range(addr, gate_base, VEC3072 * 4)) classify_write = WRITE_GATE;
            else if (full_layer && in_range(addr, up_base, VEC3072 * 4)) classify_write = WRITE_UP;
            else if (full_layer && in_range(addr, silu_hidden_base, VEC3072 * 4)) classify_write = WRITE_SILU_HIDDEN;
            else if (full_layer && in_range(addr, down_base, VEC1024 * 4)) classify_write = WRITE_DOWN;
            else if (full_layer && in_range(addr, layer_base, VEC1024 * 4)) classify_write = WRITE_LAYER;
            else classify_write = 0;
        end
    endfunction

    function automatic logic address_known(input logic [63 : 0] addr);
        begin
            address_known =
                in_range(addr, embedding_weight_row_addr, EMBED_WEIGHT_WORDS * 4) ||
                in_range(addr, embedding_scale_row_addr, EMBED_SCALE_WORDS * 4) ||
                in_range(addr, INPUT_HIDDEN_BASE, VEC1024 * 4) ||
                in_range(addr, `QMAP_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES) ||
                in_range(addr, QKV_QMAP_BASE, QKV_IMAGE_BYTES) ||
                in_range(addr, `QMAP_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES) ||
                (full_layer && (
                    in_range(addr, `QMAP_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES) ||
                    in_range(addr, `QMAP_FINAL_TOKEN_BASE_ADDR, TAIL_IMAGE_BYTES) ||
                    is_cache_addr(addr) ||
                    in_range(addr, `QMAP_O_PROJ_WEIGHT_BASE_ADDR, OPROJ_WEIGHT_WORDS * 4) ||
                    in_range(addr, `QMAP_O_PROJ_SCALE_BASE_ADDR, OPROJ_SCALE_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * 4) ||
                    in_range(addr, `QMAP_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * 4)
                ));
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input logic [63 : 0] addr);
        integer index;
        begin
            memory_word = 32'hDEAD_BEEF;
            if (in_range(addr, embedding_weight_row_addr, EMBED_WEIGHT_WORDS * 4)) begin
                index = (addr - embedding_weight_row_addr) >> 2;
                memory_word = embedding_weight[index];
            end else if (in_range(addr, embedding_scale_row_addr, EMBED_SCALE_WORDS * 4)) begin
                index = (addr - embedding_scale_row_addr) >> 2;
                memory_word = embedding_scale[index];
            end else if (in_range(addr, INPUT_HIDDEN_BASE, VEC1024 * 4)) begin
                index = (addr - INPUT_HIDDEN_BASE) >> 2;
                memory_word = input_hidden_mem[index];
            end else if (in_range(addr, `QMAP_INPUT_NORM_BASE_ADDR, INPUT_NORM_IMAGE_BYTES)) begin
                index = (addr - `QMAP_INPUT_NORM_BASE_ADDR) >> 2;
                memory_word = input_norm_image[index];
            end else if (in_range(addr, QKV_QMAP_BASE, QKV_IMAGE_BYTES)) begin
                index = (addr - QKV_QMAP_BASE) >> 2;
                memory_word = qkv_image[index];
            end else if (in_range(addr, `QMAP_ATTN_FRONTEND_BASE_ADDR, FRONT_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2;
                memory_word = frontend_image[index];
            end else if (full_layer && in_range(addr, `QMAP_ATTN_SCORE_VALUE_BASE_ADDR, SCORE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_ATTN_SCORE_VALUE_BASE_ADDR) >> 2;
                memory_word = score_image[index];
            end else if (full_layer && in_range(addr, `QMAP_O_PROJ_BASE_ADDR, OPROJ_IMAGE_BYTES)) begin
                index = (addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
                memory_word = oproj_image[index];
            end else if (full_layer && in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2;
                memory_word = post_image[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2;
                memory_word = gate_image[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2;
                memory_word = silu_image[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_BASE_ADDR) >> 2;
                memory_word = down_image[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                memory_word = residual_image[index];
            end else if (full_layer && is_cache_addr(addr)) begin
                memory_word = cache_word(addr);
            end else if (full_layer && in_range(addr, `QMAP_O_PROJ_WEIGHT_BASE_ADDR, OPROJ_WEIGHT_WORDS * 4)) begin
                index = (addr - `QMAP_O_PROJ_WEIGHT_BASE_ADDR) >> 2;
                memory_word = oproj_weight_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_O_PROJ_SCALE_BASE_ADDR, OPROJ_SCALE_WORDS * 4)) begin
                index = (addr - `QMAP_O_PROJ_SCALE_BASE_ADDR) >> 2;
                memory_word = oproj_scale_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_GATE_WEIGHT_BASE_ADDR) >> 2;
                memory_word = gate_weight_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_GATE_SCALE_BASE_ADDR) >> 2;
                memory_word = gate_scale_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_UP_WEIGHT_BASE_ADDR) >> 2;
                memory_word = up_weight_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_UP_SCALE_BASE_ADDR) >> 2;
                memory_word = up_scale_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR) >> 2;
                memory_word = down_weight_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * 4)) begin
                index = (addr - `QMAP_MLP_DOWN_SCALE_BASE_ADDR) >> 2;
                memory_word = down_scale_mem[index];
            end else if (full_layer && in_range(addr, `QMAP_FINAL_TOKEN_BASE_ADDR, TAIL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_FINAL_TOKEN_BASE_ADDR) >> 2;
                memory_word = tail_image[index];
            end
        end
    endfunction

    task automatic fail(input string message);
        begin
            $display("FAIL: %s", message);
            fail_count = fail_count + 1;
        end
    endtask

    task automatic check(input logic condition, input string message);
        begin
            if (!condition) fail(message);
        end
    endtask

    qmap_one_token_top dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_embedding_enable(1'b1),
        .i_input_token_id(token_id),
        .i_embedding_weight_base_addr(EMBED_WEIGHT_BASE),
        .i_embedding_scale_base_addr(EMBED_SCALE_BASE),
        .i_layer_start_index(5'd0),
        .i_layer_count(5'd1),
        .i_position(8'd4),
        .i_input_hidden_base_addr(INPUT_HIDDEN_BASE),
        .i_output_hidden_base_addr(OUTPUT_HIDDEN_BASE),
        .i_kv_cache_base_addr(KV_CACHE_BASE),
        .i_final_tail_qmap_base_addr(`QMAP_FINAL_TOKEN_BASE_ADDR),
        .i_final_hidden_base_override_valid(1'b0),
        .i_final_hidden_base_override_addr('0),
        .i_qkv_qmap_base_addr_table(qkv_table),
        .i_input_norm_qmap_base_addr_table(input_norm_table),
        .i_attn_frontend_qmap_base_addr_table(frontend_table),
        .i_attn_score_value_qmap_base_addr_table(score_table),
        .i_o_proj_qmap_base_addr_table(oproj_table),
        .i_post_attn_norm_qmap_base_addr_table(post_table),
        .i_mlp_gate_up_qmap_base_addr_table(gate_table),
        .i_mlp_silu_mul_qmap_base_addr_table(silu_table),
        .i_mlp_down_qmap_base_addr_table(down_table),
        .i_mlp_residual_add_qmap_base_addr_table(residual_table),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_phase_debug(phase_debug),
        .o_scheduler_done_pulse(scheduler_done_pulse),
        .o_tail_start_pulse(tail_start_pulse),
        .o_tail_done_pulse(),
        .o_tail_active(),
        .o_active_layer_index(),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_last_layer_output_base_addr(),
        .o_layer0_active_stage_debug(),
        .o_layer0_state_debug(),
        .o_layer0_stage_done_mask(stage_done_mask),
        .o_layer0_stage_error_mask(stage_error_mask),
        .o_layer0_full_stage_done_mask(full_stage_done_mask),
        .o_layer0_full_stage_error_mask(full_stage_error_mask),
        .o_body_stage_done_mask(body_stage_done_mask),
        .o_body_stage_error_mask(body_stage_error_mask),
        .o_qkv_rows_done(),
        .o_qkv_last_row_sum_q26(),
        .o_qkv_last_output_q12_12(),
        .o_scheduler_mem_read_burst_count(scheduler_rd_reqs),
        .o_scheduler_mem_read_word_count(scheduler_rd_words),
        .o_scheduler_mem_write_req_count(scheduler_wr_reqs),
        .o_scheduler_mem_write_word_count(scheduler_wr_words),
        .o_tail_error(tail_error),
        .o_tail_norm_saturation(),
        .o_tail_effective_final_hidden_base_addr(),
        .o_tail_best_token_id(),
        .o_tail_best_score_q26(),
        .o_tail_tiles_started(),
        .o_tail_tiles_completed(),
        .o_tail_norm_cycle_count(),
        .o_tail_mem_read_burst_count(tail_rd_reqs),
        .o_tail_mem_read_word_count(tail_rd_words),
        .o_tail_mem_write_word_count(tail_wr_words),
        .o_mem_read_burst_count(top_rd_reqs),
        .o_mem_read_word_count(top_rd_words),
        .o_mem_write_req_count(top_wr_reqs),
        .o_mem_write_word_count(top_wr_words),
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

    always #5 clk = ~clk;

    assign mem_rd_req_ready = !rd_active &&
        (!stall_enable || ((cycle_count % 7) != 2));
    assign mem_rd_rsp_valid = rd_active &&
        (!stall_enable || ((cycle_count % 5) != 1));
    assign mem_rd_rsp_data = memory_word(rd_base + (rd_index * 4));
    assign mem_rd_rsp_last = rd_active && (rd_index == (rd_words - 1));
    assign mem_wr_req_ready = !wr_active && !wr_done_pending &&
        (!stall_enable || ((cycle_count % 6) != 3));
    assign mem_wr_data_ready = wr_active &&
        (!stall_enable || ((cycle_count % 5) != 2));
    assign mem_wr_error = 1'b0;

    always @(posedge clk or negedge rst_n) begin
        logic [63 : 0] response_addr;
        logic [31 : 0] expected_word;
        integer image_index;
        integer expected_index;
        integer cache_element_index;
        integer cache_kind;
        integer cache_kind_index;
        integer cache_head;
        integer cache_position;
        integer cache_dim;
        integer cache_vector_index;
        if (!rst_n) begin
            cycle_count <= 0;
            rd_active <= 1'b0;
            rd_base <= '0;
            rd_words <= 0;
            rd_index <= 0;
            wr_active <= 1'b0;
            wr_base <= '0;
            wr_words <= 0;
            wr_index <= 0;
            wr_kind <= 0;
            wr_done_pending <= 1'b0;
            wr_done_kind <= 0;
            mem_wr_done <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;

            if (wr_done_pending) begin
                mem_wr_done <= 1'b1;
                wr_done_pending <= 1'b0;
                if ((wr_done_kind >= 1) && (wr_done_kind <= WRITE_LAYER)) begin
                    write_done_cycle[wr_done_kind] = cycle_count;
                end
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,write_done,0x%016h,%0d,0x00000000,1\n",
                            cycle_count, wr_base, wr_done_kind);
                end
                case (wr_done_kind)
                    WRITE_EMBEDDING: embedding_done_cycle = cycle_count;
                    WRITE_INPUT_NORM: input_norm_done_cycle = cycle_count;
                    WRITE_Q,
                    WRITE_K,
                    WRITE_V: last_qkv_done_cycle = cycle_count;
                    WRITE_CACHE,
                    WRITE_Q_ROPE,
                    WRITE_ATTN_OUT,
                    WRITE_O_PROJ,
                    WRITE_POST_HIDDEN,
                    WRITE_POST_NORM,
                    WRITE_GATE,
                    WRITE_UP,
                    WRITE_SILU_HIDDEN,
                    WRITE_DOWN,
                    WRITE_LAYER: begin
                    end
                    default: fail("write completion had an unknown kind");
                endcase
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                if ((mem_rd_req_len_bytes == 0) || (mem_rd_req_len_bytes[1 : 0] != 0)) begin
                    fail("read request length was zero or unaligned");
                end
                if (!address_known(mem_rd_req_addr) ||
                    !address_known(mem_rd_req_addr + mem_rd_req_len_bytes - 1)) begin
                    fail("read request targeted an unknown memory range");
                    unknown_read_count = unknown_read_count + 1;
                end
                if (in_range(mem_rd_req_addr, INPUT_HIDDEN_BASE, VEC1024 * 4)) begin
                    if (first_input_hidden_read_cycle < 0) begin
                        first_input_hidden_read_cycle = cycle_count;
                    end
                    if (embedding_done_cycle < 0) begin
                        fail("input RMSNorm read hidden data before embedding write response");
                    end
                end
                if (in_range(mem_rd_req_addr, input_norm_output_base, VEC1024 * 4)) begin
                    if (first_input_norm_read_cycle < 0) begin
                        first_input_norm_read_cycle = cycle_count;
                    end
                    if (input_norm_done_cycle < 0) begin
                        fail("QKV read input-norm data before RMSNorm write response");
                    end
                end
                if (full_layer &&
                    (in_range(mem_rd_req_addr, q_base, VEC2048 * 4) ||
                     in_range(mem_rd_req_addr, k_base, VEC1024 * 4) ||
                     in_range(mem_rd_req_addr, v_base, VEC1024 * 4))) begin
                    if (first_frontend_read_cycle < 0) first_frontend_read_cycle = cycle_count;
                    if (last_qkv_done_cycle < 0) fail("attention frontend read Q/K/V before QKV write response");
                end
                if (full_layer && in_range(mem_rd_req_addr, q_rope_base, VEC2048 * 4)) begin
                    if (first_score_q_rope_read_cycle < 0) first_score_q_rope_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_Q_ROPE] < 0) fail("attention score read Q RoPE before write response");
                end
                if (full_layer && is_cache_addr(mem_rd_req_addr)) begin
                    if (first_score_cache_read_cycle < 0) first_score_cache_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_CACHE] < 0) fail("attention score/value read cache before append response");
                end
                if (full_layer && in_range(mem_rd_req_addr, attn_out_base, VEC2048 * 4)) begin
                    if (first_oproj_read_cycle < 0) first_oproj_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_ATTN_OUT] < 0) fail("o_proj read attention output before write response");
                end
                if (full_layer && in_range(mem_rd_req_addr, oproj_base, VEC1024 * 4)) begin
                    if (first_post_read_cycle < 0) first_post_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_O_PROJ] < 0) fail("post-attention stage read o_proj before write response");
                end
                if (full_layer && in_range(mem_rd_req_addr, post_norm_base, VEC1024 * 4)) begin
                    if (first_gate_read_cycle < 0) first_gate_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_POST_NORM] < 0) fail("gate/up read post norm before write response");
                end
                if (full_layer &&
                    (in_range(mem_rd_req_addr, gate_base, VEC3072 * 4) ||
                     in_range(mem_rd_req_addr, up_base, VEC3072 * 4))) begin
                    if (first_silu_read_cycle < 0) first_silu_read_cycle = cycle_count;
                    if ((write_done_cycle[WRITE_GATE] < 0) ||
                        (write_done_cycle[WRITE_UP] < 0)) fail("SiLU read gate/up before both write responses");
                end
                if (full_layer && in_range(mem_rd_req_addr, silu_hidden_base, VEC3072 * 4)) begin
                    if (first_down_read_cycle < 0) first_down_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_SILU_HIDDEN] < 0) fail("down projection read MLP hidden before write response");
                end
                if (full_layer && in_range(mem_rd_req_addr, post_hidden_base, VEC1024 * 4)) begin
                    if (first_residual_post_read_cycle < 0) first_residual_post_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_POST_HIDDEN] < 0) fail("final residual read post hidden before write response");
                end
                if (full_layer && in_range(mem_rd_req_addr, down_base, VEC1024 * 4)) begin
                    if (first_residual_down_read_cycle < 0) first_residual_down_read_cycle = cycle_count;
                    if (write_done_cycle[WRITE_DOWN] < 0) fail("final residual read down output before write response");
                end
                rd_active <= 1'b1;
                rd_base <= mem_rd_req_addr;
                rd_words <= mem_rd_req_len_bytes >> 2;
                rd_index <= 0;
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,read_req,0x%016h,%0d,0x%08h,0\n",
                            cycle_count, mem_rd_req_addr,
                            mem_rd_req_len_bytes >> 2, mem_rd_req_len_bytes);
                end
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                response_addr = rd_base + (rd_index * 4);
                if (!address_known(response_addr)) begin
                    fail("read response came from an unknown address");
                end
                if (mem_rd_rsp_last != (rd_index == (rd_words - 1))) begin
                    fail("read response last was misaligned");
                end
                if (mem_rd_rsp_last) begin
                    if (trace_fd != 0) begin
                        $fwrite(trace_fd, "%0d,read_last,0x%016h,%0d,0x%08h,1\n",
                                cycle_count, response_addr, rd_index, mem_rd_rsp_data);
                    end
                    rd_active <= 1'b0;
                    rd_index <= 0;
                end else begin
                    rd_index <= rd_index + 1;
                end
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                wr_kind <= classify_write(mem_wr_req_addr);
                wr_base <= mem_wr_req_addr;
                wr_words <= mem_wr_req_len_bytes >> 2;
                wr_index <= 0;
                wr_active <= 1'b1;
                if ((classify_write(mem_wr_req_addr) == 0) ||
                    (mem_wr_req_len_bytes == 0) ||
                    (mem_wr_req_len_bytes[1 : 0] != 0)) begin
                    fail("write request address or length was invalid");
                end
                if ((classify_write(mem_wr_req_addr) >= 1) &&
                    (classify_write(mem_wr_req_addr) <= WRITE_LAYER)) begin
                    write_req_count[classify_write(mem_wr_req_addr)] =
                        write_req_count[classify_write(mem_wr_req_addr)] + 1;
                end
                case (classify_write(mem_wr_req_addr))
                    WRITE_EMBEDDING: begin
                        embedding_write_reqs = embedding_write_reqs + 1;
                        if (mem_wr_req_len_bytes != VEC1024 * 4) begin
                            fail("embedding write request length mismatch");
                        end
                    end
                    WRITE_INPUT_NORM: begin
                        input_norm_write_reqs = input_norm_write_reqs + 1;
                        if (mem_wr_req_len_bytes != VEC1024 * 4) begin
                            fail("input RMSNorm write request length mismatch");
                        end
                    end
                    WRITE_Q: q_write_reqs = q_write_reqs + 1;
                    WRITE_K: k_write_reqs = k_write_reqs + 1;
                    WRITE_V: v_write_reqs = v_write_reqs + 1;
                    WRITE_CACHE: begin
                        if (mem_wr_req_addr !== cache_expected_addr[write_word_count[WRITE_CACHE]]) begin
                            fail("KV-cache write address mismatch");
                        end
                        if (mem_wr_req_len_bytes != 4) fail("KV-cache write length mismatch");
                    end
                    WRITE_Q_ROPE,
                    WRITE_ATTN_OUT: begin
                        if (mem_wr_req_len_bytes != VEC2048 * 4) fail("2048-word write length mismatch");
                    end
                    WRITE_O_PROJ,
                    WRITE_POST_HIDDEN,
                    WRITE_POST_NORM,
                    WRITE_DOWN,
                    WRITE_LAYER: begin
                        if (mem_wr_req_len_bytes != VEC1024 * 4) fail("1024-word write length mismatch");
                    end
                    WRITE_GATE,
                    WRITE_UP,
                    WRITE_SILU_HIDDEN: begin
                        if (mem_wr_req_len_bytes != VEC3072 * 4) fail("3072-word write length mismatch");
                    end
                    default: begin
                    end
                endcase
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,write_req,0x%016h,%0d,0x%08h,0\n",
                            cycle_count, mem_wr_req_addr,
                            classify_write(mem_wr_req_addr), mem_wr_req_len_bytes);
                end
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                expected_word = 32'hDEAD_BEEF;
                expected_index = wr_index;
                case (wr_kind)
                    WRITE_EMBEDDING: begin
                        expected_word = embedding_expected[expected_index];
                        input_hidden_mem[expected_index] = mem_wr_data;
                        embedding_write_words = embedding_write_words + 1;
                    end
                    WRITE_INPUT_NORM: begin
                        expected_word = input_norm_expected[expected_index];
                        image_index = ((wr_base - `QMAP_INPUT_NORM_BASE_ADDR) >> 2) + expected_index;
                        input_norm_image[image_index] = mem_wr_data;
                        input_norm_write_words = input_norm_write_words + 1;
                    end
                    WRITE_Q: begin
                        expected_index = ((wr_base - q_base) >> 2) + wr_index;
                        expected_word = qkv_expected[expected_index];
                        image_index = ((wr_base - QKV_QMAP_BASE) >> 2) + wr_index;
                        qkv_image[image_index] = mem_wr_data;
                        q_write_words = q_write_words + 1;
                    end
                    WRITE_K: begin
                        expected_index = ((wr_base - k_base) >> 2) + wr_index;
                        expected_word = qkv_expected[VEC2048 + expected_index];
                        image_index = ((wr_base - QKV_QMAP_BASE) >> 2) + wr_index;
                        qkv_image[image_index] = mem_wr_data;
                        k_write_words = k_write_words + 1;
                    end
                    WRITE_V: begin
                        expected_index = ((wr_base - v_base) >> 2) + wr_index;
                        expected_word = qkv_expected[VEC2048 + VEC1024 + expected_index];
                        image_index = ((wr_base - QKV_QMAP_BASE) >> 2) + wr_index;
                        qkv_image[image_index] = mem_wr_data;
                        v_write_words = v_write_words + 1;
                    end
                    WRITE_CACHE: begin
                        expected_index = write_word_count[WRITE_CACHE];
                        expected_word = cache_expected_data[expected_index];
                        cache_element_index = (wr_base - KV_CACHE_BASE) >> 2;
                        cache_kind = cache_element_index / CACHE_KIND_WORDS;
                        cache_kind_index = cache_element_index % CACHE_KIND_WORDS;
                        cache_head = cache_kind_index / (MAX_CONTEXT * HEAD_DIM);
                        cache_position = (cache_kind_index % (MAX_CONTEXT * HEAD_DIM)) / HEAD_DIM;
                        cache_dim = cache_kind_index % HEAD_DIM;
                        cache_vector_index = ((cache_position * NUM_KV_HEADS + cache_head) * HEAD_DIM) + cache_dim;
                        if ((cache_position >= CACHE_LENGTH) || (cache_head >= NUM_KV_HEADS) ||
                            (cache_dim >= HEAD_DIM) || (cache_kind > 1)) begin
                            fail("KV-cache write was outside modeled range");
                        end else if (cache_kind == 0) begin
                            k_cache_mem[cache_vector_index] = mem_wr_data[23 : 0];
                        end else begin
                            v_cache_mem[cache_vector_index] = mem_wr_data[23 : 0];
                        end
                    end
                    WRITE_Q_ROPE: begin
                        expected_word = q_rope_expected[expected_index];
                        image_index = ((wr_base - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2) + wr_index;
                        frontend_image[image_index] = mem_wr_data;
                    end
                    WRITE_ATTN_OUT: begin
                        expected_word = attn_out_expected[expected_index];
                        image_index = ((wr_base - `QMAP_ATTN_SCORE_VALUE_BASE_ADDR) >> 2) + wr_index;
                        score_image[image_index] = mem_wr_data;
                    end
                    WRITE_O_PROJ: begin
                        expected_word = oproj_expected[expected_index];
                        image_index = ((wr_base - `QMAP_O_PROJ_BASE_ADDR) >> 2) + wr_index;
                        oproj_image[image_index] = mem_wr_data;
                    end
                    WRITE_POST_HIDDEN: begin
                        expected_word = post_hidden_expected[expected_index];
                        image_index = ((wr_base - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2) + wr_index;
                        post_image[image_index] = mem_wr_data;
                    end
                    WRITE_POST_NORM: begin
                        expected_word = post_norm_expected[expected_index];
                        image_index = ((wr_base - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2) + wr_index;
                        post_image[image_index] = mem_wr_data;
                    end
                    WRITE_GATE: begin
                        expected_word = gate_expected[expected_index];
                        image_index = ((wr_base - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2) + wr_index;
                        gate_image[image_index] = mem_wr_data;
                    end
                    WRITE_UP: begin
                        expected_word = up_expected[expected_index];
                        image_index = ((wr_base - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2) + wr_index;
                        gate_image[image_index] = mem_wr_data;
                    end
                    WRITE_SILU_HIDDEN: begin
                        expected_word = silu_hidden_expected[expected_index];
                        image_index = ((wr_base - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2) + wr_index;
                        silu_image[image_index] = mem_wr_data;
                    end
                    WRITE_DOWN: begin
                        expected_word = down_expected[expected_index];
                        image_index = ((wr_base - `QMAP_MLP_DOWN_BASE_ADDR) >> 2) + wr_index;
                        down_image[image_index] = mem_wr_data;
                    end
                    WRITE_LAYER: begin
                        expected_word = layer_expected[expected_index];
                        image_index = ((wr_base - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2) + wr_index;
                        residual_image[image_index] = mem_wr_data;
                        input_hidden_mem[expected_index] = mem_wr_data;
                    end
                    default: fail("write data arrived for an unknown request");
                endcase
                if ((wr_kind >= 1) && (wr_kind <= WRITE_LAYER)) begin
                    write_word_count[wr_kind] = write_word_count[wr_kind] + 1;
                end
                if (mem_wr_data !== expected_word) begin
                    mismatch_count = mismatch_count + 1;
                    if (mismatch_count <= 16) begin
                        $display("FAIL: write mismatch kind=%0d index=%0d expected=0x%08h observed=0x%08h",
                                 wr_kind, expected_index, expected_word, mem_wr_data);
                    end
                end
                if (mem_wr_data_last != (wr_index == (wr_words - 1))) begin
                    fail("write data last was misaligned");
                end
                if (mem_wr_data_last) begin
                    if (trace_fd != 0) begin
                        $fwrite(trace_fd, "%0d,write_last,0x%016h,%0d,0x%08h,1\n",
                                cycle_count, wr_base + (wr_index * 4), wr_kind, mem_wr_data);
                    end
                    wr_active <= 1'b0;
                    wr_done_pending <= 1'b1;
                    wr_done_kind <= wr_kind;
                    wr_index <= 0;
                end else begin
                    wr_index <= wr_index + 1;
                end
            end

            if (scheduler_done_pulse) begin
                scheduler_done_cycle = cycle_count;
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,scheduler_done,0x0000000000000000,0,0x00000000,1\n",
                            cycle_count);
                end
            end
            if (tail_start_pulse) begin
                tail_start_count = tail_start_count + 1;
                tail_start_cycle = cycle_count;
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,tail_start,0x0000000000000000,0,0x00000000,1\n",
                            cycle_count);
                end
            end
            if (done) begin
                top_done_cycle = cycle_count;
                if (trace_fd != 0) begin
                    $fwrite(trace_fd, "%0d,top_done,0x0000000000000000,0,0x00000000,1\n",
                            cycle_count);
                end
            end
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        fail_count = 0;
        mismatch_count = 0;
        unknown_read_count = 0;
        embedding_write_reqs = 0;
        embedding_write_words = 0;
        input_norm_write_reqs = 0;
        input_norm_write_words = 0;
        q_write_reqs = 0;
        q_write_words = 0;
        k_write_reqs = 0;
        k_write_words = 0;
        v_write_reqs = 0;
        v_write_words = 0;
        embedding_done_cycle = -1;
        first_input_hidden_read_cycle = -1;
        input_norm_done_cycle = -1;
        first_input_norm_read_cycle = -1;
        last_qkv_done_cycle = -1;
        scheduler_done_cycle = -1;
        top_done_cycle = -1;
        tail_start_count = 0;
        tail_start_cycle = -1;
        first_frontend_read_cycle = -1;
        first_score_q_rope_read_cycle = -1;
        first_score_cache_read_cycle = -1;
        first_oproj_read_cycle = -1;
        first_post_read_cycle = -1;
        first_gate_read_cycle = -1;
        first_silu_read_cycle = -1;
        first_down_read_cycle = -1;
        first_residual_post_read_cycle = -1;
        first_residual_down_read_cycle = -1;
        for (init_kind = 1; init_kind <= WRITE_LAYER; init_kind = init_kind + 1) begin
            write_req_count[init_kind] = 0;
            write_word_count[init_kind] = 0;
            write_done_cycle[init_kind] = -1;
        end
        full_layer = $test$plusargs("full_layer");
        stall_enable = !$test$plusargs("nostall");

        vector_dir = "vectors";
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir)) begin
        end
        frontend_hex = "source_vectors/qmap_attention_frontend_image_words32.hex";
        if (!$value$plusargs("FRONTEND_HEX=%s", frontend_hex)) begin
        end
        tail_hex = "source_vectors/qmap_final_token_tail_compact_image_words32.hex";
        if (!$value$plusargs("TAIL_HEX=%s", tail_hex)) begin
        end

        vector_path = {vector_dir, "/embedding/embedding_weight_words32.hex"};
        $readmemh(vector_path, embedding_weight);
        vector_path = {vector_dir, "/embedding/embedding_scale_words32.hex"};
        $readmemh(vector_path, embedding_scale);
        vector_path = {vector_dir, "/embedding/embedding_expected_q14_10.hex"};
        $readmemh(vector_path, embedding_expected);
        vector_path = {vector_dir, "/embedding/embedding_token_id.hex"};
        $readmemh(vector_path, token_file);
        if (full_layer) begin
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_input_rmsnorm_image_words32.hex"};
            $readmemh(vector_path, input_norm_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_input_rmsnorm_expected_words32.hex"};
            $readmemh(vector_path, input_norm_expected);
            vector_path = {vector_dir, "/layer0/qmap/layer0_qkv_from_embedding_rmsnorm_full_image_words32.hex"};
            $readmemh(vector_path, qkv_image);
            vector_path = {vector_dir, "/layer0/qmap/layer0_qkv_from_embedding_rmsnorm_full_expected_words32.hex"};
            $readmemh(vector_path, qkv_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_attention_frontend_image_words32.hex"};
            $readmemh(vector_path, frontend_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_attention_score_value_image_words32.hex"};
            $readmemh(vector_path, score_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_o_proj_image_words32.hex"};
            $readmemh(vector_path, oproj_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_post_attention_residual_norm_image_words32.hex"};
            $readmemh(vector_path, post_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_gate_up_image_words32.hex"};
            $readmemh(vector_path, gate_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_silu_mul_image_words32.hex"};
            $readmemh(vector_path, silu_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_down_image_words32.hex"};
            $readmemh(vector_path, down_image);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_residual_add_image_words32.hex"};
            $readmemh(vector_path, residual_image);

            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_attention_score_stage_real_k_cache.hex"};
            $readmemh(vector_path, k_cache_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_attention_softmax_value_stage_real_v_cache.hex"};
            $readmemh(vector_path, v_cache_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_o_proj_stage_real_weight_words32.hex"};
            $readmemh(vector_path, oproj_weight_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_o_proj_stage_real_scale_words32.hex"};
            $readmemh(vector_path, oproj_scale_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_gate_up_proj_stage_real_gate_weight_words32.hex"};
            $readmemh(vector_path, gate_weight_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_gate_up_proj_stage_real_gate_scale_words32.hex"};
            $readmemh(vector_path, gate_scale_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_gate_up_proj_stage_real_up_weight_words32.hex"};
            $readmemh(vector_path, up_weight_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_gate_up_proj_stage_real_up_scale_words32.hex"};
            $readmemh(vector_path, up_scale_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_down_proj_stage_real_weight_words32.hex"};
            $readmemh(vector_path, down_weight_mem);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_mlp_down_proj_stage_real_scale_words32.hex"};
            $readmemh(vector_path, down_scale_mem);

            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_kv_cache_append_real_expected_addr.hex"};
            $readmemh(vector_path, cache_expected_addr);
            vector_path = {vector_dir, "/layer0/sim_vectors/layer0_chained_kv_cache_append_real_expected_data.hex"};
            $readmemh(vector_path, cache_expected_data);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_attention_frontend_q_rope_expected_words32.hex"};
            $readmemh(vector_path, q_rope_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_attention_score_value_attn_out_expected_words32.hex"};
            $readmemh(vector_path, attn_out_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_o_proj_expected_words32.hex"};
            $readmemh(vector_path, oproj_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_post_attention_residual_norm_expected_hidden_words32.hex"};
            $readmemh(vector_path, post_hidden_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_post_attention_residual_norm_expected_norm_words32.hex"};
            $readmemh(vector_path, post_norm_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_gate_up_expected_gate_words32.hex"};
            $readmemh(vector_path, gate_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_gate_up_expected_up_words32.hex"};
            $readmemh(vector_path, up_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_silu_mul_expected_hidden_words32.hex"};
            $readmemh(vector_path, silu_hidden_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_down_expected_words32.hex"};
            $readmemh(vector_path, down_expected);
            vector_path = {vector_dir, "/layer0/sim_vectors/qmap_layer0_chained_mlp_residual_add_expected_words32.hex"};
            $readmemh(vector_path, layer_expected);
            $readmemh(tail_hex, tail_image);
        end else begin
            vector_path = {vector_dir, "/layer0_input_rmsnorm_image_words32.hex"};
            $readmemh(vector_path, input_norm_image);
            vector_path = {vector_dir, "/layer0_input_rmsnorm_expected_words32.hex"};
            $readmemh(vector_path, input_norm_expected);
            vector_path = {vector_dir, "/layer0_qkv_from_embedding_rmsnorm_image_words32.hex"};
            $readmemh(vector_path, qkv_image);
            vector_path = {vector_dir, "/layer0_qkv_from_embedding_rmsnorm_expected_words32.hex"};
            $readmemh(vector_path, qkv_expected);
            $readmemh(frontend_hex, frontend_image);
        end

        token_id = token_file[0];
        embedding_weight_row_addr = EMBED_WEIGHT_BASE + (token_file[0] * 512);
        embedding_scale_row_addr = EMBED_SCALE_BASE + (token_file[0] * 32);
        input_norm_output_base = input_norm_desc_base(INPUT_NORM_SLOT_OUTPUT);
        q_base = qkv_desc_base(QKV_SLOT_Q_OUT);
        k_base = qkv_desc_base(QKV_SLOT_K_OUT);
        v_base = qkv_desc_base(QKV_SLOT_V_OUT);
        if (full_layer) begin
            q_rope_base = frontend_desc_base(FRONT_SLOT_Q_ROPE);
            attn_out_base = score_desc_base(SCORE_SLOT_ATTN_OUT);
            oproj_base = oproj_desc_base(OPROJ_SLOT_OUTPUT);
            post_hidden_base = post_desc_base(POST_SLOT_HIDDEN);
            post_norm_base = post_desc_base(POST_SLOT_NORM);
            gate_base = gate_desc_base(GATE_SLOT_GATE_OUTPUT);
            up_base = gate_desc_base(GATE_SLOT_UP_OUTPUT);
            silu_hidden_base = silu_desc_base(SILU_SLOT_HIDDEN);
            down_base = down_desc_base(DOWN_SLOT_OUTPUT);
            layer_base = residual_desc_base(RESIDUAL_SLOT_OUTPUT);

            patch_frontend_base(FRONT_SLOT_Q_FLAT, q_base);
            patch_frontend_base(FRONT_SLOT_K_FLAT, k_base);
            patch_frontend_base(FRONT_SLOT_V_FLAT, v_base);
            patch_score_base(SCORE_SLOT_Q_ROPE, q_rope_base);
            patch_oproj_base(OPROJ_SLOT_ACTIVATION, attn_out_base);
            patch_post_base(POST_SLOT_RESIDUAL, INPUT_HIDDEN_BASE);
            patch_post_base(POST_SLOT_O_PROJ, oproj_base);
            patch_gate_base(GATE_SLOT_ACTIVATION, post_norm_base);
            patch_silu_base(SILU_SLOT_GATE, gate_base);
            patch_silu_base(SILU_SLOT_UP, up_base);
            patch_down_base(DOWN_SLOT_ACTIVATION, silu_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_POST_ATTN, post_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_DOWN, down_base);
            tail_image[desc_index(TAIL_SLOT_FINAL_GAMMA, DESC_DTYPE_WORD)] = 32'd0;
        end else begin
            frontend_image[desc_index(FRONT_SLOT_COS, DESC_DTYPE_WORD)] = 32'd5;
        end

        qkv_table = '0;
        input_norm_table = '0;
        frontend_table = '0;
        score_table = '0;
        oproj_table = '0;
        post_table = '0;
        gate_table = '0;
        silu_table = '0;
        down_table = '0;
        residual_table = '0;
        qkv_table[0 +: ADDR_WIDTH] = QKV_QMAP_BASE;
        input_norm_table[0 +: ADDR_WIDTH] = `QMAP_INPUT_NORM_BASE_ADDR;
        frontend_table[0 +: ADDR_WIDTH] = `QMAP_ATTN_FRONTEND_BASE_ADDR;
        score_table[0 +: ADDR_WIDTH] = `QMAP_ATTN_SCORE_VALUE_BASE_ADDR;
        oproj_table[0 +: ADDR_WIDTH] = `QMAP_O_PROJ_BASE_ADDR;
        post_table[0 +: ADDR_WIDTH] = `QMAP_POST_ATTN_NORM_BASE_ADDR;
        gate_table[0 +: ADDR_WIDTH] = `QMAP_MLP_GATE_UP_BASE_ADDR;
        silu_table[0 +: ADDR_WIDTH] = `QMAP_MLP_SILU_MUL_BASE_ADDR;
        down_table[0 +: ADDR_WIDTH] = `QMAP_MLP_DOWN_BASE_ADDR;
        residual_table[0 +: ADDR_WIDTH] = `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR;

        trace_fd = $fopen("timing_trace.csv", "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not create timing_trace.csv");
            $finish(1);
        end
        $fwrite(trace_fd, "cycle,event,address,index_or_kind,data_or_length,last\n");

        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (!done && (cycle_count < 12000000)) begin
            @(posedge clk);
            #1;
        end
        check(done, "embedding -> input RMSNorm -> QKV top timed out");
        if (top_done_cycle < 0) begin
            top_done_cycle = cycle_count;
            if (trace_fd != 0) begin
                $fwrite(trace_fd, "%0d,top_done,0x0000000000000000,0,0x00000000,1\n",
                        cycle_count);
            end
        end
        #1;
        check(!busy, "top remained busy when done asserted");
        check(embedding_write_reqs == 1 && embedding_write_words == VEC1024,
              "embedding write counts mismatch");
        check(input_norm_write_reqs == 1 && input_norm_write_words == VEC1024,
              "input RMSNorm write counts mismatch");
        check(q_write_reqs == VEC2048 && q_write_words == VEC2048,
              "Q projection write counts mismatch");
        check(k_write_reqs == VEC1024 && k_write_words == VEC1024,
              "K projection write counts mismatch");
        check(v_write_reqs == VEC1024 && v_write_words == VEC1024,
              "V projection write counts mismatch");
        check(mismatch_count == 0, "focused chain output words did not match golden data");
        check(unknown_read_count == 0, "focused chain issued unknown memory reads");
        check(embedding_done_cycle >= 0 &&
              first_input_hidden_read_cycle > embedding_done_cycle,
              "input RMSNorm did not wait for embedding write completion");
        check(input_norm_done_cycle >= 0 &&
              first_input_norm_read_cycle > input_norm_done_cycle,
              "QKV did not wait for input RMSNorm write completion");

        if (full_layer) begin
            check(error && tail_error,
                  "full Layer 0 run should stop only on the intentionally invalid tail gamma");
            check(layers_started == 1 && layers_completed == 1,
                  "full Layer 0 scheduler layer counters mismatch");
            check(layer_done_mask[0] && (layer_error_mask == 0),
                  "full Layer 0 scheduler done/error masks mismatch");
            check(stage_done_mask == 2'b11 && stage_error_mask == 0,
                  "input-norm/QKV stage masks mismatch");
            check(full_stage_done_mask == 4'hF && full_stage_error_mask == 0,
                  "attention full-stage masks mismatch");
            check(body_stage_done_mask == 5'h1F && body_stage_error_mask == 0,
                  "Layer 0 body-stage masks mismatch");
            check(tail_start_count == 1 && tail_wr_words == 0,
                  "invalid final tail launch/write counts mismatch");

            check(write_req_count[WRITE_CACHE] == 2*VEC1024 &&
                  write_word_count[WRITE_CACHE] == 2*VEC1024,
                  "KV-cache write counts mismatch");
            check(write_req_count[WRITE_Q_ROPE] == 1 &&
                  write_word_count[WRITE_Q_ROPE] == VEC2048,
                  "Q RoPE write counts mismatch");
            check(write_req_count[WRITE_ATTN_OUT] == 1 &&
                  write_word_count[WRITE_ATTN_OUT] == VEC2048,
                  "attention output write counts mismatch");
            check(write_req_count[WRITE_O_PROJ] == 1 &&
                  write_word_count[WRITE_O_PROJ] == VEC1024,
                  "o_proj write counts mismatch");
            check(write_req_count[WRITE_POST_HIDDEN] == 1 &&
                  write_word_count[WRITE_POST_HIDDEN] == VEC1024 &&
                  write_req_count[WRITE_POST_NORM] == 1 &&
                  write_word_count[WRITE_POST_NORM] == VEC1024,
                  "post-attention write counts mismatch");
            check(write_req_count[WRITE_GATE] == 1 &&
                  write_word_count[WRITE_GATE] == VEC3072 &&
                  write_req_count[WRITE_UP] == 1 &&
                  write_word_count[WRITE_UP] == VEC3072,
                  "gate/up write counts mismatch");
            check(write_req_count[WRITE_SILU_HIDDEN] == 1 &&
                  write_word_count[WRITE_SILU_HIDDEN] == VEC3072,
                  "SiLU/multiply write counts mismatch");
            check(write_req_count[WRITE_DOWN] == 1 &&
                  write_word_count[WRITE_DOWN] == VEC1024 &&
                  write_req_count[WRITE_LAYER] == 1 &&
                  write_word_count[WRITE_LAYER] == VEC1024,
                  "down/final residual write counts mismatch");

            check(scheduler_rd_reqs == EXPECTED_FULL_SCHED_RD_REQS &&
                  scheduler_rd_words == EXPECTED_FULL_SCHED_RD_WORDS &&
                  scheduler_wr_reqs == EXPECTED_FULL_SCHED_WR_REQS &&
                  scheduler_wr_words == EXPECTED_FULL_SCHED_WR_WORDS,
                  "full Layer 0 scheduler memory counters mismatch");
            check(top_rd_reqs == (EXPECTED_FULL_SCHED_RD_REQS + 2 + tail_rd_reqs) &&
                  top_rd_words == (EXPECTED_FULL_SCHED_RD_WORDS + 136 + tail_rd_words) &&
                  top_wr_reqs == (EXPECTED_FULL_SCHED_WR_REQS + 1) &&
                  top_wr_words == (EXPECTED_FULL_SCHED_WR_WORDS + VEC1024),
                  "full Layer 0 aggregate top counters mismatch");

            check(first_frontend_read_cycle > last_qkv_done_cycle,
                  "attention frontend did not wait for QKV completion");
            check(first_score_q_rope_read_cycle > write_done_cycle[WRITE_Q_ROPE] &&
                  first_score_cache_read_cycle > write_done_cycle[WRITE_CACHE],
                  "attention score/value did not wait for frontend writes");
            check(first_oproj_read_cycle > write_done_cycle[WRITE_ATTN_OUT],
                  "o_proj did not wait for attention output");
            check(first_post_read_cycle > write_done_cycle[WRITE_O_PROJ],
                  "post-attention stage did not wait for o_proj");
            check(first_gate_read_cycle > write_done_cycle[WRITE_POST_NORM],
                  "gate/up did not wait for post norm");
            check(first_silu_read_cycle > write_done_cycle[WRITE_GATE] &&
                  first_silu_read_cycle > write_done_cycle[WRITE_UP],
                  "SiLU did not wait for gate/up");
            check(first_down_read_cycle > write_done_cycle[WRITE_SILU_HIDDEN],
                  "down projection did not wait for MLP hidden");
            check(first_residual_post_read_cycle > write_done_cycle[WRITE_POST_HIDDEN] &&
                  first_residual_down_read_cycle > write_done_cycle[WRITE_DOWN],
                  "final residual did not wait for both inputs");
            check(scheduler_done_cycle > write_done_cycle[WRITE_LAYER] &&
                  tail_start_cycle > scheduler_done_cycle &&
                  top_done_cycle > tail_start_cycle,
                  "Layer 0 scheduler/tail/top completion ordering mismatch");
        end else begin
            check(error, "focused chain should stop on the intentionally invalid frontend descriptor");
            check(layers_started == 1 && layers_completed == 0,
                  "scheduler layer started/completed counters mismatch");
            check(layer_done_mask == 0 && layer_error_mask[0],
                  "scheduler layer done/error masks mismatch");
            check(stage_done_mask == 2'b01 && stage_error_mask == 2'b10,
                  "input-norm/QKV scheduler stage masks mismatch");
            check(full_stage_done_mask == 4'h0 && full_stage_error_mask == 4'h1,
                  "frontend error stage masks mismatch");
            check(tail_start_count == 0, "final-token tail unexpectedly started");
            check(scheduler_rd_reqs == EXPECTED_SCHED_RD_REQS &&
                  scheduler_rd_words == EXPECTED_SCHED_RD_WORDS &&
                  scheduler_wr_reqs == EXPECTED_SCHED_WR_REQS &&
                  scheduler_wr_words == EXPECTED_SCHED_WR_WORDS,
                  "scheduler memory counters mismatch");
            check(top_rd_reqs == EXPECTED_TOP_RD_REQS &&
                  top_rd_words == EXPECTED_TOP_RD_WORDS &&
                  top_wr_reqs == EXPECTED_TOP_WR_REQS &&
                  top_wr_words == EXPECTED_TOP_WR_WORDS,
                  "aggregate top memory counters mismatch");
            check(last_qkv_done_cycle >= 0 &&
                  scheduler_done_cycle > last_qkv_done_cycle &&
                  top_done_cycle > scheduler_done_cycle,
                  "scheduler/top completion ordering mismatch");
        end

        if (trace_fd != 0) $fclose(trace_fd);
        if (fail_count != 0) begin
            $display("FAIL: focused embedding Layer 0 chain saw %0d failure(s), mismatches=%0d",
                     fail_count, mismatch_count);
            $finish(1);
        end

        if (full_layer) begin
            $display("PASS: tied-Q4 embedding fed the complete Layer 0 scheduler exactly.");
        end else begin
            $display("PASS: tied-Q4 embedding fed Layer 0 input RMSNorm and full QKV exactly.");
        end
        $display("  cycles embedding_done/input_read = %0d/%0d",
                 embedding_done_cycle, first_input_hidden_read_cycle);
        $display("  cycles norm_done/qkv_read       = %0d/%0d",
                 input_norm_done_cycle, first_input_norm_read_cycle);
        $display("  cycles qkv_done/scheduler/top   = %0d/%0d/%0d",
                 last_qkv_done_cycle, scheduler_done_cycle, top_done_cycle);
        $display("  scheduler rd/wr = %0d/%0d, %0d/%0d",
                 scheduler_rd_reqs, scheduler_rd_words,
                 scheduler_wr_reqs, scheduler_wr_words);
        $display("  top rd/wr       = %0d/%0d, %0d/%0d",
                 top_rd_reqs, top_rd_words, top_wr_reqs, top_wr_words);
        $finish;
    end

endmodule

`default_nettype wire
