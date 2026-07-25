`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// Focused integration proof for persistent KV-cache behavior.  The real
// QMAP attention frontend and score/value wrappers share one memory model.
// The model is initialized once and is deliberately not cleared between
// runtime positions 0, 1, and 255.
module tb_qmap_attention_persistent_kv_two_token;

    localparam int ADDR_WIDTH       = 64;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int NUM_LAYERS       = 2;
    localparam int NUM_Q_HEADS      = 2;
    localparam int NUM_KV_HEADS     = 1;
    localparam int HEAD_DIM         = 4;
    localparam int MAX_CONTEXT      = 256;
    localparam int IN_WIDTH         = 24;
    localparam int OUT_WIDTH        = 24;
    localparam int EXP_LUT_SIZE     = 257;
    localparam int LAYER_INDEX_W    = 1;
    localparam int POSITION_INDEX_W = 8;
    localparam int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT         = NUM_KV_HEADS * HEAD_DIM;
    localparam int KV_REPEAT        = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam int KV_KIND_WORDS    = NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM;
    localparam int KV_LAYER_WORDS   = 2 * KV_KIND_WORDS;
    localparam int KV_WORDS         = NUM_LAYERS * KV_LAYER_WORDS;
    localparam int ROPE_WORDS       = MAX_CONTEXT * HEAD_DIM;
    localparam int FRONT_IMAGE_BYTES = 16'h1000;
    localparam int SCORE_IMAGE_BYTES = 16'h1000;
    localparam int FRONT_WORDS       = FRONT_IMAGE_BYTES / 4;
    localparam int SCORE_WORDS       = SCORE_IMAGE_BYTES / 4;

    localparam int FRONT_DESC_OFFSET = 16'h0100;
    localparam int SCORE_DESC_OFFSET = 16'h0100;
    localparam int DESC_BYTES        = 128;
    localparam int FRONT_METADATA_OFFSET = 16'h0700;
    localparam int FRONT_Q_OFFSET        = 16'h0800;
    localparam int FRONT_K_OFFSET        = 16'h0840;
    localparam int FRONT_V_OFFSET        = 16'h0880;
    localparam int FRONT_Q_GAMMA_OFFSET  = 16'h08C0;
    localparam int FRONT_K_GAMMA_OFFSET  = 16'h0900;
    localparam int SCORE_METADATA_OFFSET = 16'h0400;
    localparam int SCORE_EXP_OFFSET      = 16'h0500;

    localparam logic [ADDR_WIDTH-1:0] FRONT_QMAP_BASE = 64'h0000_0004_3000_0000;
    localparam logic [ADDR_WIDTH-1:0] SCORE_QMAP_BASE = 64'h0000_0004_3001_0000;
    localparam logic [ADDR_WIDTH-1:0] COS_BASE        = 64'h0000_0004_3010_0000;
    localparam logic [ADDR_WIDTH-1:0] SIN_BASE        = 64'h0000_0004_3011_0000;
    localparam logic [ADDR_WIDTH-1:0] KV_BASE         = 64'h0000_0004_3020_0000;
    localparam logic [ADDR_WIDTH-1:0] DESCRIPTOR_KV_POISON = 64'h0000_0004_3021_0000;
    localparam logic [ADDR_WIDTH-1:0] QROPE_BASE      = 64'h0000_0004_3030_0000;
    localparam logic [ADDR_WIDTH-1:0] ATTN_OUT_BASE   = 64'h0000_0004_3030_1000;

    localparam logic [1:0] ACTIVE_NONE  = 2'd0;
    localparam logic [1:0] ACTIVE_FRONT = 2'd1;
    localparam logic [1:0] ACTIVE_SCORE = 2'd2;

    logic clk;
    logic rst_n;
    logic [1:0] active_dut;

    logic front_start;
    logic front_busy;
    logic front_done;
    logic front_error;
    logic front_saturation;
    logic front_norm_saturation;
    logic front_rope_saturation;
    logic [31:0] front_cache_write_count;
    logic [31:0] front_qrope_write_count;
    logic front_rd_req_valid;
    logic front_rd_req_ready;
    logic [ADDR_WIDTH-1:0] front_rd_req_addr;
    logic [15:0] front_rd_req_len;
    logic front_rd_rsp_valid;
    logic front_rd_rsp_ready;
    logic [31:0] front_rd_rsp_data;
    logic front_rd_rsp_last;
    logic front_wr_req_valid;
    logic front_wr_req_ready;
    logic [ADDR_WIDTH-1:0] front_wr_req_addr;
    logic [15:0] front_wr_req_len;
    logic [31:0] front_wr_data;
    logic front_wr_data_valid;
    logic front_wr_data_ready;
    logic front_wr_data_last;
    logic front_wr_done;
    logic front_wr_error;
    logic [POSITION_INDEX_W-1:0] front_runtime_position;

    logic score_start;
    logic score_busy;
    logic score_done;
    logic score_error;
    logic score_saturation;
    logic [31:0] score_count;
    logic [31:0] score_k_read_count;
    logic [31:0] score_v_read_count;
    logic [31:0] score_attn_capture_count;
    logic [31:0] score_attn_write_count;
    logic score_rd_req_valid;
    logic score_rd_req_ready;
    logic [ADDR_WIDTH-1:0] score_rd_req_addr;
    logic [15:0] score_rd_req_len;
    logic score_rd_rsp_valid;
    logic score_rd_rsp_ready;
    logic [31:0] score_rd_rsp_data;
    logic score_rd_rsp_last;
    logic score_wr_req_valid;
    logic score_wr_req_ready;
    logic [ADDR_WIDTH-1:0] score_wr_req_addr;
    logic [15:0] score_wr_req_len;
    logic [31:0] score_wr_data;
    logic score_wr_data_valid;
    logic score_wr_data_ready;
    logic score_wr_data_last;
    logic score_wr_done;
    logic score_wr_error;
    logic [POSITION_INDEX_W-1:0] score_runtime_position;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1:0] mem_rd_req_addr;
    logic [15:0] mem_rd_req_len;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [31:0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;
    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [ADDR_WIDTH-1:0] mem_wr_req_addr;
    logic [15:0] mem_wr_req_len;
    logic [31:0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    logic [31:0] front_mem [0:FRONT_WORDS-1];
    logic [31:0] score_mem [0:SCORE_WORDS-1];
    logic [31:0] cos_mem [0:ROPE_WORDS-1];
    logic [31:0] sin_mem [0:ROPE_WORDS-1];
    logic [31:0] kv_mem [0:KV_WORDS-1];
    logic [31:0] qrope_mem [0:Q_COUNT-1];
    logic [31:0] attn_mem [0:Q_COUNT-1];
    logic [31:0] k_pos0_snapshot [0:KV_COUNT-1];
    logic [31:0] v_pos0_snapshot [0:KV_COUNT-1];
    logic [31:0] qrope_pos0_snapshot [0:Q_COUNT-1];

    logic rd_active;
    logic [ADDR_WIDTH-1:0] rd_base_addr;
    integer rd_word_count;
    integer rd_word_index;
    logic wr_active;
    logic [ADDR_WIDTH-1:0] wr_base_addr;
    integer wr_word_count;
    integer wr_word_index;
    integer wr_done_countdown;

    integer mismatch_count;
    integer print_count;
    integer frontend_expected_position;
    integer score_expected_cache_length;
    integer frontend_cache_req_count;
    integer frontend_cache_data_count;
    integer frontend_qrope_req_count;
    integer frontend_qrope_data_count;
    integer score_k_req_count;
    integer score_v_req_count;
    integer score_attn_req_count;
    integer score_attn_data_count;
    logic [ADDR_WIDTH-1:0] score_first_k_addr;
    logic [ADDR_WIDTH-1:0] score_last_k_addr;
    logic [ADDR_WIDTH-1:0] score_first_v_addr;
    logic [ADDR_WIDTH-1:0] score_last_v_addr;

    qmap_attention_frontend_compute_path #(
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CONTEXT(MAX_CONTEXT)
    ) frontend_dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(front_start),
        .i_qmap_base_addr(FRONT_QMAP_BASE),
        .i_runtime_context_valid(1'b1),
        .i_runtime_layer_id({LAYER_INDEX_W{1'b0}}),
        .i_runtime_position(front_runtime_position),
        .i_runtime_kv_cache_base_addr(KV_BASE),
        .o_busy(front_busy),
        .o_done(front_done),
        .o_error(front_error),
        .o_saturation(front_saturation),
        .o_norm_saturation(front_norm_saturation),
        .o_rope_saturation(front_rope_saturation),
        .o_cache_write_count(front_cache_write_count),
        .o_q_rope_write_word_count(front_qrope_write_count),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(front_rd_req_valid),
        .i_mem_rd_req_ready(front_rd_req_ready),
        .o_mem_rd_req_addr(front_rd_req_addr),
        .o_mem_rd_req_len_bytes(front_rd_req_len),
        .i_mem_rd_rsp_valid(front_rd_rsp_valid),
        .o_mem_rd_rsp_ready(front_rd_rsp_ready),
        .i_mem_rd_rsp_data(front_rd_rsp_data),
        .i_mem_rd_rsp_last(front_rd_rsp_last),
        .o_mem_wr_req_valid(front_wr_req_valid),
        .i_mem_wr_req_ready(front_wr_req_ready),
        .o_mem_wr_req_addr(front_wr_req_addr),
        .o_mem_wr_req_len_bytes(front_wr_req_len),
        .o_mem_wr_data(front_wr_data),
        .o_mem_wr_data_valid(front_wr_data_valid),
        .i_mem_wr_data_ready(front_wr_data_ready),
        .o_mem_wr_data_last(front_wr_data_last),
        .i_mem_wr_done(front_wr_done),
        .i_mem_wr_error(front_wr_error)
    );

    qmap_attention_score_value_compute_path #(
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .MAX_CONTEXT(MAX_CONTEXT)
    ) score_dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(score_start),
        .i_qmap_base_addr(SCORE_QMAP_BASE),
        .i_runtime_context_valid(1'b1),
        .i_runtime_layer_id({LAYER_INDEX_W{1'b0}}),
        .i_runtime_position(score_runtime_position),
        .i_runtime_kv_cache_base_addr(KV_BASE),
        .o_busy(score_busy),
        .o_done(score_done),
        .o_error(score_error),
        .o_saturation(score_saturation),
        .o_score_count(score_count),
        .o_k_read_count(score_k_read_count),
        .o_v_read_count(score_v_read_count),
        .o_attn_out_capture_count(score_attn_capture_count),
        .o_attn_out_write_word_count(score_attn_write_count),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(score_rd_req_valid),
        .i_mem_rd_req_ready(score_rd_req_ready),
        .o_mem_rd_req_addr(score_rd_req_addr),
        .o_mem_rd_req_len_bytes(score_rd_req_len),
        .i_mem_rd_rsp_valid(score_rd_rsp_valid),
        .o_mem_rd_rsp_ready(score_rd_rsp_ready),
        .i_mem_rd_rsp_data(score_rd_rsp_data),
        .i_mem_rd_rsp_last(score_rd_rsp_last),
        .o_mem_wr_req_valid(score_wr_req_valid),
        .i_mem_wr_req_ready(score_wr_req_ready),
        .o_mem_wr_req_addr(score_wr_req_addr),
        .o_mem_wr_req_len_bytes(score_wr_req_len),
        .o_mem_wr_data(score_wr_data),
        .o_mem_wr_data_valid(score_wr_data_valid),
        .i_mem_wr_data_ready(score_wr_data_ready),
        .o_mem_wr_data_last(score_wr_data_last),
        .i_mem_wr_done(score_wr_done),
        .i_mem_wr_error(score_wr_error)
    );

    always @* begin
        mem_rd_req_valid = 1'b0;
        mem_rd_req_addr = '0;
        mem_rd_req_len = 16'd0;
        mem_rd_rsp_ready = 1'b0;
        mem_wr_req_valid = 1'b0;
        mem_wr_req_addr = '0;
        mem_wr_req_len = 16'd0;
        mem_wr_data = 32'd0;
        mem_wr_data_valid = 1'b0;
        mem_wr_data_last = 1'b0;
        case (active_dut)
            ACTIVE_FRONT: begin
                mem_rd_req_valid = front_rd_req_valid;
                mem_rd_req_addr = front_rd_req_addr;
                mem_rd_req_len = front_rd_req_len;
                mem_rd_rsp_ready = front_rd_rsp_ready;
                mem_wr_req_valid = front_wr_req_valid;
                mem_wr_req_addr = front_wr_req_addr;
                mem_wr_req_len = front_wr_req_len;
                mem_wr_data = front_wr_data;
                mem_wr_data_valid = front_wr_data_valid;
                mem_wr_data_last = front_wr_data_last;
            end
            ACTIVE_SCORE: begin
                mem_rd_req_valid = score_rd_req_valid;
                mem_rd_req_addr = score_rd_req_addr;
                mem_rd_req_len = score_rd_req_len;
                mem_rd_rsp_ready = score_rd_rsp_ready;
                mem_wr_req_valid = score_wr_req_valid;
                mem_wr_req_addr = score_wr_req_addr;
                mem_wr_req_len = score_wr_req_len;
                mem_wr_data = score_wr_data;
                mem_wr_data_valid = score_wr_data_valid;
                mem_wr_data_last = score_wr_data_last;
            end
            default: begin
            end
        endcase
    end

    assign front_rd_req_ready = (active_dut == ACTIVE_FRONT) ? mem_rd_req_ready : 1'b0;
    assign front_rd_rsp_valid = (active_dut == ACTIVE_FRONT) ? mem_rd_rsp_valid : 1'b0;
    assign front_rd_rsp_data  = mem_rd_rsp_data;
    assign front_rd_rsp_last  = mem_rd_rsp_last;
    assign front_wr_req_ready = (active_dut == ACTIVE_FRONT) ? mem_wr_req_ready : 1'b0;
    assign front_wr_data_ready = (active_dut == ACTIVE_FRONT) ? mem_wr_data_ready : 1'b0;
    assign front_wr_done = (active_dut == ACTIVE_FRONT) ? mem_wr_done : 1'b0;
    assign front_wr_error = (active_dut == ACTIVE_FRONT) ? mem_wr_error : 1'b0;

    assign score_rd_req_ready = (active_dut == ACTIVE_SCORE) ? mem_rd_req_ready : 1'b0;
    assign score_rd_rsp_valid = (active_dut == ACTIVE_SCORE) ? mem_rd_rsp_valid : 1'b0;
    assign score_rd_rsp_data  = mem_rd_rsp_data;
    assign score_rd_rsp_last  = mem_rd_rsp_last;
    assign score_wr_req_ready = (active_dut == ACTIVE_SCORE) ? mem_wr_req_ready : 1'b0;
    assign score_wr_data_ready = (active_dut == ACTIVE_SCORE) ? mem_wr_data_ready : 1'b0;
    assign score_wr_done = (active_dut == ACTIVE_SCORE) ? mem_wr_done : 1'b0;
    assign score_wr_error = (active_dut == ACTIVE_SCORE) ? mem_wr_error : 1'b0;

    assign mem_rd_req_ready = rst_n && !rd_active && !mem_rd_rsp_valid;
    assign mem_wr_req_ready = rst_n && !wr_active && (wr_done_countdown == 0);
    assign mem_wr_data_ready = rst_n && wr_active;

    function automatic logic in_region(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [ADDR_WIDTH-1:0] base,
        input integer byte_count
    );
        begin
            in_region = (addr >= base) && (addr < (base + byte_count));
        end
    endfunction

    function automatic logic range_in_region(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [15:0] length,
        input logic [ADDR_WIDTH-1:0] base,
        input integer byte_count
    );
        begin
            range_in_region =
                (length != 0) &&
                (addr >= base) &&
                ((addr + length) <= (base + byte_count));
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1:0] kv_addr(
        input integer layer,
        input integer kind,
        input integer head,
        input integer position,
        input integer dim
    );
        integer word_index;
        begin
            word_index =
                (layer * KV_LAYER_WORDS) +
                (kind * KV_KIND_WORDS) +
                (head * MAX_CONTEXT * HEAD_DIM) +
                (position * HEAD_DIM) + dim;
            kv_addr = KV_BASE + (word_index * 4);
        end
    endfunction

    function automatic integer kv_word_index(
        input integer layer,
        input integer kind,
        input integer head,
        input integer position,
        input integer dim
    );
        begin
            kv_word_index =
                (layer * KV_LAYER_WORDS) +
                (kind * KV_KIND_WORDS) +
                (head * MAX_CONTEXT * HEAD_DIM) +
                (position * HEAD_DIM) + dim;
        end
    endfunction

    function automatic logic signed [23:0] qk_value(input integer dim);
        begin
            case (dim)
                0: qk_value = 24'sd4096;
                1: qk_value = -24'sd4096;
                2: qk_value = 24'sd2048;
                default: qk_value = -24'sd2048;
            endcase
        end
    endfunction

    function automatic logic signed [23:0] token_v_value(
        input integer position,
        input integer dim
    );
        integer units;
        begin
            case (position)
                0: units = dim + 1;
                1: units = dim + 3;
                255: units = dim + 5;
                default: units = 0;
            endcase
            token_v_value = units * 256;
        end
    endfunction

    function automatic logic [31:0] expected_attn_word(
        input integer runtime_position,
        input integer dim
    );
        integer value;
        begin
            case (runtime_position)
                0: value = token_v_value(0, dim);
                1: value = (token_v_value(0, dim) + token_v_value(1, dim)) / 2;
                255: value =
                    (token_v_value(0, dim) + token_v_value(1, dim) +
                     token_v_value(255, dim)) / 256;
                default: value = 0;
            endcase
            expected_attn_word = value;
        end
    endfunction

    function automatic logic [31:0] read_word(input logic [ADDR_WIDTH-1:0] addr);
        integer index;
        begin
            if (in_region(addr, FRONT_QMAP_BASE, FRONT_IMAGE_BYTES)) begin
                index = (addr - FRONT_QMAP_BASE) >> 2;
                read_word = front_mem[index];
            end
            else if (in_region(addr, SCORE_QMAP_BASE, SCORE_IMAGE_BYTES)) begin
                index = (addr - SCORE_QMAP_BASE) >> 2;
                read_word = score_mem[index];
            end
            else if (in_region(addr, COS_BASE, ROPE_WORDS * 4)) begin
                index = (addr - COS_BASE) >> 2;
                read_word = cos_mem[index];
            end
            else if (in_region(addr, SIN_BASE, ROPE_WORDS * 4)) begin
                index = (addr - SIN_BASE) >> 2;
                read_word = sin_mem[index];
            end
            else if (in_region(addr, KV_BASE, KV_WORDS * 4)) begin
                index = (addr - KV_BASE) >> 2;
                read_word = kv_mem[index];
            end
            else if (in_region(addr, QROPE_BASE, Q_COUNT * 4)) begin
                index = (addr - QROPE_BASE) >> 2;
                read_word = qrope_mem[index];
            end
            else begin
                read_word = 32'hBAD0_ADD0;
            end
        end
    endfunction

    function automatic logic supported_read_range(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [15:0] length
    );
        begin
            supported_read_range =
                range_in_region(addr, length, FRONT_QMAP_BASE, FRONT_IMAGE_BYTES) ||
                range_in_region(addr, length, SCORE_QMAP_BASE, SCORE_IMAGE_BYTES) ||
                range_in_region(addr, length, COS_BASE, ROPE_WORDS * 4) ||
                range_in_region(addr, length, SIN_BASE, ROPE_WORDS * 4) ||
                range_in_region(addr, length, KV_BASE, KV_WORDS * 4) ||
                range_in_region(addr, length, QROPE_BASE, Q_COUNT * 4);
        end
    endfunction

    task automatic record_failure(input string message);
        begin
            mismatch_count = mismatch_count + 1;
            if (print_count < 64) begin
                $display("FAIL: %s", message);
                print_count = print_count + 1;
            end
        end
    endtask

    task automatic set_descriptor(
        input integer image_select,
        input integer slot,
        input logic [31:0] tensor_id,
        input logic [31:0] role,
        input logic [31:0] dtype,
        input logic [31:0] rank,
        input logic [31:0] element_bits,
        input logic [63:0] base_addr,
        input logic [63:0] nbytes,
        input logic [31:0] dim0,
        input logic [31:0] dim1,
        input logic [31:0] dim2,
        input logic [31:0] dim3,
        input logic [63:0] stride0,
        input logic [63:0] stride1,
        input logic [63:0] stride2,
        input logic [63:0] stride3,
        input logic [31:0] aux0,
        input logic [31:0] aux1,
        input logic [31:0] aux2,
        input logic [31:0] aux3
    );
        logic [31:0] words [0:31];
        integer i;
        integer base_word;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                words[i] = 32'd0;
            end
            words[0] = tensor_id;
            words[1] = role;
            words[2] = dtype;
            words[3] = rank;
            words[4] = `QMAP_TENSOR_F_ROW_MAJOR;
            words[5] = element_bits;
            words[6] = 32'd0;
            words[7] = `QMAP_NO_TENSOR_ID;
            words[8] = base_addr[31:0];
            words[9] = base_addr[63:32];
            words[10] = nbytes[31:0];
            words[11] = nbytes[63:32];
            words[12] = dim0;
            words[13] = dim1;
            words[14] = dim2;
            words[15] = dim3;
            words[16] = stride0[31:0];
            words[17] = stride0[63:32];
            words[18] = stride1[31:0];
            words[19] = stride1[63:32];
            words[20] = stride2[31:0];
            words[21] = stride2[63:32];
            words[22] = stride3[31:0];
            words[23] = stride3[63:32];
            words[24] = aux0;
            words[25] = aux1;
            words[26] = aux2;
            words[27] = aux3;
            if (image_select == ACTIVE_FRONT) begin
                base_word = (FRONT_DESC_OFFSET + (slot * DESC_BYTES)) >> 2;
                for (i = 0; i < 32; i = i + 1) begin
                    front_mem[base_word + i] = words[i];
                end
            end
            else begin
                base_word = (SCORE_DESC_OFFSET + (slot * DESC_BYTES)) >> 2;
                for (i = 0; i < 32; i = i + 1) begin
                    score_mem[base_word + i] = words[i];
                end
            end
        end
    endtask

    task automatic build_qmap_images;
        integer i;
        integer q_head;
        integer dim;
        logic signed [23:0] value24;
        begin
            front_mem[0] = `QMAP_MAGIC;
            front_mem[1] = `QMAP_VERSION;
            front_mem[2] = `QMAP_HEADER_BYTES;
            front_mem[3] = `QMAP_DESCRIPTOR_BYTES;
            front_mem[4] = `QMAP_ATTN_FRONTEND_DESCRIPTOR_COUNT;
            front_mem[5] = `QMAP_ATTN_FRONTEND_DESCRIPTOR_CAPACITY;
            front_mem[6] = (FRONT_QMAP_BASE + FRONT_DESC_OFFSET);
            front_mem[7] = (FRONT_QMAP_BASE + FRONT_DESC_OFFSET) >> 32;
            front_mem[8] = (FRONT_QMAP_BASE + FRONT_METADATA_OFFSET);
            front_mem[9] = (FRONT_QMAP_BASE + FRONT_METADATA_OFFSET) >> 32;
            front_mem[10] = FRONT_QMAP_BASE;
            front_mem[11] = FRONT_QMAP_BASE >> 32;
            front_mem[12] = FRONT_IMAGE_BYTES;
            front_mem[13] = 32'd0;
            front_mem[14] = 32'd0;
            front_mem[15] = 32'd0;

            score_mem[0] = `QMAP_MAGIC;
            score_mem[1] = `QMAP_VERSION;
            score_mem[2] = `QMAP_HEADER_BYTES;
            score_mem[3] = `QMAP_DESCRIPTOR_BYTES;
            score_mem[4] = `QMAP_ATTN_SCORE_VALUE_DESCRIPTOR_COUNT;
            score_mem[5] = `QMAP_ATTN_SCORE_VALUE_DESCRIPTOR_CAPACITY;
            score_mem[6] = (SCORE_QMAP_BASE + SCORE_DESC_OFFSET);
            score_mem[7] = (SCORE_QMAP_BASE + SCORE_DESC_OFFSET) >> 32;
            score_mem[8] = (SCORE_QMAP_BASE + SCORE_METADATA_OFFSET);
            score_mem[9] = (SCORE_QMAP_BASE + SCORE_METADATA_OFFSET) >> 32;
            score_mem[10] = SCORE_QMAP_BASE;
            score_mem[11] = SCORE_QMAP_BASE >> 32;
            score_mem[12] = SCORE_IMAGE_BYTES;
            score_mem[13] = 32'd0;
            score_mem[14] = 32'd0;
            score_mem[15] = 32'd0;

            set_descriptor(ACTIVE_FRONT, 0, `QMAP_TENSOR_ID_ATTN_METADATA,
                `QMAP_ROLE_METADATA, `QMAP_DTYPE_U32, 1, 32,
                FRONT_QMAP_BASE + FRONT_METADATA_OFFSET, 64, 16, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 7);
            set_descriptor(ACTIVE_FRONT, 1, `QMAP_TENSOR_ID_ATTN_Q_FLAT,
                `QMAP_ROLE_ACTIVATION, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                FRONT_QMAP_BASE + FRONT_Q_OFFSET, Q_COUNT * 4, Q_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 2, `QMAP_TENSOR_ID_ATTN_K_FLAT,
                `QMAP_ROLE_ACTIVATION, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                FRONT_QMAP_BASE + FRONT_K_OFFSET, KV_COUNT * 4, KV_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 3, `QMAP_TENSOR_ID_ATTN_V_FLAT,
                `QMAP_ROLE_ACTIVATION, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                FRONT_QMAP_BASE + FRONT_V_OFFSET, KV_COUNT * 4, KV_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 4, `QMAP_TENSOR_ID_ATTN_Q_GAMMA,
                `QMAP_ROLE_PARAMETER, `QMAP_DTYPE_I16_Q8_7, 1, 16,
                FRONT_QMAP_BASE + FRONT_Q_GAMMA_OFFSET, HEAD_DIM * 4,
                HEAD_DIM, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 5, `QMAP_TENSOR_ID_ATTN_K_GAMMA,
                `QMAP_ROLE_PARAMETER, `QMAP_DTYPE_I16_Q8_7, 1, 16,
                FRONT_QMAP_BASE + FRONT_K_GAMMA_OFFSET, HEAD_DIM * 4,
                HEAD_DIM, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 6, `QMAP_TENSOR_ID_ATTN_ROPE_COS,
                `QMAP_ROLE_ROPE_TABLE, `QMAP_DTYPE_I16_Q1_15, 2, 16,
                COS_BASE, ROPE_WORDS * 4, MAX_CONTEXT, HEAD_DIM, 0, 0,
                HEAD_DIM * 4, 4, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 7, `QMAP_TENSOR_ID_ATTN_ROPE_SIN,
                `QMAP_ROLE_ROPE_TABLE, `QMAP_DTYPE_I16_Q1_15, 2, 16,
                SIN_BASE, ROPE_WORDS * 4, MAX_CONTEXT, HEAD_DIM, 0, 0,
                HEAD_DIM * 4, 4, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 8, `QMAP_TENSOR_ID_ATTN_KV_CACHE,
                `QMAP_ROLE_KV_CACHE, `QMAP_DTYPE_I32_Q12_12, 4, 24,
                DESCRIPTOR_KV_POISON, KV_LAYER_WORDS * 4, 2, NUM_KV_HEADS, MAX_CONTEXT, HEAD_DIM,
                KV_KIND_WORDS * 4, MAX_CONTEXT * HEAD_DIM * 4,
                HEAD_DIM * 4, 4, 0, 0, 0, 0);
            set_descriptor(ACTIVE_FRONT, 9, `QMAP_TENSOR_ID_ATTN_Q_ROPE,
                `QMAP_ROLE_OUTPUT, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                QROPE_BASE, Q_COUNT * 4, Q_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);

            set_descriptor(ACTIVE_SCORE, 0, `QMAP_TENSOR_ID_ATTN_SV_METADATA,
                `QMAP_ROLE_METADATA, `QMAP_DTYPE_U32, 1, 32,
                SCORE_QMAP_BASE + SCORE_METADATA_OFFSET, 64, 16, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 4, 0);
            set_descriptor(ACTIVE_SCORE, 1, `QMAP_TENSOR_ID_ATTN_SV_Q_ROPE,
                `QMAP_ROLE_ACTIVATION, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                QROPE_BASE, Q_COUNT * 4, Q_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_SCORE, 2, `QMAP_TENSOR_ID_ATTN_SV_KV_CACHE,
                `QMAP_ROLE_KV_CACHE, `QMAP_DTYPE_I32_Q12_12, 4, 24,
                DESCRIPTOR_KV_POISON, KV_LAYER_WORDS * 4, 2, NUM_KV_HEADS, MAX_CONTEXT, HEAD_DIM,
                KV_KIND_WORDS * 4, MAX_CONTEXT * HEAD_DIM * 4,
                HEAD_DIM * 4, 4, 0, 0, 0, 0);
            set_descriptor(ACTIVE_SCORE, 3, `QMAP_TENSOR_ID_ATTN_SV_EXP_LUT,
                `QMAP_ROLE_PARAMETER, `QMAP_DTYPE_U32_Q0_20, 1, 24,
                SCORE_QMAP_BASE + SCORE_EXP_OFFSET, EXP_LUT_SIZE * 4,
                EXP_LUT_SIZE, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0);
            set_descriptor(ACTIVE_SCORE, 4, `QMAP_TENSOR_ID_ATTN_SV_ATTN_OUT,
                `QMAP_ROLE_OUTPUT, `QMAP_DTYPE_I32_Q12_12, 1, 24,
                ATTN_OUT_BASE, Q_COUNT * 4, Q_COUNT, 0, 0, 0,
                4, 0, 0, 0, 0, 0, 0, 0);

            for (q_head = 0; q_head < NUM_Q_HEADS; q_head = q_head + 1) begin
                for (dim = 0; dim < HEAD_DIM; dim = dim + 1) begin
                    value24 = qk_value(dim);
                    front_mem[(FRONT_Q_OFFSET >> 2) + q_head * HEAD_DIM + dim] =
                        {{8{value24[23]}}, value24};
                end
            end
            for (dim = 0; dim < HEAD_DIM; dim = dim + 1) begin
                value24 = qk_value(dim);
                front_mem[(FRONT_K_OFFSET >> 2) + dim] = {{8{value24[23]}}, value24};
                front_mem[(FRONT_Q_GAMMA_OFFSET >> 2) + dim] = 32'd128;
                front_mem[(FRONT_K_GAMMA_OFFSET >> 2) + dim] = 32'd128;
            end
            for (i = 0; i < EXP_LUT_SIZE; i = i + 1) begin
                score_mem[(SCORE_EXP_OFFSET >> 2) + i] = 32'h0010_0000;
            end
        end
    endtask

    task automatic load_frontend_v(input integer runtime_position);
        integer dim;
        logic signed [23:0] value24;
        begin
            for (dim = 0; dim < HEAD_DIM; dim = dim + 1) begin
                value24 = token_v_value(runtime_position, dim);
                front_mem[(FRONT_V_OFFSET >> 2) + dim] = {{8{value24[23]}}, value24};
            end
        end
    endtask

    task automatic reset_frontend_scoreboard(input integer runtime_position);
        begin
            frontend_expected_position = runtime_position;
            frontend_cache_req_count = 0;
            frontend_cache_data_count = 0;
            frontend_qrope_req_count = 0;
            frontend_qrope_data_count = 0;
        end
    endtask

    task automatic reset_score_scoreboard(input integer runtime_position);
        integer i;
        begin
            score_expected_cache_length = runtime_position + 1;
            score_k_req_count = 0;
            score_v_req_count = 0;
            score_attn_req_count = 0;
            score_attn_data_count = 0;
            score_first_k_addr = '0;
            score_last_k_addr = '0;
            score_first_v_addr = '0;
            score_last_v_addr = '0;
            for (i = 0; i < Q_COUNT; i = i + 1) begin
                attn_mem[i] = 32'hDEAD_BEEF;
            end
        end
    endtask

    task automatic audit_cache_read(input logic [ADDR_WIDTH-1:0] addr);
        integer request_index;
        integer q_head;
        integer position;
        integer dim;
        integer kv_head;
        integer kind;
        logic [ADDR_WIDTH-1:0] expected_addr;
        begin
            kind = ((addr - KV_BASE) >> 2) / KV_KIND_WORDS;
            if (kind == 0) begin
                request_index = score_k_req_count;
                q_head = request_index / (score_expected_cache_length * HEAD_DIM);
                position = (request_index / HEAD_DIM) % score_expected_cache_length;
                dim = request_index % HEAD_DIM;
                kv_head = q_head / KV_REPEAT;
                expected_addr = kv_addr(0, 0, kv_head, position, dim);
                if (addr !== expected_addr) begin
                    record_failure($sformatf(
                        "K read %0d addr 0x%016h expected 0x%016h",
                        request_index, addr, expected_addr));
                end
                if (score_k_req_count == 0) begin
                    score_first_k_addr = addr;
                end
                score_last_k_addr = addr;
                score_k_req_count = score_k_req_count + 1;
            end
            else if (kind == 1) begin
                request_index = score_v_req_count;
                q_head = request_index / (HEAD_DIM * score_expected_cache_length);
                dim = (request_index / score_expected_cache_length) % HEAD_DIM;
                position = request_index % score_expected_cache_length;
                kv_head = q_head / KV_REPEAT;
                expected_addr = kv_addr(0, 1, kv_head, position, dim);
                if (addr !== expected_addr) begin
                    record_failure($sformatf(
                        "V read %0d addr 0x%016h expected 0x%016h",
                        request_index, addr, expected_addr));
                end
                if (score_v_req_count == 0) begin
                    score_first_v_addr = addr;
                end
                score_last_v_addr = addr;
                score_v_req_count = score_v_req_count + 1;
            end
            else begin
                record_failure($sformatf("score read outside layer 0 KV window: 0x%016h", addr));
            end
        end
    endtask

    task automatic audit_write_request(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [15:0] length
    );
        integer request_index;
        integer kind;
        integer head;
        integer dim;
        logic [ADDR_WIDTH-1:0] expected_addr;
        begin
            if (active_dut == ACTIVE_FRONT) begin
                if (in_region(addr, KV_BASE, KV_WORDS * 4)) begin
                    request_index = frontend_cache_req_count;
                    kind = request_index / KV_COUNT;
                    head = (request_index % KV_COUNT) / HEAD_DIM;
                    dim = request_index % HEAD_DIM;
                    expected_addr = kv_addr(0, kind, head, frontend_expected_position, dim);
                    if ((length != 4) || (addr !== expected_addr)) begin
                        record_failure($sformatf(
                            "frontend cache write %0d addr/len 0x%016h/%0d expected 0x%016h/4",
                            request_index, addr, length, expected_addr));
                    end
                    frontend_cache_req_count = frontend_cache_req_count + 1;
                end
                else if (addr == QROPE_BASE) begin
                    if (length != (Q_COUNT * 4)) begin
                        record_failure($sformatf("Q RoPE write length %0d expected %0d",
                            length, Q_COUNT * 4));
                    end
                    frontend_qrope_req_count = frontend_qrope_req_count + 1;
                end
                else begin
                    record_failure($sformatf("unexpected frontend write request 0x%016h", addr));
                end
            end
            else if (active_dut == ACTIVE_SCORE) begin
                if ((addr !== ATTN_OUT_BASE) || (length != (Q_COUNT * 4))) begin
                    record_failure($sformatf(
                        "score output write addr/len 0x%016h/%0d expected 0x%016h/%0d",
                        addr, length, ATTN_OUT_BASE, Q_COUNT * 4));
                end
                score_attn_req_count = score_attn_req_count + 1;
            end
        end
    endtask

    task automatic store_word(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [31:0] data
    );
        integer index;
        begin
            if (in_region(addr, KV_BASE, KV_WORDS * 4)) begin
                index = (addr - KV_BASE) >> 2;
                kv_mem[index] = data;
                frontend_cache_data_count = frontend_cache_data_count + 1;
            end
            else if (in_region(addr, QROPE_BASE, Q_COUNT * 4)) begin
                index = (addr - QROPE_BASE) >> 2;
                if (index != frontend_qrope_data_count) begin
                    record_failure($sformatf("Q RoPE write index %0d expected %0d",
                        index, frontend_qrope_data_count));
                end
                qrope_mem[index] = data;
                frontend_qrope_data_count = frontend_qrope_data_count + 1;
            end
            else if (in_region(addr, ATTN_OUT_BASE, Q_COUNT * 4)) begin
                index = (addr - ATTN_OUT_BASE) >> 2;
                if (index != score_attn_data_count) begin
                    record_failure($sformatf("attention output write index %0d expected %0d",
                        index, score_attn_data_count));
                end
                attn_mem[index] = data;
                score_attn_data_count = score_attn_data_count + 1;
            end
            else begin
                record_failure($sformatf("write to unsupported address 0x%016h", addr));
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_active <= 1'b0;
            rd_base_addr <= '0;
            rd_word_count <= 0;
            rd_word_index <= 0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            wr_active <= 1'b0;
            wr_base_addr <= '0;
            wr_word_count <= 0;
            wr_word_index <= 0;
            wr_done_countdown <= 0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
        end
        else begin
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                if (((mem_rd_req_addr & 3) != 0) || ((mem_rd_req_len & 3) != 0) ||
                    !supported_read_range(mem_rd_req_addr, mem_rd_req_len)) begin
                    record_failure($sformatf("unsupported read addr/len 0x%016h/%0d",
                        mem_rd_req_addr, mem_rd_req_len));
                end
                if ((active_dut == ACTIVE_SCORE) &&
                    in_region(mem_rd_req_addr, KV_BASE, KV_WORDS * 4)) begin
                    if (mem_rd_req_len != 4) begin
                        record_failure("KV-cache read length was not 4 bytes");
                    end
                    audit_cache_read(mem_rd_req_addr);
                end
                rd_active <= 1'b1;
                rd_base_addr <= mem_rd_req_addr;
                rd_word_count <= mem_rd_req_len >> 2;
                rd_word_index <= 0;
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                if (mem_rd_rsp_last) begin
                    mem_rd_rsp_valid <= 1'b0;
                    mem_rd_rsp_last <= 1'b0;
                    rd_active <= 1'b0;
                end
                else begin
                    rd_word_index <= rd_word_index + 1;
                    mem_rd_rsp_data <= read_word(rd_base_addr + ((rd_word_index + 1) * 4));
                    mem_rd_rsp_last <= ((rd_word_index + 1) == (rd_word_count - 1));
                end
            end
            else if (rd_active && !mem_rd_rsp_valid) begin
                mem_rd_rsp_valid <= 1'b1;
                mem_rd_rsp_data <= read_word(rd_base_addr);
                mem_rd_rsp_last <= (rd_word_count == 1);
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                if (((mem_wr_req_addr & 3) != 0) || ((mem_wr_req_len & 3) != 0) ||
                    (mem_wr_req_len == 0)) begin
                    record_failure($sformatf("bad write addr/len 0x%016h/%0d",
                        mem_wr_req_addr, mem_wr_req_len));
                end
                audit_write_request(mem_wr_req_addr, mem_wr_req_len);
                wr_active <= 1'b1;
                wr_base_addr <= mem_wr_req_addr;
                wr_word_count <= mem_wr_req_len >> 2;
                wr_word_index <= 0;
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                store_word(wr_base_addr + (wr_word_index * 4), mem_wr_data);
                if (mem_wr_data_last !== (wr_word_index == (wr_word_count - 1))) begin
                    record_failure($sformatf("write last mismatch at word %0d/%0d",
                        wr_word_index, wr_word_count));
                end
                if (mem_wr_data_last) begin
                    wr_active <= 1'b0;
                    wr_done_countdown <= 2;
                end
                else begin
                    wr_word_index <= wr_word_index + 1;
                end
            end

            if (wr_done_countdown > 0) begin
                if (wr_done_countdown == 1) begin
                    mem_wr_done <= 1'b1;
                end
                wr_done_countdown <= wr_done_countdown - 1;
            end
        end
    end

    task automatic run_frontend(input integer runtime_position);
        integer wait_cycles;
        begin
            @(negedge clk);
            load_frontend_v(runtime_position);
            reset_frontend_scoreboard(runtime_position);
            front_runtime_position = runtime_position[POSITION_INDEX_W-1:0];
            active_dut = ACTIVE_FRONT;
            front_start = 1'b1;
            @(negedge clk);
            front_start = 1'b0;
            wait_cycles = 0;
            while ((front_done !== 1'b1) && (wait_cycles < 200000)) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (front_done !== 1'b1) begin
                record_failure($sformatf("frontend position %0d timed out", runtime_position));
                $finish(1);
            end
            if (front_error || front_saturation || front_norm_saturation || front_rope_saturation) begin
                record_failure($sformatf(
                    "frontend position %0d status error=%0b sat=%0b norm_sat=%0b rope_sat=%0b",
                    runtime_position, front_error, front_saturation,
                    front_norm_saturation, front_rope_saturation));
            end
            if ((frontend_cache_req_count != (2 * KV_COUNT)) ||
                (frontend_cache_data_count != (2 * KV_COUNT)) ||
                (front_cache_write_count != (2 * KV_COUNT))) begin
                record_failure($sformatf(
                    "frontend position %0d cache counts req/data/dut=%0d/%0d/%0d expected %0d",
                    runtime_position, frontend_cache_req_count, frontend_cache_data_count,
                    front_cache_write_count, 2 * KV_COUNT));
            end
            if ((frontend_qrope_req_count != 1) ||
                (frontend_qrope_data_count != Q_COUNT) ||
                (front_qrope_write_count != Q_COUNT)) begin
                record_failure($sformatf(
                    "frontend position %0d Q RoPE counts req/data/dut=%0d/%0d/%0d",
                    runtime_position, frontend_qrope_req_count,
                    frontend_qrope_data_count, front_qrope_write_count));
            end
            active_dut = ACTIVE_NONE;
            @(negedge clk);
        end
    endtask

    task automatic run_score(input integer runtime_position);
        integer wait_cycles;
        integer expected_cache_reads;
        integer q_head;
        integer dim;
        integer output_index;
        logic [ADDR_WIDTH-1:0] expected_last_k;
        logic [ADDR_WIDTH-1:0] expected_last_v;
        logic [31:0] expected_word;
        begin
            @(negedge clk);
            reset_score_scoreboard(runtime_position);
            score_runtime_position = runtime_position[POSITION_INDEX_W-1:0];
            active_dut = ACTIVE_SCORE;
            score_start = 1'b1;
            @(negedge clk);
            score_start = 1'b0;
            wait_cycles = 0;
            while ((score_done !== 1'b1) && (wait_cycles < 500000)) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (score_done !== 1'b1) begin
                record_failure($sformatf("score position %0d timed out", runtime_position));
                $finish(1);
            end
            expected_cache_reads = NUM_Q_HEADS * (runtime_position + 1) * HEAD_DIM;
            if (score_error || score_saturation) begin
                record_failure($sformatf("score position %0d status error=%0b sat=%0b",
                    runtime_position, score_error, score_saturation));
            end
            if (score_dut.metadata_cache_length !== (runtime_position + 1)) begin
                record_failure($sformatf("score position %0d cache_length=%0d expected %0d",
                    runtime_position, score_dut.metadata_cache_length, runtime_position + 1));
            end
            if ((score_k_req_count != expected_cache_reads) ||
                (score_v_req_count != expected_cache_reads) ||
                (score_k_read_count != expected_cache_reads) ||
                (score_v_read_count != expected_cache_reads)) begin
                record_failure($sformatf(
                    "score position %0d K/V req/rsp=%0d/%0d/%0d/%0d expected %0d",
                    runtime_position, score_k_req_count, score_v_req_count,
                    score_k_read_count, score_v_read_count, expected_cache_reads));
            end
            if (score_count != (NUM_Q_HEADS * (runtime_position + 1))) begin
                record_failure($sformatf("score position %0d score count %0d expected %0d",
                    runtime_position, score_count, NUM_Q_HEADS * (runtime_position + 1)));
            end
            if ((score_attn_req_count != 1) || (score_attn_data_count != Q_COUNT) ||
                (score_attn_capture_count != Q_COUNT) || (score_attn_write_count != Q_COUNT)) begin
                record_failure($sformatf(
                    "score position %0d output counts req/data/capture/dut=%0d/%0d/%0d/%0d",
                    runtime_position, score_attn_req_count, score_attn_data_count,
                    score_attn_capture_count, score_attn_write_count));
            end
            expected_last_k = kv_addr(0, 0, 0, runtime_position, HEAD_DIM - 1);
            expected_last_v = kv_addr(0, 1, 0, runtime_position, HEAD_DIM - 1);
            if ((score_first_k_addr !== kv_addr(0, 0, 0, 0, 0)) ||
                (score_last_k_addr !== expected_last_k) ||
                (score_first_v_addr !== kv_addr(0, 1, 0, 0, 0)) ||
                (score_last_v_addr !== expected_last_v)) begin
                record_failure($sformatf(
                    "score position %0d cache endpoints K=0x%016h..0x%016h V=0x%016h..0x%016h",
                    runtime_position, score_first_k_addr, score_last_k_addr,
                    score_first_v_addr, score_last_v_addr));
            end
            for (q_head = 0; q_head < NUM_Q_HEADS; q_head = q_head + 1) begin
                for (dim = 0; dim < HEAD_DIM; dim = dim + 1) begin
                    output_index = q_head * HEAD_DIM + dim;
                    expected_word = expected_attn_word(runtime_position, dim);
                    if (attn_mem[output_index] !== expected_word) begin
                        record_failure($sformatf(
                            "score position %0d output[%0d]=0x%08h expected 0x%08h",
                            runtime_position, output_index, attn_mem[output_index], expected_word));
                    end
                end
            end
            active_dut = ACTIVE_NONE;
            @(negedge clk);
        end
    endtask

    task automatic snapshot_position_zero;
        integer i;
        integer nonzero_k;
        begin
            nonzero_k = 0;
            for (i = 0; i < KV_COUNT; i = i + 1) begin
                k_pos0_snapshot[i] = kv_mem[kv_word_index(0, 0, 0, 0, i)];
                v_pos0_snapshot[i] = kv_mem[kv_word_index(0, 1, 0, 0, i)];
                if (k_pos0_snapshot[i] != 0) begin
                    nonzero_k = nonzero_k + 1;
                end
                if (v_pos0_snapshot[i] !== expected_attn_word(0, i)) begin
                    record_failure($sformatf("position 0 V[%0d]=0x%08h expected 0x%08h",
                        i, v_pos0_snapshot[i], expected_attn_word(0, i)));
                end
            end
            for (i = 0; i < Q_COUNT; i = i + 1) begin
                qrope_pos0_snapshot[i] = qrope_mem[i];
            end
            if (nonzero_k == 0) begin
                record_failure("position 0 K snapshot was all zero; retention proof is not meaningful");
            end
        end
    endtask

    task automatic check_position_zero_retained_after_position_one;
        integer i;
        begin
            for (i = 0; i < KV_COUNT; i = i + 1) begin
                if (kv_mem[kv_word_index(0, 0, 0, 0, i)] !== k_pos0_snapshot[i]) begin
                    record_failure($sformatf("position 0 K[%0d] changed after position 1 append", i));
                end
                if (kv_mem[kv_word_index(0, 1, 0, 0, i)] !== v_pos0_snapshot[i]) begin
                    record_failure($sformatf("position 0 V[%0d] changed after position 1 append", i));
                end
                if (kv_mem[kv_word_index(0, 0, 0, 1, i)] !== k_pos0_snapshot[i]) begin
                    record_failure($sformatf("position 1 K[%0d] differs from identical position 0 K", i));
                end
                if (kv_mem[kv_word_index(0, 1, 0, 1, i)] !==
                    expected_attn_word(1, i) * 2 - expected_attn_word(0, i)) begin
                    record_failure($sformatf("position 1 V[%0d] data mismatch", i));
                end
            end
            for (i = 0; i < Q_COUNT; i = i + 1) begin
                if (qrope_mem[i] !== qrope_pos0_snapshot[i]) begin
                    record_failure($sformatf("Q RoPE[%0d] changed for identical token geometry", i));
                end
            end
        end
    endtask

    task automatic check_sparse_positions_unchanged;
        integer position;
        integer dim;
        begin
            for (position = 2; position < 255; position = position + 1) begin
                for (dim = 0; dim < HEAD_DIM; dim = dim + 1) begin
                    if (kv_mem[kv_word_index(0, 0, 0, position, dim)] !== 32'd0) begin
                        record_failure($sformatf("unexpected K data at untouched position %0d dim %0d",
                            position, dim));
                    end
                    if (kv_mem[kv_word_index(0, 1, 0, position, dim)] !== 32'd0) begin
                        record_failure($sformatf("unexpected V data at untouched position %0d dim %0d",
                            position, dim));
                    end
                end
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    integer init_index;
    initial begin
        rst_n = 1'b0;
        active_dut = ACTIVE_NONE;
        front_start = 1'b0;
        score_start = 1'b0;
        front_runtime_position = '0;
        score_runtime_position = '0;
        mismatch_count = 0;
        print_count = 0;
        frontend_expected_position = 0;
        score_expected_cache_length = 1;
        frontend_cache_req_count = 0;
        frontend_cache_data_count = 0;
        frontend_qrope_req_count = 0;
        frontend_qrope_data_count = 0;
        score_k_req_count = 0;
        score_v_req_count = 0;
        score_attn_req_count = 0;
        score_attn_data_count = 0;

        for (init_index = 0; init_index < FRONT_WORDS; init_index = init_index + 1) begin
            front_mem[init_index] = 32'd0;
        end
        for (init_index = 0; init_index < SCORE_WORDS; init_index = init_index + 1) begin
            score_mem[init_index] = 32'd0;
        end
        for (init_index = 0; init_index < ROPE_WORDS; init_index = init_index + 1) begin
            cos_mem[init_index] = 32'd32767;
            sin_mem[init_index] = 32'd0;
        end
        for (init_index = 0; init_index < KV_WORDS; init_index = init_index + 1) begin
            kv_mem[init_index] = 32'd0;
        end
        for (init_index = 0; init_index < Q_COUNT; init_index = init_index + 1) begin
            qrope_mem[init_index] = 32'd0;
            attn_mem[init_index] = 32'd0;
            qrope_pos0_snapshot[init_index] = 32'd0;
        end

        build_qmap_images();
        repeat (5) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_frontend(0);
        snapshot_position_zero();
        run_score(0);

        run_frontend(1);
        check_position_zero_retained_after_position_one();
        run_score(1);

        check_sparse_positions_unchanged();
        run_frontend(255);
        check_position_zero_retained_after_position_one();
        check_sparse_positions_unchanged();
        run_score(255);

        if (mismatch_count == 0) begin
            $display("PASS: persistent KV frontend+score two-token proof passed; pos0 retained, pos1 cache_length=2, and pos255 boundary/output were exact.");
            $finish(0);
        end
        else begin
            $display("FAIL: persistent KV proof completed with %0d mismatches", mismatch_count);
            $finish(1);
        end
    end

endmodule

`default_nettype wire
