`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

`ifdef QMAP_FINAL_TOKEN_TB_FULL_VOCAB
`ifndef QMAP_FINAL_TOKEN_TB_MAX_TILES
`define QMAP_FINAL_TOKEN_TB_MAX_TILES 9496
`endif
`ifndef QMAP_FINAL_TOKEN_TB_ENABLE_INVALID_RUN
`define QMAP_FINAL_TOKEN_TB_ENABLE_INVALID_RUN 0
`endif
`endif

`ifndef QMAP_FINAL_TOKEN_TB_MAX_TILES
`define QMAP_FINAL_TOKEN_TB_MAX_TILES 64
`endif

`ifndef QMAP_FINAL_TOKEN_TB_ENABLE_INVALID_RUN
`define QMAP_FINAL_TOKEN_TB_ENABLE_INVALID_RUN 1
`endif

module tb_qmap_final_token_tail_compute_path #(
    parameter int MAX_TILES          = `QMAP_FINAL_TOKEN_TB_MAX_TILES,
    parameter int ENABLE_INVALID_RUN = `QMAP_FINAL_TOKEN_TB_ENABLE_INVALID_RUN
);

    localparam int ADDR_WIDTH       = 64;
    localparam int TILE_COUNT_WIDTH = $clog2(MAX_TILES + 1);
    localparam int TILE_ROWS        = 16;
    localparam int SCAN_ROWS        = MAX_TILES * TILE_ROWS;
    localparam int INPUT_SIZE       = 1024;
    localparam int GROUP_SIZE       = 64;
    localparam int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE;
    localparam int HIDDEN_WIDTH     = 24;
    localparam int GAMMA_WIDTH      = 16;
    localparam int NORM_WIDTH       = 24;
    localparam int WEIGHT_WIDTH     = 4;
    localparam int SCALE_WIDTH      = 16;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;
    localparam int PARTIAL_WIDTH    = NORM_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;
    localparam int TOKEN_ID_WIDTH   = 32;
    localparam int WEIGHT_ROW_BYTES = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES  = (GROUP_COUNT * SCALE_WIDTH) / 8;
    localparam int TILE_WEIGHT_BYTES = TILE_ROWS * WEIGHT_ROW_BYTES;
    localparam int TILE_SCALE_BYTES  = TILE_ROWS * SCALE_ROW_BYTES;
    localparam int WEIGHT_BURSTS_PER_TILE = TILE_WEIGHT_BYTES / MAX_READ_BYTES;
    localparam int SCALE_BURSTS_PER_TILE = 1;
    localparam int BURSTS_PER_TILE = WEIGHT_BURSTS_PER_TILE + SCALE_BURSTS_PER_TILE;
    localparam int WORDS_PER_TILE = (TILE_WEIGHT_BYTES + TILE_SCALE_BYTES) / 4;
    localparam int WEIGHT_BURSTS_PER_ROW = 1;
    localparam int SCALE_BURSTS_PER_ROW = 1;
    localparam int BURSTS_PER_ROW =
        WEIGHT_BURSTS_PER_ROW + SCALE_BURSTS_PER_ROW;
    localparam int WEIGHT_WORDS = SCAN_ROWS * WEIGHT_ROW_BYTES / 4;
    localparam int SCALE_WORDS = SCAN_ROWS * SCALE_ROW_BYTES / 4;
    localparam int QMAP_IMAGE_BYTES = 32'h0000_4000;
    localparam int QMAP_WORDS = QMAP_IMAGE_BYTES / 4;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int SLOT_NORM_OUTPUT = 1;
    localparam int SLOT_OUTPUT = 4;
    localparam int SLOT_FINAL_HIDDEN = 6;
    localparam int SLOT_FINAL_GAMMA = 7;
    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] TILE_SCALE_BYTES_U16 = TILE_SCALE_BYTES;
    localparam int REGION_QMAP = 0;
    localparam int REGION_WEIGHT = 1;
    localparam int REGION_SCALE = 2;
    localparam int WRITE_KIND_NONE = 0;
    localparam int WRITE_KIND_NORM = 1;
    localparam int WRITE_KIND_OUTPUT = 2;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic norm_saturation;
    logic [TOKEN_ID_WIDTH-1 : 0] best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] best_score_q26;
    logic [31 : 0] tiles_started;
    logic [31 : 0] tiles_completed;
    logic [31 : 0] norm_cycle_count;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_word_count;

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

    logic [31 : 0] qmap_mem [0:QMAP_WORDS-1];
    logic [31 : 0] weight_words_mem [0:WEIGHT_WORDS-1];
    logic [31 : 0] scale_words_mem [0:SCALE_WORDS-1];
    logic signed [ROW_ACC_WIDTH-1 : 0] expected_logits_mem [0:SCAN_ROWS-1];
    logic signed [NORM_WIDTH-1 : 0] final_norm_expected_mem [0:INPUT_SIZE-1];
    logic [31 : 0] expected_words_mem [0:2];
    logic [TOKEN_ID_WIDTH-1 : 0] scan_base_token_mem [0:0];
    logic [ADDR_WIDTH-1 : 0] weight_base_addr_mem [0:0];
    logic [ADDR_WIDTH-1 : 0] scale_base_addr_mem [0:0];

    string vector_dir;
    string lm_prefix;
    string final_norm_prefix;
    string qmap_image_file;
    string qmap_expected_file;
    string tracefile;
    string wavefile;
    integer trace_fd;
    integer trace_every_cycle;
    integer fast_memory;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer run_index;
    integer done_seen_count;
    integer last_done_cycle;
    integer spurious_start_seen_busy;
    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_hidden_req_count;
    integer mem_gamma_req_count;
    integer mem_norm_read_req_count;
    integer mem_weight_req_count;
    integer mem_scale_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer norm_write_word_count;
    integer output_write_word_count;
    integer norm_write_mismatch_count;
    integer core_update_count;
    integer checked_logit_count;
    integer max_abs_logit_diff;
    integer run_scheduler_req_count;
    integer run_checked_logit_base;
    integer run_core_update_base;
    integer run_wr_req_base;
    integer run_norm_wr_base;
    integer run_output_wr_base;
    integer run_write_word_base;
    integer element_index;
    integer row_index;
    integer active_region;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer response_delay;
    integer req_stall_active;
    integer rsp_stall_active;
    integer wr_req_stall_active;
    integer wr_data_stall_active;
    integer active_write_index;
    integer active_write_words_left;
    integer active_write_kind;
    integer write_done_delay;
    longint signed logit_diff;
    logic read_active;
    logic write_active;
    logic [ADDR_WIDTH-1 : 0] stalled_req_addr;
    logic [15 : 0] stalled_req_len;
    logic [31 : 0] stalled_rsp_data;
    logic stalled_rsp_last;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_last;
    logic [TOKEN_ID_WIDTH-1 : 0] token_base;
    logic [ADDR_WIDTH-1 : 0] weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] scale_base_addr;
    logic [31 : 0] expected_norm_index;
    logic [31 : 0] expected_output_index;
    logic [31 : 0] expected_run_token;
    logic signed [ROW_ACC_WIDTH-1 : 0] expected_run_score_q26;
    logic signed [63 : 0] expected_run_score_ext;
    logic [31 : 0] last_success_token;
    logic signed [ROW_ACC_WIDTH-1 : 0] last_success_score_q26;

    qmap_final_token_tail_compute_path #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(8),
        .MAX_TILES       (MAX_TILES),
        .TILE_COUNT_WIDTH(TILE_COUNT_WIDTH),
        .TILE_ROWS       (TILE_ROWS),
        .INPUT_SIZE      (INPUT_SIZE),
        .GROUP_SIZE      (GROUP_SIZE),
        .GROUP_COUNT     (GROUP_COUNT),
        .MEM_DATA_WIDTH  (MEM_DATA_WIDTH),
        .MAX_READ_BYTES  (MAX_READ_BYTES)
    ) dut (
        .i_clk                  (clk),
        .i_rst_n                (rst_n),
        .i_start                (start),
        .i_qmap_base_addr       (`QMAP_FINAL_TOKEN_BASE_ADDR),
        .i_final_hidden_base_override_valid(1'b0),
        .i_final_hidden_base_override_addr ('0),
        .o_busy                 (busy),
        .o_done                 (done),
        .o_error                (error),
        .o_norm_saturation      (norm_saturation),
        .o_effective_final_hidden_base_addr(),
        .o_best_token_id        (best_token_id),
        .o_best_score_q26       (best_score_q26),
        .o_tiles_started        (tiles_started),
        .o_tiles_completed      (tiles_completed),
        .o_norm_cycle_count     (norm_cycle_count),
        .o_mem_read_burst_count (mem_read_burst_count),
        .o_mem_read_word_count  (mem_read_word_count),
        .o_mem_write_word_count (mem_write_word_count),
        .o_mem_rd_req_valid     (mem_rd_req_valid),
        .i_mem_rd_req_ready     (mem_rd_req_ready),
        .o_mem_rd_req_addr      (mem_rd_req_addr),
        .o_mem_rd_req_len_bytes (mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid     (mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready     (mem_rd_rsp_ready),
        .i_mem_rd_rsp_data      (mem_rd_rsp_data),
        .i_mem_rd_rsp_last      (mem_rd_rsp_last),
        .o_mem_wr_req_valid     (mem_wr_req_valid),
        .i_mem_wr_req_ready     (mem_wr_req_ready),
        .o_mem_wr_req_addr      (mem_wr_req_addr),
        .o_mem_wr_req_len_bytes (mem_wr_req_len_bytes),
        .o_mem_wr_data          (mem_wr_data),
        .o_mem_wr_data_valid    (mem_wr_data_valid),
        .i_mem_wr_data_ready    (mem_wr_data_ready),
        .o_mem_wr_data_last     (mem_wr_data_last),
        .i_mem_wr_done          (mem_wr_done),
        .i_mem_wr_error         (mem_wr_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        integer enable_dumpwaves;
`ifdef QMAP_FINAL_TOKEN_TB_FULL_VOCAB
        enable_dumpwaves = 0;
`else
        enable_dumpwaves = 1;
`endif
        wavefile = "FPGA_Project/wave/qmap_final_token_tail_compute_path.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        if ($value$plusargs("dumpwaves=%d", enable_dumpwaves)) begin
        end
        if (enable_dumpwaves != 0) begin
            $dumpfile(wavefile);
            $dumpvars(0, clk);
            $dumpvars(0, rst_n);
            $dumpvars(0, start);
            $dumpvars(0, busy);
            $dumpvars(0, done);
            $dumpvars(0, error);
            $dumpvars(0, norm_saturation);
            $dumpvars(0, mem_rd_req_valid);
            $dumpvars(0, mem_rd_req_ready);
            $dumpvars(0, mem_rd_rsp_valid);
            $dumpvars(0, mem_rd_rsp_ready);
            $dumpvars(0, mem_wr_req_valid);
            $dumpvars(0, mem_wr_req_ready);
            $dumpvars(0, mem_wr_data_valid);
            $dumpvars(0, mem_wr_data_ready);
            $dumpvars(0, best_token_id);
            $dumpvars(0, best_score_q26);
            $dumpvars(0, dut.state);
            $dumpvars(0, dut.final_norm_state_debug);
            $dumpvars(0, dut.lm_head_path.state);
            $dumpvars(0, dut.lm_head_path.scheduler_state_debug);
            $dumpvars(0, dut.lm_head_path.scheduler_engine_state_debug);
        end
    end

    function automatic integer descriptor_word_index(input integer slot, input integer word_offset);
        begin
            descriptor_word_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    function automatic logic [63 : 0] descriptor_base_addr(input integer slot);
        begin
            descriptor_base_addr = {
                qmap_mem[descriptor_word_index(slot, DESC_BASE_HI_WORD)],
                qmap_mem[descriptor_word_index(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [31 : 0] norm_expected_word(input integer index);
        begin
            norm_expected_word = {
                {8{final_norm_expected_mem[index][NORM_WIDTH-1]}},
                final_norm_expected_mem[index]
            };
        end
    endfunction

    function automatic logic req_ready_pattern(input integer cycle, input integer count);
        begin
            if (fast_memory != 0) begin
                req_ready_pattern = 1'b1;
            end
            else begin
                req_ready_pattern = ((cycle % 11) != 3) && (((cycle + count) % 17) != 5);
            end
        end
    endfunction

    function automatic integer response_latency_pattern(input integer count, input integer cycle);
        begin
            if (fast_memory != 0) begin
                response_latency_pattern = 0;
            end
            else begin
                response_latency_pattern = 1 + ((count + cycle) % 5);
            end
        end
    endfunction

    function automatic logic response_gap_pattern(input integer cycle, input integer word_count);
        begin
            if (fast_memory != 0) begin
                response_gap_pattern = 1'b1;
            end
            else begin
                response_gap_pattern = ((cycle % 13) != 4) && (((cycle + word_count) % 29) != 7);
            end
        end
    endfunction

    function automatic logic wr_req_ready_pattern(input integer cycle, input integer count);
        begin
            if (fast_memory != 0) begin
                wr_req_ready_pattern = 1'b1;
            end
            else begin
                wr_req_ready_pattern = ((cycle + count) % 9) != 2;
            end
        end
    endfunction

    function automatic logic wr_data_ready_pattern(input integer cycle, input integer count);
        begin
            if (fast_memory != 0) begin
                wr_data_ready_pattern = 1'b1;
            end
            else begin
                wr_data_ready_pattern = ((cycle % 7) != 1) && (((cycle + count) % 13) != 4);
            end
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input integer region, input integer word_index);
        begin
            case (region)
                REGION_QMAP: memory_word = qmap_mem[word_index];
                REGION_WEIGHT: memory_word = weight_words_mem[word_index];
                REGION_SCALE: memory_word = scale_words_mem[word_index];
                default: memory_word = 32'hDEAD_BAD1;
            endcase
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(qmap_expected_file, expected_words_mem);
            $readmemh({vector_dir, "/", final_norm_prefix, "_expected.hex"}, final_norm_expected_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_weight_words32.hex"}, weight_words_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_scale_words32.hex"}, scale_words_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_expected_scan_logits_q26.hex"}, expected_logits_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_scan_base_token.hex"}, scan_base_token_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_weight_base_addr.hex"}, weight_base_addr_mem);
            $readmemh({vector_dir, "/", lm_prefix, "_scale_base_addr.hex"}, scale_base_addr_mem);
            token_base = scan_base_token_mem[0];
            weight_base_addr = weight_base_addr_mem[0];
            scale_base_addr = scale_base_addr_mem[0];
            expected_norm_index = (descriptor_base_addr(SLOT_NORM_OUTPUT) - `QMAP_FINAL_TOKEN_BASE_ADDR) >> 2;
            expected_output_index = (descriptor_base_addr(SLOT_OUTPUT) - `QMAP_FINAL_TOKEN_BASE_ADDR) >> 2;
        end
    endtask

    task clear_write_targets;
        begin
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                qmap_mem[expected_norm_index + element_index] = 32'hA5A5_0000 | element_index[15:0];
            end
            qmap_mem[expected_output_index + 0] = 32'hFFFF_FFFF;
            qmap_mem[expected_output_index + 1] = 32'hFFFF_FFFF;
            qmap_mem[expected_output_index + 2] = 32'hFFFF_FFFF;
        end
    endtask

    task calculate_expected_for_tiles(input integer tile_count);
        begin
            expected_run_token = token_base;
            expected_run_score_q26 = expected_logits_mem[0];
            for (element_index = 1; element_index < tile_count*TILE_ROWS; element_index = element_index + 1) begin
                if ($signed(expected_logits_mem[element_index]) > expected_run_score_q26) begin
                    expected_run_score_q26 = expected_logits_mem[element_index];
                    expected_run_token = token_base + element_index;
                end
            end
            expected_run_score_ext =
                {{(64-ROW_ACC_WIDTH){expected_run_score_q26[ROW_ACC_WIDTH-1]}}, expected_run_score_q26};
        end
    endtask

    task check_scheduler_request_address;
        integer row_id;
        integer row_request_id;
        integer row_token;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        logic [15 : 0] expected_len;
        begin
            row_id = run_scheduler_req_count / BURSTS_PER_ROW;
            row_request_id =
                run_scheduler_req_count % BURSTS_PER_ROW;
            row_token = token_base + row_id;

            if (row_id >= (MAX_TILES * TILE_ROWS)) begin
                $display("FAIL: extra LM scheduler request row_id=%0d max_rows=%0d",
                         row_id, MAX_TILES * TILE_ROWS);
                mismatch_count = mismatch_count + 1;
            end

            if (row_request_id == 0) begin
                expected_addr =
                    weight_base_addr +
                    (row_token * WEIGHT_ROW_BYTES);
                expected_len = WEIGHT_ROW_BYTES;
                mem_weight_req_count = mem_weight_req_count + 1;
            end
            else begin
                expected_addr =
                    scale_base_addr +
                    (row_token * SCALE_ROW_BYTES);
                expected_len = SCALE_ROW_BYTES;
                mem_scale_req_count = mem_scale_req_count + 1;
            end

            if ((mem_rd_req_addr !== expected_addr) || (mem_rd_req_len_bytes !== expected_len)) begin
                if (print_count < 32) begin
                    $display("FAIL: LM request mismatch row %0d request %0d addr=0x%016h expected=0x%016h len=%0d expected_len=%0d",
                             row_id, row_request_id,
                             mem_rd_req_addr, expected_addr,
                             mem_rd_req_len_bytes, expected_len);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            run_scheduler_req_count = run_scheduler_req_count + 1;
        end
    endtask

    task classify_and_check_read_request;
        logic [ADDR_WIDTH-1 : 0] qmap_end;
        logic [ADDR_WIDTH-1 : 0] weight_end;
        logic [ADDR_WIDTH-1 : 0] scale_end;
        begin
            qmap_end = `QMAP_FINAL_TOKEN_BASE_ADDR + QMAP_IMAGE_BYTES;
            weight_end = weight_base_addr + (WEIGHT_WORDS * 4);
            scale_end = scale_base_addr + (SCALE_WORDS * 4);

            if ((mem_rd_req_addr >= `QMAP_FINAL_TOKEN_BASE_ADDR) &&
                ((mem_rd_req_addr + mem_rd_req_len_bytes) <= qmap_end)) begin
                mem_qmap_req_count = mem_qmap_req_count + 1;
                if ((mem_rd_req_addr >= descriptor_base_addr(SLOT_FINAL_HIDDEN)) &&
                    (mem_rd_req_addr < (descriptor_base_addr(SLOT_FINAL_HIDDEN) + INPUT_SIZE*4))) begin
                    mem_hidden_req_count = mem_hidden_req_count + 1;
                end
                if ((mem_rd_req_addr >= descriptor_base_addr(SLOT_FINAL_GAMMA)) &&
                    (mem_rd_req_addr < (descriptor_base_addr(SLOT_FINAL_GAMMA) + INPUT_SIZE*4))) begin
                    mem_gamma_req_count = mem_gamma_req_count + 1;
                end
                if ((mem_rd_req_addr >= descriptor_base_addr(SLOT_NORM_OUTPUT)) &&
                    (mem_rd_req_addr < (descriptor_base_addr(SLOT_NORM_OUTPUT) + INPUT_SIZE*4))) begin
                    mem_norm_read_req_count = mem_norm_read_req_count + 1;
                end
            end
            else if ((mem_rd_req_addr >= weight_base_addr) &&
                     ((mem_rd_req_addr + mem_rd_req_len_bytes) <= weight_end)) begin
                check_scheduler_request_address();
            end
            else if ((mem_rd_req_addr >= scale_base_addr) &&
                     ((mem_rd_req_addr + mem_rd_req_len_bytes) <= scale_end)) begin
                check_scheduler_request_address();
            end
            else begin
                if (print_count < 32) begin
                    $display("FAIL: read request outside known regions addr=0x%016h len=%0d",
                             mem_rd_req_addr, mem_rd_req_len_bytes);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_tile_logits;
        integer local_index;
        logic signed [ROW_ACC_WIDTH-1 : 0] observed_logit;
        begin
            local_index =
                dut.lm_head_path.scheduler_row_result_token -
                token_base;
            observed_logit =
                $signed(
                    dut.lm_head_path.scheduler_row_result_score);
            logit_diff =
                observed_logit - expected_logits_mem[local_index];
            if (logit_diff < 0) begin
                logit_diff = -logit_diff;
            end
            if (logit_diff > max_abs_logit_diff) begin
                max_abs_logit_diff = logit_diff;
            end
            if (observed_logit !==
                expected_logits_mem[local_index]) begin
                if (print_count < 32) begin
                    $display("FAIL: logit row %0d token %0d actual=%0d expected=%0d",
                             local_index,
                             dut.lm_head_path.scheduler_row_result_token,
                             observed_logit,
                             expected_logits_mem[local_index]);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            checked_logit_count = checked_logit_count + 1;
        end
    endtask

    task check_successful_run;
        integer expected_scheduler_reqs;
        begin
            expected_scheduler_reqs =
                MAX_TILES * TILE_ROWS * BURSTS_PER_ROW;
            if (error != 1'b0) begin
                $display("FAIL: final-token tail error high at done");
                mismatch_count = mismatch_count + 1;
            end
            if (norm_saturation != 1'b0) begin
                $display("FAIL: final RMSNorm saturation asserted");
                mismatch_count = mismatch_count + 1;
            end
            if (best_token_id != expected_run_token) begin
                $display("FAIL: best token actual=%0d expected=%0d", best_token_id, expected_run_token);
                mismatch_count = mismatch_count + 1;
            end
            if (best_score_q26 != expected_run_score_q26) begin
                $display("FAIL: best score actual=%0d expected=%0d", best_score_q26, expected_run_score_q26);
                mismatch_count = mismatch_count + 1;
            end
            if ((tiles_started != MAX_TILES) || (tiles_completed != MAX_TILES)) begin
                $display("FAIL: tile count started=%0d completed=%0d expected=%0d",
                         tiles_started, tiles_completed, MAX_TILES);
                mismatch_count = mismatch_count + 1;
            end
            if (run_scheduler_req_count != expected_scheduler_reqs) begin
                $display("FAIL: scheduler read request count actual=%0d expected=%0d",
                         run_scheduler_req_count, expected_scheduler_reqs);
                mismatch_count = mismatch_count + 1;
            end
            if ((core_update_count - run_core_update_base) != MAX_TILES) begin
                $display("FAIL: tile update count actual=%0d expected=%0d",
                         core_update_count - run_core_update_base, MAX_TILES);
                mismatch_count = mismatch_count + 1;
            end
            if ((checked_logit_count - run_checked_logit_base) != MAX_TILES*TILE_ROWS) begin
                $display("FAIL: checked logit count actual=%0d expected=%0d",
                         checked_logit_count - run_checked_logit_base, MAX_TILES*TILE_ROWS);
                mismatch_count = mismatch_count + 1;
            end
            if ((norm_write_word_count - run_norm_wr_base) != INPUT_SIZE) begin
                $display("FAIL: norm write word count actual=%0d expected=%0d",
                         norm_write_word_count - run_norm_wr_base, INPUT_SIZE);
                mismatch_count = mismatch_count + 1;
            end
            if ((output_write_word_count - run_output_wr_base) != 3) begin
                $display("FAIL: output write word count actual=%0d expected=3",
                         output_write_word_count - run_output_wr_base);
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count - run_wr_req_base) != 2) begin
                $display("FAIL: write request count actual=%0d expected=2",
                         mem_wr_req_count - run_wr_req_base);
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_word_count_total - run_write_word_base) != (INPUT_SIZE + 3)) begin
                $display("FAIL: write data word count actual=%0d expected=%0d",
                         mem_wr_word_count_total - run_write_word_base, INPUT_SIZE + 3);
                mismatch_count = mismatch_count + 1;
            end
            if (norm_write_mismatch_count != 0) begin
                $display("FAIL: norm write mismatches=%0d", norm_write_mismatch_count);
                mismatch_count = mismatch_count + 1;
            end
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                if (qmap_mem[expected_norm_index + element_index] !== norm_expected_word(element_index)) begin
                    if (print_count < 32) begin
                        $display("FAIL: final_norm memory word %0d actual=%08h expected=%08h",
                                 element_index,
                                 qmap_mem[expected_norm_index + element_index],
                                 norm_expected_word(element_index));
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
            if ((qmap_mem[expected_output_index + 0] !== expected_run_token) ||
                (qmap_mem[expected_output_index + 1] !== expected_run_score_ext[31 : 0]) ||
                (qmap_mem[expected_output_index + 2] !== expected_run_score_ext[63 : 32])) begin
                $display("FAIL: output token/score memory mismatch actual=%08h_%08h_%08h expected=%08h_%08h_%08h",
                         qmap_mem[expected_output_index + 2],
                         qmap_mem[expected_output_index + 1],
                         qmap_mem[expected_output_index + 0],
                         expected_run_score_ext[63 : 32],
                         expected_run_score_ext[31 : 0],
                         expected_run_token);
                mismatch_count = mismatch_count + 1;
            end
            last_success_token = best_token_id;
            last_success_score_q26 = best_score_q26;
        end
    endtask

    task run_success(input integer run_id);
        integer wait_cycles;
        integer wait_limit;
        begin
            run_index = run_id;
            run_scheduler_req_count = 0;
            run_checked_logit_base = checked_logit_count;
            run_core_update_base = core_update_count;
            run_wr_req_base = mem_wr_req_count;
            run_norm_wr_base = norm_write_word_count;
            run_output_wr_base = output_write_word_count;
            run_write_word_base = mem_wr_word_count_total;
            clear_write_targets();
            calculate_expected_for_tiles(MAX_TILES);

            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            wait_cycles = 0;
            wait_limit = 50000 + (MAX_TILES * 25000);
            while ((done != 1'b1) && (wait_cycles < wait_limit)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if ((wait_cycles == 20) && (busy == 1'b1)) begin
                    start <= 1'b1;
                    spurious_start_seen_busy <= 1;
                end
                else if (wait_cycles == 21) begin
                    start <= 1'b0;
                end
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for qmap final-token tail done after %0d cycles", wait_limit);
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end
            check_successful_run();
            @(posedge clk);
        end
    endtask

    task run_invalid_gamma_dtype(input integer run_id);
        integer wait_cycles;
        integer wr_req_before;
        integer wr_word_before;
        begin
            run_index = run_id;
            wr_req_before = mem_wr_req_count;
            wr_word_before = mem_wr_word_count_total;
            qmap_mem[descriptor_word_index(SLOT_FINAL_GAMMA, DESC_DTYPE_WORD)] = 32'd0;
            clear_write_targets();

            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            wait_cycles = 0;
            while ((done != 1'b1) && (wait_cycles < 20000)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for invalid gamma descriptor run");
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end
            if (error != 1'b1) begin
                $display("FAIL: invalid gamma descriptor did not raise error");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count != wr_req_before) || (mem_wr_word_count_total != wr_word_before)) begin
                $display("FAIL: invalid descriptor issued writes req_before=%0d req_after=%0d word_before=%0d word_after=%0d",
                         wr_req_before, mem_wr_req_count, wr_word_before, mem_wr_word_count_total);
                mismatch_count = mismatch_count + 1;
            end
            if ((best_token_id != 32'd0) || (best_score_q26 != 'd0)) begin
                $display("FAIL: invalid descriptor exposed stale best token=%0d score=%0d",
                         best_token_id, best_score_q26);
                mismatch_count = mismatch_count + 1;
            end
            @(posedge clk);
        end
    endtask

    task check_request_stability;
        begin
            if (req_stall_active != 0) begin
                if ((mem_rd_req_addr !== stalled_req_addr) || (mem_rd_req_len_bytes !== stalled_req_len)) begin
                    $display("FAIL: read request changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_response_stability;
        begin
            if (rsp_stall_active != 0) begin
                if ((mem_rd_rsp_data !== stalled_rsp_data) || (mem_rd_rsp_last !== stalled_rsp_last)) begin
                    $display("FAIL: read response changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_write_request_stability;
        begin
            if (wr_req_stall_active != 0) begin
                if ((mem_wr_req_addr !== stalled_wr_req_addr) || (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    $display("FAIL: write request changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_write_data_stability;
        begin
            if (wr_data_stall_active != 0) begin
                if ((mem_wr_data !== stalled_wr_data) || (mem_wr_data_last !== stalled_wr_last)) begin
                    $display("FAIL: write data changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1000;
            mem_req_fire_count <= 0;
            mem_rsp_fire_count <= 0;
            mem_qmap_req_count <= 0;
            mem_hidden_req_count <= 0;
            mem_gamma_req_count <= 0;
            mem_norm_read_req_count <= 0;
            mem_weight_req_count <= 0;
            mem_scale_req_count <= 0;
            mem_wr_req_count <= 0;
            mem_wr_word_count_total <= 0;
            norm_write_word_count <= 0;
            output_write_word_count <= 0;
            norm_write_mismatch_count <= 0;
            core_update_count <= 0;
            checked_logit_count <= 0;
            max_abs_logit_diff <= 0;
            req_stall_active <= 0;
            rsp_stall_active <= 0;
            wr_req_stall_active <= 0;
            wr_data_stall_active <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if (mem_rd_req_valid == 1'b1) begin
                check_request_stability();
                if (mem_rd_req_ready == 1'b1) begin
                    classify_and_check_read_request();
                    mem_req_fire_count <= mem_req_fire_count + 1;
                    req_stall_active <= 0;
                end
                else begin
                    req_stall_active <= 1;
                    stalled_req_addr <= mem_rd_req_addr;
                    stalled_req_len <= mem_rd_req_len_bytes;
                end
            end
            else begin
                req_stall_active <= 0;
            end

            if (mem_rd_rsp_valid == 1'b1) begin
                check_response_stability();
                if (mem_rd_rsp_ready == 1'b1) begin
                    mem_rsp_fire_count <= mem_rsp_fire_count + 1;
                    rsp_stall_active <= 0;
                end
                else begin
                    rsp_stall_active <= 1;
                    stalled_rsp_data <= mem_rd_rsp_data;
                    stalled_rsp_last <= mem_rd_rsp_last;
                end
            end
            else begin
                rsp_stall_active <= 0;
            end

            if (mem_wr_req_valid == 1'b1) begin
                check_write_request_stability();
                if (mem_wr_req_ready == 1'b1) begin
                    mem_wr_req_count <= mem_wr_req_count + 1;
                    wr_req_stall_active <= 0;
                end
                else begin
                    wr_req_stall_active <= 1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
            end
            else begin
                wr_req_stall_active <= 0;
            end

            if (mem_wr_data_valid == 1'b1) begin
                check_write_data_stability();
                if (mem_wr_data_ready == 1'b1) begin
                    mem_wr_word_count_total <= mem_wr_word_count_total + 1;
                    wr_data_stall_active <= 0;
                end
                else begin
                    wr_data_stall_active <= 1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_last <= mem_wr_data_last;
                end
            end
            else begin
                wr_data_stall_active <= 0;
            end

            if (dut.lm_head_path.scheduler_row_result_valid) begin
                check_tile_logits();
            end
            if (dut.lm_head_path.scheduler_tile_complete_pulse) begin
                core_update_count <= core_update_count + 1;
            end

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                if (cycle_count == (last_done_cycle + 1)) begin
                    $display("FAIL: adjacent done pulses at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
                last_done_cycle <= cycle_count;
            end

            if ((trace_fd != 0) &&
                ((trace_every_cycle != 0) ||
                 ((mem_rd_req_valid == 1'b1) && (mem_rd_req_ready == 1'b1)) ||
                 ((mem_wr_req_valid == 1'b1) && (mem_wr_req_ready == 1'b1)) ||
                 ((mem_wr_data_valid == 1'b1) && (mem_wr_data_ready == 1'b1)) ||
                 (mem_wr_done == 1'b1) ||
                 (done == 1'b1) ||
                 dut.lm_head_path.scheduler_row_result_valid)) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    run_index,
                    start,
                    busy,
                    done,
                    error,
                    norm_saturation,
                    dut.state,
                    dut.final_norm_state_debug,
                    dut.lm_head_path.state,
                    dut.lm_head_path.scheduler_state_debug,
                    mem_rd_req_valid,
                    mem_rd_req_ready,
                    mem_rd_req_addr,
                    mem_rd_req_len_bytes,
                    mem_rd_rsp_valid,
                    mem_rd_rsp_ready,
                    mem_rd_rsp_last,
                    mem_wr_req_valid,
                    mem_wr_req_ready,
                    mem_wr_data_valid,
                    mem_wr_data_ready,
                    mem_wr_done,
                    active_write_kind,
                    dut.lm_head_path.scheduler_current_tile,
                    best_token_id,
                    best_score_q26,
                    tiles_started,
                    tiles_completed,
                    norm_cycle_count,
                    mem_read_burst_count,
                    mem_read_word_count,
                    mem_write_word_count,
                    norm_write_word_count
                );
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_req_ready <= 1'b0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            read_active <= 1'b0;
            active_region <= REGION_QMAP;
            active_read_index <= 0;
            active_words_left <= 0;
            active_total_words <= 0;
            response_delay <= 0;
        end
        else begin
            mem_rd_req_ready <=
                (read_active == 1'b0) &&
                (mem_rd_rsp_valid == 1'b0) &&
                req_ready_pattern(cycle_count, mem_req_fire_count);

            if ((mem_rd_req_valid == 1'b1) && (mem_rd_req_ready == 1'b1)) begin
                read_active <= 1'b1;
                response_delay <= response_latency_pattern(mem_req_fire_count, cycle_count);
                active_words_left <= (mem_rd_req_len_bytes + 3) >> 2;
                active_total_words <= (mem_rd_req_len_bytes + 3) >> 2;
                if ((mem_rd_req_addr >= `QMAP_FINAL_TOKEN_BASE_ADDR) &&
                    (mem_rd_req_addr < (`QMAP_FINAL_TOKEN_BASE_ADDR + QMAP_IMAGE_BYTES))) begin
                    active_region <= REGION_QMAP;
                    active_read_index <= (mem_rd_req_addr - `QMAP_FINAL_TOKEN_BASE_ADDR) >> 2;
                end
                else if ((mem_rd_req_addr >= weight_base_addr) &&
                         (mem_rd_req_addr < (weight_base_addr + (WEIGHT_WORDS * 4)))) begin
                    active_region <= REGION_WEIGHT;
                    active_read_index <= (mem_rd_req_addr - weight_base_addr) >> 2;
                end
                else begin
                    active_region <= REGION_SCALE;
                    active_read_index <= (mem_rd_req_addr - scale_base_addr) >> 2;
                end
            end

            if ((mem_rd_rsp_valid == 1'b1) && (mem_rd_rsp_ready == 1'b1)) begin
                mem_rd_rsp_valid <= 1'b0;
                if (active_words_left <= 1) begin
                    read_active <= 1'b0;
                    active_words_left <= 0;
                    mem_rd_rsp_last <= 1'b0;
                end
                else begin
                    active_read_index <= active_read_index + 1;
                    active_words_left <= active_words_left - 1;
                end
            end
            else if ((read_active == 1'b1) && (mem_rd_rsp_valid == 1'b0)) begin
                if (response_delay > 0) begin
                    response_delay <= response_delay - 1;
                end
                else if (response_gap_pattern(cycle_count, active_total_words - active_words_left)) begin
                    mem_rd_rsp_valid <= 1'b1;
                    mem_rd_rsp_data <= memory_word(active_region, active_read_index);
                    mem_rd_rsp_last <= (active_words_left == 1);
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_wr_req_ready <= 1'b0;
            mem_wr_data_ready <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            write_active <= 1'b0;
            active_write_index <= 0;
            active_write_words_left <= 0;
            active_write_kind <= WRITE_KIND_NONE;
            write_done_delay <= 0;
        end
        else begin
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            mem_wr_req_ready <=
                (write_active == 1'b0) &&
                wr_req_ready_pattern(cycle_count, mem_wr_req_count);
            mem_wr_data_ready <=
                (write_active == 1'b1) &&
                wr_data_ready_pattern(cycle_count, mem_wr_word_count_total);

            if ((mem_wr_req_valid == 1'b1) && (mem_wr_req_ready == 1'b1)) begin
                write_active <= 1'b1;
                active_write_index <= (mem_wr_req_addr - `QMAP_FINAL_TOKEN_BASE_ADDR) >> 2;
                active_write_words_left <= (mem_wr_req_len_bytes + 3) >> 2;
                if (mem_wr_req_addr == descriptor_base_addr(SLOT_NORM_OUTPUT)) begin
                    active_write_kind <= WRITE_KIND_NORM;
                    if (mem_wr_req_len_bytes != 16'd4096) begin
                        $display("FAIL: norm write length actual=%0d expected=4096", mem_wr_req_len_bytes);
                        mismatch_count = mismatch_count + 1;
                    end
                end
                else if (mem_wr_req_addr == descriptor_base_addr(SLOT_OUTPUT)) begin
                    active_write_kind <= WRITE_KIND_OUTPUT;
                    if (mem_wr_req_len_bytes != 16'd12) begin
                        $display("FAIL: output write length actual=%0d expected=12", mem_wr_req_len_bytes);
                        mismatch_count = mismatch_count + 1;
                    end
                end
                else begin
                    active_write_kind <= WRITE_KIND_NONE;
                    $display("FAIL: write request address outside expected targets addr=0x%016h", mem_wr_req_addr);
                    mismatch_count = mismatch_count + 1;
                end
            end

            if ((mem_wr_data_valid == 1'b1) && (mem_wr_data_ready == 1'b1)) begin
                if (active_write_kind == WRITE_KIND_NORM) begin
                    if (mem_wr_data !== norm_expected_word(active_write_index - expected_norm_index)) begin
                        if (print_count < 32) begin
                            $display("FAIL: norm write data index=%0d actual=%08h expected=%08h",
                                     active_write_index - expected_norm_index,
                                     mem_wr_data,
                                     norm_expected_word(active_write_index - expected_norm_index));
                            print_count = print_count + 1;
                        end
                        norm_write_mismatch_count <= norm_write_mismatch_count + 1;
                    end
                    norm_write_word_count <= norm_write_word_count + 1;
                end
                else if (active_write_kind == WRITE_KIND_OUTPUT) begin
                    output_write_word_count <= output_write_word_count + 1;
                end
                qmap_mem[active_write_index] <= mem_wr_data;
                active_write_index <= active_write_index + 1;
                if (active_write_words_left == 1) begin
                    if (mem_wr_data_last != 1'b1) begin
                        $display("FAIL: final write word missing last");
                        mismatch_count = mismatch_count + 1;
                    end
                    write_active <= 1'b0;
                    active_write_words_left <= 0;
                    active_write_kind <= WRITE_KIND_NONE;
                    write_done_delay <= 2;
                end
                else begin
                    if (mem_wr_data_last == 1'b1) begin
                        $display("FAIL: early write last with %0d words left", active_write_words_left);
                        mismatch_count = mismatch_count + 1;
                    end
                    active_write_words_left <= active_write_words_left - 1;
                end
            end

            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) begin
                    mem_wr_done <= 1'b1;
                end
            end
        end
    end

    initial begin
        vector_dir = "FPGA_Project/sim/vectors";
`ifdef QMAP_FINAL_TOKEN_TB_FULL_VOCAB
        lm_prefix = "lm_head_argmax_full_vocab_real";
        final_norm_prefix = "final_rmsnorm_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_full_vocab_image_words32.hex";
        qmap_expected_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_full_vocab_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_final_token_tail_full_vocab_trace.csv";
        trace_every_cycle = 0;
        fast_memory = 1;
`ifdef QMAP_FINAL_TOKEN_TB_LAYER2_CHAINED
        lm_prefix = "lm_head_argmax_layer2_chained_full_vocab_real";
        final_norm_prefix = "final_rmsnorm_layer2_chained_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_layer2_chained_full_vocab_image_words32.hex";
        qmap_expected_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_layer2_chained_full_vocab_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_final_token_tail_layer2_chained_full_vocab_xsim_trace.csv";
`endif
`else
        lm_prefix = "lm_head_argmax_stage_real";
        final_norm_prefix = "final_rmsnorm_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_compact_image_words32.hex";
        qmap_expected_file = "FPGA_Project/sim/vectors/qmap_final_token_tail_compact_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_final_token_tail_compute_path_trace.csv";
        trace_every_cycle = 1;
        fast_memory = 0;
`endif
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("lm_prefix=%s", lm_prefix)) begin
        end
        if ($value$plusargs("final_norm_prefix=%s", final_norm_prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("qmap_expected=%s", qmap_expected_file)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end
        if ($value$plusargs("trace_every_cycle=%d", trace_every_cycle)) begin
        end
        if ($value$plusargs("fast_memory=%d", fast_memory)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,run,start,busy,done,error,norm_saturation,tail_state,norm_state,lm_state,lm_sched_state,rd_req_valid,rd_req_ready,rd_req_addr,rd_req_len,rd_rsp_valid,rd_rsp_ready,rd_rsp_last,wr_req_valid,wr_req_ready,wr_data_valid,wr_data_ready,wr_done,active_write_kind,current_tile,best_token,best_score,tiles_started,tiles_completed,norm_cycle_count,mem_read_burst_count,mem_read_word_count,mem_write_word_count,norm_write_word_count\n"
        );

        load_vectors();

        mismatch_count = 0;
        print_count = 0;
        run_index = 0;
        run_scheduler_req_count = 0;
        last_success_token = 32'd0;
        last_success_score_q26 = '0;
        start = 1'b0;
        spurious_start_seen_busy = 0;
        rst_n = 1'b0;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_success(1);
        if (ENABLE_INVALID_RUN != 0) begin
            run_invalid_gamma_dtype(2);
        end

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_final_token_tail_compute_path final-token tail test");
        $display("  final norm prefix       = %s", final_norm_prefix);
        $display("  lm prefix               = %s", lm_prefix);
        $display("  expected token          = %0d", expected_run_token);
        $display("  expected score q26      = %0d", expected_run_score_q26);
        $display("  success token           = %0d", last_success_token);
        $display("  success score q26       = %0d", last_success_score_q26);
        $display("  read req/rsp fires      = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap hidden/gamma/norm  = %0d / %0d / %0d / %0d",
                 mem_qmap_req_count, mem_hidden_req_count, mem_gamma_req_count, mem_norm_read_req_count);
        $display("  weight/scale reqs       = %0d / %0d", mem_weight_req_count, mem_scale_req_count);
        $display("  write reqs/words        = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  norm/output write words = %0d / %0d", norm_write_word_count, output_write_word_count);
        $display("  tile updates            = %0d", core_update_count);
        $display("  checked logits          = %0d", checked_logit_count);
        $display("  max_abs_logit_diff      = %0d", max_abs_logit_diff);
        $display("  norm write mismatches   = %0d", norm_write_mismatch_count);
        $display("  spurious start covered  = %0d", spurious_start_seen_busy);
        $display("  done seen count         = %0d", done_seen_count);
        $display("  total cycles waited     = %0d", cycle_count);
        $display("  trace                   = %s", tracefile);

        if (done_seen_count != (1 + ENABLE_INVALID_RUN)) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=%0d",
                     done_seen_count, 1 + ENABLE_INVALID_RUN);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_hidden_req_count != 4 || mem_gamma_req_count != 4 || mem_norm_read_req_count != 4) begin
            $display("FAIL: final-token qmap read count mismatch hidden=%0d gamma=%0d norm_read=%0d",
                     mem_hidden_req_count, mem_gamma_req_count, mem_norm_read_req_count);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_weight_req_count !=
            (MAX_TILES * TILE_ROWS * WEIGHT_BURSTS_PER_ROW)) begin
            $display("FAIL: weight request count actual=%0d expected=%0d",
                     mem_weight_req_count,
                     MAX_TILES * TILE_ROWS *
                     WEIGHT_BURSTS_PER_ROW);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_scale_req_count !=
            (MAX_TILES * TILE_ROWS * SCALE_BURSTS_PER_ROW)) begin
            $display("FAIL: scale request count actual=%0d expected=%0d",
                     mem_scale_req_count,
                     MAX_TILES * TILE_ROWS *
                     SCALE_BURSTS_PER_ROW);
            mismatch_count = mismatch_count + 1;
        end
        if (max_abs_logit_diff != 0) begin
            $display("FAIL: expected exact logits, max_abs_logit_diff=%0d", max_abs_logit_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_final_token_tail_compute_path found %0d mismatches", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_final_token_tail_compute_path matched final RMSNorm write-back and LM-head token/score.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
