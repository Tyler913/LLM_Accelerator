`timescale 1ns/1ps
`default_nettype none

module tb_lm_head_argmax_tile_scheduler;

    localparam int ADDR_WIDTH       = 64;
    localparam int MAX_TILES        = 64;
    localparam int TILE_COUNT_WIDTH = $clog2(MAX_TILES + 1);
    localparam int TILE_ROWS        = 16;
`ifdef LM_HEAD_ROW_PARALLEL
    localparam int ROW_PARALLEL     = `LM_HEAD_ROW_PARALLEL;
`else
    localparam int ROW_PARALLEL     = TILE_ROWS;
`endif
`ifdef LM_HEAD_GROUP_PARALLEL
    localparam int GROUP_PARALLEL   = `LM_HEAD_GROUP_PARALLEL;
`else
    localparam int GROUP_PARALLEL   = 4;
`endif
    localparam int SCAN_ROWS        = MAX_TILES * TILE_ROWS;
    localparam int INPUT_SIZE       = 1024;
    localparam int GROUP_SIZE       = 64;
    localparam int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH        = 24;
    localparam int WEIGHT_WIDTH     = 4;
    localparam int SCALE_WIDTH      = 16;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;
    localparam int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;
    localparam int TOKEN_ID_WIDTH   = 32;
    localparam int WEIGHT_ROW_BYTES = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES  = (GROUP_COUNT * SCALE_WIDTH) / 8;
    localparam int ROW_BATCH_COUNT = TILE_ROWS / ROW_PARALLEL;
    localparam int BATCH_WEIGHT_BYTES = ROW_PARALLEL * WEIGHT_ROW_BYTES;
    localparam int BATCH_SCALE_BYTES  = ROW_PARALLEL * SCALE_ROW_BYTES;
    localparam int WEIGHT_BURSTS_PER_BATCH =
        (BATCH_WEIGHT_BYTES + MAX_READ_BYTES - 1) / MAX_READ_BYTES;
    localparam int SCALE_BURSTS_PER_BATCH = 1;
    localparam int BURSTS_PER_BATCH =
        WEIGHT_BURSTS_PER_BATCH + SCALE_BURSTS_PER_BATCH;
    localparam int WEIGHT_BURSTS_PER_TILE =
        ROW_BATCH_COUNT * WEIGHT_BURSTS_PER_BATCH;
    localparam int SCALE_BURSTS_PER_TILE =
        ROW_BATCH_COUNT * SCALE_BURSTS_PER_BATCH;
    localparam int BURSTS_PER_TILE =
        WEIGHT_BURSTS_PER_TILE + SCALE_BURSTS_PER_TILE;
    localparam int TILE_WEIGHT_BYTES = TILE_ROWS * WEIGHT_ROW_BYTES;
    localparam int TILE_SCALE_BYTES  = TILE_ROWS * SCALE_ROW_BYTES;
    localparam int WORDS_PER_TILE = (TILE_WEIGHT_BYTES + TILE_SCALE_BYTES) / 4;
    localparam int WEIGHT_WORDS = SCAN_ROWS * WEIGHT_ROW_BYTES / 4;
    localparam int SCALE_WORDS = SCAN_ROWS * SCALE_ROW_BYTES / 4;
    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] BATCH_SCALE_BYTES_U16 = BATCH_SCALE_BYTES;

    logic clk;
    logic rst_n;
    logic start;
    logic [TOKEN_ID_WIDTH-1 : 0] token_base;
    logic [TILE_COUNT_WIDTH-1 : 0] run_tile_count;
    logic [INPUT_SIZE*ACT_WIDTH-1 : 0] activation_flat;
    logic [ADDR_WIDTH-1 : 0] weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] scale_base_addr;

    logic mem_req_valid;
    logic mem_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_req_addr;
    logic [15 : 0] mem_req_len_bytes;
    logic mem_rsp_valid;
    logic mem_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] mem_rsp_data;
    logic mem_rsp_last;

    logic busy;
    logic done;
    logic error;
    logic [TOKEN_ID_WIDTH-1 : 0] best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] best_score_q26;
    logic [TILE_COUNT_WIDTH-1 : 0] current_tile_index;
    logic [31 : 0] tiles_started;
    logic [31 : 0] tiles_completed;
    logic [31 : 0] compute_cycle_count;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;

    logic signed [ACT_WIDTH-1 : 0] activation_mem [0:INPUT_SIZE-1];
    logic [31 : 0] weight_words_mem [0:WEIGHT_WORDS-1];
    logic [31 : 0] scale_words_mem [0:SCALE_WORDS-1];
    logic signed [ROW_ACC_WIDTH-1 : 0] expected_logits_mem [0:SCAN_ROWS-1];
    logic [TOKEN_ID_WIDTH-1 : 0] scan_base_token_mem [0:0];
    logic [ADDR_WIDTH-1 : 0] weight_base_addr_mem [0:0];
    logic [ADDR_WIDTH-1 : 0] scale_base_addr_mem [0:0];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer row_index;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer run_index;
    integer current_run_tile_count;
    integer expected_run_token;
    integer signed expected_run_score;
    integer spurious_start_seen_busy;
    integer done_seen_count;
    integer last_done_cycle;
    integer core_update_count;
    integer checked_logit_count;
    integer max_abs_logit_diff;
    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_weight_req_count;
    integer mem_scale_req_count;
    integer run_req_fire_count;
    integer run_checked_logit_base;
    integer run_core_update_base;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer response_delay;
    integer req_stall_active;
    integer rsp_stall_active;
    longint signed logit_diff;
    logic active_is_weight;
    logic read_active;
    logic [ADDR_WIDTH-1 : 0] stalled_req_addr;
    logic [15 : 0] stalled_req_len;
    logic [31 : 0] stalled_rsp_data;
    logic stalled_rsp_last;

    lm_head_argmax_tile_scheduler #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .MAX_TILES       (MAX_TILES),
        .TILE_COUNT_WIDTH(TILE_COUNT_WIDTH),
        .TILE_ROWS       (TILE_ROWS),
        .ROW_PARALLEL    (ROW_PARALLEL),
        .INPUT_SIZE      (INPUT_SIZE),
        .GROUP_SIZE      (GROUP_SIZE),
        .GROUP_COUNT     (GROUP_COUNT),
        .GROUP_PARALLEL  (GROUP_PARALLEL),
        .ACT_WIDTH       (ACT_WIDTH),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH),
        .SCALE_WIDTH     (SCALE_WIDTH),
        .ROW_ACC_WIDTH   (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH  (TOKEN_ID_WIDTH),
        .MEM_DATA_WIDTH  (MEM_DATA_WIDTH),
        .MAX_READ_BYTES  (MAX_READ_BYTES)
    ) dut (
        .i_clk                  (clk),
        .i_rst_n                (rst_n),
        .i_start                (start),
        .i_token_base           (token_base),
        .i_tile_count           (run_tile_count),
        .i_activation_flat      (activation_flat),
        .i_weight_base_addr     (weight_base_addr),
        .i_scale_base_addr      (scale_base_addr),
        .o_mem_req_valid        (mem_req_valid),
        .i_mem_req_ready        (mem_req_ready),
        .o_mem_req_addr         (mem_req_addr),
        .o_mem_req_len_bytes    (mem_req_len_bytes),
        .i_mem_rsp_valid        (mem_rsp_valid),
        .o_mem_rsp_ready        (mem_rsp_ready),
        .i_mem_rsp_data         (mem_rsp_data),
        .i_mem_rsp_last         (mem_rsp_last),
        .o_busy                 (busy),
        .o_done                 (done),
        .o_error                (error),
        .o_best_token_id        (best_token_id),
        .o_best_score_q26       (best_score_q26),
        .o_current_tile_index   (current_tile_index),
        .o_tiles_started        (tiles_started),
        .o_tiles_completed      (tiles_completed),
        .o_compute_cycle_count  (compute_cycle_count),
        .o_mem_read_burst_count (mem_read_burst_count),
        .o_mem_read_word_count  (mem_read_word_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/lm_head_argmax_tile_scheduler.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, mem_req_valid);
        $dumpvars(0, mem_req_ready);
        $dumpvars(0, mem_rsp_valid);
        $dumpvars(0, mem_rsp_ready);
        $dumpvars(0, mem_rsp_last);
        $dumpvars(0, current_tile_index);
        $dumpvars(0, best_token_id);
        $dumpvars(0, best_score_q26);
        $dumpvars(0, dut.state);
        $dumpvars(0, dut.tile_stage.argmax_core.current_state);
        $dumpvars(0, dut.tile_stage.tile_reader.state);
    end

    function automatic logic req_ready_pattern(input integer cycle, input integer count);
        begin
            req_ready_pattern =
                ((cycle % 11) != 3) &&
                (((cycle + count) % 17) != 5);
        end
    endfunction

    function automatic integer response_latency_pattern(input integer count, input integer cycle);
        begin
            response_latency_pattern = 1 + ((count + cycle) % 5);
        end
    endfunction

    function automatic logic response_gap_pattern(input integer cycle, input integer word_count);
        begin
            response_gap_pattern =
                ((cycle % 13) != 4) &&
                (((cycle + word_count) % 29) != 7);
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input logic is_weight, input integer word_index);
        begin
            if (is_weight == 1'b1) begin
                memory_word = weight_words_mem[word_index];
            end
            else begin
                memory_word = scale_words_mem[word_index];
            end
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_activation.hex"}, activation_mem);
            $readmemh({vector_dir, "/", prefix, "_weight_words32.hex"}, weight_words_mem);
            $readmemh({vector_dir, "/", prefix, "_scale_words32.hex"}, scale_words_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_scan_logits_q26.hex"}, expected_logits_mem);
            $readmemh({vector_dir, "/", prefix, "_scan_base_token.hex"}, scan_base_token_mem);
            $readmemh({vector_dir, "/", prefix, "_weight_base_addr.hex"}, weight_base_addr_mem);
            $readmemh({vector_dir, "/", prefix, "_scale_base_addr.hex"}, scale_base_addr_mem);
        end
    endtask

    task pack_inputs;
        begin
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                activation_flat[element_index*ACT_WIDTH +: ACT_WIDTH] = activation_mem[element_index];
            end
            token_base = scan_base_token_mem[0];
            weight_base_addr = weight_base_addr_mem[0];
            scale_base_addr = scale_base_addr_mem[0];
        end
    endtask

    task calculate_expected_for_tiles(input integer tile_count);
        begin
            expected_run_token = token_base;
            expected_run_score = expected_logits_mem[0];
            for (element_index = 1; element_index < tile_count*TILE_ROWS; element_index = element_index + 1) begin
                if ($signed(expected_logits_mem[element_index]) > expected_run_score) begin
                    expected_run_score = expected_logits_mem[element_index];
                    expected_run_token = token_base + element_index;
                end
            end
        end
    endtask

    task check_request_address;
        integer tile_id;
        integer batch_id;
        integer burst_id;
        integer tile_token;
        integer burst_byte_offset;
        integer remaining_weight_bytes;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        logic [15 : 0] expected_len;
        begin
            tile_id = run_req_fire_count / BURSTS_PER_TILE;
            batch_id =
                (run_req_fire_count % BURSTS_PER_TILE) / BURSTS_PER_BATCH;
            burst_id = run_req_fire_count % BURSTS_PER_BATCH;
            tile_token =
                token_base +
                (tile_id * TILE_ROWS) +
                (batch_id * ROW_PARALLEL);

            if (tile_id >= current_run_tile_count) begin
                if (print_count < 32) begin
                    $display("FAIL: extra memory request run %0d tile_id=%0d tile_count=%0d",
                             run_index, tile_id, current_run_tile_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end

            if (burst_id < WEIGHT_BURSTS_PER_BATCH) begin
                burst_byte_offset = burst_id * MAX_READ_BYTES;
                remaining_weight_bytes =
                    BATCH_WEIGHT_BYTES - burst_byte_offset;
                expected_addr =
                    weight_base_addr +
                    (tile_token * WEIGHT_ROW_BYTES) +
                    burst_byte_offset;
                expected_len =
                    (remaining_weight_bytes > MAX_READ_BYTES)
                    ? MAX_READ_BYTES_U16
                    : remaining_weight_bytes[15 : 0];
                mem_weight_req_count = mem_weight_req_count + 1;
            end
            else begin
                expected_addr =
                    scale_base_addr +
                    (tile_token * SCALE_ROW_BYTES);
                expected_len = BATCH_SCALE_BYTES_U16;
                mem_scale_req_count = mem_scale_req_count + 1;
            end

            if ((mem_req_addr !== expected_addr) || (mem_req_len_bytes !== expected_len)) begin
                if (print_count < 32) begin
                    $display("FAIL: request mismatch run %0d tile %0d burst %0d addr=0x%016h expected=0x%016h len=%0d expected_len=%0d",
                             run_index, tile_id, burst_id, mem_req_addr, expected_addr,
                             mem_req_len_bytes, expected_len);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_request_stability;
        begin
            if (req_stall_active != 0) begin
                if ((mem_req_addr !== stalled_req_addr) ||
                    (mem_req_len_bytes !== stalled_req_len)) begin
                    if (print_count < 32) begin
                        $display("FAIL: memory request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_response_stability;
        begin
            if (rsp_stall_active != 0) begin
                if ((mem_rsp_data !== stalled_rsp_data) ||
                    (mem_rsp_last !== stalled_rsp_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: memory response changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_tile_logits;
        integer local_index;
        logic signed [ROW_ACC_WIDTH-1 : 0] observed_logit;
        begin
            for (row_index = 0; row_index < ROW_PARALLEL; row_index = row_index + 1) begin
                local_index =
                    (dut.current_tile_index * TILE_ROWS) +
                    (dut.row_batch_index * ROW_PARALLEL) +
                    row_index;
                observed_logit =
                    $signed(dut.tile_stage.argmax_core.tile_output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH]);
                logit_diff = observed_logit - expected_logits_mem[local_index];
                if (logit_diff < 0) begin
                    logit_diff = -logit_diff;
                end
                if (logit_diff > max_abs_logit_diff) begin
                    max_abs_logit_diff = logit_diff;
                end
                if (observed_logit !== expected_logits_mem[local_index]) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: logit run %0d tile %0d row %0d token %0d actual=%0d expected=%0d",
                            run_index,
                            dut.current_tile_index,
                            row_index,
                            token_base + local_index,
                            observed_logit,
                            expected_logits_mem[local_index]
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
                checked_logit_count = checked_logit_count + 1;
            end
        end
    endtask

    task check_run_result(input integer expected_tiles);
        begin
            if (done != 1'b1) begin
                $display("FAIL: check_run_result called without done");
                mismatch_count = mismatch_count + 1;
            end
            if (error != 1'b0) begin
                $display("FAIL: scheduler error high at done for run %0d", run_index);
                mismatch_count = mismatch_count + 1;
            end
            if (best_token_id != expected_run_token) begin
                $display("FAIL: best token run %0d actual=%0d expected=%0d",
                         run_index, best_token_id, expected_run_token);
                mismatch_count = mismatch_count + 1;
            end
            if (best_score_q26 != expected_run_score) begin
                $display("FAIL: best score run %0d actual=%0d expected=%0d",
                         run_index, best_score_q26, expected_run_score);
                mismatch_count = mismatch_count + 1;
            end
            if (tiles_started != expected_tiles) begin
                $display("FAIL: tiles_started run %0d actual=%0d expected=%0d",
                         run_index, tiles_started, expected_tiles);
                mismatch_count = mismatch_count + 1;
            end
            if (tiles_completed != expected_tiles) begin
                $display("FAIL: tiles_completed run %0d actual=%0d expected=%0d",
                         run_index, tiles_completed, expected_tiles);
                mismatch_count = mismatch_count + 1;
            end
            if (mem_read_burst_count != expected_tiles*BURSTS_PER_TILE) begin
                $display("FAIL: per-run burst count run %0d actual=%0d expected=%0d",
                         run_index, mem_read_burst_count, expected_tiles*BURSTS_PER_TILE);
                mismatch_count = mismatch_count + 1;
            end
            if (mem_read_word_count != expected_tiles*WORDS_PER_TILE) begin
                $display("FAIL: per-run word count run %0d actual=%0d expected=%0d",
                         run_index, mem_read_word_count, expected_tiles*WORDS_PER_TILE);
                mismatch_count = mismatch_count + 1;
            end
            if ((core_update_count - run_core_update_base) !=
                expected_tiles*ROW_BATCH_COUNT) begin
                $display("FAIL: per-run tile update count run %0d actual=%0d expected=%0d",
                         run_index, core_update_count - run_core_update_base,
                         expected_tiles*ROW_BATCH_COUNT);
                mismatch_count = mismatch_count + 1;
            end
            if ((checked_logit_count - run_checked_logit_base) != expected_tiles*TILE_ROWS) begin
                $display("FAIL: per-run checked logit count run %0d actual=%0d expected=%0d",
                         run_index, checked_logit_count - run_checked_logit_base, expected_tiles*TILE_ROWS);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task run_scan(input integer run_id, input integer tile_count);
        integer wait_cycles;
        begin
            run_index = run_id;
            current_run_tile_count = tile_count;
            run_req_fire_count = 0;
            run_checked_logit_base = checked_logit_count;
            run_core_update_base = core_update_count;
            calculate_expected_for_tiles(tile_count);

            @(posedge clk);
            run_tile_count <= tile_count;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            wait_cycles = 0;
            while ((done != 1'b1) && (wait_cycles < 4000000)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
                if ((run_id == 1) && (wait_cycles == 20) && (busy == 1'b1)) begin
                    start <= 1'b1;
                    spurious_start_seen_busy <= 1;
                end
                else if ((run_id == 1) && (wait_cycles == 21)) begin
                    start <= 1'b0;
                end
            end

            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for scheduler run %0d done", run_id);
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end

            check_run_result(tile_count);
            @(posedge clk);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1000;
            core_update_count <= 0;
            checked_logit_count <= 0;
            max_abs_logit_diff <= 0;
            mem_req_fire_count <= 0;
            mem_rsp_fire_count <= 0;
            mem_weight_req_count <= 0;
            mem_scale_req_count <= 0;
            req_stall_active <= 0;
            rsp_stall_active <= 0;
            stalled_req_addr <= 'd0;
            stalled_req_len <= 'd0;
            stalled_rsp_data <= 'd0;
            stalled_rsp_last <= 1'b0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if (mem_req_valid == 1'b1) begin
                check_request_stability();
                if (mem_req_ready == 1'b1) begin
                    check_request_address();
                    mem_req_fire_count <= mem_req_fire_count + 1;
                    run_req_fire_count <= run_req_fire_count + 1;
                    req_stall_active <= 0;
                end
                else begin
                    req_stall_active <= 1;
                    stalled_req_addr <= mem_req_addr;
                    stalled_req_len <= mem_req_len_bytes;
                end
            end
            else begin
                req_stall_active <= 0;
            end

            if (mem_rsp_valid == 1'b1) begin
                check_response_stability();
                if (mem_rsp_ready == 1'b1) begin
                    mem_rsp_fire_count <= mem_rsp_fire_count + 1;
                    rsp_stall_active <= 0;
                end
                else begin
                    rsp_stall_active <= 1;
                    stalled_rsp_data <= mem_rsp_data;
                    stalled_rsp_last <= mem_rsp_last;
                end
            end
            else begin
                rsp_stall_active <= 0;
            end

            if (dut.tile_stage.argmax_core.current_state == dut.tile_stage.argmax_core.UPDATE_BEST) begin
                check_tile_logits();
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

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    run_index,
                    start,
                    busy,
                    done,
                    error,
                    dut.state,
                    dut.tile_stage.argmax_core.current_state,
                    dut.tile_stage.tile_reader.state,
                    mem_req_valid,
                    mem_req_ready,
                    mem_req_valid && mem_req_ready,
                    mem_req_addr,
                    mem_req_len_bytes,
                    mem_rsp_valid,
                    mem_rsp_ready,
                    mem_rsp_valid && mem_rsp_ready,
                    mem_rsp_last,
                    read_active,
                    response_delay,
                    active_words_left,
                    current_tile_index,
                    best_token_id,
                    best_score_q26,
                    tiles_started,
                    tiles_completed,
                    compute_cycle_count,
                    mem_read_burst_count,
                    mem_read_word_count,
                    core_update_count,
                    run_req_fire_count
                );
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_req_ready <= 1'b0;
            mem_rsp_valid <= 1'b0;
            mem_rsp_data <= 32'd0;
            mem_rsp_last <= 1'b0;
            read_active <= 1'b0;
            active_is_weight <= 1'b0;
            active_read_index <= 0;
            active_words_left <= 0;
            active_total_words <= 0;
            response_delay <= 0;
        end
        else begin
            mem_req_ready <=
                (read_active == 1'b0) &&
                (mem_rsp_valid == 1'b0) &&
                req_ready_pattern(cycle_count, mem_req_fire_count);

            if ((mem_req_valid == 1'b1) && (mem_req_ready == 1'b1)) begin
                read_active <= 1'b1;
                response_delay <= response_latency_pattern(mem_req_fire_count, cycle_count);
                active_words_left <= (mem_req_len_bytes + 3) >> 2;
                active_total_words <= (mem_req_len_bytes + 3) >> 2;
                if ((mem_req_addr >= (weight_base_addr + (token_base * WEIGHT_ROW_BYTES))) &&
                    (mem_req_addr < (weight_base_addr + ((token_base + SCAN_ROWS) * WEIGHT_ROW_BYTES)))) begin
                    active_is_weight <= 1'b1;
                    active_read_index <= (mem_req_addr - (weight_base_addr + (token_base * WEIGHT_ROW_BYTES))) >> 2;
                end
                else begin
                    active_is_weight <= 1'b0;
                    active_read_index <= (mem_req_addr - (scale_base_addr + (token_base * SCALE_ROW_BYTES))) >> 2;
                end
            end

            if ((mem_rsp_valid == 1'b1) && (mem_rsp_ready == 1'b1)) begin
                mem_rsp_valid <= 1'b0;
                if (active_words_left <= 1) begin
                    read_active <= 1'b0;
                    active_words_left <= 0;
                    mem_rsp_last <= 1'b0;
                end
                else begin
                    active_read_index <= active_read_index + 1;
                    active_words_left <= active_words_left - 1;
                end
            end
            else if ((read_active == 1'b1) && (mem_rsp_valid == 1'b0)) begin
                if (response_delay > 0) begin
                    response_delay <= response_delay - 1;
                end
                else if (response_gap_pattern(cycle_count, active_total_words - active_words_left)) begin
                    mem_rsp_valid <= 1'b1;
                    mem_rsp_data <= memory_word(active_is_weight, active_read_index);
                    mem_rsp_last <= (active_words_left == 1);
                end
            end
        end
    end

    initial begin
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "lm_head_argmax_stage_real";
        tracefile = "FPGA_Project/sim/lm_head_argmax_tile_scheduler_trace.csv";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,run,start,busy,done,error,sched_state,core_state,reader_state,mem_req_valid,mem_req_ready,mem_req_fire,mem_req_addr,mem_req_len,mem_rsp_valid,mem_rsp_ready,mem_rsp_fire,mem_rsp_last,read_active,response_delay,active_words_left,current_tile,best_token,best_score,tiles_started,tiles_completed,compute_cycle_count,mem_read_burst_count,mem_read_word_count,core_update_count,run_req_fire_count\n"
        );

        load_vectors();
        pack_inputs();

        mismatch_count = 0;
        print_count = 0;
        run_index = 0;
        current_run_tile_count = 0;
        run_req_fire_count = 0;
        run_tile_count = 'd0;
        start = 1'b0;
        spurious_start_seen_busy = 0;
        rst_n = 1'b0;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_scan(1, MAX_TILES);
        run_scan(2, 23);

        $fclose(trace_fd);
        trace_fd = 0;

        $display("lm_head_argmax_tile_scheduler memory-backed Q4 LM-head tile scheduler test");
        $display("  final expected token     = %0d", expected_run_token);
        $display("  final expected score q26 = %0d", expected_run_score);
        $display("  final best token         = %0d", best_token_id);
        $display("  final best score q26     = %0d", best_score_q26);
        $display("  mem req fires            = %0d", mem_req_fire_count);
        $display("  mem rsp fires            = %0d", mem_rsp_fire_count);
        $display("  mem weight reqs          = %0d", mem_weight_req_count);
        $display("  mem scale reqs           = %0d", mem_scale_req_count);
        $display("  tile updates             = %0d", core_update_count);
        $display("  checked logits           = %0d", checked_logit_count);
        $display("  max_abs_logit_diff       = %0d", max_abs_logit_diff);
        $display("  spurious start covered   = %0d", spurious_start_seen_busy);
        $display("  done seen count          = %0d", done_seen_count);
        $display("  total cycles waited      = %0d", cycle_count);
        $display("  row parallel             = %0d", ROW_PARALLEL);
        $display("  group parallel           = %0d", GROUP_PARALLEL);
        $display("  trace                    = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_req_fire_count != (MAX_TILES + 23)*BURSTS_PER_TILE) begin
            $display("FAIL: memory request count mismatch actual=%0d expected=%0d",
                     mem_req_fire_count, (MAX_TILES + 23)*BURSTS_PER_TILE);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_rsp_fire_count != (MAX_TILES + 23)*WORDS_PER_TILE) begin
            $display("FAIL: memory response word count mismatch actual=%0d expected=%0d",
                     mem_rsp_fire_count, (MAX_TILES + 23)*WORDS_PER_TILE);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_weight_req_count != (MAX_TILES + 23)*WEIGHT_BURSTS_PER_TILE) begin
            $display("FAIL: weight request count mismatch actual=%0d expected=%0d",
                     mem_weight_req_count, (MAX_TILES + 23)*WEIGHT_BURSTS_PER_TILE);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_scale_req_count != (MAX_TILES + 23)*SCALE_BURSTS_PER_TILE) begin
            $display("FAIL: scale request count mismatch actual=%0d expected=%0d",
                     mem_scale_req_count, (MAX_TILES + 23)*SCALE_BURSTS_PER_TILE);
            mismatch_count = mismatch_count + 1;
        end
        if (core_update_count != (MAX_TILES + 23)*ROW_BATCH_COUNT) begin
            $display("FAIL: tile update count mismatch actual=%0d expected=%0d",
                     core_update_count, (MAX_TILES + 23)*ROW_BATCH_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (checked_logit_count != (MAX_TILES + 23)*TILE_ROWS) begin
            $display("FAIL: checked logit count mismatch actual=%0d expected=%0d",
                     checked_logit_count, (MAX_TILES + 23)*TILE_ROWS);
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
            $display("FAIL: lm_head_argmax_tile_scheduler found %0d mismatches", mismatch_count);
            $finish(1);
        end

        $display("PASS: lm_head_argmax_tile_scheduler matched memory-backed Q4 LM-head logits and global greedy argmax.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
