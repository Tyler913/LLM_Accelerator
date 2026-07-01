`timescale 1ns/1ps
`default_nettype none

module tb_lm_head_argmax_stage;

    localparam int SCAN_ROWS      = 1024;
    localparam int TILE_ROWS      = 16;
    localparam int INPUT_SIZE     = 1024;
    localparam int GROUP_SIZE     = 64;
    localparam int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH      = 24;
    localparam int WEIGHT_WIDTH   = 4;
    localparam int SCALE_WIDTH    = 16;
    localparam int PARTIAL_WIDTH  = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH   = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH  = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;
    localparam int TOKEN_ID_WIDTH = 32;
    localparam int TILE_COUNT     = SCAN_ROWS / TILE_ROWS;
    localparam int TILE_INDEX_W   = $clog2(TILE_COUNT);
    localparam int TILE_WEIGHT_BITS = TILE_ROWS * INPUT_SIZE * WEIGHT_WIDTH;
    localparam int TILE_SCALE_BITS  = TILE_ROWS * GROUP_COUNT * SCALE_WIDTH;

    logic clk;
    logic rst_n;
    logic start;
    logic [TOKEN_ID_WIDTH-1 : 0] token_base;
    logic [INPUT_SIZE*ACT_WIDTH-1 : 0] activation_flat;

    logic tile_req_valid;
    logic tile_req_ready;
    logic [TILE_INDEX_W-1 : 0] tile_index;
    logic [TOKEN_ID_WIDTH-1 : 0] tile_token_base;
    logic tile_valid;
    logic tile_ready;
    logic [TILE_WEIGHT_BITS-1 : 0] tile_weight_flat;
    logic [TILE_SCALE_BITS-1 : 0] tile_scale_flat;

    logic busy;
    logic done;
    logic error;
    logic [TOKEN_ID_WIDTH-1 : 0] best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] best_score_q26;
    logic [31 : 0] tiles_requested;
    logic [31 : 0] tiles_completed;
    logic [31 : 0] compute_cycle_count;

    logic signed [ACT_WIDTH-1 : 0] activation_mem [0:INPUT_SIZE-1];
    logic [TILE_WEIGHT_BITS-1 : 0] weight_tile_mem [0:TILE_COUNT-1];
    logic [TILE_SCALE_BITS-1 : 0] scale_tile_mem [0:TILE_COUNT-1];
    logic signed [ROW_ACC_WIDTH-1 : 0] expected_logits_mem [0:SCAN_ROWS-1];
    logic [TOKEN_ID_WIDTH-1 : 0] expected_best_token_mem [0:0];
    logic signed [ROW_ACC_WIDTH-1 : 0] expected_best_score_mem [0:0];
    logic [TOKEN_ID_WIDTH-1 : 0] scan_base_token_mem [0:0];

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
    integer spurious_start_seen_busy;
    integer done_seen_count;
    integer last_done_cycle;
    integer req_fire_count;
    integer resp_fire_count;
    integer checked_logit_count;
    integer output_update_count;
    integer expected_tile_index;
    integer response_delay;
    integer pending_tile_index;
    integer max_abs_logit_diff;
    longint signed logit_diff;
    logic pending_response;

    logic req_stall_active;
    logic [TILE_INDEX_W-1 : 0] stalled_req_index;
    logic [TOKEN_ID_WIDTH-1 : 0] stalled_req_token_base;
    logic resp_stall_active;
    logic [TILE_WEIGHT_BITS-1 : 0] stalled_resp_weight;
    logic [TILE_SCALE_BITS-1 : 0] stalled_resp_scale;

    lm_head_argmax_stage #(
        .SCAN_ROWS     (SCAN_ROWS),
        .TILE_ROWS     (TILE_ROWS),
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .ACT_WIDTH     (ACT_WIDTH),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .ROW_ACC_WIDTH (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH(TOKEN_ID_WIDTH),
        .TILE_COUNT    (TILE_COUNT),
        .TILE_INDEX_W  (TILE_INDEX_W)
    ) dut (
        .i_clk                (clk),
        .i_rst_n              (rst_n),
        .i_start              (start),
        .i_token_base         (token_base),
        .i_activation_flat    (activation_flat),
        .o_tile_req_valid     (tile_req_valid),
        .i_tile_req_ready     (tile_req_ready),
        .o_tile_index         (tile_index),
        .o_tile_token_base    (tile_token_base),
        .i_tile_valid         (tile_valid),
        .o_tile_ready         (tile_ready),
        .i_tile_weight_flat   (tile_weight_flat),
        .i_tile_scale_flat    (tile_scale_flat),
        .o_busy               (busy),
        .o_done               (done),
        .o_error              (error),
        .o_best_token_id      (best_token_id),
        .o_best_score_q26     (best_score_q26),
        .o_tiles_requested    (tiles_requested),
        .o_tiles_completed    (tiles_completed),
        .o_compute_cycle_count(compute_cycle_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/lm_head_argmax_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, tile_req_valid);
        $dumpvars(0, tile_req_ready);
        $dumpvars(0, tile_index);
        $dumpvars(0, tile_token_base);
        $dumpvars(0, tile_valid);
        $dumpvars(0, tile_ready);
        $dumpvars(0, best_token_id);
        $dumpvars(0, best_score_q26);
        $dumpvars(0, tiles_requested);
        $dumpvars(0, tiles_completed);
        $dumpvars(0, compute_cycle_count);
        $dumpvars(0, dut.current_state);
        $dumpvars(0, dut.lm_head_tile.current_state);
    end

    function automatic logic req_ready_pattern(input integer cycle, input integer request_count);
        begin
            req_ready_pattern =
                (cycle > 16) &&
                ((cycle % 5) != 2) &&
                ((cycle % 17) != 9) &&
                !((request_count >= 20) && (request_count <= 28) && ((cycle % 4) != 0));
        end
    endfunction

    function automatic integer response_latency_pattern(input integer tile_id, input integer cycle);
        begin
            response_latency_pattern = 1 + ((tile_id * 3 + cycle) % 7);
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_activation.hex"}, activation_mem);
            $readmemh({vector_dir, "/", prefix, "_weight_tiles.hex"}, weight_tile_mem);
            $readmemh({vector_dir, "/", prefix, "_scale_tiles.hex"}, scale_tile_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_scan_logits_q26.hex"}, expected_logits_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_best_token.hex"}, expected_best_token_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_best_score_q26.hex"}, expected_best_score_mem);
            $readmemh({vector_dir, "/", prefix, "_scan_base_token.hex"}, scan_base_token_mem);
        end
    endtask

    task pack_inputs;
        begin
            activation_flat = '0;
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                activation_flat[element_index*ACT_WIDTH +: ACT_WIDTH] =
                    activation_mem[element_index];
            end
            token_base = scan_base_token_mem[0];
        end
    endtask

    task pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task check_request_stability;
        begin
            if (req_stall_active == 1'b1) begin
                if ((tile_index !== stalled_req_index) ||
                    (tile_token_base !== stalled_req_token_base)) begin
                    if (print_count < 32) begin
                        $display("FAIL: tile request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_response_stability;
        begin
            if (resp_stall_active == 1'b1) begin
                if ((tile_weight_flat !== stalled_resp_weight) ||
                    (tile_scale_flat !== stalled_resp_scale)) begin
                    if (print_count < 32) begin
                        $display("FAIL: tile response changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_tile_logits;
        integer local_index;
        longint signed observed_logit;
        begin
            for (row_index = 0; row_index < TILE_ROWS; row_index = row_index + 1) begin
                local_index = (dut.tile_index_reg * TILE_ROWS) + row_index;
                observed_logit =
                    $signed(dut.tile_output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH]);
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
                            dut.tile_index_reg,
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

    task check_final_result;
        input integer checked_run;
        begin
            if (error) begin
                $display("FAIL: dut error asserted after run %0d", checked_run);
                mismatch_count = mismatch_count + 1;
            end
            if (best_token_id !== expected_best_token_mem[0]) begin
                $display("FAIL: best token mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, best_token_id, expected_best_token_mem[0]);
                mismatch_count = mismatch_count + 1;
            end
            if (best_score_q26 !== expected_best_score_mem[0]) begin
                $display("FAIL: best score mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, best_score_q26, expected_best_score_mem[0]);
                mismatch_count = mismatch_count + 1;
            end
            if (tiles_requested != TILE_COUNT) begin
                $display("FAIL: tiles_requested mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, tiles_requested, TILE_COUNT);
                mismatch_count = mismatch_count + 1;
            end
            if (tiles_completed != TILE_COUNT) begin
                $display("FAIL: tiles_completed mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, tiles_completed, TILE_COUNT);
                mismatch_count = mismatch_count + 1;
            end
            if (compute_cycle_count < 16000) begin
                $display("FAIL: compute-cycle coverage unexpectedly short after run %0d: %0d",
                         checked_run, compute_cycle_count);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            spurious_start_seen_busy <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1000;
            req_fire_count <= 0;
            resp_fire_count <= 0;
            checked_logit_count <= 0;
            output_update_count <= 0;
            max_abs_logit_diff <= 0;
            req_stall_active <= 1'b0;
            stalled_req_index <= 'd0;
            stalled_req_token_base <= 'd0;
            resp_stall_active <= 1'b0;
            stalled_resp_weight <= 'd0;
            stalled_resp_scale <= 'd0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end

            if (tile_req_valid == 1'b1) begin
                check_request_stability();
                if (tile_req_ready == 1'b1) begin
                    expected_tile_index = req_fire_count % TILE_COUNT;
                    if (tile_index != expected_tile_index[TILE_INDEX_W-1 : 0]) begin
                        $display("FAIL: request index mismatch actual=%0d expected=%0d",
                                 tile_index, expected_tile_index);
                        mismatch_count <= mismatch_count + 1;
                    end
                    if (tile_token_base != (token_base + (expected_tile_index * TILE_ROWS))) begin
                        $display("FAIL: request token base mismatch actual=%0d expected=%0d",
                                 tile_token_base, token_base + (expected_tile_index * TILE_ROWS));
                        mismatch_count <= mismatch_count + 1;
                    end
                    req_fire_count <= req_fire_count + 1;
                    req_stall_active <= 1'b0;
                end
                else begin
                    req_stall_active <= 1'b1;
                    stalled_req_index <= tile_index;
                    stalled_req_token_base <= tile_token_base;
                end
            end
            else begin
                req_stall_active <= 1'b0;
            end

            if (tile_valid == 1'b1) begin
                check_response_stability();
                if (tile_ready == 1'b1) begin
                    resp_fire_count <= resp_fire_count + 1;
                    resp_stall_active <= 1'b0;
                end
                else begin
                    resp_stall_active <= 1'b1;
                    stalled_resp_weight <= tile_weight_flat;
                    stalled_resp_scale <= tile_scale_flat;
                end
            end
            else begin
                resp_stall_active <= 1'b0;
            end

            if (dut.current_state == dut.UPDATE_BEST) begin
                check_tile_logits();
                output_update_count <= output_update_count + 1;
            end

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                last_done_cycle <= cycle_count;
                if ((cycle_count - last_done_cycle) == 1) begin
                    $display("FAIL: done stayed high for adjacent cycles at cycle %0d", cycle_count);
                    mismatch_count <= mismatch_count + 1;
                end
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    run_index,
                    start,
                    busy,
                    done,
                    error,
                    dut.current_state,
                    tile_req_valid,
                    tile_req_ready,
                    tile_req_valid && tile_req_ready,
                    tile_index,
                    tile_token_base,
                    tile_valid,
                    tile_ready,
                    tile_valid && tile_ready,
                    pending_response,
                    response_delay,
                    dut.lm_head_tile.current_state,
                    best_token_id,
                    best_score_q26,
                    tiles_requested,
                    tiles_completed,
                    compute_cycle_count
                );
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            tile_req_ready <= 1'b0;
            tile_valid <= 1'b0;
            tile_weight_flat <= 'd0;
            tile_scale_flat <= 'd0;
            pending_response <= 1'b0;
            response_delay <= 0;
            pending_tile_index <= 0;
        end
        else begin
            tile_req_ready <= req_ready_pattern(cycle_count, req_fire_count);

            if ((tile_req_valid == 1'b1) && (tile_req_ready == 1'b1)) begin
                pending_response <= 1'b1;
                response_delay <= response_latency_pattern(tile_index, cycle_count);
                pending_tile_index <= tile_index;
            end

            if ((tile_valid == 1'b1) && (tile_ready == 1'b1)) begin
                tile_valid <= 1'b0;
                pending_response <= 1'b0;
                tile_weight_flat <= 'd0;
                tile_scale_flat <= 'd0;
            end
            else if ((pending_response == 1'b1) && (tile_valid == 1'b0)) begin
                if (response_delay <= 0) begin
                    tile_valid <= 1'b1;
                    tile_weight_flat <= weight_tile_mem[pending_tile_index];
                    tile_scale_flat <= scale_tile_mem[pending_tile_index];
                end
                else begin
                    response_delay <= response_delay - 1;
                end
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        token_base = 'd0;
        activation_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;
        run_index = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "lm_head_argmax_stage_real";
        tracefile = "FPGA_Project/sim/lm_head_argmax_stage_trace.csv";
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
            "cycle,run,start,busy,done,error,state,req_valid,req_ready,req_fire,tile_index,tile_token_base,resp_valid,resp_ready,resp_fire,pending_response,response_delay,core_state,best_token,best_score,tiles_requested,tiles_completed,compute_cycle_count\n"
        );

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_index = 1;
        pulse_start();

        repeat (128) @(negedge clk);
        if (done != 1'b1) begin
            pulse_start();
        end

        while ((done != 1'b1) && (cycle_count < 80000)) begin
            @(negedge clk);
        end
        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for lm_head_argmax_stage run 1 done");
            $finish(1);
        end
        @(posedge clk);
        #1;
        check_final_result(1);

        run_index = 2;
        repeat (12) @(negedge clk);
        pulse_start();

        while ((done != 1'b1) && (cycle_count < 160000)) begin
            @(negedge clk);
        end
        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for lm_head_argmax_stage run 2 done");
            $finish(1);
        end
        @(posedge clk);
        #1;
        check_final_result(2);

        $fclose(trace_fd);

        $display("lm_head_argmax_stage real Q4 LM-head scan test");
        $display("  expected token         = %0d", expected_best_token_mem[0]);
        $display("  expected score q26     = %0d", expected_best_score_mem[0]);
        $display("  best token             = %0d", best_token_id);
        $display("  best score q26         = %0d", best_score_q26);
        $display("  req fires              = %0d", req_fire_count);
        $display("  resp fires             = %0d", resp_fire_count);
        $display("  tile updates           = %0d", output_update_count);
        $display("  checked logits         = %0d", checked_logit_count);
        $display("  max_abs_logit_diff     = %0d", max_abs_logit_diff);
        $display("  compute cycles         = %0d", compute_cycle_count);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  done seen count        = %0d", done_seen_count);
        $display("  total cycles waited    = %0d", cycle_count);
        $display("  trace                  = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy == 0) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end
        if (req_fire_count != 2*TILE_COUNT) begin
            $display("FAIL: request fire count mismatch actual=%0d expected=%0d",
                     req_fire_count, 2*TILE_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (resp_fire_count != 2*TILE_COUNT) begin
            $display("FAIL: response fire count mismatch actual=%0d expected=%0d",
                     resp_fire_count, 2*TILE_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (output_update_count != 2*TILE_COUNT) begin
            $display("FAIL: tile update count mismatch actual=%0d expected=%0d",
                     output_update_count, 2*TILE_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (checked_logit_count != 2*SCAN_ROWS) begin
            $display("FAIL: checked logit count mismatch actual=%0d expected=%0d",
                     checked_logit_count, 2*SCAN_ROWS);
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted after final done cleanup");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d lm_head_argmax_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: lm_head_argmax_stage matched Q4 LM-head logits and greedy argmax.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
