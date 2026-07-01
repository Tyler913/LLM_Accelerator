`timescale 1ns/1ps
`default_nettype none

// Real-vector attention-score test.
//
// Coverage intent:
// - K-cache request order is q_head -> position -> dim, with GQA kv_head=q_head/2.
// - The memory model injects request backpressure and variable response latency.
// - Score output uses ready/valid and is checked while stalled.
// - Raw and scaled scores must match the Python fixed-point golden vectors exactly.

module tb_attention_score_stage;

    localparam int NUM_Q_HEADS      = 16;
    localparam int NUM_KV_HEADS     = 8;
    localparam int HEAD_DIM         = 128;
    localparam int MAX_CONTEXT      = 8;
    localparam int CACHE_LENGTH     = 5;
    localparam int IN_WIDTH         = 24;
    localparam int SCORE_WIDTH      = 64;
    localparam int SCALE_WIDTH      = 32;
    localparam int Q_HEAD_INDEX_W   = 4;
    localparam int KV_HEAD_INDEX_W  = 3;
    localparam int POSITION_INDEX_W = 3;
    localparam int CACHE_LENGTH_W   = 4;
    localparam int DIM_INDEX_W      = 7;
    localparam int KV_REPEAT        = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM;
    localparam int K_CACHE_COUNT    = CACHE_LENGTH * NUM_KV_HEADS * HEAD_DIM;
    localparam int TOTAL_SCORES     = NUM_Q_HEADS * CACHE_LENGTH;
    localparam int TOTAL_REQUESTS   = TOTAL_SCORES * HEAD_DIM;

    logic clk;
    logic rst_n;
    logic start;
    logic [CACHE_LENGTH_W-1 : 0] cache_length;
    logic signed [SCALE_WIDTH-1 : 0] score_scale_q0_31;
    logic [Q_COUNT*IN_WIDTH-1 : 0] q_rope_flat;

    logic busy;
    logic done;
    logic error;

    logic k_req_valid;
    logic k_req_ready;
    logic [KV_HEAD_INDEX_W-1 : 0] k_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] k_req_position;
    logic [DIM_INDEX_W-1 : 0] k_req_dim;

    logic k_rsp_valid;
    logic k_rsp_ready;
    logic signed [IN_WIDTH-1 : 0] k_rsp_data;

    logic score_valid;
    logic score_ready;
    logic [Q_HEAD_INDEX_W-1 : 0] score_q_head;
    logic [KV_HEAD_INDEX_W-1 : 0] score_kv_head;
    logic [POSITION_INDEX_W-1 : 0] score_position;
    logic signed [SCORE_WIDTH-1 : 0] score_raw;
    logic signed [SCORE_WIDTH-1 : 0] score_scaled;
    logic score_last;
    logic [31 : 0] dut_k_request_count;
    logic [31 : 0] dut_k_response_count;
    logic [31 : 0] dut_score_count;

    logic signed [IN_WIDTH-1 : 0] q_input_mem [0:Q_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] k_cache_mem [0:K_CACHE_COUNT-1];
    logic signed [SCORE_WIDTH-1 : 0] expected_raw_mem [0:TOTAL_SCORES-1];
    logic signed [SCORE_WIDTH-1 : 0] expected_scaled_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] expected_q_head_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] expected_kv_head_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] expected_position_mem [0:TOTAL_SCORES-1];
    logic [SCALE_WIDTH-1 : 0] score_scale_mem [0:0];
    logic [15 : 0] cache_length_mem [0:0];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer cycle_count;
    integer request_accept_count;
    integer response_accept_count;
    integer score_accept_count;
    integer request_stall_cycle_count;
    integer score_stall_cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer expected_q_head;
    integer expected_position;
    integer expected_dim;
    integer expected_kv_head;
    integer k_linear_index;
    integer response_latency_max;
    integer response_latency_sum;

    logic response_pending;
    integer response_latency_countdown;
    logic signed [IN_WIDTH-1 : 0] pending_response_data;

    logic request_stall_active;
    logic [KV_HEAD_INDEX_W-1 : 0] stalled_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] stalled_req_position;
    logic [DIM_INDEX_W-1 : 0] stalled_req_dim;

    logic score_stall_active;
    logic [Q_HEAD_INDEX_W-1 : 0] stalled_score_q_head;
    logic [KV_HEAD_INDEX_W-1 : 0] stalled_score_kv_head;
    logic [POSITION_INDEX_W-1 : 0] stalled_score_position;
    logic signed [SCORE_WIDTH-1 : 0] stalled_score_raw;
    logic signed [SCORE_WIDTH-1 : 0] stalled_score_scaled;
    logic stalled_score_last;

    longint signed scaled_diff;
    longint signed max_abs_scaled_diff;

    attention_score_stage #(
        .NUM_Q_HEADS     (NUM_Q_HEADS),
        .NUM_KV_HEADS    (NUM_KV_HEADS),
        .HEAD_DIM        (HEAD_DIM),
        .MAX_CONTEXT     (MAX_CONTEXT),
        .IN_WIDTH        (IN_WIDTH),
        .SCORE_WIDTH     (SCORE_WIDTH),
        .SCALE_WIDTH     (SCALE_WIDTH),
        .Q_HEAD_INDEX_W  (Q_HEAD_INDEX_W),
        .KV_HEAD_INDEX_W (KV_HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .CACHE_LENGTH_W  (CACHE_LENGTH_W),
        .DIM_INDEX_W     (DIM_INDEX_W)
    ) dut (
        .i_clk             (clk),
        .i_rst_n           (rst_n),
        .i_start           (start),
        .i_cache_length    (cache_length),
        .i_score_scale_q0_31(score_scale_q0_31),
        .i_q_rope_flat     (q_rope_flat),
        .o_busy            (busy),
        .o_done            (done),
        .o_error           (error),
        .o_k_req_valid     (k_req_valid),
        .i_k_req_ready     (k_req_ready),
        .o_k_req_kv_head   (k_req_kv_head),
        .o_k_req_position  (k_req_position),
        .o_k_req_dim       (k_req_dim),
        .i_k_rsp_valid     (k_rsp_valid),
        .o_k_rsp_ready     (k_rsp_ready),
        .i_k_rsp_data      (k_rsp_data),
        .o_score_valid     (score_valid),
        .i_score_ready     (score_ready),
        .o_score_q_head    (score_q_head),
        .o_score_kv_head   (score_kv_head),
        .o_score_position  (score_position),
        .o_score_raw       (score_raw),
        .o_score_scaled    (score_scaled),
        .o_score_last      (score_last),
        .o_k_request_count (dut_k_request_count),
        .o_k_response_count(dut_k_response_count),
        .o_score_count     (dut_score_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/attention_score_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_attention_score_stage);
    end

    function automatic logic request_ready_pattern(input integer cycle, input integer accepted);
        begin
            request_ready_pattern =
                (cycle > 8) &&
                ((cycle % 4) != 1) &&
                ((cycle % 17) != 9) &&
                !((accepted >= 5000) && (accepted <= 5030) && ((cycle % 5) != 0));
        end
    endfunction

    function automatic logic score_ready_pattern(input integer cycle, input integer accepted);
        begin
            score_ready_pattern =
                (cycle > 16) &&
                ((cycle % 6) != 2) &&
                ((cycle % 19) != 7) &&
                !((accepted >= 40) && (accepted <= 45) && ((cycle % 4) != 0));
        end
    endfunction

    function automatic integer response_latency_pattern(input integer accepted);
        begin
            if ((accepted % 37) == 11) begin
                response_latency_pattern = 7;
            end
            else if ((accepted % 13) == 5) begin
                response_latency_pattern = 4;
            end
            else if ((accepted % 5) == 0) begin
                response_latency_pattern = 2;
            end
            else begin
                response_latency_pattern = 1;
            end
        end
    endfunction

    assign k_req_ready =
        rst_n &&
        request_ready_pattern(cycle_count, request_accept_count) &&
        (response_pending == 1'b0) &&
        (k_rsp_valid == 1'b0);

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_q_input.hex"}, q_input_mem);
            $readmemh({vector_dir, "/", prefix, "_k_cache.hex"}, k_cache_mem);
            $readmemh({vector_dir, "/", prefix, "_score_scale_q0_31.hex"}, score_scale_mem);
            $readmemh({vector_dir, "/", prefix, "_cache_length.hex"}, cache_length_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_raw.hex"}, expected_raw_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_scaled.hex"}, expected_scaled_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_q_head.hex"}, expected_q_head_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_kv_head.hex"}, expected_kv_head_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_position.hex"}, expected_position_mem);
        end
    endtask

    task pack_inputs;
        begin
            q_rope_flat = '0;
            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                q_rope_flat[element_index*IN_WIDTH +: IN_WIDTH] = q_input_mem[element_index];
            end
            score_scale_q0_31 = score_scale_mem[0];
            cache_length = cache_length_mem[0][CACHE_LENGTH_W-1 : 0];
        end
    endtask

    task check_request_stability;
        begin
            if (request_stall_active == 1'b1) begin
                if ((k_req_kv_head !== stalled_req_kv_head) ||
                    (k_req_position !== stalled_req_position) ||
                    (k_req_dim !== stalled_req_dim)) begin
                    if (print_count < 32) begin
                        $display("FAIL: K request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_score_stability;
        begin
            if (score_stall_active == 1'b1) begin
                if ((score_q_head !== stalled_score_q_head) ||
                    (score_kv_head !== stalled_score_kv_head) ||
                    (score_position !== stalled_score_position) ||
                    (score_raw !== stalled_score_raw) ||
                    (score_scaled !== stalled_score_scaled) ||
                    (score_last !== stalled_score_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: score stream changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_request;
        begin
            if (request_accept_count >= TOTAL_REQUESTS) begin
                if (print_count < 32) begin
                    $display("FAIL: extra K request at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                expected_q_head = request_accept_count / (CACHE_LENGTH * HEAD_DIM);
                expected_position = (request_accept_count / HEAD_DIM) % CACHE_LENGTH;
                expected_dim = request_accept_count % HEAD_DIM;
                expected_kv_head = expected_q_head / KV_REPEAT;

                if ((k_req_kv_head != expected_kv_head) ||
                    (k_req_position != expected_position) ||
                    (k_req_dim != expected_dim)) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: K request %0d mismatch kv=%0d exp=%0d pos=%0d exp=%0d dim=%0d exp=%0d",
                            request_accept_count,
                            k_req_kv_head,
                            expected_kv_head,
                            k_req_position,
                            expected_position,
                            k_req_dim,
                            expected_dim
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_score;
        begin
            if (score_accept_count >= TOTAL_SCORES) begin
                if (print_count < 32) begin
                    $display("FAIL: extra score output at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((score_q_head !== expected_q_head_mem[score_accept_count][Q_HEAD_INDEX_W-1 : 0]) ||
                    (score_kv_head !== expected_kv_head_mem[score_accept_count][KV_HEAD_INDEX_W-1 : 0]) ||
                    (score_position !== expected_position_mem[score_accept_count][POSITION_INDEX_W-1 : 0]) ||
                    (score_raw !== expected_raw_mem[score_accept_count]) ||
                    (score_scaled !== expected_scaled_mem[score_accept_count]) ||
                    (score_last !== (score_accept_count == (TOTAL_SCORES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: score %0d mismatch q=%0d exp=%0d kv=%0d exp=%0d pos=%0d exp=%0d raw=%0d exp=%0d scaled=%0d exp=%0d last=%0d",
                            score_accept_count,
                            score_q_head,
                            expected_q_head_mem[score_accept_count],
                            score_kv_head,
                            expected_kv_head_mem[score_accept_count],
                            score_position,
                            expected_position_mem[score_accept_count],
                            score_raw,
                            expected_raw_mem[score_accept_count],
                            score_scaled,
                            expected_scaled_mem[score_accept_count],
                            score_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end

                scaled_diff = score_scaled - expected_scaled_mem[score_accept_count];
                if (scaled_diff < 0) begin
                    scaled_diff = -scaled_diff;
                end
                if (scaled_diff > max_abs_scaled_diff) begin
                    max_abs_scaled_diff = scaled_diff;
                end
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            request_accept_count <= 0;
            response_accept_count <= 0;
            score_accept_count <= 0;
            request_stall_cycle_count <= 0;
            score_stall_cycle_count <= 0;
            response_latency_max <= 0;
            response_latency_sum <= 0;
            response_pending <= 1'b0;
            response_latency_countdown <= 0;
            pending_response_data <= 'd0;
            k_rsp_valid <= 1'b0;
            k_rsp_data <= 'd0;
            request_stall_active <= 1'b0;
            stalled_req_kv_head <= 'd0;
            stalled_req_position <= 'd0;
            stalled_req_dim <= 'd0;
            score_stall_active <= 1'b0;
            stalled_score_q_head <= 'd0;
            stalled_score_kv_head <= 'd0;
            stalled_score_position <= 'd0;
            stalled_score_raw <= 'd0;
            stalled_score_scaled <= 'd0;
            stalled_score_last <= 1'b0;
            max_abs_scaled_diff <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((k_rsp_valid == 1'b1) && (k_rsp_ready == 1'b1)) begin
                k_rsp_valid <= 1'b0;
                response_accept_count <= response_accept_count + 1;
            end

            if ((response_pending == 1'b1) && (k_rsp_valid == 1'b0)) begin
                if (response_latency_countdown == 0) begin
                    k_rsp_valid <= 1'b1;
                    k_rsp_data <= pending_response_data;
                    response_pending <= 1'b0;
                end
                else begin
                    response_latency_countdown <= response_latency_countdown - 1;
                end
            end

            if (k_req_valid == 1'b1) begin
                if (k_req_ready == 1'b0) begin
                    check_request_stability();
                    request_stall_cycle_count <= request_stall_cycle_count + 1;
                    request_stall_active <= 1'b1;
                    stalled_req_kv_head <= k_req_kv_head;
                    stalled_req_position <= k_req_position;
                    stalled_req_dim <= k_req_dim;
                end
                else begin
                    request_stall_active <= 1'b0;
                end
            end
            else begin
                request_stall_active <= 1'b0;
            end

            if ((k_req_valid == 1'b1) && (k_req_ready == 1'b1)) begin
                check_request();
                k_linear_index =
                    ((k_req_position * NUM_KV_HEADS + k_req_kv_head) * HEAD_DIM) +
                    k_req_dim;
                pending_response_data <= k_cache_mem[k_linear_index];
                response_pending <= 1'b1;
                response_latency_countdown <= response_latency_pattern(request_accept_count);
                response_latency_sum <= response_latency_sum +
                    response_latency_pattern(request_accept_count);
                if (response_latency_pattern(request_accept_count) > response_latency_max) begin
                    response_latency_max <= response_latency_pattern(request_accept_count);
                end
                request_accept_count <= request_accept_count + 1;
            end

            if (score_valid == 1'b1) begin
                check_score_stability();
                if (score_ready == 1'b1) begin
                    check_score();
                    score_accept_count <= score_accept_count + 1;
                    score_stall_active <= 1'b0;
                end
                else begin
                    score_stall_cycle_count <= score_stall_cycle_count + 1;
                    score_stall_active <= 1'b1;
                    stalled_score_q_head <= score_q_head;
                    stalled_score_kv_head <= score_kv_head;
                    stalled_score_position <= score_position;
                    stalled_score_raw <= score_raw;
                    stalled_score_scaled <= score_scaled;
                    stalled_score_last <= score_last;
                end
            end
            else begin
                score_stall_active <= 1'b0;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%06h,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,0x%016h,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    k_req_valid,
                    k_req_ready,
                    k_req_valid && k_req_ready,
                    k_req_kv_head,
                    k_req_position,
                    k_req_dim,
                    k_rsp_valid,
                    k_rsp_ready,
                    k_rsp_valid && k_rsp_ready,
                    k_rsp_data,
                    score_valid,
                    score_ready,
                    score_valid && score_ready,
                    score_q_head,
                    score_kv_head,
                    score_position,
                    score_raw,
                    score_scaled,
                    score_last,
                    dut_k_request_count,
                    dut_k_response_count,
                    dut_score_count
                );
            end
        end
    end

    always @(negedge clk) begin
        if (rst_n == 1'b0) begin
            score_ready = 1'b0;
        end
        else begin
            score_ready = score_ready_pattern(cycle_count, score_accept_count);
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        score_ready = 1'b0;
        cache_length = 'd0;
        score_scale_q0_31 = 'd0;
        q_rope_flat = 'd0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "attention_score_stage_real";
        tracefile = "FPGA_Project/sim/attention_score_stage_trace.csv";
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
            "cycle,req_valid,req_ready,req_accept,req_kv_head,req_position,req_dim,rsp_valid,rsp_ready,rsp_accept,rsp_data,score_valid,score_ready,score_accept,score_q_head,score_kv_head,score_position,score_raw,score_scaled,score_last,k_request_count,k_response_count,score_count\n"
        );

        load_vectors();
        pack_inputs();

        if (cache_length != CACHE_LENGTH) begin
            $display("FAIL: cache_length vector mismatch actual=%0d expected=%0d",
                     cache_length, CACHE_LENGTH);
            $finish(1);
        end

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 200000)) begin
            @(negedge clk);
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for attention_score_stage done");
            $finish(1);
        end

        @(negedge clk);
        score_ready = 1'b0;
        $fclose(trace_fd);

        $display("attention_score_stage real Layer 0 current-token test");
        $display("  K requests accepted  = %0d", request_accept_count);
        $display("  K responses accepted = %0d", response_accept_count);
        $display("  scores accepted      = %0d", score_accept_count);
        $display("  request stall cycles = %0d", request_stall_cycle_count);
        $display("  score stall cycles   = %0d", score_stall_cycle_count);
        $display("  response latency max = %0d", response_latency_max);
        $display("  response latency sum = %0d", response_latency_sum);
        $display("  max_abs_scaled_diff  = %0d", max_abs_scaled_diff);
        $display("  cycles waited        = %0d", cycle_count);
        $display("  trace                = %s", tracefile);

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end
        if (request_accept_count != TOTAL_REQUESTS) begin
            $display("FAIL: request count mismatch actual=%0d expected=%0d",
                     request_accept_count, TOTAL_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (response_accept_count != TOTAL_REQUESTS) begin
            $display("FAIL: response count mismatch actual=%0d expected=%0d",
                     response_accept_count, TOTAL_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (score_accept_count != TOTAL_SCORES) begin
            $display("FAIL: score count mismatch actual=%0d expected=%0d",
                     score_accept_count, TOTAL_SCORES);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_k_request_count != TOTAL_REQUESTS) begin
            $display("FAIL: dut_k_request_count mismatch actual=%0d expected=%0d",
                     dut_k_request_count, TOTAL_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_k_response_count != TOTAL_REQUESTS) begin
            $display("FAIL: dut_k_response_count mismatch actual=%0d expected=%0d",
                     dut_k_response_count, TOTAL_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_score_count != TOTAL_SCORES) begin
            $display("FAIL: dut_score_count mismatch actual=%0d expected=%0d",
                     dut_score_count, TOTAL_SCORES);
            mismatch_count = mismatch_count + 1;
        end
        if (request_stall_cycle_count < 100) begin
            $display("FAIL: request backpressure coverage too weak: %0d",
                     request_stall_cycle_count);
            mismatch_count = mismatch_count + 1;
        end
        if (score_stall_cycle_count < 5) begin
            $display("FAIL: score backpressure coverage too weak: %0d",
                     score_stall_cycle_count);
            mismatch_count = mismatch_count + 1;
        end
        if (response_latency_max < 7) begin
            $display("FAIL: response latency coverage too weak: max=%0d",
                     response_latency_max);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d attention_score_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: attention_score_stage matched exact raw/scaled scores under request and score backpressure.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
