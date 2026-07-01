`timescale 1ns/1ps
`default_nettype none

module tb_attention_softmax_value_stage;

    localparam int NUM_Q_HEADS      = 16;
    localparam int NUM_KV_HEADS     = 8;
    localparam int HEAD_DIM         = 128;
    localparam int MAX_CONTEXT      = 8;
    localparam int CACHE_LENGTH     = 5;
    localparam int SCORE_WIDTH      = 64;
    localparam int VALUE_WIDTH      = 24;
    localparam int OUT_WIDTH        = 24;
    localparam int EXP_WIDTH        = 24;
    localparam int EXP_LUT_SIZE     = 257;
    localparam int PROB_WIDTH       = 24;
    localparam int Q_HEAD_INDEX_W   = 4;
    localparam int KV_HEAD_INDEX_W  = 3;
    localparam int POSITION_INDEX_W = 3;
    localparam int CACHE_LENGTH_W   = 4;
    localparam int DIM_INDEX_W      = 7;
    localparam int KV_REPEAT        = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam int TOTAL_SCORES     = NUM_Q_HEADS * CACHE_LENGTH;
    localparam int TOTAL_V_REQUESTS = NUM_Q_HEADS * HEAD_DIM * CACHE_LENGTH;
    localparam int TOTAL_OUTPUTS    = NUM_Q_HEADS * HEAD_DIM;
    localparam int V_CACHE_COUNT    = CACHE_LENGTH * NUM_KV_HEADS * HEAD_DIM;

    logic clk;
    logic rst_n;
    logic start;
    logic [CACHE_LENGTH_W-1 : 0] cache_length;
    logic [EXP_LUT_SIZE*EXP_WIDTH-1 : 0] exp_lut_flat;

    logic score_valid;
    logic score_ready;
    logic [Q_HEAD_INDEX_W-1 : 0] score_q_head_in;
    logic [KV_HEAD_INDEX_W-1 : 0] score_kv_head_in;
    logic [POSITION_INDEX_W-1 : 0] score_position_in;
    logic signed [SCORE_WIDTH-1 : 0] score_scaled_in;
    logic score_last_in;

    logic v_req_valid;
    logic v_req_ready;
    logic [KV_HEAD_INDEX_W-1 : 0] v_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] v_req_position;
    logic [DIM_INDEX_W-1 : 0] v_req_dim;
    logic [PROB_WIDTH-1 : 0] v_req_prob;

    logic v_rsp_valid;
    logic v_rsp_ready;
    logic signed [VALUE_WIDTH-1 : 0] v_rsp_data;

    logic out_valid;
    logic out_ready;
    logic [Q_HEAD_INDEX_W-1 : 0] out_q_head;
    logic [DIM_INDEX_W-1 : 0] out_dim;
    logic signed [OUT_WIDTH-1 : 0] out_data;
    logic out_last;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] dut_score_count;
    logic [31 : 0] dut_v_request_count;
    logic [31 : 0] dut_v_response_count;
    logic [31 : 0] dut_output_count;

    logic signed [SCORE_WIDTH-1 : 0] score_input_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] score_q_head_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] score_kv_head_mem [0:TOTAL_SCORES-1];
    logic [7 : 0] score_position_mem [0:TOTAL_SCORES-1];
    logic signed [VALUE_WIDTH-1 : 0] v_cache_mem [0:V_CACHE_COUNT-1];
    logic [EXP_WIDTH-1 : 0] exp_lut_mem [0:EXP_LUT_SIZE-1];
    logic [15 : 0] cache_length_mem [0:0];
    logic [PROB_WIDTH-1 : 0] expected_prob_mem [0:TOTAL_SCORES-1];
    logic [15 : 0] expected_lut_index_mem [0:TOTAL_SCORES-1];
    logic signed [OUT_WIDTH-1 : 0] expected_out_mem [0:TOTAL_OUTPUTS-1];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer cycle_count;
    integer score_accept_count;
    integer v_request_accept_count;
    integer v_response_accept_count;
    integer output_accept_count;
    integer score_gap_cycles;
    integer v_request_stall_cycles;
    integer output_stall_cycles;
    integer response_latency_max;
    integer response_latency_sum;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer expected_q_head;
    integer expected_kv_head;
    integer expected_position;
    integer expected_dim;
    integer expected_prob_index;
    integer v_linear_index;

    logic response_pending;
    integer response_latency_countdown;
    logic signed [VALUE_WIDTH-1 : 0] pending_response_data;

    logic v_req_stall_active;
    logic [KV_HEAD_INDEX_W-1 : 0] stalled_v_req_kv_head;
    logic [POSITION_INDEX_W-1 : 0] stalled_v_req_position;
    logic [DIM_INDEX_W-1 : 0] stalled_v_req_dim;
    logic [PROB_WIDTH-1 : 0] stalled_v_req_prob;

    logic out_stall_active;
    logic [Q_HEAD_INDEX_W-1 : 0] stalled_out_q_head;
    logic [DIM_INDEX_W-1 : 0] stalled_out_dim;
    logic signed [OUT_WIDTH-1 : 0] stalled_out_data;
    logic stalled_out_last;

    longint signed output_diff;
    longint signed max_abs_output_diff;

    attention_softmax_value_stage #(
        .NUM_Q_HEADS      (NUM_Q_HEADS),
        .NUM_KV_HEADS     (NUM_KV_HEADS),
        .HEAD_DIM         (HEAD_DIM),
        .MAX_CONTEXT      (MAX_CONTEXT),
        .SCORE_WIDTH      (SCORE_WIDTH),
        .VALUE_WIDTH      (VALUE_WIDTH),
        .OUT_WIDTH        (OUT_WIDTH),
        .EXP_WIDTH        (EXP_WIDTH),
        .EXP_LUT_SIZE     (EXP_LUT_SIZE),
        .PROB_WIDTH       (PROB_WIDTH),
        .Q_HEAD_INDEX_W   (Q_HEAD_INDEX_W),
        .KV_HEAD_INDEX_W  (KV_HEAD_INDEX_W),
        .POSITION_INDEX_W (POSITION_INDEX_W),
        .CACHE_LENGTH_W   (CACHE_LENGTH_W),
        .DIM_INDEX_W      (DIM_INDEX_W)
    ) dut (
        .i_clk              (clk),
        .i_rst_n            (rst_n),
        .i_start            (start),
        .i_cache_length     (cache_length),
        .i_exp_lut_flat     (exp_lut_flat),
        .i_score_valid      (score_valid),
        .o_score_ready      (score_ready),
        .i_score_q_head     (score_q_head_in),
        .i_score_kv_head    (score_kv_head_in),
        .i_score_position   (score_position_in),
        .i_score_scaled     (score_scaled_in),
        .i_score_last       (score_last_in),
        .o_v_req_valid      (v_req_valid),
        .i_v_req_ready      (v_req_ready),
        .o_v_req_kv_head    (v_req_kv_head),
        .o_v_req_position   (v_req_position),
        .o_v_req_dim        (v_req_dim),
        .o_v_req_prob       (v_req_prob),
        .i_v_rsp_valid      (v_rsp_valid),
        .o_v_rsp_ready      (v_rsp_ready),
        .i_v_rsp_data       (v_rsp_data),
        .o_out_valid        (out_valid),
        .i_out_ready        (out_ready),
        .o_out_q_head       (out_q_head),
        .o_out_dim          (out_dim),
        .o_out_data         (out_data),
        .o_out_last         (out_last),
        .o_busy             (busy),
        .o_done             (done),
        .o_error            (error),
        .o_saturation       (saturation),
        .o_score_count      (dut_score_count),
        .o_v_request_count  (dut_v_request_count),
        .o_v_response_count (dut_v_response_count),
        .o_output_count     (dut_output_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/attention_softmax_value_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_attention_softmax_value_stage);
    end

    function automatic logic score_drive_pattern(input integer cycle, input integer accepted);
        begin
            score_drive_pattern =
                (cycle > 8) &&
                ((cycle % 7) != 2) &&
                ((cycle % 17) != 5) &&
                !((accepted >= 31) && (accepted <= 37) && ((cycle % 3) != 0));
        end
    endfunction

    function automatic logic v_ready_pattern(input integer cycle, input integer accepted);
        begin
            v_ready_pattern =
                (cycle > 20) &&
                ((cycle % 5) != 1) &&
                ((cycle % 23) != 10) &&
                !((accepted >= 5000) && (accepted <= 5030) && ((cycle % 4) != 0));
        end
    endfunction

    function automatic logic out_ready_pattern(input integer cycle, input integer accepted);
        begin
            out_ready_pattern =
                (cycle > 32) &&
                ((cycle % 6) != 3) &&
                ((cycle % 29) != 12) &&
                !((accepted >= 1000) && (accepted <= 1015) && ((cycle % 5) != 0));
        end
    endfunction

    function automatic integer response_latency_pattern(input integer accepted);
        begin
            if ((accepted % 41) == 17) begin
                response_latency_pattern = 8;
            end
            else if ((accepted % 11) == 4) begin
                response_latency_pattern = 5;
            end
            else if ((accepted % 3) == 0) begin
                response_latency_pattern = 2;
            end
            else begin
                response_latency_pattern = 1;
            end
        end
    endfunction

    assign v_req_ready =
        rst_n &&
        v_ready_pattern(cycle_count, v_request_accept_count) &&
        (response_pending == 1'b0) &&
        (v_rsp_valid == 1'b0);

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_score_input.hex"}, score_input_mem);
            $readmemh({vector_dir, "/", prefix, "_score_q_head.hex"}, score_q_head_mem);
            $readmemh({vector_dir, "/", prefix, "_score_kv_head.hex"}, score_kv_head_mem);
            $readmemh({vector_dir, "/", prefix, "_score_position.hex"}, score_position_mem);
            $readmemh({vector_dir, "/", prefix, "_v_cache.hex"}, v_cache_mem);
            $readmemh({vector_dir, "/", prefix, "_exp_lut.hex"}, exp_lut_mem);
            $readmemh({vector_dir, "/", prefix, "_cache_length.hex"}, cache_length_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_prob.hex"}, expected_prob_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_lut_index.hex"}, expected_lut_index_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_out.hex"}, expected_out_mem);
        end
    endtask

    task pack_exp_lut;
        begin
            exp_lut_flat = '0;
            for (element_index = 0; element_index < EXP_LUT_SIZE; element_index = element_index + 1) begin
                exp_lut_flat[element_index*EXP_WIDTH +: EXP_WIDTH] = exp_lut_mem[element_index];
            end
            cache_length = cache_length_mem[0][CACHE_LENGTH_W-1 : 0];
        end
    endtask

    task drive_score_input;
        begin
            if (score_valid && !score_ready) begin
                score_valid = score_valid;
            end
            else if ((score_accept_count < TOTAL_SCORES) &&
                     score_drive_pattern(cycle_count, score_accept_count)) begin
                score_valid = 1'b1;
                score_q_head_in = score_q_head_mem[score_accept_count][Q_HEAD_INDEX_W-1 : 0];
                score_kv_head_in = score_kv_head_mem[score_accept_count][KV_HEAD_INDEX_W-1 : 0];
                score_position_in = score_position_mem[score_accept_count][POSITION_INDEX_W-1 : 0];
                score_scaled_in = score_input_mem[score_accept_count];
                score_last_in = (score_accept_count == (TOTAL_SCORES - 1));
            end
            else begin
                score_valid = 1'b0;
                score_last_in = 1'b0;
            end
        end
    endtask

    task check_v_req_stability;
        begin
            if (v_req_stall_active == 1'b1) begin
                if ((v_req_kv_head !== stalled_v_req_kv_head) ||
                    (v_req_position !== stalled_v_req_position) ||
                    (v_req_dim !== stalled_v_req_dim) ||
                    (v_req_prob !== stalled_v_req_prob)) begin
                    if (print_count < 32) begin
                        $display("FAIL: V request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_out_stability;
        begin
            if (out_stall_active == 1'b1) begin
                if ((out_q_head !== stalled_out_q_head) ||
                    (out_dim !== stalled_out_dim) ||
                    (out_data !== stalled_out_data) ||
                    (out_last !== stalled_out_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: output changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_v_request;
        begin
            if (v_request_accept_count >= TOTAL_V_REQUESTS) begin
                if (print_count < 32) begin
                    $display("FAIL: extra V request at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                expected_q_head = v_request_accept_count / (HEAD_DIM * CACHE_LENGTH);
                expected_dim = (v_request_accept_count / CACHE_LENGTH) % HEAD_DIM;
                expected_position = v_request_accept_count % CACHE_LENGTH;
                expected_kv_head = expected_q_head / KV_REPEAT;
                expected_prob_index = expected_q_head * CACHE_LENGTH + expected_position;

                if ((v_req_kv_head != expected_kv_head) ||
                    (v_req_position != expected_position) ||
                    (v_req_dim != expected_dim) ||
                    (v_req_prob !== expected_prob_mem[expected_prob_index])) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: V request %0d mismatch kv=%0d exp=%0d pos=%0d exp=%0d dim=%0d exp=%0d prob=0x%06h exp=0x%06h",
                            v_request_accept_count,
                            v_req_kv_head,
                            expected_kv_head,
                            v_req_position,
                            expected_position,
                            v_req_dim,
                            expected_dim,
                            v_req_prob,
                            expected_prob_mem[expected_prob_index]
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_output;
        begin
            if (output_accept_count >= TOTAL_OUTPUTS) begin
                if (print_count < 32) begin
                    $display("FAIL: extra output at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                expected_q_head = output_accept_count / HEAD_DIM;
                expected_dim = output_accept_count % HEAD_DIM;

                if ((out_q_head != expected_q_head) ||
                    (out_dim != expected_dim) ||
                    (out_data !== expected_out_mem[output_accept_count]) ||
                    (out_last !== (output_accept_count == (TOTAL_OUTPUTS - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: output %0d mismatch q=%0d exp=%0d dim=%0d exp=%0d data=%0d exp=%0d last=%0d",
                            output_accept_count,
                            out_q_head,
                            expected_q_head,
                            out_dim,
                            expected_dim,
                            out_data,
                            expected_out_mem[output_accept_count],
                            out_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end

                output_diff = out_data - expected_out_mem[output_accept_count];
                if (output_diff < 0) begin
                    output_diff = -output_diff;
                end
                if (output_diff > max_abs_output_diff) begin
                    max_abs_output_diff = output_diff;
                end
            end
        end
    endtask

    always @(negedge clk) begin
        if (rst_n == 1'b0) begin
            score_valid = 1'b0;
            score_q_head_in = 'd0;
            score_kv_head_in = 'd0;
            score_position_in = 'd0;
            score_scaled_in = 'd0;
            score_last_in = 1'b0;
            v_rsp_valid = 1'b0;
            v_rsp_data = 'd0;
            response_pending = 1'b0;
            response_latency_countdown = 0;
            pending_response_data = 'd0;
            out_ready = 1'b0;
        end
        else begin
            drive_score_input();
            if (v_rsp_valid == 1'b1) begin
                v_rsp_valid = 1'b0;
            end
            else if (response_pending == 1'b1) begin
                if (response_latency_countdown == 0) begin
                    v_rsp_valid = 1'b1;
                    v_rsp_data = pending_response_data;
                    response_pending = 1'b0;
                end
                else begin
                    response_latency_countdown = response_latency_countdown - 1;
                end
            end
            out_ready = out_ready_pattern(cycle_count, output_accept_count);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            score_accept_count <= 0;
            v_request_accept_count <= 0;
            v_response_accept_count <= 0;
            output_accept_count <= 0;
            score_gap_cycles <= 0;
            v_request_stall_cycles <= 0;
            output_stall_cycles <= 0;
            response_latency_max <= 0;
            response_latency_sum <= 0;
            v_req_stall_active <= 1'b0;
            stalled_v_req_kv_head <= 'd0;
            stalled_v_req_position <= 'd0;
            stalled_v_req_dim <= 'd0;
            stalled_v_req_prob <= 'd0;
            out_stall_active <= 1'b0;
            stalled_out_q_head <= 'd0;
            stalled_out_dim <= 'd0;
            stalled_out_data <= 'd0;
            stalled_out_last <= 1'b0;
            max_abs_output_diff <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((score_accept_count < TOTAL_SCORES) && (score_valid == 1'b0)) begin
                score_gap_cycles <= score_gap_cycles + 1;
            end

            if ((score_valid == 1'b1) && (score_ready == 1'b1)) begin
                if ((score_q_head_in !== score_q_head_mem[score_accept_count][Q_HEAD_INDEX_W-1 : 0]) ||
                    (score_kv_head_in !== score_kv_head_mem[score_accept_count][KV_HEAD_INDEX_W-1 : 0]) ||
                    (score_position_in !== score_position_mem[score_accept_count][POSITION_INDEX_W-1 : 0]) ||
                    (score_scaled_in !== score_input_mem[score_accept_count]) ||
                    (score_last_in !== (score_accept_count == (TOTAL_SCORES - 1)))) begin
                    if (print_count < 32) begin
                        $display("FAIL: score driver mismatch at accepted index %0d", score_accept_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
                score_accept_count <= score_accept_count + 1;
            end

            if ((v_rsp_valid == 1'b1) && (v_rsp_ready == 1'b1)) begin
                v_response_accept_count <= v_response_accept_count + 1;
            end

            if (v_req_valid == 1'b1) begin
                if (v_req_ready == 1'b0) begin
                    check_v_req_stability();
                    v_request_stall_cycles <= v_request_stall_cycles + 1;
                    v_req_stall_active <= 1'b1;
                    stalled_v_req_kv_head <= v_req_kv_head;
                    stalled_v_req_position <= v_req_position;
                    stalled_v_req_dim <= v_req_dim;
                    stalled_v_req_prob <= v_req_prob;
                end
                else begin
                    v_req_stall_active <= 1'b0;
                end
            end
            else begin
                v_req_stall_active <= 1'b0;
            end

            if ((v_req_valid == 1'b1) && (v_req_ready == 1'b1)) begin
                check_v_request();
                v_linear_index =
                    ((v_req_position * NUM_KV_HEADS + v_req_kv_head) * HEAD_DIM) +
                    v_req_dim;
                pending_response_data <= v_cache_mem[v_linear_index];
                response_pending <= 1'b1;
                response_latency_countdown <= response_latency_pattern(v_request_accept_count);
                response_latency_sum <= response_latency_sum +
                    response_latency_pattern(v_request_accept_count);
                if (response_latency_pattern(v_request_accept_count) > response_latency_max) begin
                    response_latency_max <= response_latency_pattern(v_request_accept_count);
                end
                v_request_accept_count <= v_request_accept_count + 1;
            end

            if (out_valid == 1'b1) begin
                check_out_stability();
                if (out_ready == 1'b1) begin
                    check_output();
                    output_accept_count <= output_accept_count + 1;
                    out_stall_active <= 1'b0;
                end
                else begin
                    output_stall_cycles <= output_stall_cycles + 1;
                    out_stall_active <= 1'b1;
                    stalled_out_q_head <= out_q_head;
                    stalled_out_dim <= out_dim;
                    stalled_out_data <= out_data;
                    stalled_out_last <= out_last;
                end
            end
            else begin
                out_stall_active <= 1'b0;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,0x%06h,%0d,%0d,%0d,0x%06h,%0d,%0d,%0d,%0d,0x%06h,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    score_valid,
                    score_ready,
                    score_valid && score_ready,
                    score_q_head_in,
                    score_kv_head_in,
                    score_position_in,
                    score_scaled_in,
                    v_req_valid,
                    v_req_ready,
                    v_req_valid && v_req_ready,
                    v_req_kv_head,
                    v_req_position,
                    v_req_dim,
                    v_req_prob,
                    v_rsp_valid,
                    v_rsp_ready,
                    v_rsp_valid && v_rsp_ready,
                    v_rsp_data,
                    out_valid,
                    out_ready,
                    out_valid && out_ready,
                    out_q_head,
                    out_dim,
                    out_data,
                    out_last,
                    dut_v_request_count,
                    dut_output_count
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        cache_length = 'd0;
        exp_lut_flat = 'd0;
        score_valid = 1'b0;
        score_q_head_in = 'd0;
        score_kv_head_in = 'd0;
        score_position_in = 'd0;
        score_scaled_in = 'd0;
        score_last_in = 1'b0;
        out_ready = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "attention_softmax_value_stage_real";
        tracefile = "FPGA_Project/sim/attention_softmax_value_stage_trace.csv";
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
            "cycle,score_valid,score_ready,score_accept,score_q_head,score_kv_head,score_position,score_scaled,v_req_valid,v_req_ready,v_req_accept,v_req_kv_head,v_req_position,v_req_dim,v_req_prob,v_rsp_valid,v_rsp_ready,v_rsp_accept,v_rsp_data,out_valid,out_ready,out_accept,out_q_head,out_dim,out_data,out_last,dut_v_request_count,dut_output_count\n"
        );

        load_vectors();
        pack_exp_lut();

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

        while ((done != 1'b1) && (cycle_count < 250000)) begin
            @(negedge clk);
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for attention_softmax_value_stage done");
            $finish(1);
        end

        @(negedge clk);
        score_valid = 1'b0;
        out_ready = 1'b0;
        $fclose(trace_fd);

        $display("attention_softmax_value_stage real Layer 0 current-token test");
        $display("  scores accepted       = %0d", score_accept_count);
        $display("  V requests accepted   = %0d", v_request_accept_count);
        $display("  V responses accepted  = %0d", v_response_accept_count);
        $display("  outputs accepted      = %0d", output_accept_count);
        $display("  score gap cycles      = %0d", score_gap_cycles);
        $display("  V request stall cycles= %0d", v_request_stall_cycles);
        $display("  output stall cycles   = %0d", output_stall_cycles);
        $display("  response latency max  = %0d", response_latency_max);
        $display("  response latency sum  = %0d", response_latency_sum);
        $display("  max_abs_output_diff   = %0d", max_abs_output_diff);
        $display("  cycles waited         = %0d", cycle_count);
        $display("  trace                 = %s", tracefile);

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (saturation) begin
            $display("FAIL: unexpected value output saturation");
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end
        if (score_accept_count != TOTAL_SCORES) begin
            $display("FAIL: score_accept_count mismatch actual=%0d expected=%0d",
                     score_accept_count, TOTAL_SCORES);
            mismatch_count = mismatch_count + 1;
        end
        if (v_request_accept_count != TOTAL_V_REQUESTS) begin
            $display("FAIL: v_request_accept_count mismatch actual=%0d expected=%0d",
                     v_request_accept_count, TOTAL_V_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (v_response_accept_count != TOTAL_V_REQUESTS) begin
            $display("FAIL: v_response_accept_count mismatch actual=%0d expected=%0d",
                     v_response_accept_count, TOTAL_V_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (output_accept_count != TOTAL_OUTPUTS) begin
            $display("FAIL: output_accept_count mismatch actual=%0d expected=%0d",
                     output_accept_count, TOTAL_OUTPUTS);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_score_count != TOTAL_SCORES) begin
            $display("FAIL: dut_score_count mismatch actual=%0d expected=%0d",
                     dut_score_count, TOTAL_SCORES);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_v_request_count != TOTAL_V_REQUESTS) begin
            $display("FAIL: dut_v_request_count mismatch actual=%0d expected=%0d",
                     dut_v_request_count, TOTAL_V_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_v_response_count != TOTAL_V_REQUESTS) begin
            $display("FAIL: dut_v_response_count mismatch actual=%0d expected=%0d",
                     dut_v_response_count, TOTAL_V_REQUESTS);
            mismatch_count = mismatch_count + 1;
        end
        if (dut_output_count != TOTAL_OUTPUTS) begin
            $display("FAIL: dut_output_count mismatch actual=%0d expected=%0d",
                     dut_output_count, TOTAL_OUTPUTS);
            mismatch_count = mismatch_count + 1;
        end
        if (score_gap_cycles < 5) begin
            $display("FAIL: score input gap coverage too weak: %0d", score_gap_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (v_request_stall_cycles < 100) begin
            $display("FAIL: V request backpressure coverage too weak: %0d",
                     v_request_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (output_stall_cycles < 100) begin
            $display("FAIL: output backpressure coverage too weak: %0d",
                     output_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (response_latency_max < 8) begin
            $display("FAIL: response latency coverage too weak: max=%0d",
                     response_latency_max);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d attention_softmax_value_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: attention_softmax_value_stage matched exact probabilities and attn_out under V/output backpressure.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
