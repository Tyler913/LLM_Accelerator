`timescale 1ns/1ps
`default_nettype none

// Real-vector integration test for qk_norm_rope_kv_cache_stage.
//
// This checks the connected path:
//
//   Q/K projection outputs -> q_norm/k_norm -> Q/K RoPE
//   K RoPE + V projection output -> KV-cache append write stream
//
// The cache write stream is checked under backpressure and emitted to a CSV
// trace for post-simulation timing/order audit.

module tb_qk_norm_rope_kv_cache_stage;

    localparam int ADDR_WIDTH    = 64;
    localparam int DATA_WIDTH    = 32;
    localparam int NUM_Q_HEADS   = 16;
    localparam int NUM_KV_HEADS  = 8;
    localparam int HEAD_DIM      = 128;
    localparam int MAX_CONTEXT   = 256;
    localparam int IN_WIDTH      = 24;
    localparam int GAMMA_WIDTH   = 16;
    localparam int TRIG_WIDTH    = 16;
    localparam int OUT_WIDTH     = 24;
    localparam int Q_COUNT       = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT      = NUM_KV_HEADS * HEAD_DIM;
    localparam int TOTAL_WRITES  = 2 * KV_COUNT;

    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_BASE_ADDR =
        64'h0000_0004_1410_0000;
    localparam logic [4 : 0] DEFAULT_LAYER_ID = 5'd0;
    localparam logic [7 : 0] DEFAULT_POSITION = 8'd4;

    logic clk;
    logic rst_n;
    logic start;
    logic [ADDR_WIDTH-1 : 0] cache_base_addr;
    logic [4 : 0] layer_id;
    logic [7 : 0] position;

    logic [Q_COUNT*IN_WIDTH-1 : 0]       q_flat;
    logic [KV_COUNT*IN_WIDTH-1 : 0]      k_flat;
    logic [KV_COUNT*IN_WIDTH-1 : 0]      v_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]   q_gamma_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]   k_gamma_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    cos_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    sin_flat;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic norm_saturation;
    logic rope_saturation;
    logic [Q_COUNT*OUT_WIDTH-1 : 0]      q_rope_flat;
    logic [KV_COUNT*OUT_WIDTH-1 : 0]     k_rope_flat;

    logic cache_wr_valid;
    logic cache_wr_ready;
    logic [ADDR_WIDTH-1 : 0] cache_wr_addr;
    logic [DATA_WIDTH-1 : 0] cache_wr_data;
    logic cache_wr_last;
    logic cache_wr_kind;
    logic [2 : 0] cache_wr_head;
    logic [6 : 0] cache_wr_dim;
    logic [31 : 0] cache_write_count;

    logic signed [IN_WIDTH-1 : 0]        q_input_mem [0:Q_COUNT-1];
    logic signed [IN_WIDTH-1 : 0]        k_input_mem [0:KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0]        v_input_mem [0:KV_COUNT-1];
    logic [GAMMA_WIDTH-1 : 0]            q_gamma_mem [0:HEAD_DIM-1];
    logic [GAMMA_WIDTH-1 : 0]            k_gamma_mem [0:HEAD_DIM-1];
    logic signed [TRIG_WIDTH-1 : 0]      cos_mem [0:HEAD_DIM-1];
    logic signed [TRIG_WIDTH-1 : 0]      sin_mem [0:HEAD_DIM-1];
    logic signed [OUT_WIDTH-1 : 0]       q_rope_expected_mem [0:Q_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0]       k_rope_expected_mem [0:KV_COUNT-1];
    logic [ADDR_WIDTH-1 : 0]             expected_addr_mem [0:TOTAL_WRITES-1];
    logic [DATA_WIDTH-1 : 0]             expected_data_mem [0:TOTAL_WRITES-1];
    logic [3 : 0]                        expected_kind_mem [0:TOTAL_WRITES-1];
    logic [7 : 0]                        expected_head_mem [0:TOTAL_WRITES-1];
    logic [7 : 0]                        expected_dim_mem [0:TOTAL_WRITES-1];
    logic                                expected_saturation_mem [0:0];

    string vector_dir;
    string qk_prefix;
    string kv_prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer cycle_count;
    integer accepted_count;
    integer valid_cycle_count;
    integer stall_cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer max_abs_diff;
    integer diff_value;

    logic stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_addr;
    logic [DATA_WIDTH-1 : 0] stalled_data;
    logic stalled_last;
    logic stalled_kind;
    logic [2 : 0] stalled_head;
    logic [6 : 0] stalled_dim;

    logic signed [OUT_WIDTH-1 : 0] observed_value;
    logic signed [OUT_WIDTH-1 : 0] expected_value;

    qk_norm_rope_kv_cache_stage #(
        .ADDR_WIDTH  (ADDR_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .NUM_Q_HEADS (NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS),
        .HEAD_DIM    (HEAD_DIM),
        .MAX_CONTEXT (MAX_CONTEXT),
        .IN_WIDTH    (IN_WIDTH),
        .GAMMA_WIDTH (GAMMA_WIDTH),
        .TRIG_WIDTH  (TRIG_WIDTH),
        .OUT_WIDTH   (OUT_WIDTH)
    ) dut (
        .i_clk              (clk),
        .i_rst_n            (rst_n),
        .i_start            (start),
        .i_cache_base_addr  (cache_base_addr),
        .i_layer_id         (layer_id),
        .i_position         (position),
        .i_q_flat           (q_flat),
        .i_k_flat           (k_flat),
        .i_v_flat           (v_flat),
        .i_q_gamma_flat     (q_gamma_flat),
        .i_k_gamma_flat     (k_gamma_flat),
        .i_cos_flat         (cos_flat),
        .i_sin_flat         (sin_flat),
        .o_busy             (busy),
        .o_done             (done),
        .o_error            (error),
        .o_saturation       (saturation),
        .o_norm_saturation  (norm_saturation),
        .o_rope_saturation  (rope_saturation),
        .o_q_rope_flat      (q_rope_flat),
        .o_k_rope_flat      (k_rope_flat),
        .o_cache_wr_valid   (cache_wr_valid),
        .i_cache_wr_ready   (cache_wr_ready),
        .o_cache_wr_addr    (cache_wr_addr),
        .o_cache_wr_data    (cache_wr_data),
        .o_cache_wr_last    (cache_wr_last),
        .o_cache_wr_kind    (cache_wr_kind),
        .o_cache_wr_head    (cache_wr_head),
        .o_cache_wr_dim     (cache_wr_dim),
        .o_cache_write_count(cache_write_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qk_norm_rope_kv_cache_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_qk_norm_rope_kv_cache_stage);
    end

    function automatic logic ready_pattern(input integer cycle, input integer accepted);
        begin
            ready_pattern =
                (cycle > 12) &&
                ((cycle % 5) != 1) &&
                ((cycle % 13) != 8) &&
                !((accepted >= 1000) && (accepted <= 1035) && ((cycle % 4) != 0)) &&
                !((accepted >= 1700) && (accepted <= 1720) && ((cycle % 3) == 2));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", qk_prefix, "_q_input.hex"}, q_input_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_k_input.hex"}, k_input_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_v_input.hex"}, v_input_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_q_gamma.hex"}, q_gamma_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_k_gamma.hex"}, k_gamma_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_cos.hex"}, cos_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_sin.hex"}, sin_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_q_rope_expected.hex"}, q_rope_expected_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_k_rope_expected.hex"}, k_rope_expected_mem);
            $readmemh({vector_dir, "/", qk_prefix, "_saturation.hex"}, expected_saturation_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_expected_addr.hex"}, expected_addr_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_expected_data.hex"}, expected_data_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_expected_kind.hex"}, expected_kind_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_expected_head.hex"}, expected_head_mem);
            $readmemh({vector_dir, "/", kv_prefix, "_expected_dim.hex"}, expected_dim_mem);
        end
    endtask

    task pack_inputs;
        begin
            q_flat = '0;
            k_flat = '0;
            v_flat = '0;
            q_gamma_flat = '0;
            k_gamma_flat = '0;
            cos_flat = '0;
            sin_flat = '0;

            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                q_flat[element_index*IN_WIDTH +: IN_WIDTH] = q_input_mem[element_index];
            end

            for (element_index = 0; element_index < KV_COUNT; element_index = element_index + 1) begin
                k_flat[element_index*IN_WIDTH +: IN_WIDTH] = k_input_mem[element_index];
                v_flat[element_index*IN_WIDTH +: IN_WIDTH] = v_input_mem[element_index];
            end

            for (element_index = 0; element_index < HEAD_DIM; element_index = element_index + 1) begin
                q_gamma_flat[element_index*GAMMA_WIDTH +: GAMMA_WIDTH] =
                    q_gamma_mem[element_index];
                k_gamma_flat[element_index*GAMMA_WIDTH +: GAMMA_WIDTH] =
                    k_gamma_mem[element_index];
                cos_flat[element_index*TRIG_WIDTH +: TRIG_WIDTH] =
                    cos_mem[element_index];
                sin_flat[element_index*TRIG_WIDTH +: TRIG_WIDTH] =
                    sin_mem[element_index];
            end
        end
    endtask

    task check_cache_stall_stability;
        begin
            if (stall_active == 1'b1) begin
                if ((cache_wr_addr !== stalled_addr) ||
                    (cache_wr_data !== stalled_data) ||
                    (cache_wr_last !== stalled_last) ||
                    (cache_wr_kind !== stalled_kind) ||
                    (cache_wr_head !== stalled_head) ||
                    (cache_wr_dim !== stalled_dim)) begin
                    $display("FAIL: cache write stream changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_current_cache_write;
        begin
            if (accepted_count >= TOTAL_WRITES) begin
                $display("FAIL: extra cache write after expected count at cycle %0d", cycle_count);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((cache_wr_addr !== expected_addr_mem[accepted_count]) ||
                    (cache_wr_data !== expected_data_mem[accepted_count]) ||
                    ({3'd0, cache_wr_kind} !== expected_kind_mem[accepted_count]) ||
                    ({5'd0, cache_wr_head} !== expected_head_mem[accepted_count]) ||
                    ({1'd0, cache_wr_dim} !== expected_dim_mem[accepted_count]) ||
                    (cache_wr_last !== (accepted_count == (TOTAL_WRITES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: cache write %0d mismatch addr=0x%016h exp=0x%016h data=0x%08h exp=0x%08h kind=%0d exp=%0d head=%0d exp=%0d dim=%0d exp=%0d last=%0d",
                            accepted_count,
                            cache_wr_addr,
                            expected_addr_mem[accepted_count],
                            cache_wr_data,
                            expected_data_mem[accepted_count],
                            cache_wr_kind,
                            expected_kind_mem[accepted_count],
                            cache_wr_head,
                            expected_head_mem[accepted_count],
                            cache_wr_dim,
                            expected_dim_mem[accepted_count],
                            cache_wr_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task write_trace_line;
        input integer accepted;
        begin
            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,0x%016h,0x%08h,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    cache_wr_valid,
                    cache_wr_ready,
                    accepted,
                    cache_wr_addr,
                    cache_wr_data,
                    cache_wr_kind,
                    cache_wr_head,
                    cache_wr_dim,
                    cache_wr_last,
                    cache_write_count
                );
            end
        end
    endtask

    task update_diff;
        input string label;
        input integer index;
        begin
            diff_value = observed_value - expected_value;
            if (diff_value < 0) begin
                diff_value = -diff_value;
            end
            if (diff_value > max_abs_diff) begin
                max_abs_diff = diff_value;
            end
            if (observed_value !== expected_value) begin
                if (print_count < 32) begin
                    $display("%0s mismatch index %0d output=%0d expected=%0d",
                             label, index, observed_value, expected_value);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_q_rope;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                observed_value = q_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = q_rope_expected_mem[element_index];
                update_diff("Q RoPE", element_index);
            end
            $display("  Q RoPE max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    task check_k_rope;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < KV_COUNT; element_index = element_index + 1) begin
                observed_value = k_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = k_rope_expected_mem[element_index];
                update_diff("K RoPE", element_index);
            end
            $display("  K RoPE max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        cache_wr_ready = 1'b0;
        cache_base_addr = DEFAULT_BASE_ADDR;
        layer_id = DEFAULT_LAYER_ID;
        position = DEFAULT_POSITION;
        cycle_count = 0;
        accepted_count = 0;
        valid_cycle_count = 0;
        stall_cycle_count = 0;
        mismatch_count = 0;
        print_count = 0;
        stall_active = 1'b0;
        stalled_addr = '0;
        stalled_data = '0;
        stalled_last = 1'b0;
        stalled_kind = 1'b0;
        stalled_head = '0;
        stalled_dim = '0;

        vector_dir = "FPGA_Project/sim/vectors";
        qk_prefix = "qk_norm_rope_stage_128_real";
        kv_prefix = "kv_cache_append_real";
        tracefile = "FPGA_Project/sim/qk_norm_rope_kv_cache_stage_trace.csv";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("qk_prefix=%s", qk_prefix)) begin
        end
        if ($value$plusargs("kv_prefix=%s", kv_prefix)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(trace_fd, "cycle,valid,ready,accepted,addr,data,kind,head,dim,last,write_count_before\n");

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 20000)) begin
            @(negedge clk);
            cache_wr_ready = ready_pattern(cycle_count, accepted_count);

            if (cache_wr_valid == 1'b1) begin
                valid_cycle_count = valid_cycle_count + 1;
                check_cache_stall_stability();

                if (cache_wr_ready == 1'b1) begin
                    check_current_cache_write();
                    write_trace_line(1);
                    accepted_count = accepted_count + 1;
                    stall_active = 1'b0;
                end
                else begin
                    write_trace_line(0);
                    stall_cycle_count = stall_cycle_count + 1;
                    stall_active = 1'b1;
                    stalled_addr = cache_wr_addr;
                    stalled_data = cache_wr_data;
                    stalled_last = cache_wr_last;
                    stalled_kind = cache_wr_kind;
                    stalled_head = cache_wr_head;
                    stalled_dim = cache_wr_dim;
                end
            end
            else begin
                stall_active = 1'b0;
            end

            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for qk_norm_rope_kv_cache_stage done");
            $finish(1);
        end

        #1;
        cache_wr_ready = 1'b0;
        $fclose(trace_fd);

        $display("qk_norm_rope_kv_cache_stage real Layer 0 last-token test");
        $display("  accepted cache writes = %0d", accepted_count);
        $display("  cache valid cycles    = %0d", valid_cycle_count);
        $display("  cache stall cycles    = %0d", stall_cycle_count);
        $display("  cache write_count     = %0d", cache_write_count);
        $display("  cycles waited         = %0d", cycle_count);
        $display("  trace                 = %s", tracefile);

        check_q_rope();
        check_k_rope();

        if (saturation !== expected_saturation_mem[0]) begin
            $display("FAIL: saturation mismatch actual=%0d expected=%0d",
                     saturation, expected_saturation_mem[0]);
            mismatch_count = mismatch_count + 1;
        end

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end

        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end

        if (cache_wr_valid) begin
            $display("FAIL: cache_wr_valid still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end

        if (accepted_count != TOTAL_WRITES) begin
            $display("FAIL: accepted_count mismatch actual=%0d expected=%0d",
                     accepted_count, TOTAL_WRITES);
            mismatch_count = mismatch_count + 1;
        end

        if (cache_write_count != TOTAL_WRITES) begin
            $display("FAIL: cache_write_count mismatch actual=%0d expected=%0d",
                     cache_write_count, TOTAL_WRITES);
            mismatch_count = mismatch_count + 1;
        end

        if (stall_cycle_count < 100) begin
            $display("FAIL: cache backpressure coverage too weak, stall_cycle_count=%0d",
                     stall_cycle_count);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d qk_norm_rope_kv_cache_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qk_norm_rope_kv_cache_stage matched RoPE outputs and exact K/V cache writes.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
