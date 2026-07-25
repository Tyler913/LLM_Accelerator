`timescale 1ns/1ps
`default_nettype none

// Real-vector smoke test for rope_qk_layer_128.
//
// Stimulus:
//   Layer 0 last-token Q/K after q_norm/k_norm plus the current-position
//   cos/sin RoPE table.
//
// Run from the repository root:
//
//   conda run -n llm_fpga python \
//     Qwen3-0.6B-Base/python_each_module/18_export_rope_fixed_vectors.py
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_rope_qk_layer_128.vvp \
//     FPGA_Project/sim/tb_rope_qk_layer_128.sv \
//     FPGA_Project/rtl/model/attention/rope_qk_layer_128.sv
//   vvp FPGA_Project/sim/tb_rope_qk_layer_128.vvp

module tb_rope_qk_layer_128;

    localparam int NUM_Q_HEADS = 16;
    localparam int NUM_K_HEADS = 8;
    localparam int HEAD_DIM    = 128;
    localparam int IN_WIDTH    = 24;
    localparam int TRIG_WIDTH  = 16;
    localparam int OUT_WIDTH   = 24;
    localparam int Q_COUNT     = NUM_Q_HEADS * HEAD_DIM;
    localparam int K_COUNT     = NUM_K_HEADS * HEAD_DIM;

    logic clk;
    logic rst_n;
    logic start;
    logic [Q_COUNT*IN_WIDTH-1 : 0]       q_flat;
    logic [K_COUNT*IN_WIDTH-1 : 0]       k_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    cos_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    sin_flat;
    logic busy;
    logic done;
    logic saturation;
    logic [Q_COUNT*OUT_WIDTH-1 : 0]      q_rope_flat;
    logic [K_COUNT*OUT_WIDTH-1 : 0]      k_rope_flat;

    logic signed [IN_WIDTH-1 : 0]        q_input_mem [0:Q_COUNT-1];
    logic signed [IN_WIDTH-1 : 0]        k_input_mem [0:K_COUNT-1];
    logic signed [TRIG_WIDTH-1 : 0]      cos_mem [0:HEAD_DIM-1];
    logic signed [TRIG_WIDTH-1 : 0]      sin_mem [0:HEAD_DIM-1];
    logic signed [OUT_WIDTH-1 : 0]       q_expected_mem [0:Q_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0]       k_expected_mem [0:K_COUNT-1];
    logic                                expected_saturation_mem [0:0];
    logic signed [OUT_WIDTH-1 : 0]       observed_value;
    logic signed [OUT_WIDTH-1 : 0]       expected_value;

    string wavefile;
    string vector_dir;
    string prefix;

    integer element_index;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer q_max_abs_diff;
    integer k_max_abs_diff;
    integer diff_value;

    rope_qk_layer_128 #(
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_K_HEADS(NUM_K_HEADS),
        .HEAD_DIM   (HEAD_DIM),
        .IN_WIDTH   (IN_WIDTH),
        .TRIG_WIDTH (TRIG_WIDTH),
        .OUT_WIDTH  (OUT_WIDTH)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_q_flat     (q_flat),
        .i_k_flat     (k_flat),
        .i_cos_flat   (cos_flat),
        .i_sin_flat   (sin_flat),
        .o_busy       (busy),
        .o_done       (done),
        .o_saturation (saturation),
        .o_q_rope_flat(q_rope_flat),
        .o_k_rope_flat(k_rope_flat)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/rope_qk_layer_128.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_rope_qk_layer_128);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_q_input.hex"}, q_input_mem);
            $readmemh({vector_dir, "/", prefix, "_k_input.hex"}, k_input_mem);
            $readmemh({vector_dir, "/", prefix, "_cos.hex"}, cos_mem);
            $readmemh({vector_dir, "/", prefix, "_sin.hex"}, sin_mem);
            $readmemh({vector_dir, "/", prefix, "_q_expected.hex"}, q_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_k_expected.hex"}, k_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_saturation.hex"}, expected_saturation_mem);
        end
    endtask

    task pack_inputs;
        begin
            q_flat = '0;
            k_flat = '0;
            cos_flat = '0;
            sin_flat = '0;

            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                q_flat[element_index*IN_WIDTH +: IN_WIDTH] = q_input_mem[element_index];
            end

            for (element_index = 0; element_index < K_COUNT; element_index = element_index + 1) begin
                k_flat[element_index*IN_WIDTH +: IN_WIDTH] = k_input_mem[element_index];
            end

            for (element_index = 0; element_index < HEAD_DIM; element_index = element_index + 1) begin
                cos_flat[element_index*TRIG_WIDTH +: TRIG_WIDTH] = cos_mem[element_index];
                sin_flat[element_index*TRIG_WIDTH +: TRIG_WIDTH] = sin_mem[element_index];
            end
        end
    endtask

    task check_q_outputs;
        begin
            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                observed_value = q_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = q_expected_mem[element_index];
                diff_value = observed_value - expected_value;
                if (diff_value < 0) begin
                    diff_value = -diff_value;
                end
                if (diff_value > q_max_abs_diff) begin
                    q_max_abs_diff = diff_value;
                end

                if ((element_index < 8) || (observed_value !== expected_value)) begin
                    if (print_count < 40) begin
                        $display("  Q element %0d: output = %0d, expected = %0d",
                                 element_index, observed_value, expected_value);
                        print_count = print_count + 1;
                    end
                end

                if (observed_value !== expected_value) begin
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_k_outputs;
        begin
            for (element_index = 0; element_index < K_COUNT; element_index = element_index + 1) begin
                observed_value = k_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = k_expected_mem[element_index];
                diff_value = observed_value - expected_value;
                if (diff_value < 0) begin
                    diff_value = -diff_value;
                end
                if (diff_value > k_max_abs_diff) begin
                    k_max_abs_diff = diff_value;
                end

                if ((element_index < 8) || (observed_value !== expected_value)) begin
                    if (print_count < 40) begin
                        $display("  K element %0d: output = %0d, expected = %0d",
                                 element_index, observed_value, expected_value);
                        print_count = print_count + 1;
                    end
                end

                if (observed_value !== expected_value) begin
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;
        print_count = 0;
        q_max_abs_diff = 0;
        k_max_abs_diff = 0;
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "rope_qk_layer_128_real";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end

        load_vectors();
        pack_inputs();

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 4000)) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("rope_qk_layer_128 real Layer 0 last-token RoPE test");
        $display("  cycles waited after start = %0d", cycle_count);

        check_q_outputs();
        check_k_outputs();

        $display("  saturation = %0d, expected = %0d",
                 saturation, expected_saturation_mem[0]);
        if (saturation !== expected_saturation_mem[0]) begin
            mismatch_count = mismatch_count + 1;
        end

        $display("  Q output max_abs_diff = %0d", q_max_abs_diff);
        $display("  K output max_abs_diff = %0d", k_max_abs_diff);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d RoPE real-vector mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: rope_qk_layer_128 real Layer 0 last-token vector matched.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
