`timescale 1ns/1ps
`default_nettype none

// Real-vector integration test for qk_norm_rope_stage_128.
//
// Stimulus:
//   Layer 0 last-token Q/K projection outputs from the fixed QMAP/Q4 path,
//   q_norm/k_norm gamma, and current-position RoPE cos/sin.
//
// Run from the repository root:
//
//   conda run -n llm_fpga python \
//     Qwen3-0.6B-Base/python_each_module/22_export_qk_norm_rope_fixed_vectors.py
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_qk_norm_rope_stage_128.vvp \
//     FPGA_Project/sim/tb_qk_norm_rope_stage_128.sv \
//     FPGA_Project/rtl/qk_norm_rope_stage_128.sv \
//     FPGA_Project/rtl/qk_norm_128.sv \
//     FPGA_Project/rtl/rmsnorm_1024.sv \
//     FPGA_Project/rtl/rmsnorm_sum_squares_1024.sv \
//     FPGA_Project/rtl/fixed_sqrt_u64.sv \
//     FPGA_Project/rtl/fixed_udiv.sv \
//     FPGA_Project/rtl/rmsnorm_apply_1024.sv \
//     FPGA_Project/rtl/rope_qk_layer_128.sv
//   vvp FPGA_Project/sim/tb_qk_norm_rope_stage_128.vvp

module tb_qk_norm_rope_stage_128;

    localparam int NUM_Q_HEADS = 16;
    localparam int NUM_K_HEADS = 8;
    localparam int HEAD_DIM    = 128;
    localparam int IN_WIDTH    = 24;
    localparam int GAMMA_WIDTH = 16;
    localparam int TRIG_WIDTH  = 16;
    localparam int OUT_WIDTH   = 24;
    localparam int Q_COUNT     = NUM_Q_HEADS * HEAD_DIM;
    localparam int K_COUNT     = NUM_K_HEADS * HEAD_DIM;

    logic clk;
    logic rst_n;
    logic start;

    logic [Q_COUNT*IN_WIDTH-1 : 0]       q_flat;
    logic [K_COUNT*IN_WIDTH-1 : 0]       k_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]   q_gamma_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]   k_gamma_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    cos_flat;
    logic [HEAD_DIM*TRIG_WIDTH-1 : 0]    sin_flat;

    logic busy;
    logic done;
    logic saturation;
    logic norm_saturation;
    logic rope_saturation;
    logic [Q_COUNT*OUT_WIDTH-1 : 0]      q_norm_flat;
    logic [K_COUNT*OUT_WIDTH-1 : 0]      k_norm_flat;
    logic [Q_COUNT*OUT_WIDTH-1 : 0]      q_rope_flat;
    logic [K_COUNT*OUT_WIDTH-1 : 0]      k_rope_flat;
    logic [31 : 0]                       norm_heads_done;

    logic signed [IN_WIDTH-1 : 0]        q_input_mem [0:Q_COUNT-1];
    logic signed [IN_WIDTH-1 : 0]        k_input_mem [0:K_COUNT-1];
    logic [GAMMA_WIDTH-1 : 0]            q_gamma_mem [0:HEAD_DIM-1];
    logic [GAMMA_WIDTH-1 : 0]            k_gamma_mem [0:HEAD_DIM-1];
    logic signed [TRIG_WIDTH-1 : 0]      cos_mem [0:HEAD_DIM-1];
    logic signed [TRIG_WIDTH-1 : 0]      sin_mem [0:HEAD_DIM-1];
    logic signed [OUT_WIDTH-1 : 0]       q_norm_expected_mem [0:Q_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0]       k_norm_expected_mem [0:K_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0]       q_rope_expected_mem [0:Q_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0]       k_rope_expected_mem [0:K_COUNT-1];
    logic                                expected_norm_saturation_mem [0:0];
    logic                                expected_rope_saturation_mem [0:0];
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
    integer max_abs_diff;
    integer diff_value;

    qk_norm_rope_stage_128 #(
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_K_HEADS(NUM_K_HEADS),
        .HEAD_DIM   (HEAD_DIM),
        .IN_WIDTH   (IN_WIDTH),
        .GAMMA_WIDTH(GAMMA_WIDTH),
        .TRIG_WIDTH (TRIG_WIDTH),
        .OUT_WIDTH  (OUT_WIDTH)
    ) dut (
        .i_clk             (clk),
        .i_rst_n           (rst_n),
        .i_start           (start),
        .i_q_flat          (q_flat),
        .i_k_flat          (k_flat),
        .i_q_gamma_flat    (q_gamma_flat),
        .i_k_gamma_flat    (k_gamma_flat),
        .i_cos_flat        (cos_flat),
        .i_sin_flat        (sin_flat),
        .o_busy            (busy),
        .o_done            (done),
        .o_saturation      (saturation),
        .o_norm_saturation (norm_saturation),
        .o_rope_saturation (rope_saturation),
        .o_q_norm_flat     (q_norm_flat),
        .o_k_norm_flat     (k_norm_flat),
        .o_q_rope_flat     (q_rope_flat),
        .o_k_rope_flat     (k_rope_flat),
        .o_norm_heads_done (norm_heads_done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qk_norm_rope_stage_128.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_qk_norm_rope_stage_128);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_q_input.hex"}, q_input_mem);
            $readmemh({vector_dir, "/", prefix, "_k_input.hex"}, k_input_mem);
            $readmemh({vector_dir, "/", prefix, "_q_gamma.hex"}, q_gamma_mem);
            $readmemh({vector_dir, "/", prefix, "_k_gamma.hex"}, k_gamma_mem);
            $readmemh({vector_dir, "/", prefix, "_cos.hex"}, cos_mem);
            $readmemh({vector_dir, "/", prefix, "_sin.hex"}, sin_mem);
            $readmemh({vector_dir, "/", prefix, "_q_norm_expected.hex"}, q_norm_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_k_norm_expected.hex"}, k_norm_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_q_rope_expected.hex"}, q_rope_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_k_rope_expected.hex"}, k_rope_expected_mem);
            $readmemh({vector_dir, "/", prefix, "_norm_saturation.hex"}, expected_norm_saturation_mem);
            $readmemh({vector_dir, "/", prefix, "_rope_saturation.hex"}, expected_rope_saturation_mem);
            $readmemh({vector_dir, "/", prefix, "_saturation.hex"}, expected_saturation_mem);
        end
    endtask

    task pack_inputs;
        begin
            q_flat = '0;
            k_flat = '0;
            q_gamma_flat = '0;
            k_gamma_flat = '0;
            cos_flat = '0;
            sin_flat = '0;

            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                q_flat[element_index*IN_WIDTH +: IN_WIDTH] = q_input_mem[element_index];
            end

            for (element_index = 0; element_index < K_COUNT; element_index = element_index + 1) begin
                k_flat[element_index*IN_WIDTH +: IN_WIDTH] = k_input_mem[element_index];
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

    task update_diff_and_print;
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

            if ((index < 8) || (observed_value !== expected_value)) begin
                if (print_count < 48) begin
                    $display("  %0s element %0d: output = %0d, expected = %0d",
                             label, index, observed_value, expected_value);
                    print_count = print_count + 1;
                end
            end

            if (observed_value !== expected_value) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_q_norm_outputs;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                observed_value = q_norm_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = q_norm_expected_mem[element_index];
                update_diff_and_print("Q norm", element_index);
            end
            $display("  Q norm max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    task check_k_norm_outputs;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < K_COUNT; element_index = element_index + 1) begin
                observed_value = k_norm_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = k_norm_expected_mem[element_index];
                update_diff_and_print("K norm", element_index);
            end
            $display("  K norm max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    task check_q_rope_outputs;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < Q_COUNT; element_index = element_index + 1) begin
                observed_value = q_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = q_rope_expected_mem[element_index];
                update_diff_and_print("Q RoPE", element_index);
            end
            $display("  Q RoPE max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    task check_k_rope_outputs;
        begin
            max_abs_diff = 0;
            for (element_index = 0; element_index < K_COUNT; element_index = element_index + 1) begin
                observed_value = k_rope_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                expected_value = k_rope_expected_mem[element_index];
                update_diff_and_print("K RoPE", element_index);
            end
            $display("  K RoPE max_abs_diff = %0d", max_abs_diff);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;
        print_count = 0;
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "qk_norm_rope_stage_128_real";
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

        while ((done != 1'b1) && (cycle_count < 30000)) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for qk_norm_rope_stage_128");
            $finish(1);
        end

        #1;
        $display("qk_norm_rope_stage_128 real Layer 0 last-token integration test");
        $display("  cycles waited after start = %0d", cycle_count);
        $display("  norm heads done           = %0d", norm_heads_done);

        if (norm_heads_done != (NUM_Q_HEADS + NUM_K_HEADS)) begin
            $display("FAIL: norm_heads_done mismatch actual=%0d expected=%0d",
                     norm_heads_done, NUM_Q_HEADS + NUM_K_HEADS);
            mismatch_count = mismatch_count + 1;
        end

        check_q_norm_outputs();
        check_k_norm_outputs();
        check_q_rope_outputs();
        check_k_rope_outputs();

        $display("  norm_saturation = %0d, expected = %0d",
                 norm_saturation, expected_norm_saturation_mem[0]);
        $display("  rope_saturation = %0d, expected = %0d",
                 rope_saturation, expected_rope_saturation_mem[0]);
        $display("  saturation      = %0d, expected = %0d",
                 saturation, expected_saturation_mem[0]);

        if (norm_saturation !== expected_norm_saturation_mem[0]) begin
            mismatch_count = mismatch_count + 1;
        end
        if (rope_saturation !== expected_rope_saturation_mem[0]) begin
            mismatch_count = mismatch_count + 1;
        end
        if (saturation !== expected_saturation_mem[0]) begin
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d qk_norm_rope_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qk_norm_rope_stage_128 matched QMAP/Q4 fixed q/k norm + RoPE vectors.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
