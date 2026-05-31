`timescale 1ns/1ps
`default_nettype none

// Real-vector smoke test for rmsnorm_1024.
//
// Stimulus:
//   Layer 0 input_layernorm, last token of the reference prompt.
//
// Run from the repository root:
//
//   conda run -n llm_fpga python \
//     Qwen3-0.6B-Base/python_each_module/17_export_rmsnorm_fixed_vectors.py
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_rmsnorm_1024_real.vvp \
//     FPGA_Project/sim/tb_rmsnorm_1024_real.sv \
//     FPGA_Project/rtl/rmsnorm_1024.sv \
//     FPGA_Project/rtl/rmsnorm_sum_squares_1024.sv \
//     FPGA_Project/rtl/fixed_sqrt_u64.sv \
//     FPGA_Project/rtl/fixed_udiv.sv \
//     FPGA_Project/rtl/rmsnorm_apply_1024.sv
//   vvp FPGA_Project/sim/tb_rmsnorm_1024_real.vvp

module tb_rmsnorm_1024_real;

    localparam int INPUT_SIZE       = 1024;
    localparam int IN_WIDTH         = 24;
    localparam int IN_FRAC          = 10;
    localparam int GAMMA_WIDTH      = 16;
    localparam int GAMMA_FRAC       = 8;
    localparam int INV_RMS_WIDTH    = 24;
    localparam int INV_RMS_FRAC     = 16;
    localparam int OUT_WIDTH        = 24;
    localparam int OUT_FRAC         = 12;
    localparam int SUM_WIDTH        = 64;
    localparam int SUM_FRAC         = 2 * IN_FRAC;
    localparam int MEAN_SHIFT       = $clog2(INPUT_SIZE);
    localparam int RMS_WIDTH        = IN_WIDTH;
    localparam int RMS_FRAC         = IN_FRAC;
    localparam int DIV_NUM_WIDTH    = 48;
    localparam int DIV_NUM_SHIFT    = RMS_FRAC + INV_RMS_FRAC;
    localparam int EPS_Q20          = 1;

    logic                                      clk;
    logic                                      rst_n;
    logic                                      start;
    logic [INPUT_SIZE*IN_WIDTH-1 : 0]          input_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]       gamma_flat;
    logic                                      busy;
    logic                                      done;
    logic                                      saturation;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         output_flat;
    logic [SUM_WIDTH-1 : 0]                    sum_squares;
    logic [SUM_WIDTH-1 : 0]                    mean_square;
    logic [INV_RMS_WIDTH-1 : 0]                inv_rms;

    logic signed [IN_WIDTH-1 : 0]              input_mem [0:INPUT_SIZE-1];
    logic [GAMMA_WIDTH-1 : 0]                  gamma_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0]             expected_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0]             observed_mem [0:INPUT_SIZE-1];

    logic [SUM_WIDTH-1 : 0]                    expected_sum_squares_mem [0:0];
    logic [SUM_WIDTH-1 : 0]                    expected_mean_square_mem [0:0];
    logic [SUM_WIDTH-1 : 0]                    expected_sqrt_radicand_mem [0:0];
    logic [RMS_WIDTH-1 : 0]                    expected_rms_mem [0:0];
    logic [INV_RMS_WIDTH-1 : 0]                expected_inv_rms_mem [0:0];
    logic                                      expected_saturation_mem [0:0];

    string wavefile;
    string vector_dir;
    string prefix;

    integer element_index;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer max_abs_diff;
    integer diff_value;

    rmsnorm_1024 #(
        .INPUT_SIZE   (INPUT_SIZE),
        .IN_WIDTH     (IN_WIDTH),
        .IN_FRAC      (IN_FRAC),
        .GAMMA_WIDTH  (GAMMA_WIDTH),
        .GAMMA_FRAC   (GAMMA_FRAC),
        .INV_RMS_WIDTH(INV_RMS_WIDTH),
        .INV_RMS_FRAC (INV_RMS_FRAC),
        .OUT_WIDTH    (OUT_WIDTH),
        .OUT_FRAC     (OUT_FRAC),
        .SUM_WIDTH    (SUM_WIDTH),
        .SUM_FRAC     (SUM_FRAC),
        .MEAN_SHIFT   (MEAN_SHIFT),
        .RMS_WIDTH    (RMS_WIDTH),
        .RMS_FRAC     (RMS_FRAC),
        .DIV_NUM_WIDTH(DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT(DIV_NUM_SHIFT),
        .EPS_Q20      (EPS_Q20)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_input_flat (input_flat),
        .i_gamma_flat (gamma_flat),
        .o_busy       (busy),
        .o_done       (done),
        .o_saturation (saturation),
        .o_output_flat(output_flat),
        .o_sum_squares(sum_squares),
        .o_mean_square(mean_square),
        .o_inv_rms    (inv_rms)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/rmsnorm_1024_real.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_rmsnorm_1024_real);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_input.hex"}, input_mem);
            $readmemh({vector_dir, "/", prefix, "_gamma.hex"}, gamma_mem);
            $readmemh({vector_dir, "/", prefix, "_expected.hex"}, expected_mem);
            $readmemh({vector_dir, "/", prefix, "_sum_squares.hex"}, expected_sum_squares_mem);
            $readmemh({vector_dir, "/", prefix, "_mean_square.hex"}, expected_mean_square_mem);
            $readmemh({vector_dir, "/", prefix, "_sqrt_radicand.hex"}, expected_sqrt_radicand_mem);
            $readmemh({vector_dir, "/", prefix, "_rms.hex"}, expected_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_inv_rms.hex"}, expected_inv_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_saturation.hex"}, expected_saturation_mem);
        end
    endtask

    task pack_inputs;
        begin
            input_flat = '0;
            gamma_flat = '0;
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                input_flat[element_index*IN_WIDTH +: IN_WIDTH] =
                    input_mem[element_index];
                gamma_flat[element_index*GAMMA_WIDTH +: GAMMA_WIDTH] =
                    gamma_mem[element_index];
            end
        end
    endtask

    task check_scalar;
        input string name;
        input logic [SUM_WIDTH-1 : 0] observed;
        input logic [SUM_WIDTH-1 : 0] expected;
        begin
            $display("  %-14s = %0d, expected = %0d", name, observed, expected);
            if (observed !== expected) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;
        print_count = 0;
        max_abs_diff = 0;
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "rmsnorm_1024_real";
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

        while ((done != 1'b1) && (cycle_count < 3000)) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("rmsnorm_1024 real Layer 0 input_layernorm test");
        $display("  cycles waited after start = %0d", cycle_count);

        check_scalar("sum_squares", sum_squares, expected_sum_squares_mem[0]);
        check_scalar("mean_square", mean_square, expected_mean_square_mem[0]);
        check_scalar("sqrt_rad", dut.sqrt_radicand, expected_sqrt_radicand_mem[0]);
        check_scalar("rms_q10", {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, dut.rms_q10}, {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, expected_rms_mem[0]});
        check_scalar("inv_rms", {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, inv_rms}, {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, expected_inv_rms_mem[0]});

        $display("  saturation     = %0d, expected = %0d",
                 saturation, expected_saturation_mem[0]);
        if (saturation !== expected_saturation_mem[0]) begin
            mismatch_count = mismatch_count + 1;
        end

        for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
            observed_mem[element_index] =
                output_flat[element_index*OUT_WIDTH +: OUT_WIDTH];

            diff_value = observed_mem[element_index] - expected_mem[element_index];
            if (diff_value < 0) begin
                diff_value = -diff_value;
            end
            if (diff_value > max_abs_diff) begin
                max_abs_diff = diff_value;
            end

            if ((element_index < 8) || (observed_mem[element_index] !== expected_mem[element_index])) begin
                if (print_count < 32) begin
                    $display("  element %0d: output = %0d, expected = %0d",
                             element_index, observed_mem[element_index],
                             expected_mem[element_index]);
                    print_count = print_count + 1;
                end
            end

            if (observed_mem[element_index] !== expected_mem[element_index]) begin
                mismatch_count = mismatch_count + 1;
            end
        end

        $display("  output max_abs_diff = %0d", max_abs_diff);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d RMSNorm real-vector mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: rmsnorm_1024 real Layer 0 input_layernorm vector matched.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
