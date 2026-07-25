`timescale 1ns/1ps
`default_nettype none

// End-to-end smoke test for rmsnorm_1024.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_rmsnorm_1024.vvp \
//     FPGA_Project/sim/tb_rmsnorm_1024.sv \
//     FPGA_Project/rtl/model/norm/rmsnorm_1024.sv \
//     FPGA_Project/rtl/model/norm/rmsnorm_sum_squares_1024.sv \
//     FPGA_Project/rtl/lib/math/fixed_sqrt_u64.sv \
//     FPGA_Project/rtl/lib/math/fixed_udiv.sv \
//     FPGA_Project/rtl/model/norm/rmsnorm_apply_1024.sv
//   vvp FPGA_Project/sim/tb_rmsnorm_1024.vvp

module tb_rmsnorm_1024;

    localparam int INPUT_SIZE       = 8;
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

    integer element_index;
    integer cycle_count;
    integer mismatch_count;

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
        $dumpfile("FPGA_Project/wave/rmsnorm_1024.vcd");
        $dumpvars(0, tb_rmsnorm_1024);
    end

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

    task run_case;
        input string case_name;
        input logic [SUM_WIDTH-1 : 0] expected_sum_squares;
        input logic [SUM_WIDTH-1 : 0] expected_mean_square;
        input logic [INV_RMS_WIDTH-1 : 0] expected_inv_rms;
        begin
            cycle_count = 0;
            pack_inputs();

            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < 160)) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done != 1'b1) begin
                $display("FAIL: %s timed out waiting for done", case_name);
                $finish(1);
            end

            #1;
            $display("%s", case_name);
            $display("  sum_squares = %0d, expected = %0d",
                     sum_squares, expected_sum_squares);
            $display("  mean_square = %0d, expected = %0d",
                     mean_square, expected_mean_square);
            $display("  inv_rms     = %0d, expected = %0d",
                     inv_rms, expected_inv_rms);
            $display("  cycles waited after start = %0d", cycle_count);

            if (sum_squares !== expected_sum_squares) begin
                mismatch_count = mismatch_count + 1;
            end

            if (mean_square !== expected_mean_square) begin
                mismatch_count = mismatch_count + 1;
            end

            if (inv_rms !== expected_inv_rms) begin
                mismatch_count = mismatch_count + 1;
            end

            if (saturation !== 1'b0) begin
                $display("  unexpected saturation");
                mismatch_count = mismatch_count + 1;
            end

            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                observed_mem[element_index] =
                    output_flat[element_index*OUT_WIDTH +: OUT_WIDTH];
                $display("  element %0d: output = %0d, expected = %0d",
                         element_index, observed_mem[element_index],
                         expected_mem[element_index]);

                if (observed_mem[element_index] !== expected_mem[element_index]) begin
                    mismatch_count = mismatch_count + 1;
                end
            end

            repeat (3) @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        mismatch_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        $display("rmsnorm_1024 end-to-end smoke test");

        for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
            input_mem[element_index] = (element_index[0] == 1'b0) ? 24'sd1024 : -24'sd1024;
            gamma_mem[element_index] = 16'd256;
            expected_mem[element_index] = (element_index[0] == 1'b0) ? 24'sd4096 : -24'sd4096;
        end

        run_case("case +/-1.0", 64'd8388608, 64'd1048576, 24'd65536);

        for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
            input_mem[element_index] = (element_index[0] == 1'b0) ? 24'sd2048 : -24'sd2048;
            gamma_mem[element_index] = 16'd256;
            expected_mem[element_index] = (element_index[0] == 1'b0) ? 24'sd4096 : -24'sd4096;
        end

        run_case("case +/-2.0", 64'd33554432, 64'd4194304, 24'd32768);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d RMSNorm mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: rmsnorm_1024 end-to-end smoke vectors matched.");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
