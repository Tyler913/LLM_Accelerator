`timescale 1ns/1ps
`default_nettype none

// Smoke test for fixed_udiv.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_fixed_udiv.vvp \
//     FPGA_Project/sim/tb_fixed_udiv.sv \
//     FPGA_Project/rtl/lib/math/fixed_udiv.sv
//   vvp FPGA_Project/sim/tb_fixed_udiv.vvp

module tb_fixed_udiv;

    localparam int NUMERATOR_WIDTH   = 48;
    localparam int DENOMINATOR_WIDTH = 24;
    localparam int QUOTIENT_WIDTH    = 24;
    localparam int REMAINDER_WIDTH   = DENOMINATOR_WIDTH;
    localparam int ITERATION_COUNT   = QUOTIENT_WIDTH;
    localparam int ITERATION_W       = (ITERATION_COUNT <= 1) ? 1 : $clog2(ITERATION_COUNT);

    logic                                  clk;
    logic                                  rst_n;
    logic                                  start;
    logic [NUMERATOR_WIDTH-1 : 0]          numerator;
    logic [DENOMINATOR_WIDTH-1 : 0]        denominator;
    logic                                  busy;
    logic                                  done;
    logic                                  divide_by_zero;
    logic [QUOTIENT_WIDTH-1 : 0]           quotient;
    logic [REMAINDER_WIDTH-1 : 0]          remainder;

    integer cycle_count;
    integer mismatch_count;

    fixed_udiv #(
        .NUMERATOR_WIDTH  (NUMERATOR_WIDTH),
        .DENOMINATOR_WIDTH(DENOMINATOR_WIDTH),
        .QUOTIENT_WIDTH   (QUOTIENT_WIDTH),
        .REMAINDER_WIDTH  (REMAINDER_WIDTH),
        .ITERATION_COUNT  (ITERATION_COUNT),
        .ITERATION_W      (ITERATION_W)
    ) dut (
        .i_clk           (clk),
        .i_rst_n         (rst_n),
        .i_start         (start),
        .i_numerator     (numerator),
        .i_denominator   (denominator),
        .o_busy          (busy),
        .o_done          (done),
        .o_divide_by_zero(divide_by_zero),
        .o_quotient      (quotient),
        .o_remainder     (remainder)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("FPGA_Project/wave/fixed_udiv.vcd");
        $dumpvars(0, tb_fixed_udiv);
    end

    task run_case;
        input string case_name;
        input logic [NUMERATOR_WIDTH-1 : 0] case_numerator;
        input logic [DENOMINATOR_WIDTH-1 : 0] case_denominator;
        input logic [QUOTIENT_WIDTH-1 : 0] expected_quotient;
        input logic [REMAINDER_WIDTH-1 : 0] expected_remainder;
        input logic expected_divide_by_zero;
        begin
            numerator = case_numerator;
            denominator = case_denominator;
            cycle_count = 0;

            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < 80)) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done != 1'b1) begin
                $display("FAIL: %s timed out waiting for done", case_name);
                $finish(1);
            end

            #1;
            $display("  %-18s %0d / %0d = q %0d r %0d, dbz %0b, cycles = %0d",
                     case_name, numerator, denominator, quotient, remainder,
                     divide_by_zero, cycle_count);

            if ((quotient !== expected_quotient) ||
                (remainder !== expected_remainder) ||
                (divide_by_zero !== expected_divide_by_zero)) begin
                $display("    expected q %0d r %0d, dbz %0b",
                         expected_quotient, expected_remainder,
                         expected_divide_by_zero);
                mismatch_count = mismatch_count + 1;
            end

            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        numerator = 'd0;
        denominator = 'd0;
        mismatch_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        $display("fixed_udiv smoke test");

        run_case("zero_num", 48'd0, 24'd7, 24'd0, 24'd0, 1'b0);
        run_case("small_100_7", 48'd100, 24'd7, 24'd14, 24'd2, 1'b0);
        run_case("rms_1p0", 48'd67108864, 24'd1024, 24'd65536, 24'd0, 1'b0);
        run_case("rms_2p0", 48'd67108864, 24'd2048, 24'd32768, 24'd0, 1'b0);
        run_case("rms_1p5_floor", 48'd67108864, 24'd1536, 24'd43690, 24'd1024, 1'b0);
        run_case("divide_by_zero", 48'd12345, 24'd0, {QUOTIENT_WIDTH{1'b1}},
                 24'd12345, 1'b1);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d divide case(s) mismatched", mismatch_count);
            $finish(1);
        end

        $display("PASS: fixed_udiv smoke vectors matched.");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
