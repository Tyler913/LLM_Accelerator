`timescale 1ns/1ps
`default_nettype none

// Smoke test for fixed_sqrt_u64.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_fixed_sqrt_u64.vvp \
//     FPGA_Project/sim/tb_fixed_sqrt_u64.sv \
//     FPGA_Project/rtl/lib/math/fixed_sqrt_u64.sv
//   vvp FPGA_Project/sim/tb_fixed_sqrt_u64.vvp

module tb_fixed_sqrt_u64;

    localparam int IN_WIDTH        = 64;
    localparam int IN_FRAC         = 20;
    localparam int OUT_WIDTH       = 24;
    localparam int OUT_FRAC        = 10;
    localparam int ITERATION_COUNT = OUT_WIDTH;
    localparam int ITERATION_W     = (ITERATION_COUNT <= 1) ? 1 : $clog2(ITERATION_COUNT);

    logic                         clk;
    logic                         rst_n;
    logic                         start;
    logic [IN_WIDTH-1 : 0]        radicand;
    logic                         busy;
    logic                         done;
    logic [OUT_WIDTH-1 : 0]       root;

    integer cycle_count;
    integer mismatch_count;

    fixed_sqrt_u64 #(
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .OUT_WIDTH      (OUT_WIDTH),
        .OUT_FRAC       (OUT_FRAC),
        .ITERATION_COUNT(ITERATION_COUNT),
        .ITERATION_W    (ITERATION_W)
    ) dut (
        .i_clk     (clk),
        .i_rst_n   (rst_n),
        .i_start   (start),
        .i_radicand(radicand),
        .o_busy    (busy),
        .o_done    (done),
        .o_root    (root)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("FPGA_Project/wave/fixed_sqrt_u64.vcd");
        $dumpvars(0, tb_fixed_sqrt_u64);
    end

    task run_case;
        input string case_name;
        input logic [IN_WIDTH-1 : 0] case_radicand;
        input logic [OUT_WIDTH-1 : 0] expected_root;
        begin
            radicand = case_radicand;
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
            $display("  %-18s radicand = %0d, root = %0d, expected = %0d, cycles = %0d",
                     case_name, radicand, root, expected_root, cycle_count);

            if (root !== expected_root) begin
                mismatch_count = mismatch_count + 1;
            end

            repeat (2) @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        radicand = 'd0;
        mismatch_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        $display("fixed_sqrt_u64 smoke test");

        run_case("zero", 64'd0, 24'd0);
        run_case("one", 64'd1, 24'd1);
        run_case("two_floor", 64'd2, 24'd1);
        run_case("q20_2p25", 64'd2359296, 24'd1536);
        run_case("q20_9p0", 64'd9437184, 24'd3072);
        run_case("exact_12345", 64'd152399025, 24'd12345);
        run_case("floor_12344", 64'd152399024, 24'd12344);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d sqrt case(s) mismatched", mismatch_count);
            $finish(1);
        end

        $display("PASS: fixed_sqrt_u64 smoke vectors matched.");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
