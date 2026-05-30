`timescale 1ns/1ps
`default_nettype none

// Simple smoke test for q4_dot_product_64.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_q4_dot_product_64.vvp \
//     FPGA_Project/sim/tb_q4_dot_product_64.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv
//   vvp FPGA_Project/sim/tb_q4_dot_product_64.vvp
//
// The default waveform output is:
//
//   FPGA_Project/wave/q4_dot_product_64.vcd
//
// Override it with:
//
//   vvp FPGA_Project/sim/tb_q4_dot_product_64.vvp +wavefile=<path>

module tb_q4_dot_product_64;

    localparam int GROUP_SIZE    = 64;
    localparam int ACT_WIDTH     = 16;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int PARTIAL_WIDTH = 26;
    localparam int SCALED_WIDTH  = 42;

    localparam logic [GROUP_SIZE*ACT_WIDTH-1:0] TEST_ACTIVATION_FLAT =
        1024'h01f40103ff9df636ff5c009402e9ff06f191fde100a3fd120162ffe70074ff07fe9e04b3066dfdc9fd1cfeebfb3a022dffa3fe230450fca2e91dfcda014806e3043606ee01bd02d4df3e026300de032704c5ffdfffff017afc5afef6fa10011e0669edf1ecdd06e50391f9eb05dffc81ea89fb7709f6fcfcebcbe96b0b5c01b8;

    localparam logic [GROUP_SIZE*WEIGHT_WIDTH-1:0] TEST_WEIGHT_PACKED =
        256'h3913f0e3d30d1131efde3002fd1eff1e3503fc4f1012f1ed0fe150ee30023ef1;

    localparam logic [SCALE_WIDTH-1:0] TEST_SCALE_Q2_14 = 16'd122;

    localparam logic signed [PARTIAL_WIDTH-1:0] EXPECTED_PARTIAL_SUM =
        26'sd24751;

    localparam logic signed [SCALED_WIDTH-1:0] EXPECTED_SCALED_SUM_Q26 =
        42'sd3019622;

    logic clk;
    logic rst_n;
    logic start;
    logic [GROUP_SIZE*ACT_WIDTH-1:0] activation_flat;
    logic [GROUP_SIZE*WEIGHT_WIDTH-1:0] weight_packed;
    logic [SCALE_WIDTH-1:0] scale_q2_14;
    logic busy;
    logic done;
    logic signed [PARTIAL_WIDTH-1:0] partial_sum;
    logic signed [SCALED_WIDTH-1:0] scaled_sum_q26;

    string wavefile;
    int cycle_count;

    q4_dot_product_64 dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(weight_packed),
        .i_scale_q2_14(scale_q2_14),
        .o_busy(busy),
        .o_done(done),
        .o_partial_sum(partial_sum),
        .o_scaled_sum_q26(scaled_sum_q26)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/q4_dot_product_64.vcd";
        void'($value$plusargs("wavefile=%s", wavefile));
        $dumpfile(wavefile);
        $dumpvars(0, tb_q4_dot_product_64);
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        activation_flat = TEST_ACTIVATION_FLAT;
        weight_packed = TEST_WEIGHT_PACKED;
        scale_q2_14 = TEST_SCALE_Q2_14;
        cycle_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 100)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("q4_dot_product_64 smoke test");
        $display("  partial_sum    = %0d, expected = %0d",
                 partial_sum, EXPECTED_PARTIAL_SUM);
        $display("  scaled_sum_q26 = %0d, expected = %0d",
                 scaled_sum_q26, EXPECTED_SCALED_SUM_Q26);

        if (partial_sum !== EXPECTED_PARTIAL_SUM) begin
            $display("FAIL: partial_sum mismatch");
            $finish(1);
        end

        if (scaled_sum_q26 !== EXPECTED_SCALED_SUM_Q26) begin
            $display("FAIL: scaled_sum_q26 mismatch");
            $finish(1);
        end

        $display("PASS: q4_dot_product_64 dot64 smoke vector matched.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
