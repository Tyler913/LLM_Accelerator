`timescale 1ns/1ps
`default_nettype none

// Smoke test for rmsnorm_sum_squares_1024.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_rmsnorm_sum_squares_1024.vvp \
//     FPGA_Project/sim/tb_rmsnorm_sum_squares_1024.sv \
//     FPGA_Project/rtl/model/norm/rmsnorm_sum_squares_1024.sv
//   vvp FPGA_Project/sim/tb_rmsnorm_sum_squares_1024.vvp

module tb_rmsnorm_sum_squares_1024;

    localparam int INPUT_SIZE       = 8;
    localparam int IN_WIDTH         = 24;
    localparam int IN_FRAC          = 10;
    localparam int PRODUCT_WIDTH    = IN_WIDTH * 2;
    localparam int SUM_WIDTH        = 64;
    localparam int ELEMENT_INDEX_W  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);

    localparam logic [SUM_WIDTH-1 : 0] EXPECTED_SUM_SQUARES = 64'd31457454;

    logic                                      clk;
    logic                                      rst_n;
    logic                                      start;
    logic [INPUT_SIZE*IN_WIDTH-1 : 0]          input_flat;
    logic                                      busy;
    logic                                      done;
    logic [SUM_WIDTH-1 : 0]                    sum_squares;

    logic signed [IN_WIDTH-1 : 0]              input_mem [0:INPUT_SIZE-1];

    integer element_index;
    integer cycle_count;

    rmsnorm_sum_squares_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .PRODUCT_WIDTH  (PRODUCT_WIDTH),
        .SUM_WIDTH      (SUM_WIDTH),
        .ELEMENT_INDEX_W(ELEMENT_INDEX_W)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_input_flat (input_flat),
        .o_busy       (busy),
        .o_done       (done),
        .o_sum_squares(sum_squares)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("FPGA_Project/wave/rmsnorm_sum_squares_1024.vcd");
        $dumpvars(0, tb_rmsnorm_sum_squares_1024);
    end

    task pack_inputs;
        begin
            input_flat = '0;
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                input_flat[element_index*IN_WIDTH +: IN_WIDTH] =
                    input_mem[element_index];
            end
        end
    endtask

    initial begin
        input_mem[0] = 24'sd1024;
        input_mem[1] = -24'sd2048;
        input_mem[2] = 24'sd3072;
        input_mem[3] = -24'sd4096;
        input_mem[4] = 24'sd5;
        input_mem[5] = -24'sd6;
        input_mem[6] = 24'sd7;
        input_mem[7] = -24'sd8;

        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        pack_inputs();

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 32)) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("rmsnorm_sum_squares_1024 smoke test");
        $display("  sum_squares = %0d, expected = %0d",
                 sum_squares, EXPECTED_SUM_SQUARES);

        if (sum_squares !== EXPECTED_SUM_SQUARES) begin
            $display("FAIL: sum_squares mismatch");
            $finish(1);
        end

        $display("PASS: rmsnorm_sum_squares_1024 smoke vector matched.");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
