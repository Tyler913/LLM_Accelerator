`timescale 1ns/1ps
`default_nettype none

// Smoke test for rmsnorm_apply_1024.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_rmsnorm_apply_1024.vvp \
//     FPGA_Project/sim/tb_rmsnorm_apply_1024.sv \
//     FPGA_Project/rtl/rmsnorm_apply_1024.sv
//   vvp FPGA_Project/sim/tb_rmsnorm_apply_1024.vvp

module tb_rmsnorm_apply_1024;

    localparam int INPUT_SIZE       = 8;
    localparam int IN_WIDTH         = 24;
    localparam int IN_FRAC          = 10;
    localparam int INV_RMS_WIDTH    = 24;
    localparam int INV_RMS_FRAC     = 16;
    localparam int GAMMA_WIDTH      = 16;
    localparam int GAMMA_FRAC       = 8;
    localparam int OUT_WIDTH        = 24;
    localparam int OUT_FRAC         = 12;
    localparam int PRODUCT1_WIDTH   = IN_WIDTH + INV_RMS_WIDTH;
    localparam int PRODUCT2_WIDTH   = PRODUCT1_WIDTH + GAMMA_WIDTH;
    localparam int OUTPUT_SHIFT     = IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC;
    localparam int ELEMENT_INDEX_W  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);

    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {OUT_WIDTH-1{1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {OUT_WIDTH-1{1'b0}}};

    logic                                      clk;
    logic                                      rst_n;
    logic                                      start;
    logic [INPUT_SIZE*IN_WIDTH-1 : 0]          input_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]       gamma_flat;
    logic [INV_RMS_WIDTH-1 : 0]                inv_rms;
    logic                                      busy;
    logic                                      done;
    logic                                      saturation;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         output_flat;

    logic signed [IN_WIDTH-1 : 0]              input_mem [0:INPUT_SIZE-1];
    logic [GAMMA_WIDTH-1 : 0]                  gamma_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0]             expected_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0]             observed_mem [0:INPUT_SIZE-1];

    integer element_index;
    integer cycle_count;
    integer mismatch_count;

    rmsnorm_apply_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .INV_RMS_WIDTH  (INV_RMS_WIDTH),
        .INV_RMS_FRAC   (INV_RMS_FRAC),
        .GAMMA_WIDTH    (GAMMA_WIDTH),
        .GAMMA_FRAC     (GAMMA_FRAC),
        .OUT_WIDTH      (OUT_WIDTH),
        .OUT_FRAC       (OUT_FRAC),
        .PRODUCT1_WIDTH (PRODUCT1_WIDTH),
        .PRODUCT2_WIDTH (PRODUCT2_WIDTH),
        .OUTPUT_SHIFT   (OUTPUT_SHIFT),
        .ELEMENT_INDEX_W(ELEMENT_INDEX_W)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_input_flat (input_flat),
        .i_gamma_flat (gamma_flat),
        .i_inv_rms    (inv_rms),
        .o_busy       (busy),
        .o_done       (done),
        .o_saturation (saturation),
        .o_output_flat(output_flat)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("FPGA_Project/wave/rmsnorm_apply_1024.vcd");
        $dumpvars(0, tb_rmsnorm_apply_1024);
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

    task wait_done;
        begin
            cycle_count = 0;
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
        end
    endtask

    task check_outputs;
        input logic expected_saturation;
        begin
            #1;
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

            if (saturation !== expected_saturation) begin
                $display("  saturation = %0b, expected = %0b",
                         saturation, expected_saturation);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        inv_rms = 'd0;
        mismatch_count = 0;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst_n = 1'b1;

        $display("rmsnorm_apply_1024 nominal smoke test");

        input_mem[0] = 24'sd1024;
        input_mem[1] = -24'sd2048;
        input_mem[2] = 24'sd512;
        input_mem[3] = -24'sd256;
        input_mem[4] = 24'sd3072;
        input_mem[5] = 24'sd0;
        input_mem[6] = -24'sd1536;
        input_mem[7] = 24'sd2304;

        gamma_mem[0] = 16'd256;
        gamma_mem[1] = 16'd256;
        gamma_mem[2] = 16'd512;
        gamma_mem[3] = 16'd128;
        gamma_mem[4] = 16'd256;
        gamma_mem[5] = 16'd256;
        gamma_mem[6] = 16'd512;
        gamma_mem[7] = 16'd128;

        expected_mem[0] = 24'sd2048;
        expected_mem[1] = -24'sd4096;
        expected_mem[2] = 24'sd2048;
        expected_mem[3] = -24'sd256;
        expected_mem[4] = 24'sd6144;
        expected_mem[5] = 24'sd0;
        expected_mem[6] = -24'sd6144;
        expected_mem[7] = 24'sd2304;

        inv_rms = 24'd32768;
        pack_inputs();
        wait_done();
        check_outputs(1'b0);

        repeat (3) @(posedge clk);

        $display("rmsnorm_apply_1024 saturation smoke test");

        for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
            input_mem[element_index] = 24'sd0;
            gamma_mem[element_index] = 16'd0;
            expected_mem[element_index] = 24'sd0;
        end

        input_mem[0] = 24'sh7fffff;
        input_mem[1] = -24'sd8388608;
        gamma_mem[0] = 16'hffff;
        gamma_mem[1] = 16'hffff;
        expected_mem[0] = OUT_MAX;
        expected_mem[1] = OUT_MIN;

        inv_rms = 24'hffffff;
        pack_inputs();
        wait_done();
        check_outputs(1'b1);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d apply mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: rmsnorm_apply_1024 smoke vectors matched.");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
