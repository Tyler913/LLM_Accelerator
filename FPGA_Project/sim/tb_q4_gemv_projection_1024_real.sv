`timescale 1ns/1ps
`default_nettype none

// Real-vector smoke test for q4_gemv_projection_1024.
//
// Stimulus:
//   q_proj rows 0..15 from qkv_layer0_last_token_q4.npz
//
// This covers four sequential 4-row tiles:
//   tile 0: rows 0..3
//   tile 1: rows 4..7
//   tile 2: rows 8..11
//   tile 3: rows 12..15
//
// Run from the repository root:
//
//   conda run -n llm_fpga python \
//     Qwen3-0.6B-Base/python_each_module/15_export_q4_projection_vectors.py
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_q4_gemv_projection_1024_real.vvp \
//     FPGA_Project/sim/tb_q4_gemv_projection_1024_real.sv \
//     FPGA_Project/rtl/q4_gemv_projection_1024.sv \
//     FPGA_Project/rtl/q4_gemv_tile_1024.sv \
//     FPGA_Project/rtl/q4_gemv_row_1024.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv
//   vvp FPGA_Project/sim/tb_q4_gemv_projection_1024_real.vvp

module tb_q4_gemv_projection_1024_real;

    localparam int OUT_FEATURES  = 16;
    localparam int TILE_ROWS     = 4;
    localparam int INPUT_SIZE    = 1024;
    localparam int GROUP_SIZE    = 64;
    localparam int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH     = 16;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int ROW_ACC_WIDTH = 48;

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*ACT_WIDTH-1:0] activation_flat;
    logic [OUT_FEATURES*INPUT_SIZE*WEIGHT_WIDTH-1:0] weight_packed_flat;
    logic [OUT_FEATURES*GROUP_COUNT*SCALE_WIDTH-1:0] scale_flat;
    logic busy;
    logic done;
    logic [OUT_FEATURES*ROW_ACC_WIDTH-1:0] output_flat;

    logic [ACT_WIDTH-1:0] activation_mem [0:INPUT_SIZE-1];
    logic [WEIGHT_WIDTH-1:0] weight_mem [0:OUT_FEATURES*INPUT_SIZE-1];
    logic [SCALE_WIDTH-1:0] scale_mem [0:OUT_FEATURES*GROUP_COUNT-1];
    logic signed [ROW_ACC_WIDTH-1:0] expected_mem [0:OUT_FEATURES-1];
    logic signed [ROW_ACC_WIDTH-1:0] observed_mem [0:OUT_FEATURES-1];

    string wavefile;
    string vector_dir;
    string prefix;

    integer input_index;
    integer row_index;
    integer group_index;
    integer cycle_count;
    integer mismatch_count;

    q4_gemv_projection_1024 #(
        .OUT_FEATURES(OUT_FEATURES),
        .TILE_ROWS   (TILE_ROWS)
    ) dut (
        .i_clk               (clk),
        .i_rst_n             (rst_n),
        .i_start             (start),
        .i_activation_flat   (activation_flat),
        .i_weight_packed_flat(weight_packed_flat),
        .i_scale_flat        (scale_flat),
        .o_busy              (busy),
        .o_done              (done),
        .o_output_flat       (output_flat)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/q4_gemv_projection_1024_real.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_q4_gemv_projection_1024_real);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_activation.hex"}, activation_mem);
            $readmemh({vector_dir, "/", prefix, "_weight.hex"}, weight_mem);
            $readmemh({vector_dir, "/", prefix, "_scale.hex"}, scale_mem);
            $readmemh({vector_dir, "/", prefix, "_expected.hex"}, expected_mem);
        end
    endtask

    task pack_inputs;
        begin
            activation_flat = '0;
            weight_packed_flat = '0;
            scale_flat = '0;

            for (input_index = 0; input_index < INPUT_SIZE; input_index = input_index + 1) begin
                activation_flat[input_index*ACT_WIDTH +: ACT_WIDTH] =
                    activation_mem[input_index];
            end

            for (row_index = 0; row_index < OUT_FEATURES; row_index = row_index + 1) begin
                for (input_index = 0; input_index < INPUT_SIZE; input_index = input_index + 1) begin
                    weight_packed_flat[(row_index*INPUT_SIZE + input_index)*WEIGHT_WIDTH +: WEIGHT_WIDTH] =
                        weight_mem[row_index*INPUT_SIZE + input_index];
                end

                for (group_index = 0; group_index < GROUP_COUNT; group_index = group_index + 1) begin
                    scale_flat[(row_index*GROUP_COUNT + group_index)*SCALE_WIDTH +: SCALE_WIDTH] =
                        scale_mem[row_index*GROUP_COUNT + group_index];
                end
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "q4_gemv_projection_1024_real";
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

        while ((done != 1'b1) && (cycle_count < 800)) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for done");
            $finish(1);
        end

        #1;
        $display("q4_gemv_projection_1024 real q_proj rows 0..15 test");
        $display("  cycles waited after start = %0d", cycle_count);

        for (row_index = 0; row_index < OUT_FEATURES; row_index = row_index + 1) begin
            observed_mem[row_index] =
                output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH];

            $display("  row %0d: output_q26 = %0d, expected = %0d",
                     row_index, observed_mem[row_index], expected_mem[row_index]);

            if (observed_mem[row_index] !== expected_mem[row_index]) begin
                mismatch_count = mismatch_count + 1;
            end
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d output row(s) mismatched", mismatch_count);
            $finish(1);
        end

        $display("PASS: q4_gemv_projection_1024 real q_proj rows 0..15 matched.");
        $display("Waveform: %s", wavefile);

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
