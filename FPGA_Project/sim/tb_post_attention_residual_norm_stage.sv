`timescale 1ns/1ps
`default_nettype none

module tb_post_attention_residual_norm_stage;

    localparam int INPUT_SIZE       = 1024;
    localparam int RESIDUAL_WIDTH   = 24;
    localparam int RESIDUAL_FRAC    = 10;
    localparam int O_PROJ_WIDTH     = 24;
    localparam int O_PROJ_FRAC      = 12;
    localparam int GAMMA_WIDTH      = 16;
    localparam int GAMMA_FRAC       = 7;
    localparam int INV_RMS_WIDTH    = 24;
    localparam int INV_RMS_FRAC     = 16;
    localparam int NORM_OUT_WIDTH   = 24;
    localparam int NORM_OUT_FRAC    = 12;
    localparam int SUM_WIDTH        = 64;
    localparam int SUM_FRAC         = 2 * RESIDUAL_FRAC;
    localparam int MEAN_SHIFT       = $clog2(INPUT_SIZE);
    localparam int RMS_WIDTH        = RESIDUAL_WIDTH;
    localparam int RMS_FRAC         = RESIDUAL_FRAC;
    localparam int DIV_NUM_WIDTH    = 48;
    localparam int DIV_NUM_SHIFT    = RMS_FRAC + INV_RMS_FRAC;
    localparam int EPS_Q20          = 1;

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0] residual_flat;
    logic [INPUT_SIZE*O_PROJ_WIDTH-1 : 0] o_proj_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0] gamma_flat;

    logic busy;
    logic done;
    logic error;
    logic residual_saturation;
    logic norm_saturation;
    logic [31 : 0] residual_count;
    logic [31 : 0] stage_cycle_count;
    logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0] post_attention_hidden_flat;
    logic [INPUT_SIZE*NORM_OUT_WIDTH-1 : 0] post_norm_flat;
    logic [SUM_WIDTH-1 : 0] sum_squares;
    logic [SUM_WIDTH-1 : 0] mean_square;
    logic [INV_RMS_WIDTH-1 : 0] inv_rms;

    logic signed [RESIDUAL_WIDTH-1 : 0] residual_mem [0:INPUT_SIZE-1];
    logic signed [O_PROJ_WIDTH-1 : 0] o_proj_mem [0:INPUT_SIZE-1];
    logic [GAMMA_WIDTH-1 : 0] gamma_mem [0:INPUT_SIZE-1];
    logic signed [RESIDUAL_WIDTH-1 : 0] expected_residual_mem [0:INPUT_SIZE-1];
    logic signed [NORM_OUT_WIDTH-1 : 0] expected_norm_mem [0:INPUT_SIZE-1];
    logic signed [RESIDUAL_WIDTH-1 : 0] observed_residual_mem [0:INPUT_SIZE-1];
    logic signed [NORM_OUT_WIDTH-1 : 0] observed_norm_mem [0:INPUT_SIZE-1];

    logic [SUM_WIDTH-1 : 0] expected_sum_squares_mem [0:0];
    logic [SUM_WIDTH-1 : 0] expected_mean_square_mem [0:0];
    logic [SUM_WIDTH-1 : 0] expected_sqrt_radicand_mem [0:0];
    logic [RMS_WIDTH-1 : 0] expected_rms_mem [0:0];
    logic [INV_RMS_WIDTH-1 : 0] expected_inv_rms_mem [0:0];
    logic expected_residual_saturation_mem [0:0];
    logic expected_norm_saturation_mem [0:0];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer spurious_start_seen_busy;
    integer residual_max_abs_diff;
    integer norm_max_abs_diff;
    integer diff_value;

    post_attention_residual_norm_stage #(
        .INPUT_SIZE      (INPUT_SIZE),
        .RESIDUAL_WIDTH  (RESIDUAL_WIDTH),
        .RESIDUAL_FRAC   (RESIDUAL_FRAC),
        .O_PROJ_WIDTH    (O_PROJ_WIDTH),
        .O_PROJ_FRAC     (O_PROJ_FRAC),
        .GAMMA_WIDTH     (GAMMA_WIDTH),
        .GAMMA_FRAC      (GAMMA_FRAC),
        .INV_RMS_WIDTH   (INV_RMS_WIDTH),
        .INV_RMS_FRAC    (INV_RMS_FRAC),
        .NORM_OUT_WIDTH  (NORM_OUT_WIDTH),
        .NORM_OUT_FRAC   (NORM_OUT_FRAC),
        .SUM_WIDTH       (SUM_WIDTH),
        .SUM_FRAC        (SUM_FRAC),
        .MEAN_SHIFT      (MEAN_SHIFT),
        .RMS_WIDTH       (RMS_WIDTH),
        .RMS_FRAC        (RMS_FRAC),
        .DIV_NUM_WIDTH   (DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT   (DIV_NUM_SHIFT),
        .EPS_Q20         (EPS_Q20)
    ) dut (
        .i_clk                         (clk),
        .i_rst_n                       (rst_n),
        .i_start                       (start),
        .i_residual_flat               (residual_flat),
        .i_o_proj_flat                 (o_proj_flat),
        .i_gamma_flat                  (gamma_flat),
        .o_busy                        (busy),
        .o_done                        (done),
        .o_error                       (error),
        .o_residual_saturation         (residual_saturation),
        .o_norm_saturation             (norm_saturation),
        .o_residual_count              (residual_count),
        .o_cycle_count                 (stage_cycle_count),
        .o_post_attention_hidden_flat  (post_attention_hidden_flat),
        .o_post_norm_flat              (post_norm_flat),
        .o_sum_squares                 (sum_squares),
        .o_mean_square                 (mean_square),
        .o_inv_rms                     (inv_rms)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/post_attention_residual_norm_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, residual_saturation);
        $dumpvars(0, norm_saturation);
        $dumpvars(0, residual_count);
        $dumpvars(0, stage_cycle_count);
        $dumpvars(0, sum_squares);
        $dumpvars(0, mean_square);
        $dumpvars(0, inv_rms);
        $dumpvars(0, dut.current_state);
        $dumpvars(0, dut.residual_busy);
        $dumpvars(0, dut.residual_done);
        $dumpvars(0, dut.norm_busy);
        $dumpvars(0, dut.norm_done);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_residual_input.hex"}, residual_mem);
            $readmemh({vector_dir, "/", prefix, "_o_proj_input.hex"}, o_proj_mem);
            $readmemh({vector_dir, "/", prefix, "_gamma.hex"}, gamma_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_residual.hex"}, expected_residual_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_norm.hex"}, expected_norm_mem);
            $readmemh({vector_dir, "/", prefix, "_sum_squares.hex"}, expected_sum_squares_mem);
            $readmemh({vector_dir, "/", prefix, "_mean_square.hex"}, expected_mean_square_mem);
            $readmemh({vector_dir, "/", prefix, "_sqrt_radicand.hex"}, expected_sqrt_radicand_mem);
            $readmemh({vector_dir, "/", prefix, "_rms.hex"}, expected_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_inv_rms.hex"}, expected_inv_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_residual_saturation.hex"}, expected_residual_saturation_mem);
            $readmemh({vector_dir, "/", prefix, "_norm_saturation.hex"}, expected_norm_saturation_mem);
        end
    endtask

    task pack_inputs;
        begin
            residual_flat = '0;
            o_proj_flat = '0;
            gamma_flat = '0;
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                residual_flat[element_index*RESIDUAL_WIDTH +: RESIDUAL_WIDTH] =
                    residual_mem[element_index];
                o_proj_flat[element_index*O_PROJ_WIDTH +: O_PROJ_WIDTH] =
                    o_proj_mem[element_index];
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
            $display("  %-18s = %0d, expected = %0d", name, observed, expected);
            if (observed !== expected) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end
            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    dut.current_state,
                    dut.residual_busy,
                    dut.residual_done,
                    dut.norm_busy,
                    dut.norm_done,
                    residual_count,
                    stage_cycle_count,
                    residual_saturation,
                    norm_saturation,
                    sum_squares,
                    inv_rms
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        residual_flat = '0;
        o_proj_flat = '0;
        gamma_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;
        spurious_start_seen_busy = 0;
        residual_max_abs_diff = 0;
        norm_max_abs_diff = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "post_attention_residual_norm_stage_real";
        tracefile = "FPGA_Project/sim/post_attention_residual_norm_stage_trace.csv";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,start,busy,done,error,state,residual_busy,residual_done,norm_busy,norm_done,residual_count,stage_cycle_count,residual_saturation,norm_saturation,sum_squares,inv_rms\n"
        );

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        repeat (128) @(negedge clk);
        if (done != 1'b1) begin
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end

        while ((done != 1'b1) && (cycle_count < 10000)) begin
            @(negedge clk);
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for post_attention_residual_norm_stage done");
            $finish(1);
        end

        @(posedge clk);
        #1;
        $fclose(trace_fd);

        $display("post_attention_residual_norm_stage real Layer 0 current-token test");
        $display("  cycles waited          = %0d", cycle_count);
        $display("  stage cycle count      = %0d", stage_cycle_count);
        $display("  residual count         = %0d", residual_count);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  trace                  = %s", tracefile);

        check_scalar("sum_squares", sum_squares, expected_sum_squares_mem[0]);
        check_scalar("mean_square", mean_square, expected_mean_square_mem[0]);
        check_scalar("sqrt_rad", dut.inst_post_attention_rmsnorm.sqrt_radicand, expected_sqrt_radicand_mem[0]);
        check_scalar("rms_q10", {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, dut.inst_post_attention_rmsnorm.rms_q10},
                     {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, expected_rms_mem[0]});
        check_scalar("inv_rms", {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, inv_rms},
                     {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, expected_inv_rms_mem[0]});

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (residual_saturation !== expected_residual_saturation_mem[0]) begin
            $display("FAIL: residual saturation mismatch actual=%0d expected=%0d",
                     residual_saturation, expected_residual_saturation_mem[0]);
            mismatch_count = mismatch_count + 1;
        end
        if (norm_saturation !== expected_norm_saturation_mem[0]) begin
            $display("FAIL: norm saturation mismatch actual=%0d expected=%0d",
                     norm_saturation, expected_norm_saturation_mem[0]);
            mismatch_count = mismatch_count + 1;
        end
        if (residual_count != INPUT_SIZE) begin
            $display("FAIL: residual_count mismatch actual=%0d expected=%0d",
                     residual_count, INPUT_SIZE);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy == 0) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end
        if (stage_cycle_count < 2000) begin
            $display("FAIL: stage-cycle coverage unexpectedly short: %0d", stage_cycle_count);
            mismatch_count = mismatch_count + 1;
        end

        for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
            observed_residual_mem[element_index] =
                post_attention_hidden_flat[element_index*RESIDUAL_WIDTH +: RESIDUAL_WIDTH];
            observed_norm_mem[element_index] =
                post_norm_flat[element_index*NORM_OUT_WIDTH +: NORM_OUT_WIDTH];

            diff_value = observed_residual_mem[element_index] - expected_residual_mem[element_index];
            if (diff_value < 0) begin
                diff_value = -diff_value;
            end
            if (diff_value > residual_max_abs_diff) begin
                residual_max_abs_diff = diff_value;
            end

            if (observed_residual_mem[element_index] !== expected_residual_mem[element_index]) begin
                if (print_count < 32) begin
                    $display(
                        "FAIL: residual element %0d mismatch actual=%0d expected=%0d",
                        element_index,
                        observed_residual_mem[element_index],
                        expected_residual_mem[element_index]
                    );
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end

            diff_value = observed_norm_mem[element_index] - expected_norm_mem[element_index];
            if (diff_value < 0) begin
                diff_value = -diff_value;
            end
            if (diff_value > norm_max_abs_diff) begin
                norm_max_abs_diff = diff_value;
            end

            if (observed_norm_mem[element_index] !== expected_norm_mem[element_index]) begin
                if (print_count < 32) begin
                    $display(
                        "FAIL: norm element %0d mismatch actual=%0d expected=%0d",
                        element_index,
                        observed_norm_mem[element_index],
                        expected_norm_mem[element_index]
                    );
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end

        $display("  residual max_abs_diff  = %0d", residual_max_abs_diff);
        $display("  norm max_abs_diff      = %0d", norm_max_abs_diff);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d post_attention_residual_norm_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: post_attention_residual_norm_stage matched residual and post-norm vectors.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
