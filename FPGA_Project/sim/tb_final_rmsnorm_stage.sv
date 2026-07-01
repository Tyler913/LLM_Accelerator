`timescale 1ns/1ps
`default_nettype none

module tb_final_rmsnorm_stage;

    localparam int INPUT_SIZE       = 1024;
    localparam int IN_WIDTH         = 24;
    localparam int IN_FRAC          = 10;
    localparam int GAMMA_WIDTH      = 16;
    localparam int GAMMA_FRAC       = 7;
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

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*IN_WIDTH-1 : 0] input_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0] gamma_flat;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] cycle_count_dut;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0] output_flat;
    logic [SUM_WIDTH-1 : 0] sum_squares;
    logic [SUM_WIDTH-1 : 0] mean_square;
    logic [INV_RMS_WIDTH-1 : 0] inv_rms;

    logic signed [IN_WIDTH-1 : 0] input_mem [0:INPUT_SIZE-1];
    logic signed [GAMMA_WIDTH-1 : 0] gamma_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0] expected_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0] observed_mem [0:INPUT_SIZE-1];

    logic [SUM_WIDTH-1 : 0] expected_sum_squares_mem [0:0];
    logic [SUM_WIDTH-1 : 0] expected_mean_square_mem [0:0];
    logic [SUM_WIDTH-1 : 0] expected_sqrt_radicand_mem [0:0];
    logic [RMS_WIDTH-1 : 0] expected_rms_mem [0:0];
    logic [INV_RMS_WIDTH-1 : 0] expected_inv_rms_mem [0:0];
    logic expected_saturation_mem [0:0];

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
    integer done_seen_count;
    integer run_index;
    integer last_done_cycle;
    integer max_abs_diff;
    integer diff_value;

    final_rmsnorm_stage #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .GAMMA_WIDTH    (GAMMA_WIDTH),
        .GAMMA_FRAC     (GAMMA_FRAC),
        .INV_RMS_WIDTH  (INV_RMS_WIDTH),
        .INV_RMS_FRAC   (INV_RMS_FRAC),
        .OUT_WIDTH      (OUT_WIDTH),
        .OUT_FRAC       (OUT_FRAC),
        .SUM_WIDTH      (SUM_WIDTH),
        .SUM_FRAC       (SUM_FRAC),
        .MEAN_SHIFT     (MEAN_SHIFT),
        .RMS_WIDTH      (RMS_WIDTH),
        .RMS_FRAC       (RMS_FRAC),
        .DIV_NUM_WIDTH  (DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT  (DIV_NUM_SHIFT),
        .EPS_Q20        (EPS_Q20)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_input_flat (input_flat),
        .i_gamma_flat (gamma_flat),
        .o_busy       (busy),
        .o_done       (done),
        .o_error      (error),
        .o_saturation (saturation),
        .o_cycle_count(cycle_count_dut),
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
        wavefile = "FPGA_Project/wave/final_rmsnorm_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, saturation);
        $dumpvars(0, cycle_count_dut);
        $dumpvars(0, sum_squares);
        $dumpvars(0, mean_square);
        $dumpvars(0, inv_rms);
        $dumpvars(0, dut.current_state);
        $dumpvars(0, dut.norm_busy);
        $dumpvars(0, dut.norm_done);
        $dumpvars(0, dut.inst_final_rmsnorm.current_state);
        $dumpvars(0, dut.inst_final_rmsnorm.inst_rmsnorm_sum_squares_1024.element_index);
        $dumpvars(0, dut.inst_final_rmsnorm.inst_rmsnorm_apply_1024.element_index);
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

    task pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
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

    task check_outputs;
        input integer checked_run;
        begin
            max_abs_diff = 0;

            if (error) begin
                $display("FAIL: dut error asserted after run %0d", checked_run);
                mismatch_count = mismatch_count + 1;
            end
            if (saturation !== expected_saturation_mem[0]) begin
                $display("FAIL: saturation mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, saturation, expected_saturation_mem[0]);
                mismatch_count = mismatch_count + 1;
            end
            if (cycle_count_dut < 2000) begin
                $display("FAIL: cycle-count coverage unexpectedly short after run %0d: %0d",
                         checked_run, cycle_count_dut);
                mismatch_count = mismatch_count + 1;
            end

            check_scalar("sum_squares", sum_squares, expected_sum_squares_mem[0]);
            check_scalar("mean_square", mean_square, expected_mean_square_mem[0]);
            check_scalar("sqrt_rad", dut.inst_final_rmsnorm.sqrt_radicand, expected_sqrt_radicand_mem[0]);
            check_scalar("rms_q10", {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, dut.inst_final_rmsnorm.rms_q10},
                         {{(SUM_WIDTH-RMS_WIDTH){1'b0}}, expected_rms_mem[0]});
            check_scalar("inv_rms", {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, inv_rms},
                         {{(SUM_WIDTH-INV_RMS_WIDTH){1'b0}}, expected_inv_rms_mem[0]});

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

                if (observed_mem[element_index] !== expected_mem[element_index]) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: final RMSNorm run %0d element %0d mismatch actual=%0d expected=%0d",
                            checked_run,
                            element_index,
                            observed_mem[element_index],
                            expected_mem[element_index]
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end

            $display("  run %0d output max_abs_diff = %0d", checked_run, max_abs_diff);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            spurious_start_seen_busy <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1000;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                last_done_cycle <= cycle_count;
                if ((cycle_count - last_done_cycle) == 1) begin
                    $display("FAIL: done stayed high for adjacent cycles at cycle %0d", cycle_count);
                    mismatch_count <= mismatch_count + 1;
                end
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%03h,0x%03h\n",
                    cycle_count,
                    run_index,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    dut.current_state,
                    dut.inst_final_rmsnorm.current_state,
                    dut.inst_final_rmsnorm.sum_busy,
                    dut.inst_final_rmsnorm.sum_done,
                    dut.inst_final_rmsnorm.sqrt_busy,
                    dut.inst_final_rmsnorm.sqrt_done,
                    dut.inst_final_rmsnorm.div_busy,
                    dut.inst_final_rmsnorm.div_done,
                    dut.inst_final_rmsnorm.apply_busy,
                    dut.inst_final_rmsnorm.apply_done,
                    dut.inst_final_rmsnorm.inst_rmsnorm_sum_squares_1024.element_index,
                    dut.inst_final_rmsnorm.inst_rmsnorm_apply_1024.element_index
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        input_flat = '0;
        gamma_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;
        run_index = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "final_rmsnorm_stage_real";
        tracefile = "FPGA_Project/sim/final_rmsnorm_stage_trace.csv";
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
            "cycle,run,start,busy,done,error,saturation,state,norm_state,sum_busy,sum_done,sqrt_busy,sqrt_done,div_busy,div_done,apply_busy,apply_done,sum_index,apply_index\n"
        );

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_index = 1;
        pulse_start();

        repeat (256) @(negedge clk);
        if (done != 1'b1) begin
            pulse_start();
        end

        while ((done != 1'b1) && (cycle_count < 10000)) begin
            @(negedge clk);
        end
        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for final_rmsnorm_stage run 1 done");
            $finish(1);
        end
        @(posedge clk);
        #1;
        check_outputs(1);

        run_index = 2;
        repeat (8) @(negedge clk);
        pulse_start();

        while ((done != 1'b1) && (cycle_count < 20000)) begin
            @(negedge clk);
        end
        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for final_rmsnorm_stage run 2 done");
            $finish(1);
        end
        @(posedge clk);
        #1;
        check_outputs(2);

        $fclose(trace_fd);

        $display("final_rmsnorm_stage real full-model current-token test");
        $display("  done seen count        = %0d", done_seen_count);
        $display("  stage cycle count      = %0d", cycle_count_dut);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  total cycles waited    = %0d", cycle_count);
        $display("  trace                  = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy == 0) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted after final done cleanup");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d final_rmsnorm_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: final_rmsnorm_stage matched final RMSNorm outputs on two runs.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
