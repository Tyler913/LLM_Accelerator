`timescale 1ns/1ps
`default_nettype none

module tb_mlp_residual_add_stage;

    localparam int INPUT_SIZE           = 1024;
    localparam int POST_ATTENTION_WIDTH = 24;
    localparam int POST_ATTENTION_FRAC  = 10;
    localparam int DOWN_WIDTH           = 24;
    localparam int DOWN_FRAC            = 12;
    localparam int OUT_WIDTH            = 24;
    localparam int OUT_FRAC             = 10;

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*POST_ATTENTION_WIDTH-1 : 0] post_attn_hidden_flat;
    logic [INPUT_SIZE*DOWN_WIDTH-1 : 0] down_out_flat;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] output_count;
    logic [31 : 0] stage_cycle_count;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0] layer_out_flat;

    logic signed [POST_ATTENTION_WIDTH-1 : 0] post_attn_hidden_mem [0:INPUT_SIZE-1];
    logic signed [DOWN_WIDTH-1 : 0] down_out_mem [0:INPUT_SIZE-1];
    logic signed [OUT_WIDTH-1 : 0] expected_layer_out_mem [0:INPUT_SIZE-1];
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
    integer layer_max_abs_diff;
    integer diff_value;

    mlp_residual_add_stage #(
        .INPUT_SIZE          (INPUT_SIZE),
        .POST_ATTENTION_WIDTH(POST_ATTENTION_WIDTH),
        .POST_ATTENTION_FRAC (POST_ATTENTION_FRAC),
        .DOWN_WIDTH          (DOWN_WIDTH),
        .DOWN_FRAC           (DOWN_FRAC),
        .OUT_WIDTH           (OUT_WIDTH),
        .OUT_FRAC            (OUT_FRAC)
    ) dut (
        .i_clk                  (clk),
        .i_rst_n                (rst_n),
        .i_start                (start),
        .i_post_attn_hidden_flat(post_attn_hidden_flat),
        .i_down_out_flat        (down_out_flat),
        .o_busy                 (busy),
        .o_done                 (done),
        .o_error                (error),
        .o_saturation           (saturation),
        .o_output_count         (output_count),
        .o_cycle_count          (stage_cycle_count),
        .o_layer_out_flat       (layer_out_flat)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/mlp_residual_add_stage.vcd";
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
        $dumpvars(0, output_count);
        $dumpvars(0, stage_cycle_count);
        $dumpvars(0, dut.current_state);
        $dumpvars(0, dut.add_busy);
        $dumpvars(0, dut.add_done);
        $dumpvars(0, dut.inst_residual_add_1024.current_state);
        $dumpvars(0, dut.inst_residual_add_1024.element_index);
    end

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_post_attn_hidden.hex"}, post_attn_hidden_mem);
            $readmemh({vector_dir, "/", prefix, "_down_out.hex"}, down_out_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_layer_out.hex"}, expected_layer_out_mem);
            $readmemh({vector_dir, "/", prefix, "_residual_saturation.hex"}, expected_saturation_mem);
        end
    endtask

    task pack_inputs;
        begin
            post_attn_hidden_flat = '0;
            down_out_flat = '0;
            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                post_attn_hidden_flat[element_index*POST_ATTENTION_WIDTH +: POST_ATTENTION_WIDTH] =
                    post_attn_hidden_mem[element_index];
                down_out_flat[element_index*DOWN_WIDTH +: DOWN_WIDTH] =
                    down_out_mem[element_index];
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

    task check_outputs;
        input integer checked_run;
        begin
            layer_max_abs_diff = 0;
            if (error) begin
                $display("FAIL: dut error asserted after run %0d", checked_run);
                mismatch_count = mismatch_count + 1;
            end
            if (saturation !== expected_saturation_mem[0]) begin
                $display("FAIL: saturation mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, saturation, expected_saturation_mem[0]);
                mismatch_count = mismatch_count + 1;
            end
            if (output_count != INPUT_SIZE) begin
                $display("FAIL: output_count mismatch after run %0d actual=%0d expected=%0d",
                         checked_run, output_count, INPUT_SIZE);
                mismatch_count = mismatch_count + 1;
            end
            if (stage_cycle_count < INPUT_SIZE) begin
                $display("FAIL: stage-cycle coverage unexpectedly short after run %0d: %0d",
                         checked_run, stage_cycle_count);
                mismatch_count = mismatch_count + 1;
            end

            for (element_index = 0; element_index < INPUT_SIZE; element_index = element_index + 1) begin
                diff_value =
                    $signed(layer_out_flat[element_index*OUT_WIDTH +: OUT_WIDTH]) -
                    expected_layer_out_mem[element_index];
                if (diff_value < 0) begin
                    diff_value = -diff_value;
                end
                if (diff_value > layer_max_abs_diff) begin
                    layer_max_abs_diff = diff_value;
                end

                if ($signed(layer_out_flat[element_index*OUT_WIDTH +: OUT_WIDTH]) !==
                    expected_layer_out_mem[element_index]) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: layer_out run %0d element %0d mismatch actual=%0d expected=%0d",
                            checked_run,
                            element_index,
                            $signed(layer_out_flat[element_index*OUT_WIDTH +: OUT_WIDTH]),
                            expected_layer_out_mem[element_index]
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end

            $display("  run %0d layer max_abs_diff = %0d", checked_run, layer_max_abs_diff);
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
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    run_index,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    dut.current_state,
                    dut.add_busy,
                    dut.add_done,
                    output_count,
                    stage_cycle_count,
                    dut.inst_residual_add_1024.element_index
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        post_attn_hidden_flat = '0;
        down_out_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;
        run_index = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "mlp_residual_add_stage_real";
        tracefile = "FPGA_Project/sim/mlp_residual_add_stage_trace.csv";
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
            "cycle,run,start,busy,done,error,saturation,state,add_busy,add_done,output_count,stage_cycle_count,element_index\n"
        );

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_index = 1;
        pulse_start();

        repeat (128) @(negedge clk);
        if (done != 1'b1) begin
            pulse_start();
        end

        while ((done != 1'b1) && (cycle_count < 10000)) begin
            @(negedge clk);
        end
        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for mlp_residual_add_stage run 1 done");
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
            $display("FAIL: timed out waiting for mlp_residual_add_stage run 2 done");
            $finish(1);
        end
        @(posedge clk);
        #1;
        check_outputs(2);

        $fclose(trace_fd);

        $display("mlp_residual_add_stage real Layer 0 current-token test");
        $display("  done seen count        = %0d", done_seen_count);
        $display("  output count           = %0d", output_count);
        $display("  stage cycle count      = %0d", stage_cycle_count);
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
            $display("FAIL: %0d mlp_residual_add_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: mlp_residual_add_stage matched Layer 0 final residual output on two runs.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
