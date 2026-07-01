`timescale 1ns/1ps
`default_nettype none

module tb_o_proj_stage;

    localparam int INPUT_SIZE    = 2048;
    localparam int OUT_FEATURES  = 1024;
    localparam int TILE_ROWS     = 16;
    localparam int GROUP_SIZE    = 64;
    localparam int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH     = 24;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int OUT_WIDTH     = 24;
    localparam int ROW_INDEX_W   = 10;
    localparam int PARTIAL_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;
    localparam int WEIGHT_BIT_COUNT = OUT_FEATURES * INPUT_SIZE * WEIGHT_WIDTH;
    localparam int SCALE_BIT_COUNT  = OUT_FEATURES * GROUP_COUNT * SCALE_WIDTH;
    localparam int WEIGHT_WORD_COUNT = WEIGHT_BIT_COUNT / 32;
    localparam int SCALE_WORD_COUNT  = SCALE_BIT_COUNT / 32;

    logic clk;
    logic rst_n;
    logic start;
    logic [INPUT_SIZE*ACT_WIDTH-1 : 0] activation_flat;
    logic [OUT_FEATURES*INPUT_SIZE*WEIGHT_WIDTH-1 : 0] weight_flat;
    logic [OUT_FEATURES*GROUP_COUNT*SCALE_WIDTH-1 : 0] scale_flat;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic out_valid;
    logic out_ready;
    logic [ROW_INDEX_W-1 : 0] out_row;
    logic signed [OUT_WIDTH-1 : 0] out_data;
    logic out_last;
    logic [31 : 0] output_count;
    logic [31 : 0] compute_cycle_count;

    logic signed [ACT_WIDTH-1 : 0] activation_mem [0:INPUT_SIZE-1];
    logic [31 : 0] weight_word_mem [0:WEIGHT_WORD_COUNT-1];
    logic [31 : 0] scale_word_mem [0:SCALE_WORD_COUNT-1];
    logic signed [OUT_WIDTH-1 : 0] expected_out_mem [0:OUT_FEATURES-1];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer input_index;
    integer row_index;
    integer group_index;
    integer word_index;
    integer cycle_count;
    integer output_accept_count;
    integer output_stall_cycles;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    longint signed output_diff;
    longint signed max_abs_output_diff;

    logic out_stall_active;
    logic [ROW_INDEX_W-1 : 0] stalled_out_row;
    logic signed [OUT_WIDTH-1 : 0] stalled_out_data;
    logic stalled_out_last;

    o_proj_stage #(
        .INPUT_SIZE   (INPUT_SIZE),
        .OUT_FEATURES (OUT_FEATURES),
        .TILE_ROWS    (TILE_ROWS),
        .GROUP_SIZE   (GROUP_SIZE),
        .GROUP_COUNT  (GROUP_COUNT),
        .ACT_WIDTH    (ACT_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .SCALE_WIDTH  (SCALE_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH),
        .ROW_INDEX_W  (ROW_INDEX_W)
    ) dut (
        .i_clk                (clk),
        .i_rst_n              (rst_n),
        .i_start              (start),
        .i_activation_flat    (activation_flat),
        .i_weight_flat        (weight_flat),
        .i_scale_flat         (scale_flat),
        .o_busy               (busy),
        .o_done               (done),
        .o_error              (error),
        .o_saturation         (saturation),
        .o_out_valid          (out_valid),
        .i_out_ready          (out_ready),
        .o_out_row            (out_row),
        .o_out_data           (out_data),
        .o_out_last           (out_last),
        .o_output_count       (output_count),
        .o_compute_cycle_count(compute_cycle_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/o_proj_stage.vcd";
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
        $dumpvars(0, out_valid);
        $dumpvars(0, out_ready);
        $dumpvars(0, out_row);
        $dumpvars(0, out_data);
        $dumpvars(0, out_last);
        $dumpvars(0, output_count);
        $dumpvars(0, compute_cycle_count);
    end

    function automatic logic out_ready_pattern(input integer cycle, input integer accepted);
        begin
            out_ready_pattern =
                (cycle > 32) &&
                ((cycle % 5) != 2) &&
                ((cycle % 31) != 11) &&
                !((accepted >= 480) && (accepted <= 520) && ((cycle % 4) != 0));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_activation.hex"}, activation_mem);
            $readmemh({vector_dir, "/", prefix, "_weight_words32.hex"}, weight_word_mem);
            $readmemh({vector_dir, "/", prefix, "_scale_words32.hex"}, scale_word_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_q12_12.hex"}, expected_out_mem);
        end
    endtask

    task pack_inputs;
        begin
            activation_flat = '0;
            weight_flat = '0;
            scale_flat = '0;

            for (input_index = 0; input_index < INPUT_SIZE; input_index = input_index + 1) begin
                activation_flat[input_index*ACT_WIDTH +: ACT_WIDTH] =
                    activation_mem[input_index];
            end

            for (word_index = 0; word_index < WEIGHT_WORD_COUNT; word_index = word_index + 1) begin
                weight_flat[word_index*32 +: 32] = weight_word_mem[word_index];
            end

            for (word_index = 0; word_index < SCALE_WORD_COUNT; word_index = word_index + 1) begin
                scale_flat[word_index*32 +: 32] = scale_word_mem[word_index];
            end
        end
    endtask

    task check_output_stability;
        begin
            if (out_stall_active == 1'b1) begin
                if ((out_row !== stalled_out_row) ||
                    (out_data !== stalled_out_data) ||
                    (out_last !== stalled_out_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: o_proj output changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_output;
        begin
            if (output_accept_count >= OUT_FEATURES) begin
                if (print_count < 32) begin
                    $display("FAIL: extra o_proj output at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((out_row != output_accept_count) ||
                    (out_data !== expected_out_mem[output_accept_count]) ||
                    (out_last !== (output_accept_count == (OUT_FEATURES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: o_proj output %0d mismatch row=%0d data=%0d exp=%0d last=%0d",
                            output_accept_count,
                            out_row,
                            out_data,
                            expected_out_mem[output_accept_count],
                            out_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end

                output_diff = out_data - expected_out_mem[output_accept_count];
                if (output_diff < 0) begin
                    output_diff = -output_diff;
                end
                if (output_diff > max_abs_output_diff) begin
                    max_abs_output_diff = output_diff;
                end
            end
        end
    endtask

    always @(negedge clk) begin
        if (rst_n == 1'b0) begin
            out_ready = 1'b0;
        end
        else begin
            out_ready = out_ready_pattern(cycle_count, output_accept_count);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            output_accept_count <= 0;
            output_stall_cycles <= 0;
            out_stall_active <= 1'b0;
            stalled_out_row <= 'd0;
            stalled_out_data <= 'd0;
            stalled_out_last <= 1'b0;
            max_abs_output_diff <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if (out_valid == 1'b1) begin
                check_output_stability();
                if (out_ready == 1'b1) begin
                    check_output();
                    output_accept_count <= output_accept_count + 1;
                    out_stall_active <= 1'b0;
                end
                else begin
                    output_stall_cycles <= output_stall_cycles + 1;
                    out_stall_active <= 1'b1;
                    stalled_out_row <= out_row;
                    stalled_out_data <= out_data;
                    stalled_out_last <= out_last;
                end
            end
            else begin
                out_stall_active <= 1'b0;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,0x%06h,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    busy,
                    done,
                    out_valid,
                    out_ready,
                    out_valid && out_ready,
                    out_row,
                    out_data,
                    out_last,
                    output_count,
                    compute_cycle_count,
                    error,
                    saturation,
                    output_accept_count
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        out_ready = 1'b0;
        activation_flat = '0;
        weight_flat = '0;
        scale_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "o_proj_stage_real";
        tracefile = "FPGA_Project/sim/o_proj_stage_trace.csv";
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
            "cycle,busy,done,out_valid,out_ready,out_accept,out_row,out_data,out_last,output_count,compute_cycle_count,error,saturation,accepted_before\n"
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

        while ((done != 1'b1) && (cycle_count < 300000)) begin
            @(negedge clk);
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for o_proj_stage done");
            $finish(1);
        end

        @(negedge clk);
        out_ready = 1'b0;
        $fclose(trace_fd);

        $display("o_proj_stage real Layer 0 current-token test");
        $display("  outputs accepted      = %0d", output_accept_count);
        $display("  output stall cycles   = %0d", output_stall_cycles);
        $display("  compute cycles        = %0d", compute_cycle_count);
        $display("  max_abs_output_diff   = %0d", max_abs_output_diff);
        $display("  total cycles waited   = %0d", cycle_count);
        $display("  trace                 = %s", tracefile);

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (saturation) begin
            $display("FAIL: unexpected o_proj output saturation");
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end
        if (output_accept_count != OUT_FEATURES) begin
            $display("FAIL: output_accept_count mismatch actual=%0d expected=%0d",
                     output_accept_count, OUT_FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (output_count != OUT_FEATURES) begin
            $display("FAIL: dut output_count mismatch actual=%0d expected=%0d",
                     output_count, OUT_FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (output_stall_cycles < 100) begin
            $display("FAIL: output backpressure coverage too weak: %0d",
                     output_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (compute_cycle_count < 20000) begin
            $display("FAIL: compute-cycle coverage unexpectedly short: %0d",
                     compute_cycle_count);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d o_proj_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: o_proj_stage matched all 1024 Q12.12 outputs under output backpressure.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
