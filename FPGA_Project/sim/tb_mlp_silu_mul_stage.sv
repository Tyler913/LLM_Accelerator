`timescale 1ns/1ps
`default_nettype none

module tb_mlp_silu_mul_stage;

    localparam int FEATURES      = 3072;
    localparam int IN_WIDTH      = 24;
    localparam int IN_FRAC       = 12;
    localparam int SIGMOID_WIDTH = 16;
    localparam int SIGMOID_FRAC  = 16;
    localparam int LUT_INDEX_FRAC = 6;
    localparam int LUT_MIN_INT   = -8;
    localparam int LUT_MAX_INT   = 8;
    localparam int LUT_SIZE      = ((LUT_MAX_INT - LUT_MIN_INT) << LUT_INDEX_FRAC) + 1;
    localparam int OUT_WIDTH     = 24;
    localparam int ROW_INDEX_W   = 12;
    localparam int LUT_INDEX_W   = 11;

    logic clk;
    logic rst_n;
    logic start;
    logic [LUT_SIZE*SIGMOID_WIDTH-1 : 0] sigmoid_lut_flat;

    logic in_valid;
    logic in_ready;
    logic [ROW_INDEX_W-1 : 0] in_index;
    logic signed [IN_WIDTH-1 : 0] gate_data;
    logic signed [IN_WIDTH-1 : 0] up_data;
    logic in_last;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic out_valid;
    logic out_ready;
    logic [ROW_INDEX_W-1 : 0] out_index;
    logic signed [OUT_WIDTH-1 : 0] hidden_data;
    logic out_last;
    logic [31 : 0] input_count;
    logic [31 : 0] output_count;

    logic signed [IN_WIDTH-1 : 0] gate_mem [0:FEATURES-1];
    logic signed [IN_WIDTH-1 : 0] up_mem [0:FEATURES-1];
    logic [SIGMOID_WIDTH-1 : 0] sigmoid_lut_mem [0:LUT_SIZE-1];
    logic [15 : 0] expected_lut_index_mem [0:FEATURES-1];
    logic [SIGMOID_WIDTH-1 : 0] expected_sigmoid_mem [0:FEATURES-1];
    logic signed [OUT_WIDTH-1 : 0] expected_silu_gate_mem [0:FEATURES-1];
    logic signed [OUT_WIDTH-1 : 0] expected_hidden_mem [0:FEATURES-1];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer lut_index;
    integer cycle_count;
    integer input_accept_count;
    integer output_accept_count;
    integer input_stall_cycles;
    integer output_stall_cycles;
    integer input_gap_cycles;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer drive_enable;
    integer spurious_start_seen_busy;
    integer lut_mismatch_count;
    longint signed hidden_diff;
    longint signed max_abs_hidden_diff;

    logic in_stall_active;
    logic [ROW_INDEX_W-1 : 0] stalled_in_index;
    logic signed [IN_WIDTH-1 : 0] stalled_gate_data;
    logic signed [IN_WIDTH-1 : 0] stalled_up_data;
    logic stalled_in_last;

    logic out_stall_active;
    logic [ROW_INDEX_W-1 : 0] stalled_out_index;
    logic signed [OUT_WIDTH-1 : 0] stalled_hidden_data;
    logic stalled_out_last;
    logic input_fire_seen;

    mlp_silu_mul_stage #(
        .FEATURES               (FEATURES),
        .IN_WIDTH               (IN_WIDTH),
        .IN_FRAC                (IN_FRAC),
        .SIGMOID_WIDTH          (SIGMOID_WIDTH),
        .SIGMOID_FRAC           (SIGMOID_FRAC),
        .SIGMOID_LUT_INDEX_FRAC (LUT_INDEX_FRAC),
        .SIGMOID_LUT_MIN_INT    (LUT_MIN_INT),
        .SIGMOID_LUT_MAX_INT    (LUT_MAX_INT),
        .SIGMOID_LUT_SIZE       (LUT_SIZE),
        .OUT_WIDTH              (OUT_WIDTH),
        .ROW_INDEX_W            (ROW_INDEX_W),
        .LUT_INDEX_W            (LUT_INDEX_W)
    ) dut (
        .i_clk             (clk),
        .i_rst_n           (rst_n),
        .i_start           (start),
        .i_sigmoid_lut_flat(sigmoid_lut_flat),
        .i_in_valid        (in_valid),
        .o_in_ready        (in_ready),
        .i_in_index        (in_index),
        .i_gate_data       (gate_data),
        .i_up_data         (up_data),
        .i_in_last         (in_last),
        .o_busy            (busy),
        .o_done            (done),
        .o_error           (error),
        .o_saturation      (saturation),
        .o_out_valid       (out_valid),
        .i_out_ready       (out_ready),
        .o_out_index       (out_index),
        .o_hidden_data     (hidden_data),
        .o_out_last        (out_last),
        .o_input_count     (input_count),
        .o_output_count    (output_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/mlp_silu_mul_stage.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, in_valid);
        $dumpvars(0, in_ready);
        $dumpvars(0, in_index);
        $dumpvars(0, gate_data);
        $dumpvars(0, up_data);
        $dumpvars(0, in_last);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, saturation);
        $dumpvars(0, out_valid);
        $dumpvars(0, out_ready);
        $dumpvars(0, out_index);
        $dumpvars(0, hidden_data);
        $dumpvars(0, out_last);
        $dumpvars(0, input_count);
        $dumpvars(0, output_count);
        $dumpvars(0, dut.current_state);
        $dumpvars(0, dut.lut_index_comb);
        $dumpvars(0, dut.sigmoid_value_comb);
        $dumpvars(0, dut.silu_gate_comb);
    end

    function automatic logic input_valid_pattern(input integer cycle, input integer accepted);
        begin
            input_valid_pattern =
                (cycle > 20) &&
                ((cycle % 5) != 2) &&
                ((cycle % 17) != 9) &&
                !((accepted >= 640) && (accepted <= 720) && ((cycle % 4) == 1)) &&
                !((accepted >= 2100) && (accepted <= 2175) && ((cycle % 3) == 0));
        end
    endfunction

    function automatic logic out_ready_pattern(input integer cycle, input integer accepted);
        begin
            out_ready_pattern =
                (cycle > 30) &&
                ((cycle % 7) != 3) &&
                ((cycle % 23) != 11) &&
                !((accepted >= 900) && (accepted <= 980) && ((cycle % 4) != 0)) &&
                !((accepted >= 2500) && (accepted <= 2580) && ((cycle % 5) == 2));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_gate_input.hex"}, gate_mem);
            $readmemh({vector_dir, "/", prefix, "_up_input.hex"}, up_mem);
            $readmemh({vector_dir, "/", prefix, "_sigmoid_lut.hex"}, sigmoid_lut_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_lut_index.hex"}, expected_lut_index_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_sigmoid_q0_16.hex"}, expected_sigmoid_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_silu_gate_q12_12.hex"}, expected_silu_gate_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_hidden_q12_12.hex"}, expected_hidden_mem);
        end
    endtask

    task pack_lut;
        begin
            sigmoid_lut_flat = '0;
            for (lut_index = 0; lut_index < LUT_SIZE; lut_index = lut_index + 1) begin
                sigmoid_lut_flat[lut_index*SIGMOID_WIDTH +: SIGMOID_WIDTH] =
                    sigmoid_lut_mem[lut_index];
            end
        end
    endtask

    task check_input_stability;
        begin
            if (in_stall_active == 1'b1) begin
                if ((in_index !== stalled_in_index) ||
                    (gate_data !== stalled_gate_data) ||
                    (up_data !== stalled_up_data) ||
                    (in_last !== stalled_in_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: MLP SiLU/mul input changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_output_stability;
        begin
            if (out_stall_active == 1'b1) begin
                if ((out_index !== stalled_out_index) ||
                    (hidden_data !== stalled_hidden_data) ||
                    (out_last !== stalled_out_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: MLP SiLU/mul output changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_input_accept;
        begin
            if (input_accept_count >= FEATURES) begin
                if (print_count < 32) begin
                    $display("FAIL: extra MLP SiLU/mul input at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((in_index != input_accept_count) ||
                    (gate_data !== gate_mem[input_accept_count]) ||
                    (up_data !== up_mem[input_accept_count]) ||
                    (in_last !== (input_accept_count == (FEATURES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: MLP SiLU/mul input %0d mismatch idx=%0d gate=%0d exp_gate=%0d up=%0d exp_up=%0d last=%0d",
                            input_accept_count,
                            in_index,
                            gate_data,
                            gate_mem[input_accept_count],
                            up_data,
                            up_mem[input_accept_count],
                            in_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end

                if ((dut.lut_index_comb !== expected_lut_index_mem[input_accept_count][LUT_INDEX_W-1 : 0]) ||
                    (dut.sigmoid_value_comb !== expected_sigmoid_mem[input_accept_count]) ||
                    (dut.silu_gate_comb !== expected_silu_gate_mem[input_accept_count])) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: MLP SiLU/mul LUT path %0d mismatch lut=%0d exp_lut=%0d sigmoid=%0d exp_sigmoid=%0d silu=%0d exp_silu=%0d",
                            input_accept_count,
                            dut.lut_index_comb,
                            expected_lut_index_mem[input_accept_count],
                            dut.sigmoid_value_comb,
                            expected_sigmoid_mem[input_accept_count],
                            dut.silu_gate_comb,
                            expected_silu_gate_mem[input_accept_count]
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                    lut_mismatch_count = lut_mismatch_count + 1;
                end
            end
        end
    endtask

    task check_output_accept;
        begin
            if (output_accept_count >= FEATURES) begin
                if (print_count < 32) begin
                    $display("FAIL: extra MLP SiLU/mul output at cycle %0d", cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((out_index != output_accept_count) ||
                    (hidden_data !== expected_hidden_mem[output_accept_count]) ||
                    (out_last !== (output_accept_count == (FEATURES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: MLP SiLU/mul output %0d mismatch idx=%0d hidden=%0d exp_hidden=%0d last=%0d",
                            output_accept_count,
                            out_index,
                            hidden_data,
                            expected_hidden_mem[output_accept_count],
                            out_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end

                hidden_diff = hidden_data - expected_hidden_mem[output_accept_count];
                if (hidden_diff < 0) begin
                    hidden_diff = -hidden_diff;
                end
                if (hidden_diff > max_abs_hidden_diff) begin
                    max_abs_hidden_diff = hidden_diff;
                end
            end
        end
    endtask

    always @(negedge clk) begin
        if (rst_n == 1'b0) begin
            in_valid = 1'b0;
            out_ready = 1'b0;
            in_index = 'd0;
            gate_data = 'd0;
            up_data = 'd0;
            in_last = 1'b0;
        end
        else begin
            out_ready = out_ready_pattern(cycle_count, output_accept_count);

            if (drive_enable && (done == 1'b0)) begin
                if ((in_valid == 1'b0) || (in_ready == 1'b1) || (input_fire_seen == 1'b1)) begin
                    if ((input_accept_count < FEATURES) &&
                        input_valid_pattern(cycle_count, input_accept_count)) begin
                        in_valid = 1'b1;
                        in_index = input_accept_count[ROW_INDEX_W-1 : 0];
                        gate_data = gate_mem[input_accept_count];
                        up_data = up_mem[input_accept_count];
                        in_last = (input_accept_count == (FEATURES - 1));
                    end
                    else begin
                        in_valid = 1'b0;
                    end
                end
            end
            else begin
                in_valid = 1'b0;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            input_accept_count <= 0;
            output_accept_count <= 0;
            input_stall_cycles <= 0;
            output_stall_cycles <= 0;
            input_gap_cycles <= 0;
            in_stall_active <= 1'b0;
            stalled_in_index <= 'd0;
            stalled_gate_data <= 'd0;
            stalled_up_data <= 'd0;
            stalled_in_last <= 1'b0;
            out_stall_active <= 1'b0;
            stalled_out_index <= 'd0;
            stalled_hidden_data <= 'd0;
            stalled_out_last <= 1'b0;
            max_abs_hidden_diff <= 0;
            spurious_start_seen_busy <= 0;
            lut_mismatch_count <= 0;
            input_fire_seen <= 1'b0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            input_fire_seen <= in_valid && in_ready;

            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end

            if ((busy == 1'b1) && (input_accept_count < FEATURES) &&
                ((in_valid == 1'b0) || (in_ready == 1'b0))) begin
                if (in_valid == 1'b0) begin
                    input_gap_cycles <= input_gap_cycles + 1;
                end
            end

            if (in_valid == 1'b1) begin
                check_input_stability();
                if (in_ready == 1'b1) begin
                    check_input_accept();
                    input_accept_count <= input_accept_count + 1;
                    in_stall_active <= 1'b0;
                end
                else begin
                    input_stall_cycles <= input_stall_cycles + 1;
                    in_stall_active <= 1'b1;
                    stalled_in_index <= in_index;
                    stalled_gate_data <= gate_data;
                    stalled_up_data <= up_data;
                    stalled_in_last <= in_last;
                end
            end
            else begin
                in_stall_active <= 1'b0;
            end

            if (out_valid == 1'b1) begin
                check_output_stability();
                if (out_ready == 1'b1) begin
                    check_output_accept();
                    output_accept_count <= output_accept_count + 1;
                    out_stall_active <= 1'b0;
                end
                else begin
                    output_stall_cycles <= output_stall_cycles + 1;
                    out_stall_active <= 1'b1;
                    stalled_out_index <= out_index;
                    stalled_hidden_data <= hidden_data;
                    stalled_out_last <= out_last;
                end
            end
            else begin
                out_stall_active <= 1'b0;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%03h,%0d,%0d,%0d,%0d,%0d,%0d,0x%03h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    dut.current_state,
                    in_valid,
                    in_ready,
                    in_valid && in_ready,
                    in_index,
                    gate_data,
                    up_data,
                    in_last,
                    dut.lut_index_comb,
                    dut.sigmoid_value_comb,
                    dut.silu_gate_comb,
                    out_index,
                    out_valid,
                    out_ready,
                    out_valid && out_ready,
                    hidden_data,
                    out_last,
                    input_count,
                    output_count,
                    output_accept_count
                );
            end
        end
    end

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        in_valid = 1'b0;
        in_index = 'd0;
        gate_data = 'd0;
        up_data = 'd0;
        in_last = 1'b0;
        out_ready = 1'b0;
        sigmoid_lut_flat = '0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;
        drive_enable = 0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "mlp_silu_mul_stage_real";
        tracefile = "FPGA_Project/sim/mlp_silu_mul_stage_trace.csv";
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
            "cycle,start,busy,done,error,saturation,state,in_valid,in_ready,in_accept,in_index,gate_data,up_data,in_last,lut_index,sigmoid_q0_16,silu_gate_q12_12,out_index,out_valid,out_ready,out_accept,hidden_data,out_last,input_count,output_count,accepted_before\n"
        );

        load_vectors();
        pack_lut();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        drive_enable = 1;

        repeat (96) @(negedge clk);
        if (done != 1'b1) begin
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end

        while ((done != 1'b1) && (cycle_count < 30000)) begin
            @(negedge clk);
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for mlp_silu_mul_stage done");
            $finish(1);
        end

        @(posedge clk);
        drive_enable = 0;
        in_valid = 1'b0;
        out_ready = 1'b0;
        #1;
        $fclose(trace_fd);

        $display("mlp_silu_mul_stage real Layer 0 current-token test");
        $display("  inputs accepted        = %0d", input_accept_count);
        $display("  outputs accepted       = %0d", output_accept_count);
        $display("  input gap cycles       = %0d", input_gap_cycles);
        $display("  input stall cycles     = %0d", input_stall_cycles);
        $display("  output stall cycles    = %0d", output_stall_cycles);
        $display("  lut mismatch count     = %0d", lut_mismatch_count);
        $display("  max_abs_hidden_diff    = %0d", max_abs_hidden_diff);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  total cycles waited    = %0d", cycle_count);
        $display("  trace                  = %s", tracefile);

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (saturation) begin
            $display("FAIL: unexpected MLP SiLU/mul saturation");
            mismatch_count = mismatch_count + 1;
        end
        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end
        if (input_accept_count != FEATURES) begin
            $display("FAIL: input_accept_count mismatch actual=%0d expected=%0d",
                     input_accept_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (output_accept_count != FEATURES) begin
            $display("FAIL: output_accept_count mismatch actual=%0d expected=%0d",
                     output_accept_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (input_count != FEATURES) begin
            $display("FAIL: dut input_count mismatch actual=%0d expected=%0d",
                     input_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (output_count != FEATURES) begin
            $display("FAIL: dut output_count mismatch actual=%0d expected=%0d",
                     output_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (input_gap_cycles < 300) begin
            $display("FAIL: input gap coverage too weak: %0d", input_gap_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (input_stall_cycles < 300) begin
            $display("FAIL: input backpressure coverage too weak: %0d", input_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (output_stall_cycles < 500) begin
            $display("FAIL: output backpressure coverage too weak: %0d", output_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy == 0) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d mlp_silu_mul_stage mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: mlp_silu_mul_stage matched all 3072 SiLU/multiply Q12.12 outputs under input/output backpressure.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
