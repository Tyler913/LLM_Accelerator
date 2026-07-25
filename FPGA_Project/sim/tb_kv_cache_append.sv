`timescale 1ns/1ps
`default_nettype none

// Real-vector integration test for kv_cache_append.
//
// Stimulus:
//   Layer 0 last-token K after q/k norm + RoPE, plus V projection output from
//   the same Q4/QMAP fixed-point contract.
//
// This test intentionally inserts write-stream backpressure and checks:
//   - exact K-then-V write order
//   - exact cache byte addresses
//   - exact sign-extended 32-bit Q12.12 payload words
//   - last/write_count/done behavior
//   - address/data/control stability while valid is high and ready is low
//
// Run from the repository root:
//
//   conda run -n llm_fpga python \
//     Qwen3-0.6B-Base/python_each_module/23_export_kv_cache_append_vectors.py
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_kv_cache_append.vvp \
//     FPGA_Project/sim/tb_kv_cache_append.sv \
//     FPGA_Project/rtl/model/attention/kv_cache_append.sv \
//     FPGA_Project/rtl/model/attention/kv_cache_addr_gen.sv
//   vvp FPGA_Project/sim/tb_kv_cache_append.vvp

module tb_kv_cache_append;

    localparam int ADDR_WIDTH   = 64;
    localparam int DATA_WIDTH   = 32;
    localparam int NUM_LAYERS   = 28;
    localparam int NUM_KV_HEADS = 8;
    localparam int HEAD_DIM     = 128;
    localparam int MAX_CONTEXT  = 256;
    localparam int ELEMENT_WIDTH = 24;
    localparam int KV_COUNT     = NUM_KV_HEADS * HEAD_DIM;
    localparam int TOTAL_WRITES = 2 * KV_COUNT;

    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_BASE_ADDR =
        64'h0000_0004_1410_0000;
    localparam logic [4 : 0] DEFAULT_LAYER_ID = 5'd0;
    localparam logic [7 : 0] DEFAULT_POSITION = 8'd4;

    logic clk;
    logic rst_n;
    logic start;
    logic [ADDR_WIDTH-1 : 0] base_addr;
    logic [4 : 0] layer_id;
    logic [7 : 0] position;
    logic [KV_COUNT*ELEMENT_WIDTH-1 : 0] k_flat;
    logic [KV_COUNT*ELEMENT_WIDTH-1 : 0] v_flat;

    logic busy;
    logic done;
    logic error;
    logic wr_valid;
    logic wr_ready;
    logic [ADDR_WIDTH-1 : 0] wr_addr;
    logic [DATA_WIDTH-1 : 0] wr_data;
    logic wr_last;
    logic current_kind;
    logic [2 : 0] current_head;
    logic [6 : 0] current_dim;
    logic [31 : 0] write_count;

    logic signed [ELEMENT_WIDTH-1 : 0] k_input_mem [0:KV_COUNT-1];
    logic signed [ELEMENT_WIDTH-1 : 0] v_input_mem [0:KV_COUNT-1];
    logic [ADDR_WIDTH-1 : 0] expected_addr_mem [0:TOTAL_WRITES-1];
    logic [DATA_WIDTH-1 : 0] expected_data_mem [0:TOTAL_WRITES-1];
    logic [3 : 0] expected_kind_mem [0:TOTAL_WRITES-1];
    logic [7 : 0] expected_head_mem [0:TOTAL_WRITES-1];
    logic [7 : 0] expected_dim_mem [0:TOTAL_WRITES-1];

    string vector_dir;
    string prefix;
    string wavefile;
    string tracefile;

    integer element_index;
    integer cycle_count;
    integer accepted_count;
    integer valid_cycle_count;
    integer stall_cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;

    logic stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_addr;
    logic [DATA_WIDTH-1 : 0] stalled_data;
    logic stalled_last;
    logic stalled_kind;
    logic [2 : 0] stalled_head;
    logic [6 : 0] stalled_dim;

    kv_cache_append #(
        .ADDR_WIDTH   (ADDR_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .NUM_LAYERS   (NUM_LAYERS),
        .NUM_KV_HEADS (NUM_KV_HEADS),
        .HEAD_DIM     (HEAD_DIM),
        .MAX_CONTEXT  (MAX_CONTEXT),
        .ELEMENT_WIDTH(ELEMENT_WIDTH),
        .ELEMENT_BYTES(4)
    ) dut (
        .i_clk        (clk),
        .i_rst_n      (rst_n),
        .i_start      (start),
        .i_base_addr  (base_addr),
        .i_layer_id   (layer_id),
        .i_position   (position),
        .i_k_flat     (k_flat),
        .i_v_flat     (v_flat),
        .o_busy       (busy),
        .o_done       (done),
        .o_error      (error),
        .o_wr_valid   (wr_valid),
        .i_wr_ready   (wr_ready),
        .o_wr_addr    (wr_addr),
        .o_wr_data    (wr_data),
        .o_wr_last    (wr_last),
        .o_current_kind(current_kind),
        .o_current_head(current_head),
        .o_current_dim (current_dim),
        .o_write_count(write_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/kv_cache_append.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_kv_cache_append);
    end

    function automatic logic ready_pattern(input integer cycle, input integer accepted);
        begin
            ready_pattern =
                (cycle > 6) &&
                ((cycle % 7) != 2) &&
                ((cycle % 11) != 5) &&
                !((accepted >= 1010) && (accepted <= 1032) && ((cycle % 3) != 0)) &&
                !((accepted >= 1780) && (accepted <= 1795) && ((cycle % 4) == 1));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh({vector_dir, "/", prefix, "_k_input.hex"}, k_input_mem);
            $readmemh({vector_dir, "/", prefix, "_v_input.hex"}, v_input_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_addr.hex"}, expected_addr_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_data.hex"}, expected_data_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_kind.hex"}, expected_kind_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_head.hex"}, expected_head_mem);
            $readmemh({vector_dir, "/", prefix, "_expected_dim.hex"}, expected_dim_mem);
        end
    endtask

    task pack_inputs;
        begin
            k_flat = '0;
            v_flat = '0;
            for (element_index = 0; element_index < KV_COUNT; element_index = element_index + 1) begin
                k_flat[element_index*ELEMENT_WIDTH +: ELEMENT_WIDTH] =
                    k_input_mem[element_index];
                v_flat[element_index*ELEMENT_WIDTH +: ELEMENT_WIDTH] =
                    v_input_mem[element_index];
            end
        end
    endtask

    task check_stall_stability;
        begin
            if (stall_active == 1'b1) begin
                if ((wr_addr !== stalled_addr) ||
                    (wr_data !== stalled_data) ||
                    (wr_last !== stalled_last) ||
                    (current_kind !== stalled_kind) ||
                    (current_head !== stalled_head) ||
                    (current_dim !== stalled_dim)) begin
                    $display("FAIL: write stream changed while stalled at cycle %0d", cycle_count);
                    $display("  previous addr=0x%016h data=0x%08h kind=%0d head=%0d dim=%0d last=%0d",
                             stalled_addr, stalled_data, stalled_kind, stalled_head,
                             stalled_dim, stalled_last);
                    $display("  current  addr=0x%016h data=0x%08h kind=%0d head=%0d dim=%0d last=%0d",
                             wr_addr, wr_data, current_kind, current_head,
                             current_dim, wr_last);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_current_write;
        begin
            if (accepted_count >= TOTAL_WRITES) begin
                $display("FAIL: extra write after expected count at cycle %0d", cycle_count);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                if ((wr_addr !== expected_addr_mem[accepted_count]) ||
                    (wr_data !== expected_data_mem[accepted_count]) ||
                    ({3'd0, current_kind} !== expected_kind_mem[accepted_count]) ||
                    ({5'd0, current_head} !== expected_head_mem[accepted_count]) ||
                    ({1'd0, current_dim} !== expected_dim_mem[accepted_count]) ||
                    (wr_last !== (accepted_count == (TOTAL_WRITES - 1)))) begin
                    if (print_count < 32) begin
                        $display(
                            "FAIL: write %0d mismatch addr=0x%016h exp=0x%016h data=0x%08h exp=0x%08h kind=%0d exp=%0d head=%0d exp=%0d dim=%0d exp=%0d last=%0d",
                            accepted_count,
                            wr_addr,
                            expected_addr_mem[accepted_count],
                            wr_data,
                            expected_data_mem[accepted_count],
                            current_kind,
                            expected_kind_mem[accepted_count],
                            current_head,
                            expected_head_mem[accepted_count],
                            current_dim,
                            expected_dim_mem[accepted_count],
                            wr_last
                        );
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task write_trace_line;
        input integer accepted;
        begin
            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,0x%016h,0x%08h,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    wr_valid,
                    wr_ready,
                    accepted,
                    wr_addr,
                    wr_data,
                    current_kind,
                    current_head,
                    current_dim,
                    wr_last,
                    write_count
                );
            end
        end
    endtask

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        wr_ready = 1'b0;
        base_addr = DEFAULT_BASE_ADDR;
        layer_id = DEFAULT_LAYER_ID;
        position = DEFAULT_POSITION;
        cycle_count = 0;
        accepted_count = 0;
        valid_cycle_count = 0;
        stall_cycle_count = 0;
        mismatch_count = 0;
        print_count = 0;
        stall_active = 1'b0;
        stalled_addr = '0;
        stalled_data = '0;
        stalled_last = 1'b0;
        stalled_kind = 1'b0;
        stalled_head = '0;
        stalled_dim = '0;

        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "kv_cache_append_real";
        tracefile = "FPGA_Project/sim/kv_cache_append_trace.csv";
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
        $fwrite(trace_fd, "cycle,valid,ready,accepted,addr,data,kind,head,dim,last,write_count_before\n");

        load_vectors();
        pack_inputs();

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 6000)) begin
            @(negedge clk);
            wr_ready = ready_pattern(cycle_count, accepted_count);

            if (wr_valid == 1'b1) begin
                valid_cycle_count = valid_cycle_count + 1;
                check_stall_stability();

                if (wr_ready == 1'b1) begin
                    check_current_write();
                    write_trace_line(1);
                    accepted_count = accepted_count + 1;
                    stall_active = 1'b0;
                end
                else begin
                    write_trace_line(0);
                    stall_cycle_count = stall_cycle_count + 1;
                    stall_active = 1'b1;
                    stalled_addr = wr_addr;
                    stalled_data = wr_data;
                    stalled_last = wr_last;
                    stalled_kind = current_kind;
                    stalled_head = current_head;
                    stalled_dim = current_dim;
                end
            end
            else begin
                stall_active = 1'b0;
            end

            cycle_count = cycle_count + 1;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for kv_cache_append done");
            $finish(1);
        end

        #1;
        wr_ready = 1'b0;
        $fclose(trace_fd);

        $display("kv_cache_append real Layer 0 last-token test");
        $display("  accepted writes     = %0d", accepted_count);
        $display("  valid cycles        = %0d", valid_cycle_count);
        $display("  stall cycles        = %0d", stall_cycle_count);
        $display("  write_count output  = %0d", write_count);
        $display("  cycles waited       = %0d", cycle_count);
        $display("  trace               = %s", tracefile);

        if (error) begin
            $display("FAIL: dut error output asserted");
            mismatch_count = mismatch_count + 1;
        end

        if (busy) begin
            $display("FAIL: busy still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end

        if (wr_valid) begin
            $display("FAIL: wr_valid still asserted when done is high");
            mismatch_count = mismatch_count + 1;
        end

        if (accepted_count != TOTAL_WRITES) begin
            $display("FAIL: accepted_count mismatch actual=%0d expected=%0d",
                     accepted_count, TOTAL_WRITES);
            mismatch_count = mismatch_count + 1;
        end

        if (write_count != TOTAL_WRITES) begin
            $display("FAIL: write_count mismatch actual=%0d expected=%0d",
                     write_count, TOTAL_WRITES);
            mismatch_count = mismatch_count + 1;
        end

        if (stall_cycle_count < 100) begin
            $display("FAIL: backpressure coverage too weak, stall_cycle_count=%0d",
                     stall_cycle_count);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d kv_cache_append mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: kv_cache_append wrote exact K/V cache address/data stream with stall-stable timing.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
