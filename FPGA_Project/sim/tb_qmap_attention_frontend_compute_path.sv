`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_attention_frontend_compute_path;

    localparam int ADDR_WIDTH       = 64;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int NUM_Q_HEADS      = 16;
    localparam int NUM_KV_HEADS     = 8;
    localparam int HEAD_DIM         = 128;
    localparam int MAX_CONTEXT      = 256;
    localparam int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT         = NUM_KV_HEADS * HEAD_DIM;
    localparam int TOTAL_CACHE_WRITES = 2 * KV_COUNT;
    localparam int QMAP_IMAGE_BYTES = 32'h8000;
    localparam int QMAP_WORDS       = QMAP_IMAGE_BYTES / 4;
    localparam int EXPECTED_READ_REQS = 31;
    localparam int EXPECTED_READ_WORDS = 4944;
    localparam int EXPECTED_WRITE_REQS = TOTAL_CACHE_WRITES + 1;
    localparam int EXPECTED_WRITE_WORDS = TOTAL_CACHE_WRITES + Q_COUNT;

    logic clk;
    logic rst_n;
    logic start;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic norm_saturation;
    logic rope_saturation;
    logic [31 : 0] cache_write_count;
    logic [31 : 0] q_rope_write_word_count;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;
    logic [7 : 0] state_debug;
    logic [3 : 0] read_slot_debug;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;

    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [31 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    logic [31 : 0] qmap_mem [0 : QMAP_WORDS-1];
    logic [31 : 0] q_rope_expected_mem [0 : Q_COUNT-1];
    logic [63 : 0] expected_cache_addr_mem [0 : TOTAL_CACHE_WRITES-1];
    logic [31 : 0] expected_cache_data_mem [0 : TOTAL_CACHE_WRITES-1];
    logic [3 : 0] expected_cache_kind_mem [0 : TOTAL_CACHE_WRITES-1];
    logic [7 : 0] expected_cache_head_mem [0 : TOTAL_CACHE_WRITES-1];
    logic [7 : 0] expected_cache_dim_mem [0 : TOTAL_CACHE_WRITES-1];

    string qmap_image_file;
    string q_rope_expected_file;
    string tracefile;
    string wavefile;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer trace_fd;
    integer done_seen_count;
    integer spurious_start_covered;
    integer success_cache_write_accept_count;
    integer success_q_rope_write_accept_count;
    integer success_wr_req_accept_count;
    integer success_wr_data_accept_count;
    integer success_rd_req_accept_count;
    integer success_rd_rsp_accept_count;
    integer success_cache_write_count;
    integer success_q_rope_write_word_count;
    integer success_mem_read_burst_count;
    integer success_mem_read_word_count;
    integer success_mem_write_req_count;
    integer success_mem_write_word_count;
    logic success_saturation;
    logic success_norm_saturation;
    logic success_rope_saturation;

    logic rd_active;
    integer rd_delay;
    integer rd_word_index;
    integer rd_word_count;
    integer rd_base_index;
    integer rd_req_accept_count;
    integer rd_rsp_accept_count;
    integer rd_req_stall_cycles;
    integer rd_rsp_stall_cycles;
    integer rd_latency_max;

    logic wr_active;
    integer wr_word_index;
    integer wr_word_count;
    logic [ADDR_WIDTH-1 : 0] wr_base_addr;
    integer wr_req_accept_count;
    integer wr_data_accept_count;
    integer wr_req_stall_cycles;
    integer wr_data_stall_cycles;
    integer wr_done_countdown;
    integer cache_write_accept_count;
    integer q_rope_write_accept_count;

    logic rd_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_rd_req_addr;
    logic [15 : 0] stalled_rd_req_len;
    logic rd_rsp_stall_active;
    logic [31 : 0] stalled_rd_rsp_data;
    logic stalled_rd_rsp_last;
    logic wr_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic wr_data_stall_active;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_data_last;

    localparam int SLOT_COS = 6;

    qmap_attention_frontend_compute_path dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(`QMAP_ATTN_FRONTEND_BASE_ADDR),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_saturation(saturation),
        .o_norm_saturation(norm_saturation),
        .o_rope_saturation(rope_saturation),
        .o_cache_write_count(cache_write_count),
        .o_q_rope_write_word_count(q_rope_write_word_count),
        .o_mem_read_burst_count(mem_read_burst_count),
        .o_mem_read_word_count(mem_read_word_count),
        .o_mem_write_req_count(mem_write_req_count),
        .o_mem_write_word_count(mem_write_word_count),
        .o_state_debug(state_debug),
        .o_read_slot_debug(read_slot_debug),
        .o_mem_rd_req_valid(mem_rd_req_valid),
        .i_mem_rd_req_ready(mem_rd_req_ready),
        .o_mem_rd_req_addr(mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(mem_rd_rsp_data),
        .i_mem_rd_rsp_last(mem_rd_rsp_last),
        .o_mem_wr_req_valid(mem_wr_req_valid),
        .i_mem_wr_req_ready(mem_wr_req_ready),
        .o_mem_wr_req_addr(mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(mem_wr_req_len_bytes),
        .o_mem_wr_data(mem_wr_data),
        .o_mem_wr_data_valid(mem_wr_data_valid),
        .i_mem_wr_data_ready(mem_wr_data_ready),
        .o_mem_wr_data_last(mem_wr_data_last),
        .i_mem_wr_done(mem_wr_done),
        .i_mem_wr_error(mem_wr_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qmap_attention_frontend_compute_path.vcd";
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
        $dumpvars(0, state_debug);
        $dumpvars(0, read_slot_debug);
        $dumpvars(0, mem_rd_req_valid);
        $dumpvars(0, mem_rd_req_ready);
        $dumpvars(0, mem_rd_req_addr);
        $dumpvars(0, mem_rd_req_len_bytes);
        $dumpvars(0, mem_wr_req_valid);
        $dumpvars(0, mem_wr_req_ready);
        $dumpvars(0, mem_wr_req_addr);
        $dumpvars(0, mem_wr_req_len_bytes);
        $dumpvars(0, mem_wr_data_valid);
        $dumpvars(0, mem_wr_data_ready);
        $dumpvars(0, mem_wr_data);
        $dumpvars(0, mem_wr_data_last);
        $dumpvars(0, cache_write_count);
        $dumpvars(0, q_rope_write_word_count);
    end

    function automatic logic rd_ready_pattern(input integer cycle, input integer accepted);
        begin
            rd_ready_pattern =
                (cycle > 8) &&
                ((cycle % 5) != 1) &&
                ((cycle % 17) != 6) &&
                !((accepted >= 12) && (accepted <= 16) && ((cycle % 4) != 0));
        end
    endfunction

    function automatic integer rd_latency_pattern(input integer accepted);
        begin
            if ((accepted % 11) == 3) begin
                rd_latency_pattern = 5;
            end
            else if ((accepted % 5) == 2) begin
                rd_latency_pattern = 2;
            end
            else begin
                rd_latency_pattern = 1;
            end
        end
    endfunction

    function automatic logic wr_req_ready_pattern(input integer cycle, input integer accepted);
        begin
            wr_req_ready_pattern =
                (cycle > 20) &&
                ((cycle % 4) != 2) &&
                ((cycle % 23) != 7) &&
                !((accepted >= 990) && (accepted <= 1020) && ((cycle % 5) != 0));
        end
    endfunction

    function automatic logic wr_data_ready_pattern(input integer cycle, input integer accepted);
        begin
            wr_data_ready_pattern =
                (cycle > 20) &&
                ((cycle % 5) != 3) &&
                ((cycle % 29) != 11) &&
                !((accepted >= 3000) && (accepted <= 3030) && ((cycle % 4) != 0));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(q_rope_expected_file, q_rope_expected_mem);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_addr.hex", expected_cache_addr_mem);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_data.hex", expected_cache_data_mem);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_kind.hex", expected_cache_kind_mem);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_head.hex", expected_cache_head_mem);
            $readmemh("FPGA_Project/sim/vectors/kv_cache_append_real_expected_dim.hex", expected_cache_dim_mem);
        end
    endtask

    task reset_model_state;
        begin
            cycle_count = 0;
            mismatch_count = 0;
            print_count = 0;
            done_seen_count = 0;
            spurious_start_covered = 0;
            success_cache_write_accept_count = 0;
            success_q_rope_write_accept_count = 0;
            success_wr_req_accept_count = 0;
            success_wr_data_accept_count = 0;
            success_rd_req_accept_count = 0;
            success_rd_rsp_accept_count = 0;
            success_cache_write_count = 0;
            success_q_rope_write_word_count = 0;
            success_mem_read_burst_count = 0;
            success_mem_read_word_count = 0;
            success_mem_write_req_count = 0;
            success_mem_write_word_count = 0;
            success_saturation = 1'b0;
            success_norm_saturation = 1'b0;
            success_rope_saturation = 1'b0;
            rd_active = 1'b0;
            rd_delay = 0;
            rd_word_index = 0;
            rd_word_count = 0;
            rd_base_index = 0;
            rd_req_accept_count = 0;
            rd_rsp_accept_count = 0;
            rd_req_stall_cycles = 0;
            rd_rsp_stall_cycles = 0;
            rd_latency_max = 0;
            wr_active = 1'b0;
            wr_word_index = 0;
            wr_word_count = 0;
            wr_base_addr = '0;
            wr_req_accept_count = 0;
            wr_data_accept_count = 0;
            wr_req_stall_cycles = 0;
            wr_data_stall_cycles = 0;
            wr_done_countdown = 0;
            cache_write_accept_count = 0;
            q_rope_write_accept_count = 0;
            rd_req_stall_active = 1'b0;
            rd_rsp_stall_active = 1'b0;
            wr_req_stall_active = 1'b0;
            wr_data_stall_active = 1'b0;
        end
    endtask

    task check_rd_req_stability;
        begin
            if (rd_req_stall_active) begin
                if ((mem_rd_req_addr !== stalled_rd_req_addr) ||
                    (mem_rd_req_len_bytes !== stalled_rd_req_len)) begin
                    $display("FAIL: read request changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_rd_rsp_stability;
        begin
            if (rd_rsp_stall_active) begin
                if ((mem_rd_rsp_data !== stalled_rd_rsp_data) ||
                    (mem_rd_rsp_last !== stalled_rd_rsp_last)) begin
                    $display("FAIL: read response changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_wr_req_stability;
        begin
            if (wr_req_stall_active) begin
                if ((mem_wr_req_addr !== stalled_wr_req_addr) ||
                    (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    $display("FAIL: write request changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_wr_data_stability;
        begin
            if (wr_data_stall_active) begin
                if ((mem_wr_data !== stalled_wr_data) ||
                    (mem_wr_data_last !== stalled_wr_data_last)) begin
                    $display("FAIL: write data changed while stalled at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_write_word;
        logic [ADDR_WIDTH-1 : 0] word_addr;
        begin
            word_addr = wr_base_addr + (wr_word_index * 4);
            if ((word_addr >= expected_cache_addr_mem[0]) &&
                (word_addr <= expected_cache_addr_mem[TOTAL_CACHE_WRITES-1])) begin
                if (cache_write_accept_count >= TOTAL_CACHE_WRITES) begin
                    $display("FAIL: extra cache write at cycle %0d addr=0x%016h", cycle_count, word_addr);
                    mismatch_count = mismatch_count + 1;
                end
                else begin
                    if ((word_addr !== expected_cache_addr_mem[cache_write_accept_count]) ||
                        (mem_wr_data !== expected_cache_data_mem[cache_write_accept_count])) begin
                        if (print_count < 32) begin
                            $display(
                                "FAIL: cache write %0d mismatch addr=0x%016h exp=0x%016h data=0x%08h exp=0x%08h",
                                cache_write_accept_count,
                                word_addr,
                                expected_cache_addr_mem[cache_write_accept_count],
                                mem_wr_data,
                                expected_cache_data_mem[cache_write_accept_count]
                            );
                            print_count = print_count + 1;
                        end
                        mismatch_count = mismatch_count + 1;
                    end
                    cache_write_accept_count = cache_write_accept_count + 1;
                end
            end
            else if ((word_addr >= `QMAP_ATTN_FRONTEND_BASE_ADDR) &&
                     (word_addr < (`QMAP_ATTN_FRONTEND_BASE_ADDR + QMAP_IMAGE_BYTES))) begin
                if (q_rope_write_accept_count >= Q_COUNT) begin
                    $display("FAIL: extra Q RoPE write at cycle %0d addr=0x%016h", cycle_count, word_addr);
                    mismatch_count = mismatch_count + 1;
                end
                else begin
                    if (mem_wr_data !== q_rope_expected_mem[q_rope_write_accept_count]) begin
                        if (print_count < 32) begin
                            $display(
                                "FAIL: Q RoPE write %0d mismatch data=0x%08h exp=0x%08h",
                                q_rope_write_accept_count,
                                mem_wr_data,
                                q_rope_expected_mem[q_rope_write_accept_count]
                            );
                            print_count = print_count + 1;
                        end
                        mismatch_count = mismatch_count + 1;
                    end
                    qmap_mem[(word_addr - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2] = mem_wr_data;
                    q_rope_write_accept_count = q_rope_write_accept_count + 1;
                end
            end
            else begin
                $display("FAIL: write to unexpected address 0x%016h at cycle %0d", word_addr, cycle_count);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    always @* begin
        mem_rd_req_ready =
            rst_n &&
            !rd_active &&
            !mem_rd_rsp_valid &&
            rd_ready_pattern(cycle_count, rd_req_accept_count);
        mem_wr_req_ready =
            rst_n &&
            !wr_active &&
            (wr_done_countdown == 0) &&
            wr_req_ready_pattern(cycle_count, wr_req_accept_count);
        mem_wr_data_ready =
            rst_n &&
            wr_active &&
            wr_data_ready_pattern(cycle_count, wr_data_accept_count);
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (done) begin
                done_seen_count <= done_seen_count + 1;
            end

            if (mem_rd_req_valid) begin
                if (!mem_rd_req_ready) begin
                    check_rd_req_stability();
                    rd_req_stall_cycles <= rd_req_stall_cycles + 1;
                    rd_req_stall_active <= 1'b1;
                    stalled_rd_req_addr <= mem_rd_req_addr;
                    stalled_rd_req_len <= mem_rd_req_len_bytes;
                end
                else begin
                    rd_req_stall_active <= 1'b0;
                end
            end
            else begin
                rd_req_stall_active <= 1'b0;
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                if ((mem_rd_req_addr < `QMAP_ATTN_FRONTEND_BASE_ADDR) ||
                    ((mem_rd_req_addr + mem_rd_req_len_bytes) > (`QMAP_ATTN_FRONTEND_BASE_ADDR + QMAP_IMAGE_BYTES)) ||
                    ((mem_rd_req_addr & 64'h3) != 0) ||
                    ((mem_rd_req_len_bytes & 16'h3) != 0)) begin
                    $display("FAIL: bad read request addr=0x%016h len=%0d", mem_rd_req_addr, mem_rd_req_len_bytes);
                    mismatch_count <= mismatch_count + 1;
                end
                rd_active <= 1'b1;
                rd_delay <= rd_latency_pattern(rd_req_accept_count);
                if (rd_latency_pattern(rd_req_accept_count) > rd_latency_max) begin
                    rd_latency_max <= rd_latency_pattern(rd_req_accept_count);
                end
                rd_word_index <= 0;
                rd_word_count <= mem_rd_req_len_bytes >> 2;
                rd_base_index <= (mem_rd_req_addr - `QMAP_ATTN_FRONTEND_BASE_ADDR) >> 2;
                rd_req_accept_count <= rd_req_accept_count + 1;
            end

            if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                check_rd_rsp_stability();
                rd_rsp_stall_cycles <= rd_rsp_stall_cycles + 1;
                rd_rsp_stall_active <= 1'b1;
                stalled_rd_rsp_data <= mem_rd_rsp_data;
                stalled_rd_rsp_last <= mem_rd_rsp_last;
            end
            else begin
                rd_rsp_stall_active <= 1'b0;
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                rd_rsp_accept_count <= rd_rsp_accept_count + 1;
                if (mem_rd_rsp_last) begin
                    mem_rd_rsp_valid <= 1'b0;
                    mem_rd_rsp_last <= 1'b0;
                    rd_active <= 1'b0;
                end
                else begin
                    rd_word_index <= rd_word_index + 1;
                    mem_rd_rsp_data <= qmap_mem[rd_base_index + rd_word_index + 1];
                    mem_rd_rsp_last <= (rd_word_index + 1 == rd_word_count - 1);
                end
            end
            else if (rd_active && !mem_rd_rsp_valid) begin
                if (rd_delay > 0) begin
                    rd_delay <= rd_delay - 1;
                end
                else begin
                    mem_rd_rsp_valid <= 1'b1;
                    mem_rd_rsp_data <= qmap_mem[rd_base_index];
                    mem_rd_rsp_last <= (rd_word_count == 1);
                end
            end

            if (mem_wr_req_valid) begin
                if (!mem_wr_req_ready) begin
                    check_wr_req_stability();
                    wr_req_stall_cycles <= wr_req_stall_cycles + 1;
                    wr_req_stall_active <= 1'b1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
                else begin
                    wr_req_stall_active <= 1'b0;
                end
            end
            else begin
                wr_req_stall_active <= 1'b0;
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                if (((mem_wr_req_addr & 64'h3) != 0) || ((mem_wr_req_len_bytes & 16'h3) != 0)) begin
                    $display("FAIL: unaligned write request addr=0x%016h len=%0d", mem_wr_req_addr, mem_wr_req_len_bytes);
                    mismatch_count <= mismatch_count + 1;
                end
                wr_active <= 1'b1;
                wr_word_index <= 0;
                wr_word_count <= mem_wr_req_len_bytes >> 2;
                wr_base_addr <= mem_wr_req_addr;
                wr_req_accept_count <= wr_req_accept_count + 1;
            end

            if (mem_wr_data_valid) begin
                if (!mem_wr_data_ready) begin
                    check_wr_data_stability();
                    wr_data_stall_cycles <= wr_data_stall_cycles + 1;
                    wr_data_stall_active <= 1'b1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_data_last <= mem_wr_data_last;
                end
                else begin
                    wr_data_stall_active <= 1'b0;
                end
            end
            else begin
                wr_data_stall_active <= 1'b0;
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                check_write_word();
                wr_data_accept_count <= wr_data_accept_count + 1;
                if (mem_wr_data_last !== (wr_word_index == wr_word_count - 1)) begin
                    $display("FAIL: write last mismatch at word %0d of %0d", wr_word_index, wr_word_count);
                    mismatch_count <= mismatch_count + 1;
                end
                if (mem_wr_data_last) begin
                    wr_active <= 1'b0;
                    wr_done_countdown <= 2;
                end
                else begin
                    wr_word_index <= wr_word_index + 1;
                end
            end

            if (wr_done_countdown > 0) begin
                wr_done_countdown <= wr_done_countdown - 1;
                if (wr_done_countdown == 1) begin
                    mem_wr_done <= 1'b1;
                end
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,0x%08h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    busy,
                    done,
                    error,
                    saturation,
                    state_debug,
                    read_slot_debug,
                    mem_rd_req_valid,
                    mem_rd_req_addr,
                    mem_rd_req_len_bytes,
                    mem_rd_req_ready,
                    mem_rd_rsp_valid,
                    mem_rd_rsp_ready,
                    mem_rd_rsp_last,
                    mem_wr_req_addr,
                    mem_wr_req_len_bytes,
                    mem_wr_req_valid,
                    mem_wr_req_ready,
                    mem_wr_data_valid,
                    mem_wr_data,
                    mem_wr_data_ready,
                    mem_wr_data_last,
                    mem_wr_done,
                    cache_write_count,
                    q_rope_write_word_count,
                    mem_read_burst_count,
                    mem_write_req_count,
                    mem_write_word_count
                );
            end
        end
    end

    task run_success;
        begin
            repeat (4) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            repeat (50) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
                spurious_start_covered = 1;
            end

            while ((done != 1'b1) && (cycle_count < 250000)) begin
                @(negedge clk);
            end

            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for qmap_attention_frontend_compute_path done");
                $finish(1);
            end
            if (error) begin
                $display("FAIL: success run raised DUT error");
                mismatch_count = mismatch_count + 1;
            end
            success_cache_write_accept_count = cache_write_accept_count;
            success_q_rope_write_accept_count = q_rope_write_accept_count;
            success_wr_req_accept_count = wr_req_accept_count;
            success_wr_data_accept_count = wr_data_accept_count;
            success_rd_req_accept_count = rd_req_accept_count;
            success_rd_rsp_accept_count = rd_rsp_accept_count;
            success_cache_write_count = cache_write_count;
            success_q_rope_write_word_count = q_rope_write_word_count;
            success_mem_read_burst_count = mem_read_burst_count;
            success_mem_read_word_count = mem_read_word_count;
            success_mem_write_req_count = mem_write_req_count;
            success_mem_write_word_count = mem_write_word_count;
            success_saturation = saturation;
            success_norm_saturation = norm_saturation;
            success_rope_saturation = rope_saturation;
        end
    endtask

    task run_invalid_cos_dtype;
        integer cos_dtype_word_index;
        integer writes_before;
        begin
            load_vectors();
            cos_dtype_word_index = (`QMAP_ATTN_FRONTEND_DESCRIPTOR_TABLE_ADDR - `QMAP_ATTN_FRONTEND_BASE_ADDR +
                                    SLOT_COS * `QMAP_DESCRIPTOR_BYTES + 8) >> 2;
            qmap_mem[cos_dtype_word_index] = `QMAP_DTYPE_U32;
            writes_before = wr_req_accept_count;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < 270000)) begin
                @(negedge clk);
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for invalid descriptor run");
                $finish(1);
            end
            if (!error) begin
                $display("FAIL: invalid cos dtype descriptor did not raise error");
                mismatch_count = mismatch_count + 1;
            end
            if (wr_req_accept_count != writes_before) begin
                $display("FAIL: invalid descriptor issued writes before=%0d after=%0d",
                         writes_before, wr_req_accept_count);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin : main_test
        rst_n = 1'b0;
        start = 1'b0;
        mem_rd_rsp_valid = 1'b0;
        mem_rd_rsp_data = 32'd0;
        mem_rd_rsp_last = 1'b0;
        mem_wr_done = 1'b0;
        mem_wr_error = 1'b0;

        qmap_image_file = "FPGA_Project/sim/vectors/qmap_attention_frontend_image_words32.hex";
        q_rope_expected_file = "FPGA_Project/sim/vectors/qmap_attention_frontend_q_rope_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_attention_frontend_compute_path_trace.csv";
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("q_rope_expected=%s", q_rope_expected_file)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        reset_model_state();
        load_vectors();

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,busy,done,error,saturation,state,read_slot,rd_req_valid,rd_req_addr,rd_req_len,rd_req_ready,rd_rsp_valid,rd_rsp_ready,rd_rsp_last,wr_req_addr,wr_req_len,wr_req_valid,wr_req_ready,wr_data_valid,wr_data,wr_data_ready,wr_data_last,wr_done,cache_write_count,q_rope_write_word_count,mem_read_burst_count,mem_write_req_count,mem_write_word_count\n"
        );

        run_success();
        run_invalid_cos_dtype();

        @(negedge clk);
        $fclose(trace_fd);

        $display("qmap_attention_frontend_compute_path real Layer 0 current-token test");
        $display("  cache writes accepted = %0d", success_cache_write_accept_count);
        $display("  Q RoPE writes accepted= %0d", success_q_rope_write_accept_count);
        $display("  read req/rsp accepted = %0d / %0d", success_rd_req_accept_count, success_rd_rsp_accept_count);
        $display("  write req/data        = %0d / %0d", success_wr_req_accept_count, success_wr_data_accept_count);
        $display("  DUT read bursts/words = %0d / %0d", success_mem_read_burst_count, success_mem_read_word_count);
        $display("  DUT write req/words   = %0d / %0d", success_mem_write_req_count, success_mem_write_word_count);
        $display("  request stalls rd/wr  = %0d / %0d", rd_req_stall_cycles, wr_req_stall_cycles);
        $display("  data stalls rd/wr     = %0d / %0d", rd_rsp_stall_cycles, wr_data_stall_cycles);
        $display("  read latency max      = %0d", rd_latency_max);
        $display("  spurious start covered= %0d", spurious_start_covered);
        $display("  done seen count       = %0d", done_seen_count);
        $display("  total cycles waited   = %0d", cycle_count);
        $display("  trace                 = %s", tracefile);

        if (error !== 1'b1) begin
            $display("FAIL: final invalid run should leave error asserted");
            mismatch_count = mismatch_count + 1;
        end
        if (success_cache_write_accept_count != TOTAL_CACHE_WRITES) begin
            $display("FAIL: cache_write_accept_count mismatch actual=%0d expected=%0d",
                     success_cache_write_accept_count, TOTAL_CACHE_WRITES);
            mismatch_count = mismatch_count + 1;
        end
        if (success_q_rope_write_accept_count != Q_COUNT) begin
            $display("FAIL: q_rope_write_accept_count mismatch actual=%0d expected=%0d",
                     success_q_rope_write_accept_count, Q_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (success_rd_req_accept_count != EXPECTED_READ_REQS) begin
            $display("FAIL: read request count mismatch actual=%0d expected=%0d",
                     success_rd_req_accept_count, EXPECTED_READ_REQS);
            mismatch_count = mismatch_count + 1;
        end
        if (success_rd_rsp_accept_count != EXPECTED_READ_WORDS) begin
            $display("FAIL: read response word count mismatch actual=%0d expected=%0d",
                     success_rd_rsp_accept_count, EXPECTED_READ_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if (success_wr_req_accept_count != EXPECTED_WRITE_REQS) begin
            $display("FAIL: write request count mismatch actual=%0d expected=%0d",
                     success_wr_req_accept_count, EXPECTED_WRITE_REQS);
            mismatch_count = mismatch_count + 1;
        end
        if (success_wr_data_accept_count != EXPECTED_WRITE_WORDS) begin
            $display("FAIL: write data count mismatch actual=%0d expected=%0d",
                     success_wr_data_accept_count, EXPECTED_WRITE_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if (success_cache_write_count != TOTAL_CACHE_WRITES) begin
            $display("FAIL: DUT cache_write_count mismatch actual=%0d expected=%0d",
                     success_cache_write_count, TOTAL_CACHE_WRITES);
            mismatch_count = mismatch_count + 1;
        end
        if (success_q_rope_write_word_count != Q_COUNT) begin
            $display("FAIL: DUT q_rope_write_word_count mismatch actual=%0d expected=%0d",
                     success_q_rope_write_word_count, Q_COUNT);
            mismatch_count = mismatch_count + 1;
        end
        if (success_saturation) begin
            $display("FAIL: unexpected attention front-end saturation");
            mismatch_count = mismatch_count + 1;
        end
        if (success_norm_saturation || success_rope_saturation) begin
            $display("FAIL: unexpected norm/rope saturation norm=%0d rope=%0d",
                     success_norm_saturation, success_rope_saturation);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_covered != 1) begin
            $display("FAIL: spurious start coverage missing");
            mismatch_count = mismatch_count + 1;
        end
        if (wr_data_stall_cycles < 100 || wr_req_stall_cycles < 100 || rd_req_stall_cycles < 10) begin
            $display("FAIL: backpressure coverage too weak rd_req=%0d wr_req=%0d wr_data=%0d",
                     rd_req_stall_cycles, wr_req_stall_cycles, wr_data_stall_cycles);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_attention_frontend_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_attention_frontend_compute_path matched Q RoPE write-back and exact K/V cache writes.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
