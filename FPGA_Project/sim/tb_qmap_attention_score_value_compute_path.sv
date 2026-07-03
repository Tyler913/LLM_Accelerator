`timescale 1ns/1ps
`default_nettype none

module tb_qmap_attention_score_value_compute_path;

    localparam int ADDR_WIDTH       = 64;
    localparam int NUM_Q_HEADS      = 16;
    localparam int NUM_KV_HEADS     = 8;
    localparam int HEAD_DIM         = 128;
    localparam int MAX_CONTEXT      = 256;
    localparam int CACHE_LENGTH     = 5;
    localparam int IN_WIDTH         = 24;
    localparam int OUT_WIDTH        = 24;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM;
    localparam int KV_COUNT         = NUM_KV_HEADS * HEAD_DIM;
    localparam int IMAGE_BYTES      = 16'h5000;
    localparam int IMAGE_WORDS      = IMAGE_BYTES / 4;
    localparam int TOTAL_K_READS    = NUM_Q_HEADS * CACHE_LENGTH * HEAD_DIM;
    localparam int TOTAL_V_READS    = NUM_Q_HEADS * CACHE_LENGTH * HEAD_DIM;
    localparam int TOTAL_WRITES     = Q_COUNT;
    localparam int KV_REPEAT        = NUM_Q_HEADS / NUM_KV_HEADS;

    localparam logic [ADDR_WIDTH-1 : 0] QMAP_BASE_ADDR = 64'h0000_0004_0503_0000;
    localparam logic [ADDR_WIDTH-1 : 0] CACHE_BASE_ADDR = 64'h0000_0004_1410_0000;
    localparam logic [ADDR_WIDTH-1 : 0] ATTN_OUT_ADDR = 64'h0000_0004_0503_2980;
    localparam logic [ADDR_WIDTH-1 : 0] EXP_DTYPE_ADDR = QMAP_BASE_ADDR + 64'h0100 + (64'd3 * 64'd128) + 64'd8;

    logic clk;
    logic rst_n;
    logic start;

    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] score_count;
    logic [31 : 0] k_read_count;
    logic [31 : 0] v_read_count;
    logic [31 : 0] attn_out_capture_count;
    logic [31 : 0] attn_out_write_word_count;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;
    logic [7 : 0] state_debug;
    logic [3 : 0] read_slot_debug;

    logic rd_req_valid;
    logic rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] rd_req_addr;
    logic [15 : 0] rd_req_len_bytes;
    logic rd_rsp_valid;
    logic rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] rd_rsp_data;
    logic rd_rsp_last;

    logic wr_req_valid;
    logic wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] wr_req_addr;
    logic [15 : 0] wr_req_len_bytes;
    logic [31 : 0] wr_data;
    logic wr_data_valid;
    logic wr_data_ready;
    logic wr_data_last;
    logic wr_done;
    logic wr_error;

    logic [31 : 0] qmap_image_mem [0:IMAGE_WORDS-1];
    logic signed [IN_WIDTH-1 : 0] k_cache_mem [0:CACHE_LENGTH*KV_COUNT-1];
    logic signed [IN_WIDTH-1 : 0] v_cache_mem [0:CACHE_LENGTH*KV_COUNT-1];
    logic [31 : 0] expected_attn_out_mem [0:Q_COUNT-1];

    string vector_dir;
    string wavefile;
    string tracefile;
    integer trace_fd;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer done_seen_count;
    integer read_req_accept_count;
    integer read_rsp_accept_count;
    integer write_req_accept_count;
    integer write_data_accept_count;
    integer k_req_accept_count;
    integer v_req_accept_count;
    integer k_rsp_accept_count;
    integer v_rsp_accept_count;
    integer read_req_stall_cycles;
    integer write_req_stall_cycles;
    integer write_data_stall_cycles;
    integer read_latency_max;
    integer read_latency_sum;

    logic corrupt_exp_dtype;

    logic rd_active;
    logic [ADDR_WIDTH-1 : 0] rd_active_addr;
    integer rd_active_words;
    integer rd_word_index;
    integer rd_latency_countdown;
    integer rd_gap_countdown;
    logic rd_active_is_cache;
    logic rd_active_kind;

    logic wr_active;
    integer wr_done_countdown;

    logic rd_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_rd_req_addr;
    logic [15 : 0] stalled_rd_req_len;
    logic wr_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic wr_data_stall_active;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_data_last;

    integer normal_read_req_count;
    integer normal_read_rsp_count;
    integer normal_write_req_count;
    integer normal_write_data_count;
    integer normal_k_req_count;
    integer normal_v_req_count;
    integer normal_k_rsp_count;
    integer normal_v_rsp_count;
    integer normal_attn_write_count;
    integer normal_read_req_stall_cycles;
    integer normal_write_req_stall_cycles;
    integer normal_write_data_stall_cycles;
    integer normal_read_latency_max;
    integer normal_read_latency_sum;
    integer normal_done_cycle;
    integer invalid_write_req_count;
    integer invalid_write_data_count;
    integer invalid_done_cycle;

    qmap_attention_score_value_compute_path dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(QMAP_BASE_ADDR),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_saturation(saturation),
        .o_score_count(score_count),
        .o_k_read_count(k_read_count),
        .o_v_read_count(v_read_count),
        .o_attn_out_capture_count(attn_out_capture_count),
        .o_attn_out_write_word_count(attn_out_write_word_count),
        .o_mem_read_burst_count(mem_read_burst_count),
        .o_mem_read_word_count(mem_read_word_count),
        .o_mem_write_req_count(mem_write_req_count),
        .o_mem_write_word_count(mem_write_word_count),
        .o_state_debug(state_debug),
        .o_read_slot_debug(read_slot_debug),
        .o_mem_rd_req_valid(rd_req_valid),
        .i_mem_rd_req_ready(rd_req_ready),
        .o_mem_rd_req_addr(rd_req_addr),
        .o_mem_rd_req_len_bytes(rd_req_len_bytes),
        .i_mem_rd_rsp_valid(rd_rsp_valid),
        .o_mem_rd_rsp_ready(rd_rsp_ready),
        .i_mem_rd_rsp_data(rd_rsp_data),
        .i_mem_rd_rsp_last(rd_rsp_last),
        .o_mem_wr_req_valid(wr_req_valid),
        .i_mem_wr_req_ready(wr_req_ready),
        .o_mem_wr_req_addr(wr_req_addr),
        .o_mem_wr_req_len_bytes(wr_req_len_bytes),
        .o_mem_wr_data(wr_data),
        .o_mem_wr_data_valid(wr_data_valid),
        .i_mem_wr_data_ready(wr_data_ready),
        .o_mem_wr_data_last(wr_data_last),
        .i_mem_wr_done(wr_done),
        .i_mem_wr_error(wr_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qmap_attention_score_value_compute_path.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, tb_qmap_attention_score_value_compute_path);
    end

    function automatic logic rd_ready_pattern(input integer cycle, input integer accepted);
        begin
            rd_ready_pattern =
                (cycle > 8) &&
                ((cycle % 5) != 1) &&
                ((cycle % 23) != 7) &&
                !((accepted >= 12000) && (accepted <= 12040) && ((cycle % 4) != 0));
        end
    endfunction

    function automatic logic wr_req_ready_pattern(input integer cycle, input integer accepted);
        begin
            wr_req_ready_pattern =
                (cycle > 16) &&
                ((cycle % 7) != 2) &&
                ((cycle % 29) != 13);
        end
    endfunction

    function automatic logic wr_data_ready_pattern(input integer cycle, input integer accepted);
        begin
            wr_data_ready_pattern =
                (cycle > 24) &&
                ((cycle % 4) != 1) &&
                ((cycle % 19) != 8) &&
                !((accepted >= 1000) && (accepted <= 1030) && ((cycle % 5) != 0));
        end
    endfunction

    function automatic integer read_latency_pattern(input integer accepted);
        begin
            if ((accepted % 41) == 17) begin
                read_latency_pattern = 8;
            end
            else if ((accepted % 17) == 5) begin
                read_latency_pattern = 5;
            end
            else if ((accepted % 3) == 0) begin
                read_latency_pattern = 2;
            end
            else begin
                read_latency_pattern = 1;
            end
        end
    endfunction

    function automatic integer read_gap_pattern(input integer word_index);
        begin
            if ((word_index % 37) == 11) begin
                read_gap_pattern = 2;
            end
            else if ((word_index % 13) == 4) begin
                read_gap_pattern = 1;
            end
            else begin
                read_gap_pattern = 0;
            end
        end
    endfunction

    assign rd_req_ready =
        rst_n &&
        !rd_active &&
        rd_ready_pattern(cycle_count, read_req_accept_count);
    assign wr_req_ready =
        rst_n &&
        !wr_active &&
        wr_req_ready_pattern(cycle_count, write_req_accept_count);
    assign wr_data_ready =
        rst_n &&
        wr_active &&
        wr_data_ready_pattern(cycle_count, write_data_accept_count);

    function automatic logic is_qmap_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            is_qmap_addr =
                (addr >= QMAP_BASE_ADDR) &&
                (addr < (QMAP_BASE_ADDR + IMAGE_BYTES));
        end
    endfunction

    function automatic logic is_cache_addr(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            is_cache_addr =
                (addr >= CACHE_BASE_ADDR) &&
                (addr < (CACHE_BASE_ADDR + (2 * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM * 4)));
        end
    endfunction

    function automatic logic [31 : 0] qmap_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer index;
        begin
            index = (addr - QMAP_BASE_ADDR) >> 2;
            if (corrupt_exp_dtype && (addr == EXP_DTYPE_ADDR)) begin
                qmap_word = 32'd5;
            end
            else if ((index >= 0) && (index < IMAGE_WORDS)) begin
                qmap_word = qmap_image_mem[index];
            end
            else begin
                qmap_word = 32'hDEAD_BAD0;
            end
        end
    endfunction

    function automatic logic [31 : 0] cache_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer element_index;
        integer kind;
        integer rem0;
        integer head;
        integer rem1;
        integer position;
        integer dim;
        integer cache_index;
        logic signed [IN_WIDTH-1 : 0] value;
        begin
            element_index = (addr - CACHE_BASE_ADDR) >> 2;
            kind = element_index / (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            rem0 = element_index % (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM);
            head = rem0 / (MAX_CONTEXT * HEAD_DIM);
            rem1 = rem0 % (MAX_CONTEXT * HEAD_DIM);
            position = rem1 / HEAD_DIM;
            dim = rem1 % HEAD_DIM;
            cache_index = ((position * NUM_KV_HEADS + head) * HEAD_DIM) + dim;

            if ((position < CACHE_LENGTH) && (head < NUM_KV_HEADS) && (dim < HEAD_DIM)) begin
                if (kind == 0) begin
                    value = k_cache_mem[cache_index];
                end
                else begin
                    value = v_cache_mem[cache_index];
                end
                cache_word = {{8{value[IN_WIDTH-1]}}, value};
            end
            else begin
                cache_word = 32'hCAFE_BAD0;
            end
        end
    endfunction

    function automatic logic [31 : 0] read_word(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            if (is_qmap_addr(addr)) begin
                read_word = qmap_word(addr);
            end
            else if (is_cache_addr(addr)) begin
                read_word = cache_word(addr);
            end
            else begin
                read_word = 32'hBAD0_ADD0;
            end
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] expected_cache_addr(
        input integer kind,
        input integer q_head,
        input integer position,
        input integer dim
    );
        integer kv_head;
        integer element_index;
        begin
            kv_head = q_head / KV_REPEAT;
            element_index =
                (kind * NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM) +
                (kv_head * MAX_CONTEXT * HEAD_DIM) +
                (position * HEAD_DIM) +
                dim;
            expected_cache_addr = CACHE_BASE_ADDR + (element_index * 4);
        end
    endfunction

    task fail_once(input string message);
        begin
            if (print_count < 48) begin
                $display("%s", message);
                print_count = print_count + 1;
            end
            mismatch_count = mismatch_count + 1;
        end
    endtask

    task check_rd_req_stability;
        begin
            if (rd_req_stall_active) begin
                if ((rd_req_addr !== stalled_rd_req_addr) ||
                    (rd_req_len_bytes !== stalled_rd_req_len)) begin
                    fail_once("FAIL: read request changed while stalled");
                end
            end
        end
    endtask

    task check_wr_req_stability;
        begin
            if (wr_req_stall_active) begin
                if ((wr_req_addr !== stalled_wr_req_addr) ||
                    (wr_req_len_bytes !== stalled_wr_req_len)) begin
                    fail_once("FAIL: write request changed while stalled");
                end
            end
        end
    endtask

    task check_wr_data_stability;
        begin
            if (wr_data_stall_active) begin
                if ((wr_data !== stalled_wr_data) ||
                    (wr_data_last !== stalled_wr_data_last)) begin
                    fail_once("FAIL: write data changed while stalled");
                end
            end
        end
    endtask

    task check_cache_read_request(input logic [ADDR_WIDTH-1 : 0] addr);
        integer expected_q_head;
        integer expected_position;
        integer expected_dim;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            if (!is_cache_addr(addr)) begin
            end
            else if (((addr - CACHE_BASE_ADDR) >> 2) < (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM)) begin
                expected_q_head = k_req_accept_count / (CACHE_LENGTH * HEAD_DIM);
                expected_position = (k_req_accept_count / HEAD_DIM) % CACHE_LENGTH;
                expected_dim = k_req_accept_count % HEAD_DIM;
                expected_addr = expected_cache_addr(0, expected_q_head, expected_position, expected_dim);
                if (addr !== expected_addr) begin
                    fail_once("FAIL: K-cache read address/order mismatch");
                end
                k_req_accept_count = k_req_accept_count + 1;
            end
            else begin
                expected_q_head = v_req_accept_count / (HEAD_DIM * CACHE_LENGTH);
                expected_dim = (v_req_accept_count / CACHE_LENGTH) % HEAD_DIM;
                expected_position = v_req_accept_count % CACHE_LENGTH;
                expected_addr = expected_cache_addr(1, expected_q_head, expected_position, expected_dim);
                if (addr !== expected_addr) begin
                    fail_once("FAIL: V-cache read address/order mismatch");
                end
                v_req_accept_count = v_req_accept_count + 1;
            end
        end
    endtask

    task check_cache_read_response(input logic [ADDR_WIDTH-1 : 0] addr);
        begin
            if (is_cache_addr(addr)) begin
                if (((addr - CACHE_BASE_ADDR) >> 2) < (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM)) begin
                    k_rsp_accept_count = k_rsp_accept_count + 1;
                end
                else begin
                    v_rsp_accept_count = v_rsp_accept_count + 1;
                end
            end
        end
    endtask

    task clear_run_scoreboard;
        begin
            read_req_accept_count = 0;
            read_rsp_accept_count = 0;
            write_req_accept_count = 0;
            write_data_accept_count = 0;
            k_req_accept_count = 0;
            v_req_accept_count = 0;
            k_rsp_accept_count = 0;
            v_rsp_accept_count = 0;
            read_req_stall_cycles = 0;
            write_req_stall_cycles = 0;
            write_data_stall_cycles = 0;
            read_latency_max = 0;
            read_latency_sum = 0;
            rd_active = 1'b0;
            rd_rsp_valid = 1'b0;
            rd_rsp_data = 32'd0;
            rd_rsp_last = 1'b0;
            rd_active_addr = '0;
            rd_active_words = 0;
            rd_word_index = 0;
            rd_latency_countdown = 0;
            rd_gap_countdown = 0;
            rd_active_is_cache = 1'b0;
            rd_active_kind = 1'b0;
            wr_active = 1'b0;
            wr_done_countdown = 0;
            wr_done = 1'b0;
            wr_error = 1'b0;
            rd_req_stall_active = 1'b0;
            wr_req_stall_active = 1'b0;
            wr_data_stall_active = 1'b0;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            clear_run_scoreboard();
        end
        else begin
            cycle_count <= cycle_count + 1;
            wr_done <= 1'b0;

            if (done) begin
                done_seen_count <= done_seen_count + 1;
            end

            if (rd_req_valid && !rd_req_ready) begin
                check_rd_req_stability();
                read_req_stall_cycles = read_req_stall_cycles + 1;
                rd_req_stall_active = 1'b1;
                stalled_rd_req_addr = rd_req_addr;
                stalled_rd_req_len = rd_req_len_bytes;
            end
            else begin
                rd_req_stall_active = 1'b0;
            end

            if (wr_req_valid && !wr_req_ready) begin
                check_wr_req_stability();
                write_req_stall_cycles = write_req_stall_cycles + 1;
                wr_req_stall_active = 1'b1;
                stalled_wr_req_addr = wr_req_addr;
                stalled_wr_req_len = wr_req_len_bytes;
            end
            else begin
                wr_req_stall_active = 1'b0;
            end

            if (wr_data_valid && !wr_data_ready) begin
                check_wr_data_stability();
                write_data_stall_cycles = write_data_stall_cycles + 1;
                wr_data_stall_active = 1'b1;
                stalled_wr_data = wr_data;
                stalled_wr_data_last = wr_data_last;
            end
            else begin
                wr_data_stall_active = 1'b0;
            end

            if (rd_req_valid && rd_req_ready) begin
                if ((rd_req_len_bytes == 0) || ((rd_req_len_bytes % 4) != 0)) begin
                    fail_once("FAIL: invalid read request length");
                end
                check_cache_read_request(rd_req_addr);
                rd_active = 1'b1;
                rd_active_addr = rd_req_addr;
                rd_active_words = rd_req_len_bytes / 4;
                rd_word_index = 0;
                rd_latency_countdown = read_latency_pattern(read_req_accept_count);
                rd_gap_countdown = 0;
                rd_active_is_cache = is_cache_addr(rd_req_addr);
                rd_active_kind =
                    is_cache_addr(rd_req_addr) &&
                    ((((rd_req_addr - CACHE_BASE_ADDR) >> 2) >= (NUM_KV_HEADS * MAX_CONTEXT * HEAD_DIM)));
                read_latency_sum = read_latency_sum + read_latency_pattern(read_req_accept_count);
                if (read_latency_pattern(read_req_accept_count) > read_latency_max) begin
                    read_latency_max = read_latency_pattern(read_req_accept_count);
                end
                read_req_accept_count = read_req_accept_count + 1;
            end

            if (rd_active) begin
                if (rd_latency_countdown > 0) begin
                    rd_latency_countdown = rd_latency_countdown - 1;
                end
                else if (rd_rsp_valid && !rd_rsp_ready) begin
                end
                else if (rd_rsp_valid && rd_rsp_ready) begin
                    check_cache_read_response(rd_active_addr + (rd_word_index * 4));
                    read_rsp_accept_count = read_rsp_accept_count + 1;
                    if (rd_rsp_last) begin
                        rd_rsp_valid <= 1'b0;
                        rd_rsp_last <= 1'b0;
                        rd_active = 1'b0;
                    end
                    else begin
                        rd_word_index = rd_word_index + 1;
                        rd_gap_countdown = read_gap_pattern(rd_word_index);
                        rd_rsp_valid <= 1'b0;
                        rd_rsp_last <= 1'b0;
                    end
                end
                else if (!rd_rsp_valid) begin
                    if (rd_gap_countdown > 0) begin
                        rd_gap_countdown = rd_gap_countdown - 1;
                    end
                    else begin
                        rd_rsp_valid <= 1'b1;
                        rd_rsp_data <= read_word(rd_active_addr + (rd_word_index * 4));
                        rd_rsp_last <= (rd_word_index == (rd_active_words - 1));
                    end
                end
            end
            else begin
                rd_rsp_valid <= 1'b0;
                rd_rsp_last <= 1'b0;
            end

            if (wr_req_valid && wr_req_ready) begin
                if ((wr_req_addr !== ATTN_OUT_ADDR) || (wr_req_len_bytes != (Q_COUNT * 4))) begin
                    fail_once("FAIL: attention output write request mismatch");
                end
                wr_active = 1'b1;
                write_req_accept_count = write_req_accept_count + 1;
            end

            if (wr_data_valid && wr_data_ready) begin
                if (write_data_accept_count >= TOTAL_WRITES) begin
                    fail_once("FAIL: extra attention output write data");
                end
                else begin
                    if (wr_data !== expected_attn_out_mem[write_data_accept_count]) begin
                        fail_once("FAIL: attention output write data mismatch");
                    end
                    if (wr_data_last !== (write_data_accept_count == (TOTAL_WRITES - 1))) begin
                        fail_once("FAIL: attention output write last mismatch");
                    end
                end
                write_data_accept_count = write_data_accept_count + 1;
                if (wr_data_last) begin
                    wr_active = 1'b0;
                    wr_done_countdown = 3;
                end
            end

            if (wr_done_countdown > 0) begin
                wr_done_countdown = wr_done_countdown - 1;
                if (wr_done_countdown == 1) begin
                    wr_done <= 1'b1;
                end
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,0x%08h,%0d,%0d,0x%016h,%0d,%0d,%0d,0x%08h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    busy,
                    done,
                    error,
                    saturation,
                    state_debug,
                    read_slot_debug,
                    rd_req_valid,
                    rd_req_addr,
                    rd_req_len_bytes,
                    rd_req_ready,
                    rd_rsp_valid,
                    rd_rsp_ready,
                    rd_rsp_last,
                    rd_rsp_data,
                    wr_req_valid,
                    wr_req_ready,
                    wr_req_addr,
                    wr_req_len_bytes,
                    wr_data_valid,
                    wr_data_ready,
                    wr_data,
                    wr_data_last,
                    wr_done,
                    score_count,
                    k_read_count,
                    v_read_count,
                    attn_out_capture_count,
                    attn_out_write_word_count,
                    mem_read_burst_count,
                    mem_write_req_count
                );
            end
        end
    end

    task run_once(input logic corrupt_descriptor, input integer timeout_cycles);
        begin
            clear_run_scoreboard();
            corrupt_exp_dtype = corrupt_descriptor;
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < timeout_cycles)) begin
                @(negedge clk);
            end
            if (done != 1'b1) begin
                fail_once("FAIL: timed out waiting for qmap_attention_score_value_compute_path done");
                $finish(1);
            end
            @(negedge clk);
        end
    endtask

    initial begin : main_test
        vector_dir = "FPGA_Project/sim/vectors";
        tracefile = "FPGA_Project/sim/qmap_attention_score_value_compute_path_trace.csv";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        rst_n = 1'b0;
        start = 1'b0;
        corrupt_exp_dtype = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        trace_fd = 0;

        $readmemh({vector_dir, "/qmap_attention_score_value_image_words32.hex"}, qmap_image_mem);
        $readmemh({vector_dir, "/attention_score_stage_real_k_cache.hex"}, k_cache_mem);
        $readmemh({vector_dir, "/attention_softmax_value_stage_real_v_cache.hex"}, v_cache_mem);
        $readmemh({vector_dir, "/qmap_attention_score_value_attn_out_expected_words32.hex"}, expected_attn_out_mem);

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,busy,done,error,saturation,state,read_slot,rd_req_valid,rd_req_addr,rd_req_len,rd_req_ready,rd_rsp_valid,rd_rsp_ready,rd_rsp_last,rd_rsp_data,wr_req_valid,wr_req_ready,wr_req_addr,wr_req_len,wr_data_valid,wr_data_ready,wr_data,wr_data_last,wr_done,score_count,k_read_count,v_read_count,attn_capture_count,attn_write_count,mem_read_burst_count,mem_write_req_count\n"
        );

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_once(1'b0, 700000);

        normal_read_req_count = read_req_accept_count;
        normal_read_rsp_count = read_rsp_accept_count;
        normal_write_req_count = write_req_accept_count;
        normal_write_data_count = write_data_accept_count;
        normal_k_req_count = k_req_accept_count;
        normal_v_req_count = v_req_accept_count;
        normal_k_rsp_count = k_rsp_accept_count;
        normal_v_rsp_count = v_rsp_accept_count;
        normal_attn_write_count = attn_out_write_word_count;
        normal_read_req_stall_cycles = read_req_stall_cycles;
        normal_write_req_stall_cycles = write_req_stall_cycles;
        normal_write_data_stall_cycles = write_data_stall_cycles;
        normal_read_latency_max = read_latency_max;
        normal_read_latency_sum = read_latency_sum;
        normal_done_cycle = cycle_count;

        if (error) begin
            fail_once("FAIL: normal run asserted error");
        end
        if (saturation) begin
            fail_once("FAIL: normal run asserted saturation");
        end
        if (busy) begin
            fail_once("FAIL: busy still asserted when normal run done");
        end
        if (normal_k_req_count != TOTAL_K_READS) begin
            fail_once("FAIL: K read request count mismatch");
        end
        if (normal_v_req_count != TOTAL_V_READS) begin
            fail_once("FAIL: V read request count mismatch");
        end
        if (normal_k_rsp_count != TOTAL_K_READS) begin
            fail_once("FAIL: K read response count mismatch");
        end
        if (normal_v_rsp_count != TOTAL_V_READS) begin
            fail_once("FAIL: V read response count mismatch");
        end
        if (normal_write_req_count != 1) begin
            fail_once("FAIL: write request count mismatch");
        end
        if (normal_write_data_count != TOTAL_WRITES) begin
            fail_once("FAIL: write data count mismatch");
        end
        if (score_count != (NUM_Q_HEADS * CACHE_LENGTH)) begin
            fail_once("FAIL: score_count mismatch");
        end
        if (k_read_count != TOTAL_K_READS) begin
            fail_once("FAIL: DUT K read count mismatch");
        end
        if (v_read_count != TOTAL_V_READS) begin
            fail_once("FAIL: DUT V read count mismatch");
        end
        if (attn_out_capture_count != Q_COUNT) begin
            fail_once("FAIL: attn_out capture count mismatch");
        end
        if (normal_attn_write_count != Q_COUNT) begin
            fail_once("FAIL: attn_out write count mismatch");
        end
        if (normal_read_req_stall_cycles < 100) begin
            fail_once("FAIL: read request backpressure coverage too weak");
        end
        if (normal_write_data_stall_cycles < 100) begin
            fail_once("FAIL: write data backpressure coverage too weak");
        end
        if (normal_read_latency_max < 8) begin
            fail_once("FAIL: read latency coverage too weak");
        end

        run_once(1'b1, cycle_count + 5000);

        invalid_write_req_count = write_req_accept_count;
        invalid_write_data_count = write_data_accept_count;
        invalid_done_cycle = cycle_count;

        if (!error) begin
            fail_once("FAIL: invalid descriptor run did not assert error");
        end
        if (invalid_write_req_count != 0) begin
            fail_once("FAIL: invalid descriptor run issued write request");
        end
        if (invalid_write_data_count != 0) begin
            fail_once("FAIL: invalid descriptor run issued write data");
        end

        $fclose(trace_fd);

        $display("qmap_attention_score_value_compute_path real Layer 0 current-token test");
        $display("  K reads accepted       = %0d", normal_k_req_count);
        $display("  V reads accepted       = %0d", normal_v_req_count);
        $display("  attn_out writes        = %0d", normal_write_data_count);
        $display("  read req/rsp accepted  = %0d / %0d", normal_read_req_count, normal_read_rsp_count);
        $display("  write req/data         = %0d / %0d", normal_write_req_count, normal_write_data_count);
        $display("  DUT score/K/V/out      = %0d / %0d / %0d / %0d",
                 score_count, k_read_count, v_read_count, normal_attn_write_count);
        $display("  request stalls rd/wr   = %0d / %0d", normal_read_req_stall_cycles, normal_write_req_stall_cycles);
        $display("  write data stalls      = %0d", normal_write_data_stall_cycles);
        $display("  read latency max/sum   = %0d / %0d", normal_read_latency_max, normal_read_latency_sum);
        $display("  done cycles normal/bad = %0d / %0d", normal_done_cycle, invalid_done_cycle);
        $display("  trace                  = %s", tracefile);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d qmap_attention_score_value_compute_path mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_attention_score_value_compute_path matched exact attn_out write-back and exact K/V cache reads.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
