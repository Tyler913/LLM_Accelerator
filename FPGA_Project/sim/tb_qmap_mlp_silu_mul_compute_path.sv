`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_mlp_silu_mul_compute_path;

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 6;
    localparam int FEATURES         = 3072;
    localparam int IN_WIDTH         = 24;
    localparam int OUT_WIDTH        = 24;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;

    localparam int MEM_DATA_BYTES   = MEM_DATA_WIDTH / 8;
    localparam int VECTOR_BYTES     = FEATURES * MEM_DATA_BYTES;
    localparam int VECTOR_BURSTS    = VECTOR_BYTES / MAX_READ_BYTES;
    localparam int LUT_ENTRIES      = 1025;
    localparam int LUT_BYTES        = LUT_ENTRIES * MEM_DATA_BYTES;
    localparam int LUT_BURSTS       = (LUT_BYTES + MAX_READ_BYTES - 1) / MAX_READ_BYTES;
    localparam int LUT_LAST_BYTES   = LUT_BYTES - ((LUT_BURSTS - 1) * MAX_READ_BYTES);
    localparam int QMAP_IMAGE_BYTES = 32'h0000_E000;
    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_QMAP_BASE_ADDR = `QMAP_MLP_SILU_MUL_BASE_ADDR;
    localparam int QMAP_WORDS       = QMAP_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESCRIPTOR_TABLE_OFFSET_BYTES = 32'h0100;
    localparam int DESC_DTYPE_WORD  = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int SLOT_GATE        = 1;
    localparam int SLOT_UP          = 2;
    localparam int SLOT_LUT         = 3;
    localparam int SLOT_HIDDEN      = 4;

    localparam int REGION_QMAP      = 0;
    localparam int REGION_GATE      = 1;
    localparam int REGION_UP        = 2;
    localparam int REGION_LUT       = 3;

    localparam int QMAP_READER_REQS  = 1 + DESCRIPTOR_SLOTS;
    localparam int QMAP_READER_WORDS = 16 + (DESCRIPTOR_SLOTS * DESCRIPTOR_WORDS);
    localparam int NORMAL_RD_REQS    = QMAP_READER_REQS + (2 * VECTOR_BURSTS) + LUT_BURSTS;
    localparam int NORMAL_RD_WORDS   = QMAP_READER_WORDS + (2 * FEATURES) + LUT_ENTRIES;
    localparam int TOTAL_RD_REQS     = NORMAL_RD_REQS + QMAP_READER_REQS;
    localparam int TOTAL_RD_WORDS    = NORMAL_RD_WORDS + QMAP_READER_WORDS;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] stage_input_count;
    logic [31 : 0] stage_output_count;
    logic [31 : 0] stage_cycle_count;
    logic [31 : 0] output_write_word_count;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;
    logic [7 : 0] state_debug;
    logic [31 : 0] read_slot_debug;

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
    logic [31 : 0] expected_hidden_mem [0 : FEATURES-1];

    string qmap_image_file;
    string expected_hidden_file;
    string tracefile;
    string wavefile;
    integer trace_fd;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer done_seen_count;
    integer last_done_cycle;
    integer normal_done_cycle;
    integer invalid_done_cycle;
    integer spurious_start_seen_busy;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_gate_req_count;
    integer mem_up_req_count;
    integer mem_lut_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer write_mismatch_count;
    integer hidden_max_abs_diff;

    integer normal_stage_input_count;
    integer normal_stage_output_count;
    integer normal_stage_cycle_count;
    integer normal_output_write_count;
    integer normal_dut_read_bursts;
    integer normal_dut_read_words;
    integer normal_dut_write_reqs;
    integer normal_dut_write_words;
    logic normal_error;
    logic normal_saturation;
    logic invalid_error;

    logic [ADDR_WIDTH-1 : 0] qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] gate_base_addr;
    logic [ADDR_WIDTH-1 : 0] up_base_addr;
    logic [ADDR_WIDTH-1 : 0] lut_base_addr;
    logic [ADDR_WIDTH-1 : 0] hidden_base_addr;

    logic read_active;
    integer active_read_region;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer read_gap_count;
    logic [31 : 0] current_read_word;

    logic write_active;
    integer active_write_index;
    integer active_write_words_left;
    integer write_done_delay;
    longint signed output_diff;

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
    logic stalled_wr_last;
    integer last_trace_stage_output_count;

    qmap_mlp_silu_mul_compute_path #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DESCRIPTOR_SLOTS (DESCRIPTOR_SLOTS),
        .FEATURES         (FEATURES),
        .IN_WIDTH         (IN_WIDTH),
        .OUT_WIDTH        (OUT_WIDTH),
        .MEM_DATA_WIDTH   (MEM_DATA_WIDTH),
        .MAX_READ_BYTES   (MAX_READ_BYTES)
    ) dut (
        .i_clk                    (clk),
        .i_rst_n                  (rst_n),
        .i_start                  (start),
        .i_qmap_base_addr         (qmap_base_addr),
        .o_busy                   (busy),
        .o_done                   (done),
        .o_error                  (error),
        .o_saturation             (saturation),
        .o_stage_input_count      (stage_input_count),
        .o_stage_output_count     (stage_output_count),
        .o_stage_cycle_count      (stage_cycle_count),
        .o_output_write_word_count(output_write_word_count),
        .o_mem_read_burst_count   (dut_read_burst_count),
        .o_mem_read_word_count    (dut_read_word_count),
        .o_mem_write_req_count    (dut_write_req_count),
        .o_mem_write_word_count   (dut_write_word_count),
        .o_state_debug            (state_debug),
        .o_read_slot_debug        (read_slot_debug),
        .o_mem_rd_req_valid       (mem_rd_req_valid),
        .i_mem_rd_req_ready       (mem_rd_req_ready),
        .o_mem_rd_req_addr        (mem_rd_req_addr),
        .o_mem_rd_req_len_bytes   (mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid       (mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready       (mem_rd_rsp_ready),
        .i_mem_rd_rsp_data        (mem_rd_rsp_data),
        .i_mem_rd_rsp_last        (mem_rd_rsp_last),
        .o_mem_wr_req_valid       (mem_wr_req_valid),
        .i_mem_wr_req_ready       (mem_wr_req_ready),
        .o_mem_wr_req_addr        (mem_wr_req_addr),
        .o_mem_wr_req_len_bytes   (mem_wr_req_len_bytes),
        .o_mem_wr_data            (mem_wr_data),
        .o_mem_wr_data_valid      (mem_wr_data_valid),
        .i_mem_wr_data_ready      (mem_wr_data_ready),
        .o_mem_wr_data_last       (mem_wr_data_last),
        .i_mem_wr_done            (mem_wr_done),
        .i_mem_wr_error           (mem_wr_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qmap_mlp_silu_mul_compute_path.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        if ($test$plusargs("dumpwaves")) begin
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
            $dumpvars(0, stage_input_count);
            $dumpvars(0, stage_output_count);
            $dumpvars(0, output_write_word_count);
        end
    end

    assign mem_rd_req_ready =
        (!read_active) &&
        (!mem_rd_rsp_valid) &&
        ((cycle_count % 17) != 3) &&
        (((cycle_count + mem_req_fire_count) % 31) != 11);
    assign mem_wr_req_ready =
        (!write_active) &&
        ((cycle_count % 13) != 5) &&
        (((cycle_count + mem_wr_req_count) % 19) != 7);
    assign mem_wr_data_ready =
        write_active &&
        ((cycle_count % 11) != 4) &&
        (((cycle_count + mem_wr_word_count_total) % 23) != 9);

    function automatic integer descriptor_word_index(input integer slot, input integer word_offset);
        begin
            descriptor_word_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] descriptor_base_addr(input integer slot);
        begin
            descriptor_base_addr = {
                qmap_mem[descriptor_word_index(slot, DESC_BASE_HI_WORD)],
                qmap_mem[descriptor_word_index(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input integer word_index);
        begin
            if ((word_index >= 0) && (word_index < QMAP_WORDS)) begin
                memory_word = qmap_mem[word_index];
            end
            else begin
                memory_word = 32'hBAD0_BAD0;
            end
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(expected_hidden_file, expected_hidden_mem);

            gate_base_addr = descriptor_base_addr(SLOT_GATE);
            up_base_addr = descriptor_base_addr(SLOT_UP);
            lut_base_addr = descriptor_base_addr(SLOT_LUT);
            hidden_base_addr = descriptor_base_addr(SLOT_HIDDEN);
        end
    endtask

    task check_read_request_stability;
        begin
            if (rd_req_stall_active) begin
                if ((mem_rd_req_addr !== stalled_rd_req_addr) ||
                    (mem_rd_req_len_bytes !== stalled_rd_req_len)) begin
                    if (print_count < 32) begin
                        $display("FAIL: read request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_response_stability;
        begin
            if (rd_rsp_stall_active) begin
                if ((mem_rd_rsp_data !== stalled_rd_rsp_data) ||
                    (mem_rd_rsp_last !== stalled_rd_rsp_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: read response changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_write_request_stability;
        begin
            if (wr_req_stall_active) begin
                if ((mem_wr_req_addr !== stalled_wr_req_addr) ||
                    (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    if (print_count < 32) begin
                        $display("FAIL: write request changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task check_write_data_stability;
        begin
            if (wr_data_stall_active) begin
                if ((mem_wr_data !== stalled_wr_data) ||
                    (mem_wr_data_last !== stalled_wr_last)) begin
                    if (print_count < 32) begin
                        $display("FAIL: write data changed while stalled at cycle %0d", cycle_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
            end
        end
    endtask

    task classify_and_check_read_request;
        integer reader_req_index;
        integer expected_lut_len;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            active_read_region = REGION_QMAP;
            active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
            active_words_left = mem_rd_req_len_bytes / MEM_DATA_BYTES;
            active_total_words = mem_rd_req_len_bytes / MEM_DATA_BYTES;

            if ((mem_rd_req_addr >= gate_base_addr) &&
                (mem_rd_req_addr < (gate_base_addr + VECTOR_BYTES))) begin
                active_read_region = REGION_GATE;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                expected_addr = gate_base_addr + (mem_gate_req_count * MAX_READ_BYTES);
                if ((mem_gate_req_count >= VECTOR_BURSTS) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: gate read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_gate_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_gate_req_count = mem_gate_req_count + 1;
            end
            else if ((mem_rd_req_addr >= up_base_addr) &&
                     (mem_rd_req_addr < (up_base_addr + VECTOR_BYTES))) begin
                active_read_region = REGION_UP;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                expected_addr = up_base_addr + (mem_up_req_count * MAX_READ_BYTES);
                if ((mem_up_req_count >= VECTOR_BURSTS) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES) ||
                    (mem_gate_req_count != VECTOR_BURSTS)) begin
                    $display("FAIL: up read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_up_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_up_req_count = mem_up_req_count + 1;
            end
            else if ((mem_rd_req_addr >= lut_base_addr) &&
                     (mem_rd_req_addr < (lut_base_addr + LUT_BYTES))) begin
                active_read_region = REGION_LUT;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                expected_addr = lut_base_addr + (mem_lut_req_count * MAX_READ_BYTES);
                expected_lut_len = (mem_lut_req_count == (LUT_BURSTS - 1)) ? LUT_LAST_BYTES : MAX_READ_BYTES;
                if ((mem_lut_req_count >= LUT_BURSTS) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != expected_lut_len) ||
                    (mem_up_req_count != VECTOR_BURSTS)) begin
                    $display("FAIL: LUT read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d expected_len=%0d",
                             mem_lut_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes, expected_lut_len);
                    mismatch_count = mismatch_count + 1;
                end
                mem_lut_req_count = mem_lut_req_count + 1;
            end
            else begin
                reader_req_index = mem_qmap_req_count % QMAP_READER_REQS;
                if (reader_req_index == 0) begin
                    expected_addr = qmap_base_addr;
                    if ((mem_rd_req_addr !== expected_addr) ||
                        (mem_rd_req_len_bytes != `QMAP_HEADER_FETCH_BYTES)) begin
                        $display("FAIL: QMAP header read mismatch addr=0x%016h expected=0x%016h len=%0d",
                                 mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                        mismatch_count = mismatch_count + 1;
                    end
                end
                else begin
                    expected_addr = qmap_base_addr + DESCRIPTOR_TABLE_OFFSET_BYTES +
                                    ((reader_req_index - 1) * `QMAP_DESCRIPTOR_BYTES);
                    if ((mem_rd_req_addr !== expected_addr) ||
                        (mem_rd_req_len_bytes != `QMAP_DESCRIPTOR_BYTES)) begin
                        $display("FAIL: QMAP descriptor read mismatch reader_idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                                 reader_req_index, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                        mismatch_count = mismatch_count + 1;
                    end
                end
                mem_qmap_req_count = mem_qmap_req_count + 1;
            end
        end
    endtask

    task check_write_request;
        begin
            if (write_active) begin
                $display("FAIL: write request while another write is active");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_addr !== hidden_base_addr) ||
                (mem_wr_req_len_bytes != VECTOR_BYTES)) begin
                $display("FAIL: hidden write request mismatch addr=0x%016h expected=0x%016h len=%0d",
                         mem_wr_req_addr, hidden_base_addr, mem_wr_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
            write_active = 1'b1;
            active_write_index = 0;
            active_write_words_left = FEATURES;
            write_done_delay = 0;
            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        longint signed expected_signed;
        longint signed actual_signed;
        begin
            if ((active_write_index < 0) || (active_write_index >= FEATURES)) begin
                $display("FAIL: hidden write index out of range idx=%0d", active_write_index);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                expected_signed = $signed(expected_hidden_mem[active_write_index]);
                actual_signed = $signed(mem_wr_data);
                output_diff = actual_signed - expected_signed;
                if (output_diff < 0) begin
                    output_diff = -output_diff;
                end
                if (output_diff > hidden_max_abs_diff) begin
                    hidden_max_abs_diff = output_diff;
                end
                if (mem_wr_data !== expected_hidden_mem[active_write_index]) begin
                    if (print_count < 32) begin
                        $display("FAIL: hidden mismatch idx=%0d actual=0x%08h expected=0x%08h diff=%0d",
                                 active_write_index, mem_wr_data, expected_hidden_mem[active_write_index], output_diff);
                        print_count = print_count + 1;
                    end
                    write_mismatch_count = write_mismatch_count + 1;
                    mismatch_count = mismatch_count + 1;
                end
                if (mem_wr_data_last != (active_write_words_left == 1)) begin
                    $display("FAIL: hidden write last mismatch idx=%0d last=%0d words_left=%0d",
                             active_write_index, mem_wr_data_last, active_write_words_left);
                    mismatch_count = mismatch_count + 1;
                end
                active_write_words_left = active_write_words_left - 1;
                if (active_write_words_left == 0) begin
                    write_active = 1'b0;
                    write_done_delay = 3;
                end
            end

            active_write_index = active_write_index + 1;
            mem_wr_word_count_total = mem_wr_word_count_total + 1;
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

    task wait_for_next_done;
        input integer prior_done_count;
        input integer timeout_delta;
        integer deadline;
        begin
            deadline = cycle_count + timeout_delta;
            while ((done_seen_count == prior_done_count) && (cycle_count < deadline)) begin
                @(negedge clk);
            end
            if (done_seen_count == prior_done_count) begin
                $display("FAIL: timed out waiting for qmap_mlp_silu_mul_compute_path done");
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end
            @(negedge clk);
        end
    endtask

    task run_success;
        integer prior_done_count;
        begin
            prior_done_count = done_seen_count;
            pulse_start();

            repeat (128) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
            end

            wait_for_next_done(prior_done_count, 500000);
            normal_done_cycle = last_done_cycle;
            normal_stage_input_count = stage_input_count;
            normal_stage_output_count = stage_output_count;
            normal_stage_cycle_count = stage_cycle_count;
            normal_output_write_count = output_write_word_count;
            normal_dut_read_bursts = dut_read_burst_count;
            normal_dut_read_words = dut_read_word_count;
            normal_dut_write_reqs = dut_write_req_count;
            normal_dut_write_words = dut_write_word_count;
            normal_error = error;
            normal_saturation = saturation;
        end
    endtask

    task run_invalid_bad_lut_dtype;
        integer prior_done_count;
        integer dtype_word_index;
        logic [31 : 0] saved_dtype;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            dtype_word_index = descriptor_word_index(SLOT_LUT, DESC_DTYPE_WORD);
            saved_dtype = qmap_mem[dtype_word_index];
            qmap_mem[dtype_word_index] = `QMAP_DTYPE_U32;

            pulse_start();
            wait_for_next_done(prior_done_count, 100000);
            invalid_done_cycle = last_done_cycle;
            invalid_error = error;

            if (error != 1'b1) begin
                $display("FAIL: invalid LUT dtype did not assert error");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count != saved_write_req_count) ||
                (mem_wr_word_count_total != saved_write_word_count)) begin
                $display("FAIL: invalid descriptor path wrote data req=%0d/%0d words=%0d/%0d",
                         mem_wr_req_count, saved_write_req_count,
                         mem_wr_word_count_total, saved_write_word_count);
                mismatch_count = mismatch_count + 1;
            end

            qmap_mem[dtype_word_index] = saved_dtype;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1;
            last_trace_stage_output_count <= -1;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                if ((last_done_cycle != -1) && ((cycle_count - last_done_cycle) == 1)) begin
                    $display("FAIL: adjacent done pulses at cycles %0d and %0d", last_done_cycle, cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
                last_done_cycle <= cycle_count;
            end

            if ((trace_fd != 0) &&
                (((mem_rd_req_valid == 1'b1) && (mem_rd_req_ready == 1'b1)) ||
                 ((mem_rd_rsp_valid == 1'b1) && (mem_rd_rsp_ready == 1'b1) && (mem_rd_rsp_last == 1'b1)) ||
                 ((mem_wr_req_valid == 1'b1) && (mem_wr_req_ready == 1'b1)) ||
                 ((mem_wr_data_valid == 1'b1) && (mem_wr_data_ready == 1'b1) && (mem_wr_data_last == 1'b1)) ||
                 (mem_wr_done == 1'b1) ||
                 (done == 1'b1) ||
                 (error == 1'b1) ||
                 (stage_output_count != last_trace_stage_output_count))) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    state_debug,
                    read_slot_debug,
                    mem_rd_req_addr,
                    mem_rd_req_len_bytes,
                    mem_rd_req_valid && mem_rd_req_ready,
                    mem_rd_rsp_valid && mem_rd_rsp_ready && mem_rd_rsp_last,
                    mem_wr_req_addr,
                    mem_wr_req_len_bytes,
                    mem_wr_req_valid && mem_wr_req_ready,
                    mem_wr_data_valid && mem_wr_data_ready,
                    mem_wr_data_last,
                    stage_input_count,
                    stage_output_count,
                    stage_cycle_count,
                    output_write_word_count,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    mem_wr_done
                );
                last_trace_stage_output_count <= stage_output_count;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            read_active <= 1'b0;
            active_read_region <= REGION_QMAP;
            active_read_index <= 0;
            active_words_left <= 0;
            active_total_words <= 0;
            read_gap_count <= 0;
            rd_req_stall_active <= 1'b0;
            rd_rsp_stall_active <= 1'b0;
        end
        else begin
            if (mem_rd_req_valid && !mem_rd_req_ready) begin
                check_read_request_stability();
                if (!rd_req_stall_active) begin
                    rd_req_stall_active <= 1'b1;
                    stalled_rd_req_addr <= mem_rd_req_addr;
                    stalled_rd_req_len <= mem_rd_req_len_bytes;
                end
            end
            else begin
                rd_req_stall_active <= 1'b0;
            end

            if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                check_response_stability();
                if (!rd_rsp_stall_active) begin
                    rd_rsp_stall_active <= 1'b1;
                    stalled_rd_rsp_data <= mem_rd_rsp_data;
                    stalled_rd_rsp_last <= mem_rd_rsp_last;
                end
            end
            else begin
                rd_rsp_stall_active <= 1'b0;
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                classify_and_check_read_request();
                read_active <= 1'b1;
                read_gap_count <= mem_req_fire_count % 2;
                mem_req_fire_count = mem_req_fire_count + 1;
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                mem_rsp_fire_count = mem_rsp_fire_count + 1;
                if (active_words_left <= 1) begin
                    mem_rd_rsp_valid <= 1'b0;
                    read_active <= 1'b0;
                    active_words_left <= 0;
                    mem_rd_rsp_last <= 1'b0;
                end
                else begin
                    mem_rd_rsp_valid <= 1'b0;
                    active_read_index <= active_read_index + 1;
                    active_words_left <= active_words_left - 1;
                    read_gap_count <= ((active_total_words - active_words_left) % 19 == 0) ? 1 : 0;
                end
            end
            else if (read_active && !mem_rd_rsp_valid) begin
                if (read_gap_count > 0) begin
                    read_gap_count <= read_gap_count - 1;
                end
                else begin
                    current_read_word = memory_word(active_read_index);
                    mem_rd_rsp_data <= current_read_word;
                    mem_rd_rsp_last <= (active_words_left == 1);
                    mem_rd_rsp_valid <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            write_active <= 1'b0;
            active_write_index <= 0;
            active_write_words_left <= 0;
            write_done_delay <= 0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            wr_req_stall_active <= 1'b0;
            wr_data_stall_active <= 1'b0;
        end
        else begin
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (mem_wr_req_valid && !mem_wr_req_ready) begin
                check_write_request_stability();
                if (!wr_req_stall_active) begin
                    wr_req_stall_active <= 1'b1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
            end
            else begin
                wr_req_stall_active <= 1'b0;
            end

            if (mem_wr_data_valid && !mem_wr_data_ready) begin
                check_write_data_stability();
                if (!wr_data_stall_active) begin
                    wr_data_stall_active <= 1'b1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_last <= mem_wr_data_last;
                end
            end
            else begin
                wr_data_stall_active <= 1'b0;
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                check_write_request();
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                if (!write_active) begin
                    $display("FAIL: write data without active write request");
                    mismatch_count = mismatch_count + 1;
                end
                else begin
                    check_write_word();
                end
            end

            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) begin
                    mem_wr_done <= 1'b1;
                end
            end
        end
    end

    initial begin : main_test
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_mlp_silu_mul_image_words32.hex";
        expected_hidden_file = "FPGA_Project/sim/vectors/qmap_mlp_silu_mul_expected_hidden_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_mlp_silu_mul_compute_path_trace.csv";
        qmap_base_addr = DEFAULT_QMAP_BASE_ADDR;
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("expected_hidden=%s", expected_hidden_file)) begin
        end
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end
        if ($value$plusargs("qmap_base=%h", qmap_base_addr)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,start,busy,done,error,saturation,state,read_slot,rd_addr,rd_len,rd_req_fire,rd_rsp_last_fire,wr_addr,wr_len,wr_req_fire,wr_data_fire,wr_last,stage_in_count,stage_out_count,stage_cycles,output_write_count,rd_bursts,rd_words,wr_reqs,wr_words,wr_done\n"
        );

        load_vectors();

        start = 1'b0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        spurious_start_seen_busy = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_qmap_req_count = 0;
        mem_gate_req_count = 0;
        mem_up_req_count = 0;
        mem_lut_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        write_mismatch_count = 0;
        hidden_max_abs_diff = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        normal_stage_input_count = 0;
        normal_stage_output_count = 0;
        normal_stage_cycle_count = 0;
        normal_output_write_count = 0;
        normal_dut_read_bursts = 0;
        normal_dut_read_words = 0;
        normal_dut_write_reqs = 0;
        normal_dut_write_words = 0;
        normal_error = 1'b0;
        normal_saturation = 1'b0;
        invalid_error = 1'b0;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_success();
        run_invalid_bad_lut_dtype();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_mlp_silu_mul_compute_path descriptor-backed MLP SiLU/multiply test");
        $display("  qmap base                 = 0x%016h", qmap_base_addr);
        $display("  stage in/out normal       = %0d / %0d", normal_stage_input_count, normal_stage_output_count);
        $display("  stage cycles              = %0d", normal_stage_cycle_count);
        $display("  hidden words written      = %0d", normal_output_write_count);
        $display("  read req/rsp fires        = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap/gate/up/lut reqs     = %0d / %0d / %0d / %0d",
                 mem_qmap_req_count, mem_gate_req_count, mem_up_req_count, mem_lut_req_count);
        $display("  write reqs/words          = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  dut normal rd/wr          = %0d/%0d reads, %0d/%0d writes",
                 normal_dut_read_bursts, normal_dut_read_words,
                 normal_dut_write_reqs, normal_dut_write_words);
        $display("  write mismatches          = %0d", write_mismatch_count);
        $display("  max_abs hidden            = %0d", hidden_max_abs_diff);
        $display("  spurious start covered    = %0d", spurious_start_seen_busy);
        $display("  invalid descriptor error  = %0d", invalid_error);
        $display("  done cycles normal/bad    = %0d / %0d", normal_done_cycle, invalid_done_cycle);
        $display("  trace                     = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_error) begin
            $display("FAIL: error asserted on valid MLP SiLU/multiply packet");
            mismatch_count = mismatch_count + 1;
        end
        if (normal_saturation) begin
            $display("FAIL: unexpected MLP SiLU/multiply saturation");
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_stage_input_count != FEATURES) || (normal_stage_output_count != FEATURES)) begin
            $display("FAIL: stage counters mismatch in=%0d out=%0d expected=%0d",
                     normal_stage_input_count, normal_stage_output_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_output_write_count != FEATURES) begin
            $display("FAIL: hidden write counter mismatch actual=%0d expected=%0d",
                     normal_output_write_count, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_qmap_req_count != (2 * QMAP_READER_REQS)) ||
            (mem_gate_req_count != VECTOR_BURSTS) ||
            (mem_up_req_count != VECTOR_BURSTS) ||
            (mem_lut_req_count != LUT_BURSTS)) begin
            $display("FAIL: read request class counts mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_req_fire_count != TOTAL_RD_REQS) ||
            (mem_rsp_fire_count != TOTAL_RD_WORDS)) begin
            $display("FAIL: total read count mismatch req=%0d/%0d rsp=%0d/%0d",
                     mem_req_fire_count, TOTAL_RD_REQS, mem_rsp_fire_count, TOTAL_RD_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_wr_req_count != 1) || (mem_wr_word_count_total != FEATURES)) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=1/%0d",
                     mem_wr_req_count, mem_wr_word_count_total, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_dut_read_bursts != NORMAL_RD_REQS) ||
            (normal_dut_read_words != NORMAL_RD_WORDS) ||
            (normal_dut_write_reqs != 1) ||
            (normal_dut_write_words != FEATURES)) begin
            $display("FAIL: DUT normal counters mismatch rd=%0d/%0d wr=%0d/%0d expected rd=%0d/%0d wr=1/%0d",
                     normal_dut_read_bursts, normal_dut_read_words,
                     normal_dut_write_reqs, normal_dut_write_words,
                     NORMAL_RD_REQS, NORMAL_RD_WORDS, FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (hidden_max_abs_diff != 0)) begin
            $display("FAIL: exact hidden write-back expected, mismatches=%0d hidden_abs=%0d",
                     write_mismatch_count, hidden_max_abs_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end
        if (invalid_error != 1'b1) begin
            $display("FAIL: invalid descriptor path did not report error");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_mlp_silu_mul_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_mlp_silu_mul_compute_path matched exact hidden write-back and QMAP/LUT reads.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
