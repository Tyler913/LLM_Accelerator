`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_mlp_residual_add_compute_path;

    localparam int ADDR_WIDTH           = 64;
    localparam int DESCRIPTOR_SLOTS     = 5;
    localparam int INPUT_SIZE           = 1024;
    localparam int POST_ATTENTION_WIDTH = 24;
    localparam int DOWN_WIDTH           = 24;
    localparam int OUT_WIDTH            = 24;
    localparam int MEM_DATA_WIDTH       = 32;
    localparam int MAX_READ_BYTES       = 1024;
    localparam int MEM_DATA_BYTES       = MEM_DATA_WIDTH / 8;
    localparam int VECTOR_BYTES         = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int VECTOR_WORDS         = INPUT_SIZE;
    localparam int VECTOR_BURSTS        = VECTOR_BYTES / MAX_READ_BYTES;
    localparam int QMAP_IMAGE_BYTES     = 32'h0000_5000;
    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_QMAP_BASE_ADDR = `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR;
    localparam logic [ADDR_WIDTH-1 : 0] RUNTIME_OUTPUT_BASE_ADDR =
        `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR + 64'h0000_0000_0001_0000;
    localparam int QMAP_WORDS           = QMAP_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DESCRIPTOR_WORDS     = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESCRIPTOR_TABLE_OFFSET_BYTES = 32'h0100;
    localparam int DESC_DTYPE_WORD      = 2;
    localparam int DESC_BASE_LO_WORD    = 8;
    localparam int DESC_BASE_HI_WORD    = 9;
    localparam int SLOT_POST_ATTN       = 1;
    localparam int SLOT_DOWN            = 2;
    localparam int SLOT_OUTPUT          = 3;

    localparam int REGION_QMAP          = 0;
    localparam int REGION_POST_ATTN     = 1;
    localparam int REGION_DOWN          = 2;

    localparam int QMAP_READER_REQS     = 1 + DESCRIPTOR_SLOTS;
    localparam int QMAP_READER_WORDS    = 16 + (DESCRIPTOR_SLOTS * DESCRIPTOR_WORDS);
    localparam int NORMAL_RD_REQS       = QMAP_READER_REQS + (2 * VECTOR_BURSTS);
    localparam int NORMAL_RD_WORDS      = QMAP_READER_WORDS + (2 * VECTOR_WORDS);
    localparam int INVALID_RD_REQS      = QMAP_READER_REQS;
    localparam int INVALID_RD_WORDS     = QMAP_READER_WORDS;
    localparam int PROTOCOL_RD_REQS     = QMAP_READER_REQS + 1;
    localparam int PROTOCOL_RD_WORDS    = QMAP_READER_WORDS + 1;
    localparam int INVALID_OVERRIDE_RUNS = 2;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] output_count;
    logic [31 : 0] stage_cycle_count;
    logic [31 : 0] output_write_word_count;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;
    logic [7 : 0] state_debug;
    logic [1 : 0] read_slot_debug;
    logic output_base_override_valid;
    logic [ADDR_WIDTH-1 : 0] output_base_override_addr;
    logic [ADDR_WIDTH-1 : 0] effective_output_base_addr;

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
    logic [31 : 0] expected_layer_mem [0 : VECTOR_WORDS-1];
    logic [31 : 0] expected_saturation_mem [0 : 0];

    string vector_dir;
    string prefix;
    string qmap_image_file;
    string expected_layer_file;
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
    integer protocol_done_cycle;
    integer spurious_start_seen_busy;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_post_req_count;
    integer mem_down_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer write_mismatch_count;
    integer layer_max_abs_diff;

    integer normal_output_count;
    integer normal_stage_cycle_count;
    integer normal_output_write_count;
    integer normal_dut_read_bursts;
    integer normal_dut_read_words;
    integer normal_dut_write_reqs;
    integer normal_dut_write_words;
    logic normal_error;
    logic normal_saturation;
    logic invalid_error;
    logic protocol_error;
    integer protocol_dut_read_bursts;
    integer protocol_dut_read_words;
    integer runtime_write_req_delta;
    integer runtime_write_word_delta;
    integer invalid_override_error_count;
    integer invalid_override_write_req_delta;
    integer invalid_override_write_word_delta;

    logic [ADDR_WIDTH-1 : 0] qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] post_base_addr;
    logic [ADDR_WIDTH-1 : 0] down_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_base_addr;
    logic [ADDR_WIDTH-1 : 0] expected_output_base_addr;

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

    logic force_bad_payload_last;
    logic bad_payload_last_fired;
    integer last_trace_state;
    integer last_trace_write_count;

    qmap_mlp_residual_add_compute_path #(
        .ADDR_WIDTH          (ADDR_WIDTH),
        .DESCRIPTOR_SLOTS    (DESCRIPTOR_SLOTS),
        .INPUT_SIZE          (INPUT_SIZE),
        .POST_ATTENTION_WIDTH(POST_ATTENTION_WIDTH),
        .DOWN_WIDTH          (DOWN_WIDTH),
        .OUT_WIDTH           (OUT_WIDTH),
        .MEM_DATA_WIDTH      (MEM_DATA_WIDTH),
        .MAX_READ_BYTES      (MAX_READ_BYTES)
    ) dut (
        .i_clk                    (clk),
        .i_rst_n                  (rst_n),
        .i_start                  (start),
        .i_qmap_base_addr         (qmap_base_addr),
        .i_output_base_override_valid(output_base_override_valid),
        .i_output_base_override_addr(output_base_override_addr),
        .o_busy                   (busy),
        .o_done                   (done),
        .o_error                  (error),
        .o_effective_output_base_addr(effective_output_base_addr),
        .o_saturation             (saturation),
        .o_output_count           (output_count),
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
        wavefile = "FPGA_Project/wave/qmap_mlp_residual_add_compute_path.vcd";
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
            $dumpvars(0, output_count);
            $dumpvars(0, stage_cycle_count);
            $dumpvars(0, output_write_word_count);
        end
    end

    assign mem_rd_req_ready =
        (!read_active) &&
        (!mem_rd_rsp_valid) &&
        ((cycle_count % 7) != 3) &&
        (((cycle_count + mem_req_fire_count) % 17) != 5);
    assign mem_wr_req_ready =
        (!write_active) &&
        ((cycle_count % 11) != 4) &&
        (((cycle_count + mem_wr_req_count) % 19) != 8);
    assign mem_wr_data_ready =
        write_active &&
        ((cycle_count % 6) != 2) &&
        (((cycle_count + mem_wr_word_count_total) % 23) != 10);

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

    function automatic logic [31 : 0] read_qmap_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer word_index;
        begin
            word_index = (addr - qmap_base_addr) >> 2;
            if ((addr < qmap_base_addr) || (addr >= (qmap_base_addr + QMAP_IMAGE_BYTES)) ||
                (word_index < 0) || (word_index >= QMAP_WORDS)) begin
                if (print_count < 32) begin
                    $display("FAIL: read address out of QMAP image addr=0x%016h", addr);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
                read_qmap_word = 32'hBAD0_BAD0;
            end
            else begin
                read_qmap_word = qmap_mem[word_index];
            end
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(expected_layer_file, expected_layer_mem);
            $readmemh({vector_dir, "/", prefix, "_residual_saturation.hex"}, expected_saturation_mem);

            post_base_addr = descriptor_base_addr(SLOT_POST_ATTN);
            down_base_addr = descriptor_base_addr(SLOT_DOWN);
            output_base_addr = descriptor_base_addr(SLOT_OUTPUT);
            expected_output_base_addr = output_base_addr;
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
        integer post_req_index;
        integer down_req_index;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            active_read_region = REGION_QMAP;
            active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
            active_words_left = mem_rd_req_len_bytes / MEM_DATA_BYTES;
            active_total_words = mem_rd_req_len_bytes / MEM_DATA_BYTES;

            if ((mem_rd_req_addr >= post_base_addr) &&
                (mem_rd_req_addr < (post_base_addr + VECTOR_BYTES))) begin
                active_read_region = REGION_POST_ATTN;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                post_req_index = mem_post_req_count % VECTOR_BURSTS;
                expected_addr = post_base_addr + (post_req_index * MAX_READ_BYTES);
                if ((post_req_index >= VECTOR_BURSTS) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: post-attn read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_post_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_post_req_count = mem_post_req_count + 1;
            end
            else if ((mem_rd_req_addr >= down_base_addr) &&
                     (mem_rd_req_addr < (down_base_addr + VECTOR_BYTES))) begin
                active_read_region = REGION_DOWN;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                down_req_index = mem_down_req_count % VECTOR_BURSTS;
                expected_addr = down_base_addr + (down_req_index * MAX_READ_BYTES);
                if ((down_req_index >= VECTOR_BURSTS) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: down read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_down_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_down_req_count = mem_down_req_count + 1;
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

            if ((mem_rd_req_addr[1:0] != 2'b00) || ((mem_rd_req_len_bytes % 4) != 0)) begin
                $display("FAIL: unaligned read request addr=0x%016h len=%0d", mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_write_request;
        begin
            if (write_active) begin
                $display("FAIL: write request while another write is active");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_addr !== expected_output_base_addr) ||
                (mem_wr_req_len_bytes != VECTOR_BYTES)) begin
                $display("FAIL: layer_out write request mismatch addr=0x%016h expected=0x%016h len=%0d",
                         mem_wr_req_addr, expected_output_base_addr, mem_wr_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
            write_active = 1'b1;
            active_write_index = 0;
            active_write_words_left = VECTOR_WORDS;
            write_done_delay = 0;
            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        longint signed expected_signed;
        longint signed actual_signed;
        begin
            if ((active_write_index < 0) || (active_write_index >= VECTOR_WORDS)) begin
                $display("FAIL: layer_out write index out of range idx=%0d", active_write_index);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                expected_signed = $signed(expected_layer_mem[active_write_index]);
                actual_signed = $signed(mem_wr_data);
                output_diff = actual_signed - expected_signed;
                if (output_diff < 0) begin
                    output_diff = -output_diff;
                end
                if (output_diff > layer_max_abs_diff) begin
                    layer_max_abs_diff = output_diff;
                end
                if (mem_wr_data !== expected_layer_mem[active_write_index]) begin
                    if (print_count < 32) begin
                        $display("FAIL: layer_out mismatch idx=%0d actual=0x%08h expected=0x%08h diff=%0d",
                                 active_write_index, mem_wr_data, expected_layer_mem[active_write_index], output_diff);
                        print_count = print_count + 1;
                    end
                    write_mismatch_count = write_mismatch_count + 1;
                    mismatch_count = mismatch_count + 1;
                end
                if (mem_wr_data_last != (active_write_words_left == 1)) begin
                    $display("FAIL: layer_out write last mismatch idx=%0d last=%0d words_left=%0d",
                             active_write_index, mem_wr_data_last, active_write_words_left);
                    mismatch_count = mismatch_count + 1;
                end
                active_write_words_left = active_write_words_left - 1;
                if (active_write_words_left == 0) begin
                    write_active = 1'b0;
                    write_done_delay = 4;
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
                $display("FAIL: timed out waiting for qmap_mlp_residual_add_compute_path done");
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end
            @(negedge clk);
        end
    endtask

    task run_success;
        integer prior_done_count;
        begin
            output_base_override_valid = 1'b0;
            output_base_override_addr = '0;
            expected_output_base_addr = output_base_addr;
            prior_done_count = done_seen_count;
            pulse_start();

            repeat (96) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
            end

            wait_for_next_done(prior_done_count, 200000);
            normal_done_cycle = last_done_cycle;
            normal_output_count = output_count;
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

    task run_runtime_override;
        integer prior_done_count;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            expected_output_base_addr = RUNTIME_OUTPUT_BASE_ADDR;
            output_base_override_valid = 1'b1;
            output_base_override_addr = RUNTIME_OUTPUT_BASE_ADDR;
            pulse_start();

            // The accepted transaction owns the start-time address even if the
            // live configuration pins change while descriptor reads are active.
            output_base_override_valid = 1'b0;
            output_base_override_addr = RUNTIME_OUTPUT_BASE_ADDR + 64'd2;
            wait_for_next_done(prior_done_count, 200000);

            runtime_write_req_delta = mem_wr_req_count - saved_write_req_count;
            runtime_write_word_delta = mem_wr_word_count_total - saved_write_word_count;
            if (error) begin
                $display("FAIL: runtime output override run reported error");
                mismatch_count = mismatch_count + 1;
            end
            if (effective_output_base_addr !== RUNTIME_OUTPUT_BASE_ADDR) begin
                $display("FAIL: runtime effective output base mismatch actual=0x%016h expected=0x%016h",
                         effective_output_base_addr, RUNTIME_OUTPUT_BASE_ADDR);
                mismatch_count = mismatch_count + 1;
            end
            if ((runtime_write_req_delta != 1) ||
                (runtime_write_word_delta != VECTOR_WORDS)) begin
                $display("FAIL: runtime write delta mismatch req=%0d words=%0d expected=1/%0d",
                         runtime_write_req_delta, runtime_write_word_delta, VECTOR_WORDS);
                mismatch_count = mismatch_count + 1;
            end

            output_base_override_valid = 1'b0;
            output_base_override_addr = '0;
            expected_output_base_addr = output_base_addr;
        end
    endtask

    task run_invalid_runtime_override(input logic [ADDR_WIDTH-1 : 0] bad_addr);
        integer prior_done_count;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            output_base_override_valid = 1'b1;
            output_base_override_addr = bad_addr;
            pulse_start();
            output_base_override_valid = 1'b0;
            output_base_override_addr = RUNTIME_OUTPUT_BASE_ADDR;
            wait_for_next_done(prior_done_count, 80000);

            invalid_override_write_req_delta =
                invalid_override_write_req_delta +
                (mem_wr_req_count - saved_write_req_count);
            invalid_override_write_word_delta =
                invalid_override_write_word_delta +
                (mem_wr_word_count_total - saved_write_word_count);
            if (!error) begin
                $display("FAIL: invalid runtime output override 0x%016h did not report error", bad_addr);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                invalid_override_error_count = invalid_override_error_count + 1;
            end
            if (effective_output_base_addr !== bad_addr) begin
                $display("FAIL: invalid runtime output base was not latched actual=0x%016h expected=0x%016h",
                         effective_output_base_addr, bad_addr);
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count != saved_write_req_count) ||
                (mem_wr_word_count_total != saved_write_word_count)) begin
                $display("FAIL: invalid runtime output override wrote data req=%0d/%0d words=%0d/%0d",
                         mem_wr_req_count, saved_write_req_count,
                         mem_wr_word_count_total, saved_write_word_count);
                mismatch_count = mismatch_count + 1;
            end

            output_base_override_valid = 1'b0;
            output_base_override_addr = '0;
            expected_output_base_addr = output_base_addr;
        end
    endtask

    task run_invalid_bad_down_dtype;
        integer prior_done_count;
        integer dtype_word_index;
        logic [31 : 0] saved_dtype;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            dtype_word_index = descriptor_word_index(SLOT_DOWN, DESC_DTYPE_WORD);
            saved_dtype = qmap_mem[dtype_word_index];
            qmap_mem[dtype_word_index] = `QMAP_DTYPE_I32_Q14_10;

            pulse_start();
            wait_for_next_done(prior_done_count, 80000);
            invalid_done_cycle = last_done_cycle;
            invalid_error = error;

            if (error != 1'b1) begin
                $display("FAIL: invalid down dtype did not assert error");
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

    task run_bad_payload_last;
        integer prior_done_count;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            force_bad_payload_last = 1'b1;
            bad_payload_last_fired = 1'b0;

            pulse_start();
            wait_for_next_done(prior_done_count, 80000);
            protocol_done_cycle = last_done_cycle;
            protocol_error = error;
            protocol_dut_read_bursts = dut_read_burst_count;
            protocol_dut_read_words = dut_read_word_count;

            if (bad_payload_last_fired != 1'b1) begin
                $display("FAIL: bad payload-last injection did not fire");
                mismatch_count = mismatch_count + 1;
            end
            if (error != 1'b1) begin
                $display("FAIL: bad payload last did not assert error");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count != saved_write_req_count) ||
                (mem_wr_word_count_total != saved_write_word_count)) begin
                $display("FAIL: bad payload-last path wrote data req=%0d/%0d words=%0d/%0d",
                         mem_wr_req_count, saved_write_req_count,
                         mem_wr_word_count_total, saved_write_word_count);
                mismatch_count = mismatch_count + 1;
            end

            force_bad_payload_last = 1'b0;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1;
            last_trace_state <= -1;
            last_trace_write_count <= -1;
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
                    mismatch_count <= mismatch_count + 1;
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
                 (state_debug != last_trace_state) ||
                 (output_write_word_count != last_trace_write_count))) begin
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
                    output_count,
                    stage_cycle_count,
                    output_write_word_count,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    mem_wr_done,
                    bad_payload_last_fired
                );
                last_trace_state <= state_debug;
                last_trace_write_count <= output_write_word_count;
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
                read_gap_count <= mem_req_fire_count % 3;
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
                    read_gap_count <= ((active_total_words - active_words_left) % 13 == 0) ? 1 : 0;
                end
            end
            else if (read_active && !mem_rd_rsp_valid) begin
                if (read_gap_count > 0) begin
                    read_gap_count <= read_gap_count - 1;
                end
                else begin
                    current_read_word = read_qmap_word(qmap_base_addr + (active_read_index * 4));
                    mem_rd_rsp_data <= current_read_word;
                    if (force_bad_payload_last &&
                        (bad_payload_last_fired == 1'b0) &&
                        (active_read_region != REGION_QMAP)) begin
                        mem_rd_rsp_last <= 1'b1;
                        bad_payload_last_fired <= 1'b1;
                    end
                    else begin
                        mem_rd_rsp_last <= (active_words_left == 1);
                    end
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
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "mlp_residual_add_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_mlp_residual_add_image_words32.hex";
        expected_layer_file = "FPGA_Project/sim/vectors/qmap_mlp_residual_add_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_mlp_residual_add_compute_path_trace.csv";
        qmap_base_addr = DEFAULT_QMAP_BASE_ADDR;
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("expected_layer=%s", expected_layer_file)) begin
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
            "cycle,start,busy,done,error,saturation,state,read_slot,rd_addr,rd_len,rd_req_fire,rd_rsp_last_fire,wr_addr,wr_len,wr_req_fire,wr_data_fire,wr_last,output_count,stage_cycles,output_write_count,rd_bursts,rd_words,wr_reqs,wr_words,wr_done,bad_payload_last\n"
        );

        load_vectors();

        start = 1'b0;
        output_base_override_valid = 1'b0;
        output_base_override_addr = '0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        spurious_start_seen_busy = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_qmap_req_count = 0;
        mem_post_req_count = 0;
        mem_down_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        write_mismatch_count = 0;
        layer_max_abs_diff = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        protocol_done_cycle = 0;
        normal_output_count = 0;
        normal_stage_cycle_count = 0;
        normal_output_write_count = 0;
        normal_dut_read_bursts = 0;
        normal_dut_read_words = 0;
        normal_dut_write_reqs = 0;
        normal_dut_write_words = 0;
        normal_error = 1'b0;
        normal_saturation = 1'b0;
        invalid_error = 1'b0;
        protocol_error = 1'b0;
        protocol_dut_read_bursts = 0;
        protocol_dut_read_words = 0;
        runtime_write_req_delta = 0;
        runtime_write_word_delta = 0;
        invalid_override_error_count = 0;
        invalid_override_write_req_delta = 0;
        invalid_override_write_word_delta = 0;
        force_bad_payload_last = 1'b0;
        bad_payload_last_fired = 1'b0;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_success();
        run_runtime_override();
        run_invalid_runtime_override('0);
        run_invalid_runtime_override(RUNTIME_OUTPUT_BASE_ADDR + 64'd2);
        run_invalid_bad_down_dtype();
        run_bad_payload_last();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_mlp_residual_add_compute_path descriptor-backed final MLP residual test");
        $display("  qmap base               = 0x%016h", qmap_base_addr);
        $display("  output count normal       = %0d", normal_output_count);
        $display("  stage cycles normal       = %0d", normal_stage_cycle_count);
        $display("  layer words written       = %0d", normal_output_write_count);
        $display("  read req/rsp fires        = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap/post/down reqs       = %0d / %0d / %0d",
                 mem_qmap_req_count, mem_post_req_count, mem_down_req_count);
        $display("  write reqs/words          = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  dut normal rd/wr          = %0d/%0d reads, %0d/%0d writes",
                 normal_dut_read_bursts, normal_dut_read_words,
                 normal_dut_write_reqs, normal_dut_write_words);
        $display("  protocol dut rd           = %0d/%0d reads",
                 protocol_dut_read_bursts, protocol_dut_read_words);
        $display("  write mismatches          = %0d", write_mismatch_count);
        $display("  max_abs layer             = %0d", layer_max_abs_diff);
        $display("  spurious start covered    = %0d", spurious_start_seen_busy);
        $display("  invalid descriptor error  = %0d", invalid_error);
        $display("  bad payload-last error    = %0d", protocol_error);
        $display("  runtime write req/words   = %0d / %0d",
                 runtime_write_req_delta, runtime_write_word_delta);
        $display("  invalid override errors   = %0d", invalid_override_error_count);
        $display("  done cycles ok/bad/proto  = %0d / %0d / %0d",
                 normal_done_cycle, invalid_done_cycle, protocol_done_cycle);
        $display("  trace                     = %s", tracefile);

        if (done_seen_count != 6) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=6", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_error) begin
            $display("FAIL: error asserted on valid MLP residual packet");
            mismatch_count = mismatch_count + 1;
        end
        if (normal_saturation !== expected_saturation_mem[0][0]) begin
            $display("FAIL: saturation mismatch actual=%0d expected=%0d",
                     normal_saturation, expected_saturation_mem[0][0]);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_output_count != INPUT_SIZE) begin
            $display("FAIL: output count mismatch actual=%0d expected=%0d",
                     normal_output_count, INPUT_SIZE);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_stage_cycle_count < INPUT_SIZE) begin
            $display("FAIL: stage cycle count too short actual=%0d", normal_stage_cycle_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_output_write_count != VECTOR_WORDS) begin
            $display("FAIL: layer write counter mismatch actual=%0d expected=%0d",
                     normal_output_write_count, VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_qmap_req_count != ((4 + INVALID_OVERRIDE_RUNS) * QMAP_READER_REQS)) ||
            (mem_post_req_count != ((2 * VECTOR_BURSTS) + 1)) ||
            (mem_down_req_count != (2 * VECTOR_BURSTS))) begin
            $display("FAIL: read request class counts mismatch qmap=%0d post=%0d down=%0d",
                     mem_qmap_req_count, mem_post_req_count, mem_down_req_count);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_req_fire_count != ((2 * NORMAL_RD_REQS) +
                                    ((1 + INVALID_OVERRIDE_RUNS) * INVALID_RD_REQS) +
                                    PROTOCOL_RD_REQS)) ||
            (mem_rsp_fire_count != ((2 * NORMAL_RD_WORDS) +
                                    ((1 + INVALID_OVERRIDE_RUNS) * INVALID_RD_WORDS) +
                                    PROTOCOL_RD_WORDS))) begin
            $display("FAIL: total read count mismatch req=%0d/%0d rsp=%0d/%0d",
                     mem_req_fire_count,
                     (2 * NORMAL_RD_REQS) + ((1 + INVALID_OVERRIDE_RUNS) * INVALID_RD_REQS) + PROTOCOL_RD_REQS,
                     mem_rsp_fire_count,
                     (2 * NORMAL_RD_WORDS) + ((1 + INVALID_OVERRIDE_RUNS) * INVALID_RD_WORDS) + PROTOCOL_RD_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_wr_req_count != 2) || (mem_wr_word_count_total != (2 * VECTOR_WORDS))) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=2/%0d",
                     mem_wr_req_count, mem_wr_word_count_total, 2 * VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_dut_read_bursts != NORMAL_RD_REQS) ||
            (normal_dut_read_words != NORMAL_RD_WORDS) ||
            (normal_dut_write_reqs != 1) ||
            (normal_dut_write_words != VECTOR_WORDS)) begin
            $display("FAIL: DUT normal counters mismatch rd=%0d/%0d wr=%0d/%0d expected rd=%0d/%0d wr=1/%0d",
                     normal_dut_read_bursts, normal_dut_read_words,
                     normal_dut_write_reqs, normal_dut_write_words,
                     NORMAL_RD_REQS, NORMAL_RD_WORDS, VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((protocol_dut_read_bursts != PROTOCOL_RD_REQS) ||
            (protocol_dut_read_words != PROTOCOL_RD_WORDS)) begin
            $display("FAIL: protocol-error DUT read counters mismatch rd=%0d/%0d expected=%0d/%0d",
                     protocol_dut_read_bursts, protocol_dut_read_words,
                     PROTOCOL_RD_REQS, PROTOCOL_RD_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (layer_max_abs_diff != 0)) begin
            $display("FAIL: exact layer_out write-back expected, mismatches=%0d max_abs=%0d",
                     write_mismatch_count, layer_max_abs_diff);
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
        if (protocol_error != 1'b1) begin
            $display("FAIL: bad payload-last path did not report error");
            mismatch_count = mismatch_count + 1;
        end
        if ((invalid_override_error_count != INVALID_OVERRIDE_RUNS) ||
            (invalid_override_write_req_delta != 0) ||
            (invalid_override_write_word_delta != 0)) begin
            $display("FAIL: runtime output override validation coverage failed errors=%0d req=%0d words=%0d",
                     invalid_override_error_count,
                     invalid_override_write_req_delta,
                     invalid_override_write_word_delta);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_mlp_residual_add_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_mlp_residual_add_compute_path matched legacy/runtime outputs and rejected invalid overrides.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
