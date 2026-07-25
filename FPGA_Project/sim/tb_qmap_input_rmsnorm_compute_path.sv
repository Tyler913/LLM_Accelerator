`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_input_rmsnorm_compute_path;

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 5;
    localparam int INPUT_SIZE       = 1024;
    localparam int HIDDEN_WIDTH     = 24;
    localparam int GAMMA_WIDTH      = 16;
    localparam int NORM_WIDTH       = 24;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;
    localparam int MEM_DATA_BYTES   = MEM_DATA_WIDTH / 8;
    localparam int VECTOR_BYTES     = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int VECTOR_WORDS     = INPUT_SIZE;
    localparam int VECTOR_BURSTS    = VECTOR_BYTES / MAX_READ_BYTES;
    localparam int QMAP_IMAGE_BYTES = 32'h0000_5000;
    localparam int QMAP_WORDS       = QMAP_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam logic [ADDR_WIDTH-1 : 0] RUNTIME_HIDDEN_BASE_ADDR =
        `QMAP_INPUT_NORM_BASE_ADDR + 64'h0000_0000_0001_0000;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int SLOT_HIDDEN = 1;
    localparam int SLOT_GAMMA = 2;
    localparam int SLOT_NORM = 3;

    localparam int REGION_NONE = 0;
    localparam int REGION_QMAP = 1;
    localparam int REGION_HIDDEN = 2;
    localparam int REGION_GAMMA = 3;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic norm_saturation;
    logic [31 : 0] norm_cycle_count;
    logic [63 : 0] sum_squares;
    logic [63 : 0] mean_square;
    logic [23 : 0] inv_rms;
    logic [31 : 0] norm_write_word_count;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;
    logic [7 : 0] state_debug;
    logic [1 : 0] read_slot_debug;
    logic hidden_base_override_valid;
    logic [ADDR_WIDTH-1 : 0] hidden_base_override_addr;
    logic [ADDR_WIDTH-1 : 0] effective_hidden_base_addr;

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
    logic [31 : 0] expected_norm_mem [0 : VECTOR_WORDS-1];
    logic [63 : 0] expected_sum_squares_mem [0 : 0];
    logic [63 : 0] expected_mean_square_mem [0 : 0];
    logic [31 : 0] expected_inv_rms_mem [0 : 0];
    logic [31 : 0] expected_saturation_mem [0 : 0];

    string vector_dir;
    string prefix;
    string qmap_image_file;
    string expected_norm_file;
    string tracefile;
    string wavefile;
    integer trace_fd;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer done_seen_count;
    integer normal_done_cycle;
    integer invalid_done_cycle;
    integer spurious_start_seen_busy;
    integer write_mismatch_count;
    integer norm_max_abs_diff;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_hidden_req_count;
    integer mem_gamma_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer invalid_write_word_delta;
    integer runtime_hidden_req_delta;
    integer runtime_write_word_delta;
    integer invalid_override_write_word_delta;
    integer invalid_override_error_count;

    integer normal_norm_write_count;
    integer normal_dut_read_bursts;
    integer normal_dut_read_words;
    integer normal_dut_write_reqs;
    integer normal_dut_write_words;
    integer normal_mem_req_fire_count;
    integer normal_mem_rsp_fire_count;
    integer normal_mem_qmap_req_count;
    integer normal_mem_hidden_req_count;
    integer normal_mem_gamma_req_count;
    integer normal_mem_wr_req_count;
    integer normal_mem_wr_word_count_total;
    integer normal_norm_cycle_count;
    logic [63 : 0] normal_sum_squares;
    logic [63 : 0] normal_mean_square;
    logic [23 : 0] normal_inv_rms;
    logic normal_norm_saturation;

    logic [ADDR_WIDTH-1 : 0] qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] gamma_base_addr;
    logic [ADDR_WIDTH-1 : 0] norm_base_addr;
    logic [ADDR_WIDTH-1 : 0] expected_hidden_base_addr;

    logic read_active;
    integer active_read_region;
    integer active_read_index;
    integer active_words_left;
    integer read_gap_count;

    logic write_active;
    integer active_write_index;
    integer active_write_words_left;
    integer write_done_delay;
    integer output_diff;

    logic rd_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_rd_req_addr;
    logic [15 : 0] stalled_rd_req_len;
    logic wr_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic wr_data_stall_active;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_last;

    qmap_input_rmsnorm_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS),
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_WIDTH(HIDDEN_WIDTH),
        .GAMMA_WIDTH(GAMMA_WIDTH),
        .NORM_WIDTH(NORM_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .MAX_READ_BYTES(MAX_READ_BYTES)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(qmap_base_addr),
        .i_hidden_base_override_valid(hidden_base_override_valid),
        .i_hidden_base_override_addr(hidden_base_override_addr),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_effective_hidden_base_addr(effective_hidden_base_addr),
        .o_norm_saturation(norm_saturation),
        .o_norm_cycle_count(norm_cycle_count),
        .o_sum_squares(sum_squares),
        .o_mean_square(mean_square),
        .o_inv_rms(inv_rms),
        .o_norm_write_word_count(norm_write_word_count),
        .o_mem_read_burst_count(dut_read_burst_count),
        .o_mem_read_word_count(dut_read_word_count),
        .o_mem_write_req_count(dut_write_req_count),
        .o_mem_write_word_count(dut_write_word_count),
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
        wavefile = "FPGA_Project/wave/qmap_input_rmsnorm_compute_path.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        $dumpfile(wavefile);
        $dumpvars(0, clk);
        $dumpvars(0, rst_n);
        $dumpvars(0, start);
        $dumpvars(0, busy);
        $dumpvars(0, done);
        $dumpvars(0, error);
        $dumpvars(0, state_debug);
        $dumpvars(0, read_slot_debug);
        $dumpvars(0, norm_cycle_count);
        $dumpvars(0, norm_write_word_count);
    end

    assign mem_rd_req_ready = (!read_active) && ((cycle_count % 7) != 3) && ((cycle_count % 11) != 5);
    assign mem_wr_req_ready = (!write_active) && ((cycle_count % 9) != 4);
    assign mem_wr_data_ready = write_active && ((cycle_count % 6) != 2) && ((cycle_count % 13) != 7);
    assign mem_rd_rsp_valid = read_active && (read_gap_count == 0);
    assign mem_rd_rsp_data = read_active ? qmap_mem[active_read_index] : 32'd0;
    assign mem_rd_rsp_last = read_active && (active_words_left == 1);

    function automatic logic in_range(
        input logic [ADDR_WIDTH-1 : 0] addr,
        input logic [ADDR_WIDTH-1 : 0] base,
        input int bytes
    );
        begin
            in_range = (addr >= base) && (addr < (base + bytes));
        end
    endfunction

    function automatic [ADDR_WIDTH-1 : 0] descriptor_base_addr(input int slot);
        logic [31 : 0] lo;
        logic [31 : 0] hi;
        begin
            lo = qmap_mem[DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_LO_WORD];
            hi = qmap_mem[DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_HI_WORD];
            descriptor_base_addr = {hi, lo};
        end
    endfunction

    function automatic integer word_abs_diff(input logic [31 : 0] observed, input logic [31 : 0] expected);
        integer signed_observed;
        integer signed_expected;
        integer diff;
        begin
            signed_observed = $signed(observed);
            signed_expected = $signed(expected);
            diff = signed_observed - signed_expected;
            if (diff < 0) begin
                diff = -diff;
            end
            word_abs_diff = diff;
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(expected_norm_file, expected_norm_mem);
            $readmemh({vector_dir, "/", prefix, "_sum_squares.hex"}, expected_sum_squares_mem);
            $readmemh({vector_dir, "/", prefix, "_mean_square.hex"}, expected_mean_square_mem);
            $readmemh({vector_dir, "/", prefix, "_inv_rms.hex"}, expected_inv_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_saturation.hex"}, expected_saturation_mem);
            hidden_base_addr = descriptor_base_addr(SLOT_HIDDEN);
            gamma_base_addr = descriptor_base_addr(SLOT_GAMMA);
            norm_base_addr = descriptor_base_addr(SLOT_NORM);
            expected_hidden_base_addr = hidden_base_addr;
        end
    endtask

    task patch_descriptor_dtype(input int slot, input logic [31 : 0] dtype);
        begin
            qmap_mem[DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_DTYPE_WORD] = dtype;
        end
    endtask

    task fail_once(input string message);
        begin
            $display("%s", message);
            mismatch_count = mismatch_count + 1;
        end
    endtask

    task begin_read_request;
        integer words;
        begin
            if (read_active) begin
                fail_once("FAIL: read request accepted while read already active");
            end
            if ((mem_rd_req_addr[1:0] != 2'b00) || (mem_rd_req_len_bytes[1:0] != 2'b00)) begin
                fail_once("FAIL: unaligned read request");
            end
            words = mem_rd_req_len_bytes / MEM_DATA_BYTES;
            if (words <= 0) begin
                fail_once("FAIL: zero-length read request");
            end

            active_read_region = REGION_QMAP;
            if (in_range(mem_rd_req_addr, expected_hidden_base_addr, VECTOR_BYTES)) begin
                active_read_region = REGION_HIDDEN;
                mem_hidden_req_count = mem_hidden_req_count + 1;
            end
            else if (in_range(mem_rd_req_addr, gamma_base_addr, VECTOR_BYTES)) begin
                active_read_region = REGION_GAMMA;
                mem_gamma_req_count = mem_gamma_req_count + 1;
            end
            else if (in_range(mem_rd_req_addr, qmap_base_addr, QMAP_IMAGE_BYTES)) begin
                active_read_region = REGION_QMAP;
                mem_qmap_req_count = mem_qmap_req_count + 1;
            end
            else begin
                $display("FAIL: read request outside known regions addr=0x%016h len=%0d",
                         mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
                active_read_region = REGION_NONE;
            end

            if ((active_read_region == REGION_HIDDEN) &&
                !in_range(mem_rd_req_addr + mem_rd_req_len_bytes - 1,
                          expected_hidden_base_addr, VECTOR_BYTES)) begin
                $display("FAIL: hidden read request crosses selected hidden buffer addr=0x%016h len=%0d",
                         mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
            else if ((active_read_region != REGION_HIDDEN) &&
                     !in_range(mem_rd_req_addr + mem_rd_req_len_bytes - 1,
                               qmap_base_addr, QMAP_IMAGE_BYTES)) begin
                $display("FAIL: read request crosses QMAP image addr=0x%016h len=%0d",
                         mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end

            read_active <= 1'b1;
            if (active_read_region == REGION_HIDDEN) begin
                active_read_index <= ((hidden_base_addr - qmap_base_addr) >> 2) +
                                     ((mem_rd_req_addr - expected_hidden_base_addr) >> 2);
            end
            else begin
                active_read_index <= (mem_rd_req_addr - qmap_base_addr) >> 2;
            end
            active_words_left <= words;
            read_gap_count <= 1;
            mem_req_fire_count = mem_req_fire_count + 1;
        end
    endtask

    task begin_write_request;
        begin
            if (write_active) begin
                fail_once("FAIL: write request accepted while write already active");
            end
            if (mem_wr_req_addr !== norm_base_addr) begin
                $display("FAIL: write addr mismatch actual=0x%016h expected=0x%016h",
                         mem_wr_req_addr, norm_base_addr);
                mismatch_count = mismatch_count + 1;
            end
            if (mem_wr_req_len_bytes != VECTOR_BYTES) begin
                $display("FAIL: write len mismatch actual=%0d expected=%0d",
                         mem_wr_req_len_bytes, VECTOR_BYTES);
                mismatch_count = mismatch_count + 1;
            end
            write_active <= 1'b1;
            active_write_index <= 0;
            active_write_words_left <= VECTOR_WORDS;
            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        integer qmap_index;
        begin
            if (active_write_index >= VECTOR_WORDS) begin
                fail_once("FAIL: too many write-data words");
            end
            else begin
                output_diff = word_abs_diff(mem_wr_data, expected_norm_mem[active_write_index]);
                if (output_diff > norm_max_abs_diff) begin
                    norm_max_abs_diff = output_diff;
                end
                if (mem_wr_data !== expected_norm_mem[active_write_index]) begin
                    if (print_count < 16) begin
                        $display("FAIL: norm word mismatch index=%0d actual=0x%08h expected=0x%08h",
                                 active_write_index, mem_wr_data, expected_norm_mem[active_write_index]);
                        print_count = print_count + 1;
                    end
                    write_mismatch_count = write_mismatch_count + 1;
                    mismatch_count = mismatch_count + 1;
                end
                qmap_index = ((norm_base_addr - qmap_base_addr) >> 2) + active_write_index;
                qmap_mem[qmap_index] = mem_wr_data;
            end

            mem_wr_word_count_total = mem_wr_word_count_total + 1;
            if (active_write_words_left == 1) begin
                if (!mem_wr_data_last) begin
                    fail_once("FAIL: final write word missing last");
                end
                write_active <= 1'b0;
                active_write_words_left <= 0;
                write_done_delay <= 3;
            end
            else begin
                if (mem_wr_data_last) begin
                    fail_once("FAIL: early write-data last");
                end
                active_write_index <= active_write_index + 1;
                active_write_words_left <= active_write_words_left - 1;
            end
        end
    endtask

    task wait_for_done(input int max_cycles);
        integer start_cycle;
        begin
            start_cycle = cycle_count;
            while ((done !== 1'b1) && ((cycle_count - start_cycle) < max_cycles)) begin
                @(posedge clk);
            end
            if (done !== 1'b1) begin
                $display("FAIL: timed out waiting for done after %0d cycles", max_cycles);
                mismatch_count = mismatch_count + 1;
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

    task run_success;
        begin
            hidden_base_override_valid = 1'b0;
            hidden_base_override_addr = '0;
            expected_hidden_base_addr = hidden_base_addr;
            pulse_start();
            repeat (40) @(posedge clk);
            if (busy) begin
                spurious_start_seen_busy = 1;
                pulse_start();
            end
            wait_for_done(20000);
            #1;
            normal_done_cycle = cycle_count;

            if (error) begin
                fail_once("FAIL: normal run reported error");
            end
            normal_norm_write_count = norm_write_word_count;
            normal_norm_cycle_count = norm_cycle_count;
            normal_dut_read_bursts = dut_read_burst_count;
            normal_dut_read_words = dut_read_word_count;
            normal_dut_write_reqs = dut_write_req_count;
            normal_dut_write_words = dut_write_word_count;
            normal_sum_squares = sum_squares;
            normal_mean_square = mean_square;
            normal_inv_rms = inv_rms;
            normal_norm_saturation = norm_saturation;
            normal_mem_req_fire_count = mem_req_fire_count;
            normal_mem_rsp_fire_count = mem_rsp_fire_count;
            normal_mem_qmap_req_count = mem_qmap_req_count;
            normal_mem_hidden_req_count = mem_hidden_req_count;
            normal_mem_gamma_req_count = mem_gamma_req_count;
            normal_mem_wr_req_count = mem_wr_req_count;
            normal_mem_wr_word_count_total = mem_wr_word_count_total;
        end
    endtask

    task run_runtime_override;
        integer hidden_reqs_before;
        integer write_words_before;
        begin
            hidden_reqs_before = mem_hidden_req_count;
            write_words_before = mem_wr_word_count_total;
            expected_hidden_base_addr = RUNTIME_HIDDEN_BASE_ADDR;
            hidden_base_override_valid = 1'b1;
            hidden_base_override_addr = RUNTIME_HIDDEN_BASE_ADDR;
            pulse_start();

            // Change the live pins after acceptance; the transaction must retain
            // the start-time runtime address.
            hidden_base_override_valid = 1'b0;
            hidden_base_override_addr = RUNTIME_HIDDEN_BASE_ADDR + 64'd2;
            wait_for_done(20000);
            #1;

            runtime_hidden_req_delta = mem_hidden_req_count - hidden_reqs_before;
            runtime_write_word_delta = mem_wr_word_count_total - write_words_before;
            if (error) begin
                fail_once("FAIL: runtime hidden override run reported error");
            end
            if (effective_hidden_base_addr !== RUNTIME_HIDDEN_BASE_ADDR) begin
                $display("FAIL: runtime effective hidden base mismatch actual=0x%016h expected=0x%016h",
                         effective_hidden_base_addr, RUNTIME_HIDDEN_BASE_ADDR);
                mismatch_count = mismatch_count + 1;
            end
            if (runtime_hidden_req_delta != VECTOR_BURSTS) begin
                $display("FAIL: runtime hidden burst delta mismatch actual=%0d expected=%0d",
                         runtime_hidden_req_delta, VECTOR_BURSTS);
                mismatch_count = mismatch_count + 1;
            end
            if (runtime_write_word_delta != VECTOR_WORDS) begin
                $display("FAIL: runtime write word delta mismatch actual=%0d expected=%0d",
                         runtime_write_word_delta, VECTOR_WORDS);
                mismatch_count = mismatch_count + 1;
            end

            hidden_base_override_valid = 1'b0;
            hidden_base_override_addr = '0;
            expected_hidden_base_addr = hidden_base_addr;
        end
    endtask

    task run_invalid_runtime_override(input logic [ADDR_WIDTH-1 : 0] bad_addr);
        integer write_words_before;
        begin
            write_words_before = mem_wr_word_count_total;
            hidden_base_override_valid = 1'b1;
            hidden_base_override_addr = bad_addr;
            pulse_start();
            hidden_base_override_valid = 1'b0;
            hidden_base_override_addr = RUNTIME_HIDDEN_BASE_ADDR;
            wait_for_done(8000);
            #1;

            invalid_override_write_word_delta =
                invalid_override_write_word_delta +
                (mem_wr_word_count_total - write_words_before);
            if (!error) begin
                $display("FAIL: invalid runtime hidden override 0x%016h did not report error", bad_addr);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                invalid_override_error_count = invalid_override_error_count + 1;
            end
            if (effective_hidden_base_addr !== bad_addr) begin
                $display("FAIL: invalid runtime hidden base was not latched actual=0x%016h expected=0x%016h",
                         effective_hidden_base_addr, bad_addr);
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_word_count_total - write_words_before) != 0) begin
                fail_once("FAIL: invalid runtime hidden override wrote output data");
            end

            hidden_base_override_valid = 1'b0;
            hidden_base_override_addr = '0;
            expected_hidden_base_addr = hidden_base_addr;
        end
    endtask

    task run_invalid_bad_gamma_dtype;
        integer write_words_before;
        begin
            patch_descriptor_dtype(SLOT_GAMMA, `QMAP_DTYPE_U16_Q8_8);
            write_words_before = mem_wr_word_count_total;
            pulse_start();
            wait_for_done(8000);
            #1;
            invalid_done_cycle = cycle_count;
            invalid_write_word_delta = mem_wr_word_count_total - write_words_before;
            if (!error) begin
                fail_once("FAIL: bad gamma dtype did not report error");
            end
            if (invalid_write_word_delta != 0) begin
                $display("FAIL: invalid descriptor run wrote %0d words", invalid_write_word_delta);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            read_active <= 1'b0;
            active_read_region <= REGION_NONE;
            active_read_index <= 0;
            active_words_left <= 0;
            read_gap_count <= 0;
            write_active <= 1'b0;
            active_write_index <= 0;
            active_write_words_left <= 0;
            write_done_delay <= 0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            rd_req_stall_active <= 1'b0;
            wr_req_stall_active <= 1'b0;
            wr_data_stall_active <= 1'b0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (mem_rd_req_valid && !mem_rd_req_ready) begin
                if (!rd_req_stall_active) begin
                    rd_req_stall_active <= 1'b1;
                    stalled_rd_req_addr <= mem_rd_req_addr;
                    stalled_rd_req_len <= mem_rd_req_len_bytes;
                end
                else if ((mem_rd_req_addr !== stalled_rd_req_addr) ||
                         (mem_rd_req_len_bytes !== stalled_rd_req_len)) begin
                    fail_once("FAIL: read request changed while stalled");
                end
            end
            else begin
                rd_req_stall_active <= 1'b0;
            end

            if (mem_wr_req_valid && !mem_wr_req_ready) begin
                if (!wr_req_stall_active) begin
                    wr_req_stall_active <= 1'b1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
                else if ((mem_wr_req_addr !== stalled_wr_req_addr) ||
                         (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    fail_once("FAIL: write request changed while stalled");
                end
            end
            else begin
                wr_req_stall_active <= 1'b0;
            end

            if (mem_wr_data_valid && !mem_wr_data_ready) begin
                if (!wr_data_stall_active) begin
                    wr_data_stall_active <= 1'b1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_last <= mem_wr_data_last;
                end
                else if ((mem_wr_data !== stalled_wr_data) ||
                         (mem_wr_data_last !== stalled_wr_last)) begin
                    fail_once("FAIL: write data changed while stalled");
                end
            end
            else begin
                wr_data_stall_active <= 1'b0;
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                begin_read_request();
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                mem_rsp_fire_count = mem_rsp_fire_count + 1;
                if (active_words_left == 1) begin
                    read_active <= 1'b0;
                    active_words_left <= 0;
                    active_read_region <= REGION_NONE;
                end
                else begin
                    active_read_index <= active_read_index + 1;
                    active_words_left <= active_words_left - 1;
                    if ((cycle_count % 5) == 1) begin
                        read_gap_count <= 1;
                    end
                end
            end
            else if (read_gap_count > 0) begin
                read_gap_count <= read_gap_count - 1;
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                begin_write_request();
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                if (!write_active) begin
                    fail_once("FAIL: write data without active write request");
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

            if (done) begin
                done_seen_count = done_seen_count + 1;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,0x%0h,0x%0h,0x%016h,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    state_debug,
                    read_slot_debug,
                    mem_rd_req_addr,
                    mem_rd_req_len_bytes,
                    mem_rd_req_valid,
                    mem_rd_req_ready,
                    mem_rd_req_valid && mem_rd_req_ready,
                    mem_wr_req_addr,
                    mem_wr_req_len_bytes,
                    mem_wr_req_valid && mem_wr_req_ready,
                    mem_wr_data_valid,
                    mem_wr_data_ready,
                    mem_wr_data_valid && mem_wr_data_ready,
                    mem_wr_data_last,
                    norm_cycle_count,
                    norm_write_word_count,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    sum_squares,
                    inv_rms
                );
            end
        end
    end

    initial begin : main_test
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "rmsnorm_1024_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_input_rmsnorm_image_words32.hex";
        expected_norm_file = "FPGA_Project/sim/vectors/qmap_input_rmsnorm_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_input_rmsnorm_compute_path_trace.csv";
        qmap_base_addr = `QMAP_INPUT_NORM_BASE_ADDR;
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("expected_norm=%s", expected_norm_file)) begin
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
            "cycle,start,busy,done,error,state,read_slot,rd_addr,rd_len,rd_req_valid,rd_req_ready,rd_req_fire,wr_addr,wr_len,wr_req_fire,wr_data_valid,wr_data_ready,wr_data_fire,wr_last,norm_cycles,norm_write_count,rd_bursts,rd_words,wr_reqs,wr_words,sum_squares,inv_rms\n"
        );

        load_vectors();

        start = 1'b0;
        hidden_base_override_valid = 1'b0;
        hidden_base_override_addr = '0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        done_seen_count = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        spurious_start_seen_busy = 0;
        write_mismatch_count = 0;
        norm_max_abs_diff = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_qmap_req_count = 0;
        mem_hidden_req_count = 0;
        mem_gamma_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        invalid_write_word_delta = 0;
        runtime_hidden_req_delta = 0;
        runtime_write_word_delta = 0;
        invalid_override_write_word_delta = 0;
        invalid_override_error_count = 0;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_success();
        run_runtime_override();
        run_invalid_runtime_override('0);
        run_invalid_runtime_override(RUNTIME_HIDDEN_BASE_ADDR + 64'd2);
        run_invalid_bad_gamma_dtype();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_input_rmsnorm_compute_path descriptor-backed test");
        $display("  qmap base             = 0x%016h", qmap_base_addr);
        $display("  prefix                = %s", prefix);
        $display("  qmap image            = %s", qmap_image_file);
        $display("  normal done cycle     = %0d", normal_done_cycle);
        $display("  invalid done cycle    = %0d", invalid_done_cycle);
        $display("  norm cycles normal    = %0d", normal_norm_cycle_count);
        $display("  norm writes normal    = %0d", normal_norm_write_count);
        $display("  read req/rsp fires    = %0d / %0d", normal_mem_req_fire_count, normal_mem_rsp_fire_count);
        $display("  qmap/hidden/gamma req = %0d / %0d / %0d",
                 normal_mem_qmap_req_count, normal_mem_hidden_req_count, normal_mem_gamma_req_count);
        $display("  write reqs/words      = %0d / %0d", normal_mem_wr_req_count, normal_mem_wr_word_count_total);
        $display("  dut normal rd/wr      = %0d/%0d reads, %0d/%0d writes",
                 normal_dut_read_bursts, normal_dut_read_words,
                 normal_dut_write_reqs, normal_dut_write_words);
        $display("  sum_squares/inv_rms   = %0d / %0d", normal_sum_squares, normal_inv_rms);
        $display("  write mismatches      = %0d", write_mismatch_count);
        $display("  max_abs norm          = %0d", norm_max_abs_diff);
        $display("  spurious start covered= %0d", spurious_start_seen_busy);
        $display("  invalid write delta   = %0d", invalid_write_word_delta);
        $display("  runtime hidden bursts = %0d", runtime_hidden_req_delta);
        $display("  runtime write words   = %0d", runtime_write_word_delta);
        $display("  invalid override errs = %0d", invalid_override_error_count);
        $display("  trace                 = %s", tracefile);

        if (done_seen_count != 5) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=5", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_norm_write_count != VECTOR_WORDS) begin
            $display("FAIL: norm write counter mismatch actual=%0d expected=%0d",
                     normal_norm_write_count, VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_mem_hidden_req_count != VECTOR_BURSTS) ||
            (normal_mem_gamma_req_count != VECTOR_BURSTS)) begin
            $display("FAIL: input burst counts mismatch hidden=%0d gamma=%0d expected=%0d",
                     normal_mem_hidden_req_count, normal_mem_gamma_req_count, VECTOR_BURSTS);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_mem_qmap_req_count != (DESCRIPTOR_SLOTS + 1)) begin
            $display("FAIL: QMAP reader request count mismatch actual=%0d expected=%0d",
                     normal_mem_qmap_req_count, DESCRIPTOR_SLOTS + 1);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_mem_req_fire_count != (DESCRIPTOR_SLOTS + 1 + 2 * VECTOR_BURSTS)) ||
            (normal_mem_rsp_fire_count != (16 + DESCRIPTOR_SLOTS * DESCRIPTOR_WORDS + 2 * VECTOR_WORDS))) begin
            $display("FAIL: memory read count mismatch req=%0d rsp=%0d",
                     normal_mem_req_fire_count, normal_mem_rsp_fire_count);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_mem_wr_req_count != 1) || (normal_mem_wr_word_count_total != VECTOR_WORDS)) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=1/%0d",
                     normal_mem_wr_req_count, normal_mem_wr_word_count_total, VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_dut_read_bursts != (DESCRIPTOR_SLOTS + 1 + 2 * VECTOR_BURSTS)) ||
            (normal_dut_read_words != (16 + DESCRIPTOR_SLOTS * DESCRIPTOR_WORDS + 2 * VECTOR_WORDS)) ||
            (normal_dut_write_reqs != 1) ||
            (normal_dut_write_words != VECTOR_WORDS)) begin
            $display("FAIL: DUT normal counters mismatch rd=%0d/%0d wr=%0d/%0d",
                     normal_dut_read_bursts, normal_dut_read_words,
                     normal_dut_write_reqs, normal_dut_write_words);
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (norm_max_abs_diff != 0)) begin
            $display("FAIL: exact write-back expected, mismatches=%0d max_abs=%0d",
                     write_mismatch_count, norm_max_abs_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            fail_once("FAIL: busy-period spurious start was not covered");
        end
        if (normal_sum_squares !== expected_sum_squares_mem[0]) begin
            $display("FAIL: sum_squares mismatch actual=%0d expected=%0d",
                     normal_sum_squares, expected_sum_squares_mem[0]);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_mean_square !== expected_mean_square_mem[0]) begin
            $display("FAIL: mean_square mismatch actual=%0d expected=%0d",
                     normal_mean_square, expected_mean_square_mem[0]);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_inv_rms !== expected_inv_rms_mem[0][23:0]) begin
            $display("FAIL: inv_rms mismatch actual=%0d expected=%0d",
                     normal_inv_rms, expected_inv_rms_mem[0][23:0]);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_norm_saturation !== expected_saturation_mem[0][0]) begin
            $display("FAIL: saturation mismatch actual=%0d expected=%0d",
                     normal_norm_saturation, expected_saturation_mem[0][0]);
            mismatch_count = mismatch_count + 1;
        end
        if (invalid_write_word_delta != 0) begin
            fail_once("FAIL: invalid descriptor no-write check failed");
        end
        if ((invalid_override_error_count != 2) ||
            (invalid_override_write_word_delta != 0)) begin
            fail_once("FAIL: runtime hidden override validation coverage failed");
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_input_rmsnorm_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_input_rmsnorm_compute_path matched legacy/runtime hidden reads and rejected invalid overrides.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
