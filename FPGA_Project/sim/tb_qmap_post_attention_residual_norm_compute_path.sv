`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_post_attention_residual_norm_compute_path;

    localparam int ADDR_WIDTH      = 64;
    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_QMAP_BASE_ADDR = `QMAP_POST_ATTN_NORM_BASE_ADDR;
    localparam int DESCRIPTOR_SLOTS = 8;
    localparam int INPUT_SIZE      = 1024;
    localparam int RESIDUAL_WIDTH  = 24;
    localparam int O_PROJ_WIDTH    = 24;
    localparam int GAMMA_WIDTH     = 16;
    localparam int NORM_OUT_WIDTH  = 24;
    localparam int MEM_DATA_WIDTH  = 32;
    localparam int MAX_READ_BYTES  = 1024;
    localparam int MEM_DATA_BYTES  = MEM_DATA_WIDTH / 8;
    localparam int VECTOR_BYTES    = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int VECTOR_WORDS    = INPUT_SIZE;
    localparam int VECTOR_BURSTS   = VECTOR_BYTES / MAX_READ_BYTES;
    localparam int QMAP_IMAGE_BYTES = 32'h0000_8000;
    localparam int QMAP_WORDS      = QMAP_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int SLOT_RESIDUAL = 1;
    localparam int SLOT_O_PROJ = 2;
    localparam int SLOT_GAMMA = 3;
    localparam int SLOT_HIDDEN = 4;
    localparam int SLOT_NORM = 5;
    localparam int REGION_QMAP = 0;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic residual_saturation;
    logic norm_saturation;
    logic [31 : 0] residual_count;
    logic [31 : 0] stage_cycle_count;
    logic [63 : 0] sum_squares;
    logic [63 : 0] mean_square;
    logic [23 : 0] inv_rms;
    logic [31 : 0] post_hidden_write_word_count;
    logic [31 : 0] post_norm_write_word_count;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;
    logic [7 : 0] state_debug;
    logic [1 : 0] read_slot_debug;
    logic [1 : 0] write_slot_debug;

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
    logic [31 : 0] expected_hidden_mem [0 : VECTOR_WORDS-1];
    logic [31 : 0] expected_norm_mem [0 : VECTOR_WORDS-1];
    logic [63 : 0] expected_sum_squares_mem [0 : 0];
    logic [63 : 0] expected_mean_square_mem [0 : 0];
    logic [31 : 0] expected_inv_rms_mem [0 : 0];
    logic [31 : 0] expected_residual_saturation_mem [0 : 0];
    logic [31 : 0] expected_norm_saturation_mem [0 : 0];

    string vector_dir;
    string prefix;
    string qmap_image_file;
    string expected_hidden_file;
    string expected_norm_file;
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
    integer write_mismatch_count;
    integer hidden_max_abs_diff;
    integer norm_max_abs_diff;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_residual_req_count;
    integer mem_o_proj_req_count;
    integer mem_gamma_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;

    integer normal_residual_count;
    integer normal_stage_cycle_count;
    integer normal_hidden_write_count;
    integer normal_norm_write_count;
    integer normal_dut_read_bursts;
    integer normal_dut_read_words;
    integer normal_dut_write_reqs;
    integer normal_dut_write_words;
    logic [63 : 0] normal_sum_squares;
    logic [63 : 0] normal_mean_square;
    logic [23 : 0] normal_inv_rms;
    logic normal_residual_saturation;
    logic normal_norm_saturation;

    logic [ADDR_WIDTH-1 : 0] qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] residual_base_addr;
    logic [ADDR_WIDTH-1 : 0] o_proj_base_addr;
    logic [ADDR_WIDTH-1 : 0] gamma_base_addr;
    logic [ADDR_WIDTH-1 : 0] hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] norm_base_addr;

    logic read_active;
    integer active_read_region;
    integer active_read_index;
    integer active_words_left;
    integer read_gap_count;
    logic [31 : 0] current_read_word;

    logic write_active;
    integer active_write_kind;
    integer active_write_index;
    integer active_write_words_left;
    integer write_done_delay;
    integer output_diff;

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

    qmap_post_attention_residual_norm_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS),
        .INPUT_SIZE(INPUT_SIZE),
        .RESIDUAL_WIDTH(RESIDUAL_WIDTH),
        .O_PROJ_WIDTH(O_PROJ_WIDTH),
        .GAMMA_WIDTH(GAMMA_WIDTH),
        .NORM_OUT_WIDTH(NORM_OUT_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .MAX_READ_BYTES(MAX_READ_BYTES)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(qmap_base_addr),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_residual_saturation(residual_saturation),
        .o_norm_saturation(norm_saturation),
        .o_residual_count(residual_count),
        .o_stage_cycle_count(stage_cycle_count),
        .o_sum_squares(sum_squares),
        .o_mean_square(mean_square),
        .o_inv_rms(inv_rms),
        .o_post_hidden_write_word_count(post_hidden_write_word_count),
        .o_post_norm_write_word_count(post_norm_write_word_count),
        .o_mem_read_burst_count(dut_read_burst_count),
        .o_mem_read_word_count(dut_read_word_count),
        .o_mem_write_req_count(dut_write_req_count),
        .o_mem_write_word_count(dut_write_word_count),
        .o_state_debug(state_debug),
        .o_read_slot_debug(read_slot_debug),
        .o_write_slot_debug(write_slot_debug),
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
        wavefile = "FPGA_Project/wave/qmap_post_attention_residual_norm_compute_path.vcd";
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
        $dumpvars(0, write_slot_debug);
        $dumpvars(0, residual_count);
        $dumpvars(0, stage_cycle_count);
        $dumpvars(0, sum_squares);
        $dumpvars(0, mean_square);
        $dumpvars(0, inv_rms);
        $dumpvars(0, post_hidden_write_word_count);
        $dumpvars(0, post_norm_write_word_count);
    end

    assign mem_rd_req_ready = (!read_active) && ((cycle_count % 7) != 3) && ((cycle_count % 11) != 5);
    assign mem_wr_req_ready = (!write_active) && ((cycle_count % 9) != 4);
    assign mem_wr_data_ready = write_active && ((cycle_count % 6) != 2) && ((cycle_count % 13) != 7);

    function automatic [ADDR_WIDTH-1 : 0] descriptor_base_addr(input int slot);
        logic [31 : 0] lo;
        logic [31 : 0] hi;
        begin
            lo = qmap_mem[DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_LO_WORD];
            hi = qmap_mem[DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + DESC_BASE_HI_WORD];
            descriptor_base_addr = {hi, lo};
        end
    endfunction

    function automatic [31 : 0] read_qmap_word(input logic [ADDR_WIDTH-1 : 0] addr);
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
            $readmemh(expected_hidden_file, expected_hidden_mem);
            $readmemh(expected_norm_file, expected_norm_mem);
            $readmemh({vector_dir, "/", prefix, "_sum_squares.hex"}, expected_sum_squares_mem);
            $readmemh({vector_dir, "/", prefix, "_mean_square.hex"}, expected_mean_square_mem);
            $readmemh({vector_dir, "/", prefix, "_inv_rms.hex"}, expected_inv_rms_mem);
            $readmemh({vector_dir, "/", prefix, "_residual_saturation.hex"}, expected_residual_saturation_mem);
            $readmemh({vector_dir, "/", prefix, "_norm_saturation.hex"}, expected_norm_saturation_mem);

            residual_base_addr = descriptor_base_addr(SLOT_RESIDUAL);
            o_proj_base_addr = descriptor_base_addr(SLOT_O_PROJ);
            gamma_base_addr = descriptor_base_addr(SLOT_GAMMA);
            hidden_base_addr = descriptor_base_addr(SLOT_HIDDEN);
            norm_base_addr = descriptor_base_addr(SLOT_NORM);
        end
    endtask

    task check_read_request;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            active_read_region = REGION_QMAP;
            active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
            active_words_left = mem_rd_req_len_bytes / MEM_DATA_BYTES;

            if ((mem_rd_req_addr >= residual_base_addr) &&
                (mem_rd_req_addr < (residual_base_addr + VECTOR_BYTES))) begin
                expected_addr = residual_base_addr + (mem_residual_req_count * MAX_READ_BYTES);
                if ((mem_rd_req_addr !== expected_addr) || (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: residual read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_residual_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_residual_req_count = mem_residual_req_count + 1;
            end
            else if ((mem_rd_req_addr >= o_proj_base_addr) &&
                     (mem_rd_req_addr < (o_proj_base_addr + VECTOR_BYTES))) begin
                expected_addr = o_proj_base_addr + (mem_o_proj_req_count * MAX_READ_BYTES);
                if ((mem_rd_req_addr !== expected_addr) || (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: o_proj read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_o_proj_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_o_proj_req_count = mem_o_proj_req_count + 1;
            end
            else if ((mem_rd_req_addr >= gamma_base_addr) &&
                     (mem_rd_req_addr < (gamma_base_addr + VECTOR_BYTES))) begin
                expected_addr = gamma_base_addr + (mem_gamma_req_count * MAX_READ_BYTES);
                if ((mem_rd_req_addr !== expected_addr) || (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: gamma read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_gamma_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_gamma_req_count = mem_gamma_req_count + 1;
            end
            else if ((mem_rd_req_addr >= qmap_base_addr) &&
                     (mem_rd_req_addr < (qmap_base_addr + QMAP_IMAGE_BYTES))) begin
                mem_qmap_req_count = mem_qmap_req_count + 1;
            end
            else begin
                $display("FAIL: unknown read request addr=0x%016h len=%0d", mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end

            if ((mem_rd_req_addr[1:0] != 2'b00) || ((mem_rd_req_len_bytes % 4) != 0)) begin
                $display("FAIL: unaligned read request addr=0x%016h len=%0d", mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_write_request;
        begin
            if (mem_wr_req_count == 0) begin
                if ((mem_wr_req_addr !== hidden_base_addr) || (mem_wr_req_len_bytes != VECTOR_BYTES)) begin
                    $display("FAIL: hidden write request mismatch addr=0x%016h expected=0x%016h len=%0d",
                             mem_wr_req_addr, hidden_base_addr, mem_wr_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                active_write_kind = 0;
            end
            else if (mem_wr_req_count == 1) begin
                if ((mem_wr_req_addr !== norm_base_addr) || (mem_wr_req_len_bytes != VECTOR_BYTES)) begin
                    $display("FAIL: norm write request mismatch addr=0x%016h expected=0x%016h len=%0d",
                             mem_wr_req_addr, norm_base_addr, mem_wr_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                active_write_kind = 1;
            end
            else begin
                $display("FAIL: unexpected extra write request addr=0x%016h len=%0d",
                         mem_wr_req_addr, mem_wr_req_len_bytes);
                mismatch_count = mismatch_count + 1;
                active_write_kind = 2;
            end
            active_write_index = 0;
            active_write_words_left = mem_wr_req_len_bytes / MEM_DATA_BYTES;
            write_active = 1'b1;
            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        logic [31 : 0] expected_word;
        begin
            expected_word = (active_write_kind == 0) ?
                expected_hidden_mem[active_write_index] :
                expected_norm_mem[active_write_index];

            if (mem_wr_data !== expected_word) begin
                if (print_count < 32) begin
                    $display("FAIL: write data mismatch kind=%0d index=%0d actual=0x%08h expected=0x%08h",
                             active_write_kind, active_write_index, mem_wr_data, expected_word);
                    print_count = print_count + 1;
                end
                write_mismatch_count = write_mismatch_count + 1;
                mismatch_count = mismatch_count + 1;
            end

            output_diff = $signed(mem_wr_data) - $signed(expected_word);
            if (output_diff < 0) begin
                output_diff = -output_diff;
            end
            if (active_write_kind == 0) begin
                if (output_diff > hidden_max_abs_diff) begin
                    hidden_max_abs_diff = output_diff;
                end
            end
            else begin
                if (output_diff > norm_max_abs_diff) begin
                    norm_max_abs_diff = output_diff;
                end
            end

            if (active_write_index == (VECTOR_WORDS - 1)) begin
                if (mem_wr_data_last != 1'b1) begin
                    $display("FAIL: final write word missing last kind=%0d", active_write_kind);
                    mismatch_count = mismatch_count + 1;
                end
                write_active = 1'b0;
                active_write_words_left = 0;
                write_done_delay = 3;
            end
            else begin
                if (mem_wr_data_last == 1'b1) begin
                    $display("FAIL: early write last kind=%0d index=%0d", active_write_kind, active_write_index);
                    mismatch_count = mismatch_count + 1;
                end
                active_write_words_left = active_write_words_left - 1;
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
        input integer timeout_cycles;
        begin
            while ((done_seen_count == prior_done_count) && (cycle_count < timeout_cycles)) begin
                @(negedge clk);
            end
            if (done_seen_count == prior_done_count) begin
                $display("FAIL: timed out waiting for QMAP post-attention wrapper done");
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

            repeat (96) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
            end

            wait_for_next_done(prior_done_count, 200000);
            normal_done_cycle = last_done_cycle;
            normal_residual_count = residual_count;
            normal_stage_cycle_count = stage_cycle_count;
            normal_hidden_write_count = post_hidden_write_word_count;
            normal_norm_write_count = post_norm_write_word_count;
            normal_dut_read_bursts = dut_read_burst_count;
            normal_dut_read_words = dut_read_word_count;
            normal_dut_write_reqs = dut_write_req_count;
            normal_dut_write_words = dut_write_word_count;
            normal_sum_squares = sum_squares;
            normal_mean_square = mean_square;
            normal_inv_rms = inv_rms;
            normal_residual_saturation = residual_saturation;
            normal_norm_saturation = norm_saturation;
        end
    endtask

    task run_invalid_bad_gamma_dtype;
        integer prior_done_count;
        integer dtype_word_index;
        logic [31 : 0] saved_dtype;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            dtype_word_index = DESCRIPTOR_TABLE_WORD_OFFSET + (SLOT_GAMMA * DESCRIPTOR_WORDS) + DESC_DTYPE_WORD;
            saved_dtype = qmap_mem[dtype_word_index];
            qmap_mem[dtype_word_index] = `QMAP_DTYPE_U16_Q8_8;

            pulse_start();
            wait_for_next_done(prior_done_count, 220000);
            invalid_done_cycle = last_done_cycle;

            if (error != 1'b1) begin
                $display("FAIL: invalid gamma dtype did not assert error");
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
        end
        else begin
            cycle_count <= cycle_count + 1;
            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end
            if (done == 1'b1) begin
                done_seen_count = done_seen_count + 1;
                if ((last_done_cycle != -1) && ((cycle_count - last_done_cycle) == 1)) begin
                    $display("FAIL: adjacent done pulses at cycles %0d and %0d", last_done_cycle, cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
                last_done_cycle = cycle_count;
            end

            if (trace_fd != 0) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    state_debug,
                    read_slot_debug,
                    write_slot_debug,
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
                    residual_count,
                    stage_cycle_count,
                    residual_saturation,
                    norm_saturation,
                    post_hidden_write_word_count,
                    post_norm_write_word_count,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_word_count
                );
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            read_active <= 1'b0;
            active_read_region <= REGION_QMAP;
            active_read_index <= 0;
            active_words_left <= 0;
            read_gap_count <= 0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            rd_req_stall_active <= 1'b0;
            rd_rsp_stall_active <= 1'b0;
        end
        else begin
            if (mem_rd_req_valid && !mem_rd_req_ready) begin
                if (!rd_req_stall_active) begin
                    rd_req_stall_active <= 1'b1;
                    stalled_rd_req_addr <= mem_rd_req_addr;
                    stalled_rd_req_len <= mem_rd_req_len_bytes;
                end
                else if ((mem_rd_req_addr !== stalled_rd_req_addr) ||
                         (mem_rd_req_len_bytes !== stalled_rd_req_len)) begin
                    $display("FAIL: read request changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                rd_req_stall_active <= 1'b0;
            end

            if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                if (!rd_rsp_stall_active) begin
                    rd_rsp_stall_active <= 1'b1;
                    stalled_rd_rsp_data <= mem_rd_rsp_data;
                    stalled_rd_rsp_last <= mem_rd_rsp_last;
                end
                else if ((mem_rd_rsp_data !== stalled_rd_rsp_data) ||
                         (mem_rd_rsp_last !== stalled_rd_rsp_last)) begin
                    $display("FAIL: read response changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                rd_rsp_stall_active <= 1'b0;
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                check_read_request();
                read_active <= 1'b1;
                read_gap_count <= 2 + (mem_req_fire_count % 4);
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
                    read_gap_count <= (active_read_index % 5 == 0) ? 1 : 0;
                end
            end
            else if (read_active && !mem_rd_rsp_valid) begin
                if (read_gap_count > 0) begin
                    read_gap_count <= read_gap_count - 1;
                end
                else begin
                    current_read_word = read_qmap_word(qmap_base_addr + (active_read_index * 4));
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
            active_write_kind <= 0;
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
                if (!wr_req_stall_active) begin
                    wr_req_stall_active <= 1'b1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
                else if ((mem_wr_req_addr !== stalled_wr_req_addr) ||
                         (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    $display("FAIL: write request changed while stalled");
                    mismatch_count = mismatch_count + 1;
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
                    $display("FAIL: write data changed while stalled");
                    mismatch_count = mismatch_count + 1;
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
        prefix = "post_attention_residual_norm_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_image_words32.hex";
        expected_hidden_file = "FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_hidden_words32.hex";
        expected_norm_file = "FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_norm_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_post_attention_residual_norm_compute_path_trace.csv";
        qmap_base_addr = DEFAULT_QMAP_BASE_ADDR;
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("expected_hidden=%s", expected_hidden_file)) begin
        end
        if ($value$plusargs("expected_norm=%s", expected_norm_file)) begin
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
            "cycle,start,busy,done,error,state,read_slot,write_slot,rd_addr,rd_len,rd_req_valid,rd_req_ready,rd_req_fire,wr_addr,wr_len,wr_req_fire,wr_data_valid,wr_data_ready,wr_data_fire,wr_last,residual_count,stage_cycle_count,residual_saturation,norm_saturation,hidden_write_count,norm_write_count,rd_bursts,rd_words,wr_words\n"
        );

        load_vectors();

        start = 1'b0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        done_seen_count = 0;
        last_done_cycle = -1;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        spurious_start_seen_busy = 0;
        write_mismatch_count = 0;
        hidden_max_abs_diff = 0;
        norm_max_abs_diff = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_qmap_req_count = 0;
        mem_residual_req_count = 0;
        mem_o_proj_req_count = 0;
        mem_gamma_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        run_success();
        run_invalid_bad_gamma_dtype();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_post_attention_residual_norm_compute_path descriptor-backed test");
        $display("  qmap base             = 0x%016h", qmap_base_addr);
        $display("  prefix                = %s", prefix);
        $display("  qmap image            = %s", qmap_image_file);
        $display("  residual count normal = %0d", normal_residual_count);
        $display("  stage cycles normal   = %0d", normal_stage_cycle_count);
        $display("  hidden/norm writes    = %0d / %0d", normal_hidden_write_count, normal_norm_write_count);
        $display("  read req/rsp fires    = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap/res/o/gamma reqs = %0d / %0d / %0d / %0d",
                 mem_qmap_req_count, mem_residual_req_count, mem_o_proj_req_count, mem_gamma_req_count);
        $display("  write reqs/words      = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  dut normal rd/wr      = %0d/%0d reads, %0d/%0d writes",
                 normal_dut_read_bursts, normal_dut_read_words,
                 normal_dut_write_reqs, normal_dut_write_words);
        $display("  sum_squares/inv_rms   = %0d / %0d", normal_sum_squares, normal_inv_rms);
        $display("  write mismatches      = %0d", write_mismatch_count);
        $display("  max_abs hidden/norm   = %0d / %0d", hidden_max_abs_diff, norm_max_abs_diff);
        $display("  spurious start covered= %0d", spurious_start_seen_busy);
        $display("  done cycles normal/bad= %0d / %0d", normal_done_cycle, invalid_done_cycle);
        $display("  trace                 = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_residual_count != INPUT_SIZE) begin
            $display("FAIL: residual count mismatch actual=%0d expected=%0d", normal_residual_count, INPUT_SIZE);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_hidden_write_count != VECTOR_WORDS) || (normal_norm_write_count != VECTOR_WORDS)) begin
            $display("FAIL: output write counters mismatch hidden=%0d norm=%0d expected=%0d",
                     normal_hidden_write_count, normal_norm_write_count, VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_residual_req_count != VECTOR_BURSTS) ||
            (mem_o_proj_req_count != VECTOR_BURSTS) ||
            (mem_gamma_req_count != VECTOR_BURSTS)) begin
            $display("FAIL: input burst counts mismatch residual=%0d o_proj=%0d gamma=%0d expected=%0d",
                     mem_residual_req_count, mem_o_proj_req_count, mem_gamma_req_count, VECTOR_BURSTS);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_qmap_req_count != 18) begin
            $display("FAIL: QMAP reader request count mismatch actual=%0d expected=18", mem_qmap_req_count);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_req_fire_count != 30) || (mem_rsp_fire_count != 3616)) begin
            $display("FAIL: memory read count mismatch req=%0d rsp=%0d expected=30/3616",
                     mem_req_fire_count, mem_rsp_fire_count);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_wr_req_count != 2) || (mem_wr_word_count_total != (2 * VECTOR_WORDS))) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=2/%0d",
                     mem_wr_req_count, mem_wr_word_count_total, 2 * VECTOR_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_dut_read_bursts != 21) || (normal_dut_read_words != 3344) ||
            (normal_dut_write_reqs != 2) || (normal_dut_write_words != 2048)) begin
            $display("FAIL: DUT normal counters mismatch rd=%0d/%0d wr=%0d/%0d expected=21/3344 2/2048",
                     normal_dut_read_bursts, normal_dut_read_words,
                     normal_dut_write_reqs, normal_dut_write_words);
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (hidden_max_abs_diff != 0) || (norm_max_abs_diff != 0)) begin
            $display("FAIL: exact write-back expected, mismatches=%0d hidden_abs=%0d norm_abs=%0d",
                     write_mismatch_count, hidden_max_abs_diff, norm_max_abs_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
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
        if (normal_residual_saturation !== expected_residual_saturation_mem[0][0]) begin
            $display("FAIL: residual saturation mismatch actual=%0d expected=%0d",
                     normal_residual_saturation, expected_residual_saturation_mem[0][0]);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_norm_saturation !== expected_norm_saturation_mem[0][0]) begin
            $display("FAIL: norm saturation mismatch actual=%0d expected=%0d",
                     normal_norm_saturation, expected_norm_saturation_mem[0][0]);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_post_attention_residual_norm_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_post_attention_residual_norm_compute_path matched exact hidden and post-norm write-backs.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
