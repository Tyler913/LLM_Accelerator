`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_o_proj_compute_path;

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 6;
    localparam int INPUT_SIZE       = 2048;
    localparam int OUT_FEATURES     = 1024;
    localparam int GROUP_SIZE       = 64;
    localparam int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE;
    localparam int GROUP_PARALLEL   = 8;
    localparam int ACT_WIDTH        = 24;
    localparam int WEIGHT_WIDTH     = 4;
    localparam int SCALE_WIDTH      = 16;
    localparam int OUT_WIDTH        = 24;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;
    localparam int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;

    localparam int MEM_DATA_BYTES          = MEM_DATA_WIDTH / 8;
    localparam int ACTIVATION_BYTES        = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int ACTIVATION_BURSTS       = ACTIVATION_BYTES / MAX_READ_BYTES;
    localparam int WEIGHT_ROW_BYTES        = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES         = GROUP_COUNT * (SCALE_WIDTH / 8);
    localparam int WEIGHT_WORDS            = OUT_FEATURES * WEIGHT_ROW_BYTES / MEM_DATA_BYTES;
    localparam int SCALE_WORDS             = OUT_FEATURES * SCALE_ROW_BYTES / MEM_DATA_BYTES;
    localparam int QMAP_IMAGE_BYTES        = 32'h0000_5000;
    localparam int QMAP_WORDS              = QMAP_IMAGE_BYTES / 4;
    localparam int DESCRIPTOR_WORDS        = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD         = 2;
    localparam int DESC_GROUP_SIZE_WORD    = 6;
    localparam int DESC_BASE_LO_WORD       = 8;
    localparam int DESC_BASE_HI_WORD       = 9;
    localparam int SLOT_ACTIVATION         = 1;
    localparam int SLOT_WEIGHT             = 2;
    localparam int SLOT_SCALE              = 3;
    localparam int SLOT_OUTPUT             = 4;
    localparam int REGION_QMAP             = 0;
    localparam int REGION_ACTIVATION       = 1;
    localparam int REGION_WEIGHT           = 2;
    localparam int REGION_SCALE            = 3;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic saturation;
    logic [31 : 0] rows_done;
    logic signed [ROW_ACC_WIDTH-1 : 0] last_row_sum_q26;
    logic signed [31 : 0] last_output_q12_12;
    logic [31 : 0] output_write_word_count;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;
    logic [7 : 0] state_debug;
    logic [31 : 0] row_index_debug;

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
    logic [31 : 0] weight_words_mem [0 : WEIGHT_WORDS-1];
    logic [31 : 0] scale_words_mem [0 : SCALE_WORDS-1];
    logic [31 : 0] expected_words_mem [0 : OUT_FEATURES-1];

    string vector_dir;
    string prefix;
    string qmap_image_file;
    string qmap_expected_file;
    string tracefile;
    string wavefile;
    integer trace_fd;

    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer done_seen_count;
    integer last_done_cycle;
    integer spurious_start_seen_busy;
    integer normal_done_cycle;
    integer invalid_done_cycle;
    integer normal_rows_done;
    integer normal_output_write_word_count;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_qmap_req_count;
    integer mem_activation_req_count;
    integer mem_weight_req_count;
    integer mem_scale_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer write_mismatch_count;
    integer max_abs_output_diff;
    integer active_region;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer response_delay;
    integer active_write_index;
    integer active_write_words_left;
    integer write_done_delay;
    integer req_stall_active;
    integer rsp_stall_active;
    integer wr_req_stall_active;
    integer wr_data_stall_active;
    integer saved_write_req_count;
    integer saved_write_word_count;
    longint signed output_diff;

    logic read_active;
    logic write_active;
    logic [ADDR_WIDTH-1 : 0] activation_base_addr;
    logic [ADDR_WIDTH-1 : 0] weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] scale_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_base_addr;
    logic [ADDR_WIDTH-1 : 0] stalled_req_addr;
    logic [15 : 0] stalled_req_len;
    logic [31 : 0] stalled_rsp_data;
    logic stalled_rsp_last;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_last;

    qmap_o_proj_compute_path #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS),
        .INPUT_SIZE      (INPUT_SIZE),
        .OUT_FEATURES    (OUT_FEATURES),
        .GROUP_SIZE      (GROUP_SIZE),
        .GROUP_COUNT     (GROUP_COUNT),
        .GROUP_PARALLEL  (GROUP_PARALLEL),
        .ACT_WIDTH       (ACT_WIDTH),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH),
        .SCALE_WIDTH     (SCALE_WIDTH),
        .OUT_WIDTH       (OUT_WIDTH),
        .MEM_DATA_WIDTH  (MEM_DATA_WIDTH),
        .MAX_READ_BYTES  (MAX_READ_BYTES),
        .ROW_ACC_WIDTH   (ROW_ACC_WIDTH)
    ) dut (
        .i_clk                    (clk),
        .i_rst_n                  (rst_n),
        .i_start                  (start),
        .i_qmap_base_addr         (`QMAP_O_PROJ_BASE_ADDR),
        .o_busy                   (busy),
        .o_done                   (done),
        .o_error                  (error),
        .o_saturation             (saturation),
        .o_rows_done              (rows_done),
        .o_last_row_sum_q26       (last_row_sum_q26),
        .o_last_output_q12_12     (last_output_q12_12),
        .o_output_write_word_count(output_write_word_count),
        .o_mem_read_burst_count   (mem_read_burst_count),
        .o_mem_read_word_count    (mem_read_word_count),
        .o_mem_write_req_count    (mem_write_req_count),
        .o_mem_write_word_count   (mem_write_word_count),
        .o_state_debug            (state_debug),
        .o_row_index_debug        (row_index_debug),
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
        wavefile = "FPGA_Project/wave/qmap_o_proj_compute_path.vcd";
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
            $dumpvars(0, row_index_debug);
            $dumpvars(0, mem_rd_req_valid);
            $dumpvars(0, mem_rd_req_ready);
            $dumpvars(0, mem_rd_rsp_valid);
            $dumpvars(0, mem_rd_rsp_ready);
            $dumpvars(0, mem_wr_req_valid);
            $dumpvars(0, mem_wr_req_ready);
            $dumpvars(0, mem_wr_data_valid);
            $dumpvars(0, mem_wr_data_ready);
        end
    end

    function automatic integer descriptor_word_index(input integer slot, input integer word_offset);
        begin
            descriptor_word_index = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    function automatic logic [63 : 0] descriptor_base_addr(input integer slot);
        begin
            descriptor_base_addr = {
                qmap_mem[descriptor_word_index(slot, DESC_BASE_HI_WORD)],
                qmap_mem[descriptor_word_index(slot, DESC_BASE_LO_WORD)]
            };
        end
    endfunction

    function automatic logic req_ready_pattern(input integer cycle, input integer count);
        begin
            req_ready_pattern =
                ((cycle % 11) != 3) &&
                (((cycle + count) % 17) != 5);
        end
    endfunction

    function automatic integer response_latency_pattern(input integer count, input integer cycle);
        begin
            response_latency_pattern = 1 + ((count + cycle) % 7);
        end
    endfunction

    function automatic logic response_gap_pattern(input integer cycle, input integer word_count);
        begin
            response_gap_pattern =
                ((cycle % 13) != 4) &&
                (((cycle + word_count) % 29) != 7);
        end
    endfunction

    function automatic logic wr_req_ready_pattern(input integer cycle, input integer count);
        begin
            wr_req_ready_pattern = ((cycle + count) % 9) != 2;
        end
    endfunction

    function automatic logic wr_data_ready_pattern(input integer cycle, input integer count);
        begin
            wr_data_ready_pattern =
                ((cycle % 7) != 1) &&
                (((cycle + count) % 13) != 4);
        end
    endfunction

    function automatic logic [31 : 0] memory_word(input integer region, input integer word_index);
        begin
            case (region)
                REGION_QMAP: begin
                    memory_word = qmap_mem[word_index];
                end
                REGION_ACTIVATION: begin
                    memory_word = qmap_mem[word_index];
                end
                REGION_WEIGHT: begin
                    memory_word = weight_words_mem[word_index];
                end
                REGION_SCALE: begin
                    memory_word = scale_words_mem[word_index];
                end
                default: begin
                    memory_word = 32'hBAD0_BAD0;
                end
            endcase
        end
    endfunction

    task load_vectors;
        begin
            $readmemh(qmap_image_file, qmap_mem);
            $readmemh(qmap_expected_file, expected_words_mem);
            $readmemh({vector_dir, "/", prefix, "_weight_words32.hex"}, weight_words_mem);
            $readmemh({vector_dir, "/", prefix, "_scale_words32.hex"}, scale_words_mem);
            activation_base_addr = descriptor_base_addr(SLOT_ACTIVATION);
            weight_base_addr = descriptor_base_addr(SLOT_WEIGHT);
            scale_base_addr = descriptor_base_addr(SLOT_SCALE);
            output_base_addr = descriptor_base_addr(SLOT_OUTPUT);
        end
    endtask

    task check_read_request_stability;
        begin
            if (req_stall_active != 0) begin
                if ((mem_rd_req_addr !== stalled_req_addr) ||
                    (mem_rd_req_len_bytes !== stalled_req_len)) begin
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
            if (rsp_stall_active != 0) begin
                if ((mem_rd_rsp_data !== stalled_rsp_data) ||
                    (mem_rd_rsp_last !== stalled_rsp_last)) begin
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
            if (wr_req_stall_active != 0) begin
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
            if (wr_data_stall_active != 0) begin
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
        integer expected_row;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            if ((mem_rd_req_addr >= activation_base_addr) &&
                (mem_rd_req_addr < (activation_base_addr + ACTIVATION_BYTES))) begin
                active_region = REGION_ACTIVATION;
                active_read_index = (mem_rd_req_addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
                expected_addr = activation_base_addr + (mem_activation_req_count * MAX_READ_BYTES);
                if ((mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    if (print_count < 32) begin
                        $display("FAIL: activation read request mismatch index=%0d addr=0x%016h expected=0x%016h len=%0d",
                                 mem_activation_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
                mem_activation_req_count = mem_activation_req_count + 1;
            end
            else if ((mem_rd_req_addr >= weight_base_addr) &&
                     (mem_rd_req_addr < (weight_base_addr + (WEIGHT_WORDS * 4)))) begin
                active_region = REGION_WEIGHT;
                active_read_index = (mem_rd_req_addr - weight_base_addr) >> 2;
                expected_row = mem_weight_req_count;
                expected_addr = weight_base_addr + (expected_row * WEIGHT_ROW_BYTES);
                if ((expected_row >= OUT_FEATURES) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != WEIGHT_ROW_BYTES)) begin
                    if (print_count < 32) begin
                        $display("FAIL: weight read request mismatch row=%0d addr=0x%016h expected=0x%016h len=%0d",
                                 expected_row, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
                mem_weight_req_count = mem_weight_req_count + 1;
            end
            else if ((mem_rd_req_addr >= scale_base_addr) &&
                     (mem_rd_req_addr < (scale_base_addr + (SCALE_WORDS * 4)))) begin
                active_region = REGION_SCALE;
                active_read_index = (mem_rd_req_addr - scale_base_addr) >> 2;
                expected_row = mem_scale_req_count;
                expected_addr = scale_base_addr + (expected_row * SCALE_ROW_BYTES);
                if ((expected_row >= OUT_FEATURES) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != SCALE_ROW_BYTES) ||
                    (mem_weight_req_count <= mem_scale_req_count)) begin
                    if (print_count < 32) begin
                        $display("FAIL: scale read request mismatch row=%0d addr=0x%016h expected=0x%016h len=%0d weight_reqs=%0d scale_reqs=%0d",
                                 expected_row, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes,
                                 mem_weight_req_count, mem_scale_req_count);
                        print_count = print_count + 1;
                    end
                    mismatch_count = mismatch_count + 1;
                end
                mem_scale_req_count = mem_scale_req_count + 1;
            end
            else if ((mem_rd_req_addr >= `QMAP_O_PROJ_BASE_ADDR) &&
                     (mem_rd_req_addr < (`QMAP_O_PROJ_BASE_ADDR + QMAP_IMAGE_BYTES))) begin
                active_region = REGION_QMAP;
                active_read_index = (mem_rd_req_addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
                mem_qmap_req_count = mem_qmap_req_count + 1;
            end
            else begin
                active_region = REGION_QMAP;
                active_read_index = 0;
                if (print_count < 32) begin
                    $display("FAIL: read request address outside known regions addr=0x%016h len=%0d",
                             mem_rd_req_addr, mem_rd_req_len_bytes);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task clear_output_payload;
        integer idx;
        integer base_index;
        begin
            base_index = (output_base_addr - `QMAP_O_PROJ_BASE_ADDR) >> 2;
            for (idx = 0; idx < OUT_FEATURES; idx = idx + 1) begin
                qmap_mem[base_index + idx] = 32'hFFFF_FFFF;
            end
        end
    endtask

    task run_success;
        begin
            clear_output_payload();
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            wait (busy == 1'b1);
            repeat (11) @(negedge clk);
            start = 1'b1;
            spurious_start_seen_busy = 1;
            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < 2500000)) begin
                @(negedge clk);
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for qmap_o_proj_compute_path normal done");
                $finish(1);
            end
            normal_done_cycle = cycle_count;
            normal_rows_done = rows_done;
            normal_output_write_word_count = output_write_word_count;

            if (error) begin
                $display("FAIL: error asserted on valid o_proj packet");
                mismatch_count = mismatch_count + 1;
            end
            if (saturation) begin
                $display("FAIL: unexpected o_proj saturation");
                mismatch_count = mismatch_count + 1;
            end
            if (rows_done != OUT_FEATURES) begin
                $display("FAIL: rows_done mismatch actual=%0d expected=%0d", rows_done, OUT_FEATURES);
                mismatch_count = mismatch_count + 1;
            end
            if (output_write_word_count != OUT_FEATURES) begin
                $display("FAIL: output_write_word_count mismatch actual=%0d expected=%0d",
                         output_write_word_count, OUT_FEATURES);
                mismatch_count = mismatch_count + 1;
            end
            if (mem_write_req_count != 1 || mem_write_word_count != OUT_FEATURES) begin
                $display("FAIL: DUT write counters mismatch req=%0d words=%0d",
                         mem_write_req_count, mem_write_word_count);
                mismatch_count = mismatch_count + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    task run_invalid_bad_weight_group;
        begin
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            qmap_mem[descriptor_word_index(SLOT_WEIGHT, DESC_GROUP_SIZE_WORD)] = 32'd32;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            while ((done != 1'b1) && (cycle_count < (normal_done_cycle + 20000))) begin
                @(negedge clk);
            end
            if (done != 1'b1) begin
                $display("FAIL: timed out waiting for invalid descriptor done");
                $finish(1);
            end
            invalid_done_cycle = cycle_count;

            if (!error) begin
                $display("FAIL: invalid weight group_size did not assert error");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_count != saved_write_req_count) ||
                (mem_wr_word_count_total != saved_write_word_count)) begin
                $display("FAIL: invalid descriptor path wrote data req_before=%0d req_after=%0d words_before=%0d words_after=%0d",
                         saved_write_req_count, mem_wr_req_count,
                         saved_write_word_count, mem_wr_word_count_total);
                mismatch_count = mismatch_count + 1;
            end
            qmap_mem[descriptor_word_index(SLOT_WEIGHT, DESC_GROUP_SIZE_WORD)] = GROUP_SIZE;
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -100;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                if (cycle_count == (last_done_cycle + 1)) begin
                    $display("FAIL: adjacent done pulses at cycle %0d", cycle_count);
                    mismatch_count = mismatch_count + 1;
                end
                last_done_cycle <= cycle_count;
            end

            if ((trace_fd != 0) &&
                (((mem_rd_req_valid == 1'b1) && (mem_rd_req_ready == 1'b1)) ||
                 ((mem_rd_rsp_valid == 1'b1) && (mem_rd_rsp_ready == 1'b1) && (mem_rd_rsp_last == 1'b1)) ||
                 ((mem_wr_req_valid == 1'b1) && (mem_wr_req_ready == 1'b1)) ||
                 ((mem_wr_data_valid == 1'b1) && (mem_wr_data_ready == 1'b1)) ||
                 (mem_wr_done == 1'b1) ||
                 (done == 1'b1) ||
                 (state_debug == 8'd13))) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    state_debug,
                    row_index_debug,
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
                    rows_done,
                    output_write_word_count,
                    mem_read_burst_count,
                    mem_read_word_count,
                    mem_write_word_count
                );
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_req_ready <= 1'b0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            read_active <= 1'b0;
            active_region <= REGION_QMAP;
            active_read_index <= 0;
            active_words_left <= 0;
            active_total_words <= 0;
            response_delay <= 0;
            req_stall_active <= 0;
            rsp_stall_active <= 0;
        end
        else begin
            mem_rd_req_ready <=
                (read_active == 1'b0) &&
                (mem_rd_rsp_valid == 1'b0) &&
                req_ready_pattern(cycle_count, mem_req_fire_count);

            if (mem_rd_req_valid == 1'b1) begin
                check_read_request_stability();
                if (mem_rd_req_ready == 1'b1) begin
                    mem_req_fire_count = mem_req_fire_count + 1;
                    req_stall_active <= 0;
                    read_active <= 1'b1;
                    response_delay <= response_latency_pattern(mem_req_fire_count, cycle_count);
                    active_words_left <= (mem_rd_req_len_bytes + 3) >> 2;
                    active_total_words <= (mem_rd_req_len_bytes + 3) >> 2;
                    classify_and_check_read_request();
                end
                else begin
                    req_stall_active <= 1;
                    stalled_req_addr <= mem_rd_req_addr;
                    stalled_req_len <= mem_rd_req_len_bytes;
                end
            end
            else begin
                req_stall_active <= 0;
            end

            if (mem_rd_rsp_valid == 1'b1) begin
                check_response_stability();
                if (mem_rd_rsp_ready == 1'b1) begin
                    mem_rsp_fire_count = mem_rsp_fire_count + 1;
                    rsp_stall_active <= 0;
                    mem_rd_rsp_valid <= 1'b0;
                    if (active_words_left <= 1) begin
                        read_active <= 1'b0;
                        active_words_left <= 0;
                        mem_rd_rsp_last <= 1'b0;
                    end
                    else begin
                        active_read_index <= active_read_index + 1;
                        active_words_left <= active_words_left - 1;
                    end
                end
                else begin
                    rsp_stall_active <= 1;
                    stalled_rsp_data <= mem_rd_rsp_data;
                    stalled_rsp_last <= mem_rd_rsp_last;
                end
            end
            else if ((read_active == 1'b1) && (mem_rd_rsp_valid == 1'b0)) begin
                if (response_delay > 0) begin
                    response_delay <= response_delay - 1;
                end
                else if (response_gap_pattern(cycle_count, active_total_words - active_words_left)) begin
                    mem_rd_rsp_valid <= 1'b1;
                    mem_rd_rsp_data <= memory_word(active_region, active_read_index);
                    mem_rd_rsp_last <= (active_words_left == 1);
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_wr_req_ready <= 1'b0;
            mem_wr_data_ready <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            write_active <= 1'b0;
            active_write_index <= 0;
            active_write_words_left <= 0;
            write_done_delay <= 0;
            wr_req_stall_active <= 0;
            wr_data_stall_active <= 0;
        end
        else begin
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            mem_wr_req_ready <=
                (write_active == 1'b0) &&
                wr_req_ready_pattern(cycle_count, mem_wr_req_count);
            mem_wr_data_ready <=
                (write_active == 1'b1) &&
                wr_data_ready_pattern(cycle_count, mem_wr_word_count_total);

            if (mem_wr_req_valid == 1'b1) begin
                check_write_request_stability();
                if (mem_wr_req_ready == 1'b1) begin
                    mem_wr_req_count = mem_wr_req_count + 1;
                    wr_req_stall_active <= 0;
                    if (mem_wr_req_addr !== output_base_addr) begin
                        $display("FAIL: output write address mismatch actual=0x%016h expected=0x%016h",
                                 mem_wr_req_addr, output_base_addr);
                        mismatch_count = mismatch_count + 1;
                    end
                    if (mem_wr_req_len_bytes != OUT_FEATURES * 4) begin
                        $display("FAIL: output write length mismatch actual=%0d expected=%0d",
                                 mem_wr_req_len_bytes, OUT_FEATURES * 4);
                        mismatch_count = mismatch_count + 1;
                    end
                    write_active <= 1'b1;
                    active_write_index <= 0;
                    active_write_words_left <= OUT_FEATURES;
                end
                else begin
                    wr_req_stall_active <= 1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
            end
            else begin
                wr_req_stall_active <= 0;
            end

            if (mem_wr_data_valid == 1'b1) begin
                check_write_data_stability();
                if (mem_wr_data_ready == 1'b1) begin
                    mem_wr_word_count_total = mem_wr_word_count_total + 1;
                    wr_data_stall_active <= 0;
                    if (active_write_index >= OUT_FEATURES) begin
                        $display("FAIL: extra write data word index=%0d", active_write_index);
                        mismatch_count = mismatch_count + 1;
                    end
                    else begin
                        if (mem_wr_data !== expected_words_mem[active_write_index]) begin
                            if (print_count < 32) begin
                                $display("FAIL: output word %0d mismatch actual=0x%08h expected=0x%08h",
                                         active_write_index, mem_wr_data, expected_words_mem[active_write_index]);
                                print_count = print_count + 1;
                            end
                            write_mismatch_count = write_mismatch_count + 1;
                            mismatch_count = mismatch_count + 1;
                        end
                        output_diff = $signed(mem_wr_data) - $signed(expected_words_mem[active_write_index]);
                        if (output_diff < 0) begin
                            output_diff = -output_diff;
                        end
                        if (output_diff > max_abs_output_diff) begin
                            max_abs_output_diff = output_diff;
                        end
                    end

                    if (active_write_words_left == 1) begin
                        if (mem_wr_data_last != 1'b1) begin
                            $display("FAIL: final output write word missing last");
                            mismatch_count = mismatch_count + 1;
                        end
                        write_active <= 1'b0;
                        active_write_words_left <= 0;
                        write_done_delay <= 3;
                    end
                    else begin
                        if (mem_wr_data_last == 1'b1) begin
                            $display("FAIL: early output write last at index %0d", active_write_index);
                            mismatch_count = mismatch_count + 1;
                        end
                        active_write_words_left <= active_write_words_left - 1;
                    end
                    active_write_index <= active_write_index + 1;
                end
                else begin
                    wr_data_stall_active <= 1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_last <= mem_wr_data_last;
                end
            end
            else begin
                wr_data_stall_active <= 0;
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
        prefix = "o_proj_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_o_proj_image_words32.hex";
        qmap_expected_file = "FPGA_Project/sim/vectors/qmap_o_proj_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_o_proj_compute_path_trace.csv";
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("qmap_expected=%s", qmap_expected_file)) begin
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
            "cycle,start,busy,done,error,saturation,state,row,rd_addr,rd_len,rd_req_valid,rd_req_ready,rd_req_fire,wr_addr,wr_len,wr_req_fire,wr_data_valid,wr_data_ready,wr_data_fire,wr_last,rows_done,out_write_count,rd_bursts,rd_words,wr_words\n"
        );

        load_vectors();

        start = 1'b0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        write_mismatch_count = 0;
        max_abs_output_diff = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_qmap_req_count = 0;
        mem_activation_req_count = 0;
        mem_weight_req_count = 0;
        mem_scale_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        saved_write_req_count = 0;
        saved_write_word_count = 0;
        spurious_start_seen_busy = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        normal_rows_done = 0;
        normal_output_write_word_count = 0;

        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_success();
        run_invalid_bad_weight_group();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_o_proj_compute_path descriptor-backed Layer 0 o_proj test");
        $display("  rows done normal       = %0d", normal_rows_done);
        $display("  output words written   = %0d", normal_output_write_word_count);
        $display("  read req/rsp fires     = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap/activation reqs   = %0d / %0d", mem_qmap_req_count, mem_activation_req_count);
        $display("  weight/scale reqs      = %0d / %0d", mem_weight_req_count, mem_scale_req_count);
        $display("  write reqs/words       = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  write mismatches       = %0d", write_mismatch_count);
        $display("  max_abs_output_diff    = %0d", max_abs_output_diff);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  done cycles normal/bad = %0d / %0d", normal_done_cycle, invalid_done_cycle);
        $display("  trace                  = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_req_fire_count != (7 + ACTIVATION_BURSTS + OUT_FEATURES + OUT_FEATURES + 7)) begin
            $display("FAIL: read request count mismatch actual=%0d expected=%0d",
                     mem_req_fire_count, 7 + ACTIVATION_BURSTS + OUT_FEATURES + OUT_FEATURES + 7);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_rsp_fire_count != (208 + INPUT_SIZE + WEIGHT_WORDS + SCALE_WORDS + 208)) begin
            $display("FAIL: read response word count mismatch actual=%0d expected=%0d",
                     mem_rsp_fire_count, 208 + INPUT_SIZE + WEIGHT_WORDS + SCALE_WORDS + 208);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_qmap_req_count != 14) begin
            $display("FAIL: qmap request count mismatch actual=%0d expected=14", mem_qmap_req_count);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_activation_req_count != ACTIVATION_BURSTS) begin
            $display("FAIL: activation request count mismatch actual=%0d expected=%0d",
                     mem_activation_req_count, ACTIVATION_BURSTS);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_weight_req_count != OUT_FEATURES || mem_scale_req_count != OUT_FEATURES) begin
            $display("FAIL: weight/scale request count mismatch weight=%0d scale=%0d expected=%0d",
                     mem_weight_req_count, mem_scale_req_count, OUT_FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (mem_wr_req_count != 1 || mem_wr_word_count_total != OUT_FEATURES) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=1/%0d",
                     mem_wr_req_count, mem_wr_word_count_total, OUT_FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (write_mismatch_count != 0 || max_abs_output_diff != 0) begin
            $display("FAIL: expected exact output words, mismatches=%0d max_abs=%0d",
                     write_mismatch_count, max_abs_output_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_o_proj_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_o_proj_compute_path matched exact o_proj_out write-back and exact persistent Q4 row reads.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
