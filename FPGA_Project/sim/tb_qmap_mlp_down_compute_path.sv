`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_mlp_down_compute_path #(
    parameter int GROUP_PARALLEL          = 8,
    parameter int NORMAL_TIMEOUT_CYCLES   = 4000000,
    parameter int INVALID_TIMEOUT_CYCLES  = 100000
);

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 6;
    localparam int INPUT_SIZE       = 3072;
    localparam int OUT_FEATURES     = 1024;
    localparam int GROUP_SIZE       = 64;
    localparam int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH        = 24;
    localparam int WEIGHT_WIDTH     = 4;
    localparam int SCALE_WIDTH      = 16;
    localparam int OUT_WIDTH        = 24;
    localparam int MEM_DATA_WIDTH   = 32;
    localparam int MAX_READ_BYTES   = 1024;
    localparam int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;

    localparam int WORD_BYTES              = MEM_DATA_WIDTH / 8;
    localparam int ACTIVATION_BYTES        = INPUT_SIZE * WORD_BYTES;
    localparam int ACTIVATION_BURSTS       = ACTIVATION_BYTES / MAX_READ_BYTES;
    localparam int WEIGHT_ROW_BYTES        = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int WEIGHT_CHUNK_COUNT      = (WEIGHT_ROW_BYTES + MAX_READ_BYTES - 1) / MAX_READ_BYTES;
    localparam int WEIGHT_LAST_CHUNK_BYTES = WEIGHT_ROW_BYTES - ((WEIGHT_CHUNK_COUNT - 1) * MAX_READ_BYTES);
    localparam int SCALE_ROW_BYTES         = GROUP_COUNT * (SCALE_WIDTH / 8);
    localparam int WEIGHT_WORDS            = OUT_FEATURES * WEIGHT_ROW_BYTES / WORD_BYTES;
    localparam int SCALE_WORDS             = OUT_FEATURES * SCALE_ROW_BYTES / WORD_BYTES;
    localparam int OUTPUT_WORDS            = OUT_FEATURES;
    localparam int OUTPUT_BYTES            = OUTPUT_WORDS * WORD_BYTES;
    localparam int QMAP_IMAGE_BYTES        = 32'h0000_6000;
    localparam logic [ADDR_WIDTH-1 : 0] DEFAULT_QMAP_BASE_ADDR = `QMAP_MLP_DOWN_BASE_ADDR;
    localparam int QMAP_WORDS              = QMAP_IMAGE_BYTES / WORD_BYTES;
    localparam int DESCRIPTOR_WORDS        = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESCRIPTOR_TABLE_OFFSET_BYTES = 32'h0100;
    localparam int DESC_DTYPE_WORD         = 2;
    localparam int DESC_BASE_LO_WORD       = 8;
    localparam int DESC_BASE_HI_WORD       = 9;
    localparam int SLOT_ACTIVATION         = 1;
    localparam int SLOT_WEIGHT             = 2;
    localparam int SLOT_SCALE              = 3;
    localparam int SLOT_OUTPUT             = 4;

    localparam int REGION_QMAP       = 0;
    localparam int REGION_ACTIVATION = 1;
    localparam int REGION_WEIGHT     = 2;
    localparam int REGION_SCALE      = 3;

    localparam int QMAP_READER_REQS  = 1 + DESCRIPTOR_SLOTS;
    localparam int QMAP_READER_WORDS = 16 + (DESCRIPTOR_SLOTS * DESCRIPTOR_WORDS);
    localparam int NORMAL_RD_REQS    = QMAP_READER_REQS + ACTIVATION_BURSTS + (OUT_FEATURES * (WEIGHT_CHUNK_COUNT + 1));
    localparam int NORMAL_RD_WORDS   = QMAP_READER_WORDS + INPUT_SIZE + WEIGHT_WORDS + SCALE_WORDS;
    localparam int TOTAL_RD_REQS     = NORMAL_RD_REQS + QMAP_READER_REQS;
    localparam int TOTAL_RD_WORDS    = NORMAL_RD_WORDS + QMAP_READER_WORDS;

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
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;
    logic [7 : 0] state_debug;
    logic [31 : 0] row_index_debug;
    logic [1 : 0] read_slot_debug;

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
    logic [31 : 0] expected_mem [0 : OUTPUT_WORDS-1];

    string vector_dir;
    string prefix;
    string qmap_image_file;
    string expected_file;
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
    integer mem_activation_req_count;
    integer mem_weight_req_count;
    integer mem_scale_req_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer write_mismatch_count;
    integer output_max_abs_diff;

    integer normal_rows_done;
    integer normal_output_write_count;
    integer normal_dut_read_bursts;
    integer normal_dut_read_words;
    integer normal_dut_write_reqs;
    integer normal_dut_write_words;
    logic normal_error;
    logic normal_saturation;
    logic invalid_error;

    logic [ADDR_WIDTH-1 : 0] qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] activation_base_addr;
    logic [ADDR_WIDTH-1 : 0] weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] scale_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_base_addr;

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
    integer last_trace_rows_done;

    qmap_mlp_down_compute_path #(
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
        .i_qmap_base_addr         (qmap_base_addr),
        .o_busy                   (busy),
        .o_done                   (done),
        .o_error                  (error),
        .o_saturation             (saturation),
        .o_rows_done              (rows_done),
        .o_last_row_sum_q26       (last_row_sum_q26),
        .o_last_output_q12_12     (last_output_q12_12),
        .o_output_write_word_count(output_write_word_count),
        .o_mem_read_burst_count   (dut_read_burst_count),
        .o_mem_read_word_count    (dut_read_word_count),
        .o_mem_write_req_count    (dut_write_req_count),
        .o_mem_write_word_count   (dut_write_word_count),
        .o_state_debug            (state_debug),
        .o_row_index_debug        (row_index_debug),
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
        wavefile = "FPGA_Project/wave/qmap_mlp_down_compute_path.vcd";
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
            $dumpvars(0, rows_done);
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

    function automatic logic [31 : 0] memory_word(input integer region, input integer word_index);
        begin
            case (region)
                REGION_QMAP,
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
            $readmemh(expected_file, expected_mem);
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
        integer expected_row;
        integer expected_chunk;
        integer expected_len;
        integer reader_req_index;
        logic [ADDR_WIDTH-1 : 0] expected_addr;
        begin
            active_read_region = REGION_QMAP;
            active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
            active_words_left = mem_rd_req_len_bytes / WORD_BYTES;
            active_total_words = mem_rd_req_len_bytes / WORD_BYTES;

            if ((mem_rd_req_addr >= activation_base_addr) &&
                (mem_rd_req_addr < (activation_base_addr + ACTIVATION_BYTES))) begin
                active_read_region = REGION_ACTIVATION;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
                expected_addr = activation_base_addr + (mem_activation_req_count * MAX_READ_BYTES);
                if ((mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != MAX_READ_BYTES)) begin
                    $display("FAIL: activation read request mismatch idx=%0d addr=0x%016h expected=0x%016h len=%0d",
                             mem_activation_req_count, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_activation_req_count = mem_activation_req_count + 1;
            end
            else if ((mem_rd_req_addr >= weight_base_addr) &&
                     (mem_rd_req_addr < (weight_base_addr + (WEIGHT_WORDS * WORD_BYTES)))) begin
                active_read_region = REGION_WEIGHT;
                active_read_index = (mem_rd_req_addr - weight_base_addr) >> 2;
                expected_row = mem_weight_req_count / WEIGHT_CHUNK_COUNT;
                expected_chunk = mem_weight_req_count % WEIGHT_CHUNK_COUNT;
                expected_len = (expected_chunk == (WEIGHT_CHUNK_COUNT - 1)) ?
                               WEIGHT_LAST_CHUNK_BYTES :
                               MAX_READ_BYTES;
                expected_addr =
                    weight_base_addr +
                    (expected_row * WEIGHT_ROW_BYTES) +
                    (expected_chunk * MAX_READ_BYTES);
                if ((expected_row >= OUT_FEATURES) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != expected_len)) begin
                    $display("FAIL: weight read request mismatch row=%0d chunk=%0d addr=0x%016h expected=0x%016h len=%0d/%0d",
                             expected_row, expected_chunk, mem_rd_req_addr, expected_addr,
                             mem_rd_req_len_bytes, expected_len);
                    mismatch_count = mismatch_count + 1;
                end
                mem_weight_req_count = mem_weight_req_count + 1;
            end
            else if ((mem_rd_req_addr >= scale_base_addr) &&
                     (mem_rd_req_addr < (scale_base_addr + (SCALE_WORDS * WORD_BYTES)))) begin
                active_read_region = REGION_SCALE;
                active_read_index = (mem_rd_req_addr - scale_base_addr) >> 2;
                expected_row = mem_scale_req_count;
                expected_addr = scale_base_addr + (expected_row * SCALE_ROW_BYTES);
                if ((expected_row >= OUT_FEATURES) ||
                    (mem_rd_req_addr !== expected_addr) ||
                    (mem_rd_req_len_bytes != SCALE_ROW_BYTES) ||
                    (mem_weight_req_count <= (mem_scale_req_count * WEIGHT_CHUNK_COUNT))) begin
                    $display("FAIL: scale read request mismatch row=%0d addr=0x%016h expected=0x%016h len=%0d",
                             expected_row, mem_rd_req_addr, expected_addr, mem_rd_req_len_bytes);
                    mismatch_count = mismatch_count + 1;
                end
                mem_scale_req_count = mem_scale_req_count + 1;
            end
            else begin
                active_read_region = REGION_QMAP;
                active_read_index = (mem_rd_req_addr - qmap_base_addr) >> 2;
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
                $display("FAIL: write request accepted while write still active");
                mismatch_count = mismatch_count + 1;
            end
            if ((mem_wr_req_addr !== output_base_addr) ||
                (mem_wr_req_len_bytes != OUTPUT_BYTES)) begin
                $display("FAIL: output write request mismatch addr=0x%016h expected=0x%016h len=%0d/%0d",
                         mem_wr_req_addr, output_base_addr, mem_wr_req_len_bytes, OUTPUT_BYTES);
                mismatch_count = mismatch_count + 1;
            end
            write_active = 1'b1;
            active_write_index = 0;
            active_write_words_left = OUTPUT_WORDS;
            write_done_delay = 0;
            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        begin
            if (active_write_index >= OUTPUT_WORDS) begin
                $display("FAIL: extra output write word index=%0d", active_write_index);
                mismatch_count = mismatch_count + 1;
            end
            else begin
                output_diff = $signed(mem_wr_data) - $signed(expected_mem[active_write_index]);
                if (output_diff < 0) begin
                    output_diff = -output_diff;
                end
                if (output_diff > output_max_abs_diff) begin
                    output_max_abs_diff = output_diff;
                end
                if (mem_wr_data !== expected_mem[active_write_index]) begin
                    if (print_count < 32) begin
                        $display("FAIL: output mismatch idx=%0d actual=0x%08h expected=0x%08h diff=%0d",
                                 active_write_index, mem_wr_data, expected_mem[active_write_index], output_diff);
                        print_count = print_count + 1;
                    end
                    write_mismatch_count = write_mismatch_count + 1;
                    mismatch_count = mismatch_count + 1;
                end
                if (mem_wr_data_last != (active_write_words_left == 1)) begin
                    $display("FAIL: output write last mismatch idx=%0d last=%0d words_left=%0d",
                             active_write_index, mem_wr_data_last, active_write_words_left);
                    mismatch_count = mismatch_count + 1;
                end
                if (active_write_words_left <= 1) begin
                    write_active = 1'b0;
                    write_done_delay = 3;
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
        input integer timeout_delta;
        integer deadline;
        begin
            deadline = cycle_count + timeout_delta;
            while ((done_seen_count == prior_done_count) && (cycle_count < deadline)) begin
                @(negedge clk);
            end
            if (done_seen_count == prior_done_count) begin
                $display("FAIL: timed out waiting for qmap_mlp_down_compute_path done");
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

            repeat (256) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
            end

            wait_for_next_done(prior_done_count, NORMAL_TIMEOUT_CYCLES);
            normal_done_cycle = last_done_cycle;
            normal_rows_done = rows_done;
            normal_output_write_count = output_write_word_count;
            normal_dut_read_bursts = dut_read_burst_count;
            normal_dut_read_words = dut_read_word_count;
            normal_dut_write_reqs = dut_write_req_count;
            normal_dut_write_words = dut_write_word_count;
            normal_error = error;
            normal_saturation = saturation;
        end
    endtask

    task run_invalid_bad_scale_dtype;
        integer prior_done_count;
        integer dtype_word_index;
        logic [31 : 0] saved_dtype;
        integer saved_write_req_count;
        integer saved_write_word_count;
        begin
            prior_done_count = done_seen_count;
            saved_write_req_count = mem_wr_req_count;
            saved_write_word_count = mem_wr_word_count_total;
            dtype_word_index = descriptor_word_index(SLOT_SCALE, DESC_DTYPE_WORD);
            saved_dtype = qmap_mem[dtype_word_index];
            qmap_mem[dtype_word_index] = `QMAP_DTYPE_U16_Q8_8;

            pulse_start();
            wait_for_next_done(prior_done_count, INVALID_TIMEOUT_CYCLES);
            invalid_done_cycle = last_done_cycle;
            invalid_error = error;

            if (error != 1'b1) begin
                $display("FAIL: invalid scale dtype did not assert error");
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
            last_trace_rows_done <= -1;
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
                 (rows_done != last_trace_rows_done))) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    saturation,
                    state_debug,
                    row_index_debug,
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
                    rows_done,
                    output_write_word_count,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    mem_wr_done
                );
                last_trace_rows_done <= rows_done;
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
                    current_read_word = memory_word(active_read_region, active_read_index);
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
        vector_dir = "FPGA_Project/sim/vectors";
        prefix = "mlp_down_proj_stage_real";
        qmap_image_file = "FPGA_Project/sim/vectors/qmap_mlp_down_image_words32.hex";
        expected_file = "FPGA_Project/sim/vectors/qmap_mlp_down_expected_words32.hex";
        tracefile = "FPGA_Project/sim/qmap_mlp_down_compute_path_trace.csv";
        qmap_base_addr = DEFAULT_QMAP_BASE_ADDR;
        if ($value$plusargs("vectordir=%s", vector_dir)) begin
        end
        if ($value$plusargs("prefix=%s", prefix)) begin
        end
        if ($value$plusargs("qmap_image=%s", qmap_image_file)) begin
        end
        if ($value$plusargs("expected=%s", expected_file)) begin
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
            "cycle,start,busy,done,error,saturation,state,row,read_slot,rd_addr,rd_len,rd_req_fire,rd_rsp_last_fire,wr_addr,wr_len,wr_req_fire,wr_data_fire,wr_last,rows_done,output_write_count,rd_bursts,rd_words,wr_reqs,wr_words,wr_done\n"
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
        mem_activation_req_count = 0;
        mem_weight_req_count = 0;
        mem_scale_req_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        write_mismatch_count = 0;
        output_max_abs_diff = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        normal_rows_done = 0;
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
        run_invalid_bad_scale_dtype();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_mlp_down_compute_path descriptor-backed MLP down test");
        $display("  qmap base               = 0x%016h", qmap_base_addr);
        $display("  rows done normal        = %0d", normal_rows_done);
        $display("  output words written    = %0d", normal_output_write_count);
        $display("  read req/rsp fires      = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  qmap/activation reqs    = %0d / %0d", mem_qmap_req_count, mem_activation_req_count);
        $display("  weight/scale reqs       = %0d / %0d", mem_weight_req_count, mem_scale_req_count);
        $display("  write reqs/words        = %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  dut normal rd/wr        = %0d/%0d reads, %0d/%0d writes",
                 normal_dut_read_bursts, normal_dut_read_words,
                 normal_dut_write_reqs, normal_dut_write_words);
        $display("  write mismatches        = %0d", write_mismatch_count);
        $display("  max_abs output          = %0d", output_max_abs_diff);
        $display("  spurious start covered  = %0d", spurious_start_seen_busy);
        $display("  invalid descriptor error= %0d", invalid_error);
        $display("  done cycles normal/bad  = %0d / %0d", normal_done_cycle, invalid_done_cycle);
        $display("  trace                   = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_error) begin
            $display("FAIL: error asserted on valid MLP down packet");
            mismatch_count = mismatch_count + 1;
        end
        if (normal_saturation) begin
            $display("FAIL: unexpected MLP down saturation");
            mismatch_count = mismatch_count + 1;
        end
        if (normal_rows_done != OUT_FEATURES) begin
            $display("FAIL: rows_done mismatch actual=%0d expected=%0d", normal_rows_done, OUT_FEATURES);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_output_write_count != OUTPUT_WORDS) begin
            $display("FAIL: output write counter mismatch actual=%0d expected=%0d",
                     normal_output_write_count, OUTPUT_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_qmap_req_count != (2 * QMAP_READER_REQS)) ||
            (mem_activation_req_count != ACTIVATION_BURSTS) ||
            (mem_weight_req_count != (OUT_FEATURES * WEIGHT_CHUNK_COUNT)) ||
            (mem_scale_req_count != OUT_FEATURES)) begin
            $display("FAIL: read request class counts mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_req_fire_count != TOTAL_RD_REQS) ||
            (mem_rsp_fire_count != TOTAL_RD_WORDS)) begin
            $display("FAIL: total read count mismatch req=%0d/%0d rsp=%0d/%0d",
                     mem_req_fire_count, TOTAL_RD_REQS, mem_rsp_fire_count, TOTAL_RD_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((mem_wr_req_count != 1) || (mem_wr_word_count_total != OUTPUT_WORDS)) begin
            $display("FAIL: write count mismatch req=%0d words=%0d expected=1/%0d",
                     mem_wr_req_count, mem_wr_word_count_total, OUTPUT_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_dut_read_bursts != NORMAL_RD_REQS) ||
            (normal_dut_read_words != NORMAL_RD_WORDS) ||
            (normal_dut_write_reqs != 1) ||
            (normal_dut_write_words != OUTPUT_WORDS)) begin
            $display("FAIL: DUT normal counters mismatch rd=%0d/%0d wr=%0d/%0d expected rd=%0d/%0d wr=1/%0d",
                     normal_dut_read_bursts, normal_dut_read_words,
                     normal_dut_write_reqs, normal_dut_write_words,
                     NORMAL_RD_REQS, NORMAL_RD_WORDS, OUTPUT_WORDS);
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (output_max_abs_diff != 0)) begin
            $display("FAIL: exact write-back expected, mismatches=%0d abs=%0d",
                     write_mismatch_count, output_max_abs_diff);
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
            $display("FAIL: qmap_mlp_down_compute_path found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_mlp_down_compute_path matched exact output write-back and exact persistent Q4 row reads.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
