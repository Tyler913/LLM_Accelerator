`timescale 1ns/1ps
`default_nettype none

module tb_q4_embedding_lookup;

    localparam int ADDR_WIDTH = 64;
    localparam int MEM_DATA_WIDTH = 32;
    localparam int VOCAB_SIZE = 151936;
    localparam int INPUT_SIZE = 1024;
    localparam int WEIGHT_WORDS = 128;
    localparam int SCALE_WORDS = 8;
    localparam logic [ADDR_WIDTH-1 : 0] WEIGHT_BASE = 64'h0000_0004_0010_0000;
    localparam logic [ADDR_WIDTH-1 : 0] SCALE_BASE = 64'h0000_0004_04B3_0000;
    localparam logic [ADDR_WIDTH-1 : 0] OUTPUT_BASE = 64'h0000_0004_2000_0540;

    logic clk;
    logic rst_n;
    logic start;
    logic [31 : 0] token_id;
    logic busy;
    logic done;
    logic error;

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
    logic [MEM_DATA_WIDTH-1 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    logic [31 : 0] read_burst_count;
    logic [31 : 0] read_word_count;
    logic [31 : 0] write_req_count;
    logic [31 : 0] write_word_count;

    logic [31 : 0] weight_words [0 : WEIGHT_WORDS-1];
    logic [31 : 0] scale_words [0 : SCALE_WORDS-1];
    logic [31 : 0] expected_words [0 : INPUT_SIZE-1];
    logic [31 : 0] token_file [0 : 0];

    integer cycle_count;
    integer fail_count;
    integer mismatch_count;
    integer observed_read_reqs;
    integer observed_read_words;
    integer observed_write_reqs;
    integer observed_write_words;
    integer read_target;
    integer read_index;
    integer write_index;
    integer write_done_delay;
    integer weight_req_cycle;
    integer scale_req_cycle;
    integer write_req_cycle;
    integer first_write_cycle;
    integer last_write_cycle;
    integer done_cycle;
    integer done_seen;
    logic read_active;
    logic write_active;
    logic malformed_weight_last;
    logic inject_write_error;
    logic previous_rd_req_stall;
    logic [ADDR_WIDTH-1 : 0] previous_rd_req_addr;
    logic [15 : 0] previous_rd_req_len;
    logic previous_wr_req_stall;
    logic [ADDR_WIDTH-1 : 0] previous_wr_req_addr;
    logic [15 : 0] previous_wr_req_len;
    logic previous_wr_data_stall;
    logic [31 : 0] previous_wr_data;
    logic previous_wr_last;
    string vector_dir;
    string vector_path;

    q4_embedding_lookup dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_token_id(token_id),
        .i_weight_base_addr(WEIGHT_BASE),
        .i_scale_base_addr(SCALE_BASE),
        .i_output_base_addr(OUTPUT_BASE),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
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
        .i_mem_wr_error(mem_wr_error),
        .o_read_burst_count(read_burst_count),
        .o_read_word_count(read_word_count),
        .o_write_req_count(write_req_count),
        .o_write_word_count(write_word_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    assign mem_rd_req_ready = !read_active && !mem_rd_rsp_valid && ((cycle_count % 5) != 1);
    assign mem_wr_req_ready = !write_active && ((cycle_count % 7) != 2);
    assign mem_wr_data_ready = write_active && ((cycle_count % 6) != 3);

    task automatic check;
        input logic condition;
        input string message;
        begin
            if (!condition) begin
                $display("FAIL: %s", message);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task automatic wait_clk;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task automatic pulse_start;
        input logic [31 : 0] selected_token;
        begin
            @(negedge clk);
            token_id = selected_token;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic reset_case;
        begin
            rst_n = 1'b0;
            start = 1'b0;
            token_id = token_file[0];
            malformed_weight_last = 1'b0;
            inject_write_error = 1'b0;
            repeat (5) wait_clk();
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) wait_clk();
        end
    endtask

    task automatic wait_for_done;
        input integer timeout_cycles;
        integer waited;
        begin
            waited = 0;
            while (!done && (waited < timeout_cycles)) begin
                wait_clk();
                waited = waited + 1;
            end
            check(done, "embedding lookup timed out");
        end
    endtask

    initial begin
        vector_dir = "vectors";
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir)) begin
        end
        vector_path = {vector_dir, "/embedding_weight_words32.hex"};
        $readmemh(vector_path, weight_words);
        vector_path = {vector_dir, "/embedding_scale_words32.hex"};
        $readmemh(vector_path, scale_words);
        vector_path = {vector_dir, "/embedding_expected_q14_10.hex"};
        $readmemh(vector_path, expected_words);
        vector_path = {vector_dir, "/embedding_token_id.hex"};
        $readmemh(vector_path, token_file);
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
            mismatch_count <= 0;
            observed_read_reqs <= 0;
            observed_read_words <= 0;
            observed_write_reqs <= 0;
            observed_write_words <= 0;
            read_target <= 0;
            read_index <= 0;
            write_index <= 0;
            write_done_delay <= 0;
            weight_req_cycle <= -1;
            scale_req_cycle <= -1;
            write_req_cycle <= -1;
            first_write_cycle <= -1;
            last_write_cycle <= -1;
            done_cycle <= -1;
            done_seen <= 0;
            read_active <= 1'b0;
            write_active <= 1'b0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            previous_rd_req_stall <= 1'b0;
            previous_rd_req_addr <= '0;
            previous_rd_req_len <= 16'd0;
            previous_wr_req_stall <= 1'b0;
            previous_wr_req_addr <= '0;
            previous_wr_req_len <= 16'd0;
            previous_wr_data_stall <= 1'b0;
            previous_wr_data <= 32'd0;
            previous_wr_last <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (done) begin
                done_seen <= done_seen + 1;
                done_cycle <= cycle_count;
            end

            if (previous_rd_req_stall) begin
                check(mem_rd_req_valid, "read request valid dropped while stalled");
                check(mem_rd_req_addr == previous_rd_req_addr, "read request address changed while stalled");
                check(mem_rd_req_len_bytes == previous_rd_req_len, "read request length changed while stalled");
            end
            previous_rd_req_stall <= mem_rd_req_valid && !mem_rd_req_ready;
            previous_rd_req_addr <= mem_rd_req_addr;
            previous_rd_req_len <= mem_rd_req_len_bytes;

            if (previous_wr_req_stall) begin
                check(mem_wr_req_valid, "write request valid dropped while stalled");
                check(mem_wr_req_addr == previous_wr_req_addr, "write request address changed while stalled");
                check(mem_wr_req_len_bytes == previous_wr_req_len, "write request length changed while stalled");
            end
            previous_wr_req_stall <= mem_wr_req_valid && !mem_wr_req_ready;
            previous_wr_req_addr <= mem_wr_req_addr;
            previous_wr_req_len <= mem_wr_req_len_bytes;

            if (previous_wr_data_stall) begin
                check(mem_wr_data_valid, "write data valid dropped while stalled");
                check(mem_wr_data == previous_wr_data, "write data changed while stalled");
                check(mem_wr_data_last == previous_wr_last, "write data last changed while stalled");
            end
            previous_wr_data_stall <= mem_wr_data_valid && !mem_wr_data_ready;
            previous_wr_data <= mem_wr_data;
            previous_wr_last <= mem_wr_data_last;

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                observed_read_reqs <= observed_read_reqs + 1;
                read_active <= 1'b1;
                read_index <= 0;
                if (mem_rd_req_addr == (WEIGHT_BASE + (token_file[0] * 512))) begin
                    read_target <= 1;
                    weight_req_cycle <= cycle_count;
                    check(mem_rd_req_len_bytes == 16'd512, "embedding weight read length mismatch");
                end else if (mem_rd_req_addr == (SCALE_BASE + (token_file[0] * 32))) begin
                    read_target <= 2;
                    scale_req_cycle <= cycle_count;
                    check(mem_rd_req_len_bytes == 16'd32, "embedding scale read length mismatch");
                end else begin
                    read_target <= 0;
                    check(1'b0, "embedding issued an unexpected read address");
                end
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                observed_read_words <= observed_read_words + 1;
                mem_rd_rsp_valid <= 1'b0;
                if (mem_rd_rsp_last) begin
                    read_active <= 1'b0;
                end else begin
                    read_index <= read_index + 1;
                end
            end

            if (read_active && !mem_rd_rsp_valid && ((cycle_count % 4) != 2)) begin
                mem_rd_rsp_valid <= 1'b1;
                if (read_target == 1) begin
                    mem_rd_rsp_data <= weight_words[read_index];
                    mem_rd_rsp_last <= malformed_weight_last ? (read_index == 17) : (read_index == (WEIGHT_WORDS - 1));
                end else begin
                    mem_rd_rsp_data <= scale_words[read_index];
                    mem_rd_rsp_last <= (read_index == (SCALE_WORDS - 1));
                end
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                observed_write_reqs <= observed_write_reqs + 1;
                write_active <= 1'b1;
                write_index <= 0;
                write_req_cycle <= cycle_count;
                check(mem_wr_req_addr == OUTPUT_BASE, "embedding output write address mismatch");
                check(mem_wr_req_len_bytes == 16'd4096, "embedding output write length mismatch");
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                if (write_index == 0) begin
                    first_write_cycle <= cycle_count;
                end
                observed_write_words <= observed_write_words + 1;
                if (mem_wr_data !== expected_words[write_index]) begin
                    if (mismatch_count < 8) begin
                        $display("FAIL: embedding output[%0d] actual=0x%08h expected=0x%08h",
                                 write_index, mem_wr_data, expected_words[write_index]);
                    end
                    mismatch_count <= mismatch_count + 1;
                end
                check(mem_wr_data_last == (write_index == (INPUT_SIZE - 1)),
                      "embedding output last mismatch");
                if (mem_wr_data_last) begin
                    write_active <= 1'b0;
                    write_done_delay <= 3;
                    last_write_cycle <= cycle_count;
                end else begin
                    write_index <= write_index + 1;
                end
            end

            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) begin
                    mem_wr_done <= 1'b1;
                    mem_wr_error <= inject_write_error;
                end
            end
        end
    end

    initial begin
        fail_count = 0;
        rst_n = 1'b0;
        start = 1'b0;
        token_id = 32'd0;
        malformed_weight_last = 1'b0;
        inject_write_error = 1'b0;

        reset_case();
        pulse_start(token_file[0]);
        repeat (8) wait_clk();
        pulse_start(token_file[0] + 1'b1);
        wait_for_done(10000);
        wait_clk();
        check(!error, "valid embedding lookup reported an error");
        check(mismatch_count == 0, "valid embedding lookup output mismatch");
        check(read_burst_count == 32'd2, "embedding RTL read burst counter mismatch");
        check(read_word_count == 32'd136, "embedding RTL read word counter mismatch");
        check(write_req_count == 32'd1, "embedding RTL write request counter mismatch");
        check(write_word_count == 32'd1024, "embedding RTL write word counter mismatch");
        check(observed_read_reqs == 2, "memory model read request count mismatch");
        check(observed_read_words == 136, "memory model read word count mismatch");
        check(observed_write_reqs == 1, "memory model write request count mismatch");
        check(observed_write_words == 1024, "memory model write word count mismatch");
        check((weight_req_cycle >= 0) && (weight_req_cycle < scale_req_cycle),
              "embedding scale read did not follow weight read");
        check((scale_req_cycle < write_req_cycle) && (write_req_cycle < first_write_cycle),
              "embedding write did not follow scale read");
        check((first_write_cycle < last_write_cycle) && (last_write_cycle < done_cycle),
              "embedding done did not follow full output write");
        repeat (3) wait_clk();
        check(done_seen == 1, "embedding done must pulse exactly once");

        reset_case();
        pulse_start(VOCAB_SIZE);
        wait_for_done(20);
        check(error, "out-of-range token did not report an error");
        check(observed_read_reqs == 0 && observed_write_reqs == 0,
              "out-of-range token issued memory traffic");

        reset_case();
        malformed_weight_last = 1'b1;
        pulse_start(token_file[0]);
        wait_for_done(200);
        check(error, "early weight read last did not report an error");
        check(observed_read_reqs == 1, "malformed weight run issued an unexpected scale read");
        check(observed_write_reqs == 0, "malformed weight run issued a write");

        reset_case();
        inject_write_error = 1'b1;
        pulse_start(token_file[0]);
        wait_for_done(10000);
        check(error, "write response error was not propagated");
        check(mismatch_count == 0, "write-error run changed embedding data");
        check(observed_write_words == 1024, "write-error run did not send the full vector");

        if (fail_count != 0) begin
            $display("FAIL: tb_q4_embedding_lookup saw %0d failure(s).", fail_count);
            $finish(1);
        end

        $display("PASS: q4_embedding_lookup produced exact Q14.10 tied embeddings with stalls and errors covered.");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT: tb_q4_embedding_lookup did not finish");
        $finish(1);
    end

endmodule

`default_nettype wire
