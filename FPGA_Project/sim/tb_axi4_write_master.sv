`timescale 1ns/1ps
`default_nettype none

module tb_axi4_write_master;

    localparam int ADDR_WIDTH = 64;
    localparam int DATA_WIDTH = 32;
    localparam logic [1 : 0] AXI_RESP_OKAY = 2'b00;
    localparam logic [1 : 0] AXI_RESP_SLVERR = 2'b10;

    logic clk;
    logic rst_n;

    logic req_valid;
    logic req_ready;
    logic [ADDR_WIDTH-1 : 0] req_addr;
    logic [15 : 0] req_len_bytes;
    logic [DATA_WIDTH-1 : 0] wdata;
    logic wdata_valid;
    logic wdata_ready;
    logic wdata_last;
    logic done;
    logic busy;
    logic error;

    logic [ADDR_WIDTH-1 : 0] m_axi_awaddr;
    logic [7 : 0] m_axi_awlen;
    logic [2 : 0] m_axi_awsize;
    logic [1 : 0] m_axi_awburst;
    logic [2 : 0] m_axi_awprot;
    logic [3 : 0] m_axi_awcache;
    logic m_axi_awvalid;
    logic m_axi_awready;
    logic [DATA_WIDTH-1 : 0] m_axi_wdata;
    logic [DATA_WIDTH/8-1 : 0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready;
    logic [1 : 0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;

    logic enable_stalls;
    integer cycle_count;
    integer fail_count;
    integer aw_seen;
    integer active_burst_index;
    integer burst_beat_index;
    integer total_beats_seen;
    integer expected_burst_count;
    integer expected_error_burst;
    logic [31 : 0] expected_data_base;
    logic [ADDR_WIDTH-1 : 0] expected_awaddr [0 : 7];
    integer expected_awbeats [0 : 7];
    logic previous_aw_stalled;
    logic [ADDR_WIDTH-1 : 0] previous_awaddr;
    logic [7 : 0] previous_awlen;
    logic previous_w_stalled;
    logic [DATA_WIDTH-1 : 0] previous_wdata;
    logic previous_wlast;

    axi4_write_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_req_valid(req_valid),
        .o_req_ready(req_ready),
        .i_req_addr(req_addr),
        .i_req_len_bytes(req_len_bytes),
        .i_wdata(wdata),
        .i_wdata_valid(wdata_valid),
        .o_wdata_ready(wdata_ready),
        .i_wdata_last(wdata_last),
        .o_done(done),
        .o_busy(busy),
        .o_error(error),
        .o_m_axi_awaddr(m_axi_awaddr),
        .o_m_axi_awlen(m_axi_awlen),
        .o_m_axi_awsize(m_axi_awsize),
        .o_m_axi_awburst(m_axi_awburst),
        .o_m_axi_awprot(m_axi_awprot),
        .o_m_axi_awcache(m_axi_awcache),
        .o_m_axi_awvalid(m_axi_awvalid),
        .i_m_axi_awready(m_axi_awready),
        .o_m_axi_wdata(m_axi_wdata),
        .o_m_axi_wstrb(m_axi_wstrb),
        .o_m_axi_wlast(m_axi_wlast),
        .o_m_axi_wvalid(m_axi_wvalid),
        .i_m_axi_wready(m_axi_wready),
        .i_m_axi_bresp(m_axi_bresp),
        .i_m_axi_bvalid(m_axi_bvalid),
        .o_m_axi_bready(m_axi_bready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    assign m_axi_awready = !enable_stalls || ((cycle_count % 5) != 1);
    assign m_axi_wready = !enable_stalls || ((cycle_count % 7) != 2);

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

    task automatic reset_case;
        integer index;
        begin
            rst_n = 1'b0;
            req_valid = 1'b0;
            req_addr = '0;
            req_len_bytes = 16'd0;
            wdata = 32'd0;
            wdata_valid = 1'b0;
            wdata_last = 1'b0;
            enable_stalls = 1'b0;
            expected_burst_count = 0;
            expected_error_burst = -1;
            expected_data_base = 32'd0;
            for (index = 0; index < 8; index = index + 1) begin
                expected_awaddr[index] = '0;
                expected_awbeats[index] = 0;
            end
            repeat (4) wait_clk();
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) wait_clk();
        end
    endtask

    task automatic start_request;
        input logic [ADDR_WIDTH-1 : 0] address;
        input logic [15 : 0] length_bytes;
        begin
            @(negedge clk);
            req_addr = address;
            req_len_bytes = length_bytes;
            req_valid = 1'b1;
            while (!req_ready) begin
                wait_clk();
            end
            wait_clk();
            @(negedge clk);
            req_valid = 1'b0;
        end
    endtask

    task automatic send_words;
        input integer word_count;
        input logic [31 : 0] data_base;
        input logic assert_final_last;
        integer index;
        begin
            for (index = 0; index < word_count; index = index + 1) begin
                @(negedge clk);
                wdata = data_base + index;
                wdata_last = assert_final_last && (index == (word_count - 1));
                wdata_valid = 1'b1;
                @(posedge clk);
                while (!wdata_ready) begin
                    @(posedge clk);
                end
                @(negedge clk);
                wdata_valid = 1'b0;
                wdata_last = 1'b0;
            end
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
            check(done, "timed out waiting for write completion");
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
            aw_seen <= 0;
            active_burst_index <= -1;
            burst_beat_index <= 0;
            total_beats_seen <= 0;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= AXI_RESP_OKAY;
            previous_aw_stalled <= 1'b0;
            previous_awaddr <= '0;
            previous_awlen <= 8'd0;
            previous_w_stalled <= 1'b0;
            previous_wdata <= 32'd0;
            previous_wlast <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (previous_aw_stalled) begin
                check(m_axi_awvalid, "AWVALID dropped while AWREADY was low");
                check(m_axi_awaddr == previous_awaddr, "AWADDR changed while stalled");
                check(m_axi_awlen == previous_awlen, "AWLEN changed while stalled");
            end
            previous_aw_stalled <= m_axi_awvalid && !m_axi_awready;
            previous_awaddr <= m_axi_awaddr;
            previous_awlen <= m_axi_awlen;

            if (previous_w_stalled) begin
                check(m_axi_wvalid, "WVALID dropped while WREADY was low");
                check(m_axi_wdata == previous_wdata, "WDATA changed while stalled");
                check(m_axi_wlast == previous_wlast, "WLAST changed while stalled");
            end
            previous_w_stalled <= m_axi_wvalid && !m_axi_wready;
            previous_wdata <= m_axi_wdata;
            previous_wlast <= m_axi_wlast;

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end

            if (m_axi_awvalid && m_axi_awready) begin
                check(aw_seen < expected_burst_count, "unexpected extra AW burst");
                if (aw_seen < expected_burst_count) begin
                    check(m_axi_awaddr == expected_awaddr[aw_seen], "AWADDR does not match expected split");
                    check(({1'b0, m_axi_awlen} + 1'b1) == expected_awbeats[aw_seen], "AWLEN does not match expected split");
                end
                check(m_axi_awsize == 3'd2, "AWSIZE must describe 32-bit beats");
                check(m_axi_awburst == 2'b01, "AWBURST must be INCR");
                check((m_axi_awaddr[11 : 0] + (({1'b0, m_axi_awlen} + 1'b1) * 4)) <= 4096,
                      "AXI burst crossed a 4 KiB boundary");
                active_burst_index <= aw_seen;
                aw_seen <= aw_seen + 1;
                burst_beat_index <= 0;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                check(active_burst_index >= 0, "write data arrived before AW handshake");
                check(m_axi_wdata == (expected_data_base + total_beats_seen), "write data sequence mismatch");
                check(m_axi_wstrb == 4'hF, "write strobe must cover the full word");
                if (active_burst_index >= 0) begin
                    check(
                        m_axi_wlast == (burst_beat_index == (expected_awbeats[active_burst_index] - 1)),
                        "AXI WLAST does not match the current split burst"
                    );
                end
                total_beats_seen <= total_beats_seen + 1;
                if (m_axi_wlast) begin
                    m_axi_bresp <= (active_burst_index == expected_error_burst) ? AXI_RESP_SLVERR : AXI_RESP_OKAY;
                    m_axi_bvalid <= 1'b1;
                    active_burst_index <= -1;
                    burst_beat_index <= 0;
                end else begin
                    burst_beat_index <= burst_beat_index + 1;
                end
            end
        end
    end

    initial begin
        fail_count = 0;

        reset_case();
        expected_burst_count = 1;
        expected_awaddr[0] = 64'h0000_0004_0008_8000;
        expected_awbeats[0] = 4;
        expected_data_base = 32'hCAFE_1000;
        enable_stalls = 1'b1;
        start_request(expected_awaddr[0], 16'd16);
        send_words(4, expected_data_base, 1'b1);
        wait_for_done(100);
        check(!error, "single-burst request reported an error");
        check(aw_seen == 1, "single-burst AW count mismatch");
        check(total_beats_seen == 4, "single-burst data count mismatch");

        reset_case();
        expected_burst_count = 5;
        expected_awaddr[0] = 64'h0000_0004_2509_2540;
        expected_awaddr[1] = 64'h0000_0004_2509_2940;
        expected_awaddr[2] = 64'h0000_0004_2509_2D40;
        expected_awaddr[3] = 64'h0000_0004_2509_3000;
        expected_awaddr[4] = 64'h0000_0004_2509_3400;
        expected_awbeats[0] = 256;
        expected_awbeats[1] = 256;
        expected_awbeats[2] = 176;
        expected_awbeats[3] = 256;
        expected_awbeats[4] = 80;
        expected_data_base = 32'h2500_0000;
        enable_stalls = 1'b1;
        start_request(expected_awaddr[0], 16'd4096);
        send_words(1024, expected_data_base, 1'b1);
        wait_for_done(500);
        check(!error, "4 KiB split request reported an error");
        check(aw_seen == 5, "4 KiB split AW count mismatch");
        check(total_beats_seen == 1024, "4 KiB split data count mismatch");

        reset_case();
        expected_burst_count = 2;
        expected_error_burst = 1;
        expected_awaddr[0] = 64'h0000_0004_3000_0000;
        expected_awaddr[1] = 64'h0000_0004_3000_0400;
        expected_awbeats[0] = 256;
        expected_awbeats[1] = 256;
        expected_data_base = 32'hE000_0000;
        start_request(expected_awaddr[0], 16'd2048);
        send_words(512, expected_data_base, 1'b0);
        wait_for_done(100);
        check(error, "intermediate BRESP error was not propagated");
        check(aw_seen == 2, "BRESP error test AW count mismatch");
        check(total_beats_seen == 512, "BRESP error test accepted unexpected data");

        reset_case();
        expected_burst_count = 0;
        @(negedge clk);
        req_addr = 64'h0000_0004_0000_0002;
        req_len_bytes = 16'd16;
        req_valid = 1'b1;
        wait_clk();
        check(req_ready && done && error, "unaligned request must finish immediately with error");
        @(negedge clk);
        req_valid = 1'b0;
        check(aw_seen == 0, "unaligned request issued an AXI burst");

        reset_case();
        expected_burst_count = 0;
        @(negedge clk);
        req_addr = 64'h0000_0004_0000_0000;
        req_len_bytes = 16'd0;
        req_valid = 1'b1;
        wait_clk();
        check(req_ready && done && error, "zero-length request must finish immediately with error");
        @(negedge clk);
        req_valid = 1'b0;
        check(aw_seen == 0, "zero-length request issued an AXI burst");

        reset_case();
        expected_burst_count = 1;
        expected_awaddr[0] = 64'h0000_0004_0000_1000;
        expected_awbeats[0] = 4;
        expected_data_base = 32'hBAD0_0000;
        start_request(expected_awaddr[0], 16'd16);
        send_words(4, expected_data_base, 1'b0);
        wait_for_done(100);
        check(error, "missing request-level last was not detected");

        if (fail_count != 0) begin
            $display("FAIL: tb_axi4_write_master saw %0d failure(s).", fail_count);
            $finish(1);
        end

        $display("PASS: axi4_write_master split 4 KiB writes at 256 beats and 4 KiB boundaries with stalls and errors covered.");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT: tb_axi4_write_master did not finish");
        $finish(1);
    end

endmodule

`default_nettype wire
