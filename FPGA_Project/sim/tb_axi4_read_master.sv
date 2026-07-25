`timescale 1ns/1ps
`default_nettype none

module tb_axi4_read_master;

    localparam int ADDR_WIDTH = 64;
    localparam int DATA_WIDTH = 32;
    localparam logic [1 : 0] AXI_RESP_OKAY   = 2'b00;
    localparam logic [1 : 0] AXI_RESP_SLVERR = 2'b10;
    localparam int MALFORM_NONE  = 0;
    localparam int MALFORM_EARLY = 1;
    localparam int MALFORM_LATE  = 2;

    logic clk;
    logic rst_n;
    logic req_valid;
    logic req_ready;
    logic [ADDR_WIDTH-1 : 0] req_addr;
    logic [15 : 0] req_len_bytes;
    logic rsp_valid;
    logic rsp_ready;
    logic [DATA_WIDTH-1 : 0] rsp_data;
    logic rsp_last;
    logic busy;
    logic error;
    logic [ADDR_WIDTH-1 : 0] m_axi_araddr;
    logic [7 : 0] m_axi_arlen;
    logic [2 : 0] m_axi_arsize;
    logic [1 : 0] m_axi_arburst;
    logic [2 : 0] m_axi_arprot;
    logic [3 : 0] m_axi_arcache;
    logic m_axi_arvalid;
    logic m_axi_arready;
    logic [DATA_WIDTH-1 : 0] m_axi_rdata;
    logic [1 : 0] m_axi_rresp;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    logic m_axi_rready;

    logic enable_stalls;
    logic slave_active;
    integer cycle_count;
    integer fail_count;
    integer ar_seen;
    integer active_burst_index;
    integer slave_burst_beats;
    integer slave_beat_index;
    integer slave_global_beat_index;
    integer received_count;
    integer last_count;
    integer expected_burst_count;
    integer expected_error_burst;
    integer malformed_mode;
    integer malformed_burst;
    integer expected_rsp_count;
    integer expected_real_rsp_count;
    logic [31 : 0] expected_data_base;
    logic [ADDR_WIDTH-1 : 0] expected_araddr [0 : 127];
    integer expected_arbeats [0 : 127];
    logic previous_ar_stalled;
    logic [ADDR_WIDTH-1 : 0] previous_araddr;
    logic [7 : 0] previous_arlen;
    logic previous_rsp_stalled;
    logic [DATA_WIDTH-1 : 0] previous_rsp_data;
    logic previous_rsp_last;

    axi4_read_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_req_valid(req_valid),
        .o_req_ready(req_ready),
        .i_req_addr(req_addr),
        .i_req_len_bytes(req_len_bytes),
        .o_rsp_valid(rsp_valid),
        .i_rsp_ready(rsp_ready),
        .o_rsp_data(rsp_data),
        .o_rsp_last(rsp_last),
        .o_busy(busy),
        .o_error(error),
        .o_m_axi_araddr(m_axi_araddr),
        .o_m_axi_arlen(m_axi_arlen),
        .o_m_axi_arsize(m_axi_arsize),
        .o_m_axi_arburst(m_axi_arburst),
        .o_m_axi_arprot(m_axi_arprot),
        .o_m_axi_arcache(m_axi_arcache),
        .o_m_axi_arvalid(m_axi_arvalid),
        .i_m_axi_arready(m_axi_arready),
        .i_m_axi_rdata(m_axi_rdata),
        .i_m_axi_rresp(m_axi_rresp),
        .i_m_axi_rlast(m_axi_rlast),
        .i_m_axi_rvalid(m_axi_rvalid),
        .o_m_axi_rready(m_axi_rready)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    assign m_axi_arready = !slave_active &&
                           (!enable_stalls || ((cycle_count % 5) != 1));
    assign rsp_ready = !enable_stalls || ((cycle_count % 7) != 2);
    assign m_axi_rvalid = slave_active;
    assign m_axi_rdata = expected_data_base + received_count;
    assign m_axi_rresp = (slave_active && (active_burst_index == expected_error_burst)) ?
                         AXI_RESP_SLVERR : AXI_RESP_OKAY;
    assign m_axi_rlast = slave_active && (
        ((malformed_mode == MALFORM_EARLY) && (active_burst_index == malformed_burst)) ?
            (slave_beat_index == 1) :
        ((malformed_mode == MALFORM_LATE) && (active_burst_index == malformed_burst)) ?
            (slave_beat_index == slave_burst_beats) :
            (slave_beat_index == (slave_burst_beats - 1))
    );

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
            enable_stalls = 1'b0;
            expected_burst_count = 0;
            expected_error_burst = -1;
            malformed_mode = MALFORM_NONE;
            malformed_burst = -1;
            expected_rsp_count = 0;
            expected_real_rsp_count = 0;
            expected_data_base = 32'd0;
            for (index = 0; index < 128; index = index + 1) begin
                expected_araddr[index] = '0;
                expected_arbeats[index] = 0;
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

    task automatic wait_for_responses;
        input integer timeout_cycles;
        integer waited;
        begin
            waited = 0;
            while ((received_count < expected_rsp_count) && (waited < timeout_cycles)) begin
                wait_clk();
                waited = waited + 1;
            end
            check(received_count == expected_rsp_count, "timed out waiting for local read responses");
        end
    endtask

    task automatic wait_for_idle;
        input integer timeout_cycles;
        integer waited;
        begin
            waited = 0;
            while (busy && (waited < timeout_cycles)) begin
                wait_clk();
                waited = waited + 1;
            end
            check(!busy, "timed out waiting for read master to become idle");
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
            ar_seen <= 0;
            active_burst_index <= -1;
            slave_active <= 1'b0;
            slave_burst_beats <= 0;
            slave_beat_index <= 0;
            slave_global_beat_index <= 0;
            received_count <= 0;
            last_count <= 0;
            previous_ar_stalled <= 1'b0;
            previous_araddr <= '0;
            previous_arlen <= 8'd0;
            previous_rsp_stalled <= 1'b0;
            previous_rsp_data <= '0;
            previous_rsp_last <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (previous_ar_stalled) begin
                check(m_axi_arvalid, "ARVALID dropped while ARREADY was low");
                check(m_axi_araddr == previous_araddr, "ARADDR changed while stalled");
                check(m_axi_arlen == previous_arlen, "ARLEN changed while stalled");
            end
            previous_ar_stalled <= m_axi_arvalid && !m_axi_arready;
            previous_araddr <= m_axi_araddr;
            previous_arlen <= m_axi_arlen;

            if (previous_rsp_stalled) begin
                check(rsp_valid, "local response valid dropped while ready was low");
                check(rsp_data == previous_rsp_data, "local response data changed while stalled");
                check(rsp_last == previous_rsp_last, "local response last changed while stalled");
            end
            previous_rsp_stalled <= rsp_valid && !rsp_ready;
            previous_rsp_data <= rsp_data;
            previous_rsp_last <= rsp_last;

            if (m_axi_arvalid && m_axi_arready) begin
                check(ar_seen < expected_burst_count, "unexpected extra AR burst");
                if (ar_seen < expected_burst_count) begin
                    check(m_axi_araddr == expected_araddr[ar_seen], "ARADDR does not match expected split");
                    check(({1'b0, m_axi_arlen} + 1'b1) == expected_arbeats[ar_seen],
                          "ARLEN does not match expected split");
                end
                check(m_axi_arsize == 3'd2, "ARSIZE must describe 32-bit beats");
                check(m_axi_arburst == 2'b01, "ARBURST must be INCR");
                check((m_axi_araddr[11 : 0] + (({1'b0, m_axi_arlen} + 1'b1) * 4)) <= 4096,
                      "AXI read burst crossed a 4 KiB boundary");
                active_burst_index <= ar_seen;
                ar_seen <= ar_seen + 1;
                slave_burst_beats <= {1'b0, m_axi_arlen} + 1'b1;
                slave_beat_index <= 0;
                slave_active <= 1'b1;
            end

            if (m_axi_rvalid && m_axi_rready) begin
                slave_global_beat_index <= slave_global_beat_index + 1;
                if (m_axi_rlast) begin
                    slave_active <= 1'b0;
                    slave_beat_index <= 0;
                end else begin
                    slave_beat_index <= slave_beat_index + 1;
                end
            end

            if (rsp_valid && rsp_ready) begin
                check(received_count < expected_rsp_count, "unexpected extra local response");
                if (received_count < expected_real_rsp_count) begin
                    check(rsp_data == (expected_data_base + received_count), "local response data sequence mismatch");
                end else begin
                    check(rsp_data == 32'd0, "synthetic error response data must be zero");
                end
                check(rsp_last == (received_count == (expected_rsp_count - 1)),
                      "local last did not mark only the final request response");
                if (rsp_last) begin
                    last_count <= last_count + 1;
                end
                received_count <= received_count + 1;
            end
        end
    end

    initial begin
        integer index;
        fail_count = 0;

        reset_case();
        expected_burst_count = 1;
        expected_araddr[0] = 64'h0000_0004_1B10_0000;
        expected_arbeats[0] = 4;
        expected_rsp_count = 4;
        expected_real_rsp_count = 4;
        expected_data_base = 32'hCAFE_0000;
        enable_stalls = 1'b1;
        start_request(expected_araddr[0], 16'd16);
        wait_for_responses(200);
        wait_for_idle(50);
        check(!error, "single-burst request reported an error");
        check(ar_seen == 1, "single-burst AR count mismatch");
        check(last_count == 1, "single-burst request did not emit exactly one local last");

        reset_case();
        expected_burst_count = 5;
        expected_araddr[0] = 64'h0000_0004_2509_0D40;
        expected_araddr[1] = 64'h0000_0004_2509_1000;
        expected_araddr[2] = 64'h0000_0004_2509_1400;
        expected_araddr[3] = 64'h0000_0004_2509_1800;
        expected_araddr[4] = 64'h0000_0004_2509_1C00;
        expected_arbeats[0] = 176;
        expected_arbeats[1] = 256;
        expected_arbeats[2] = 256;
        expected_arbeats[3] = 256;
        expected_arbeats[4] = 80;
        expected_rsp_count = 1024;
        expected_real_rsp_count = 1024;
        expected_data_base = 32'h2500_0000;
        enable_stalls = 1'b1;
        start_request(expected_araddr[0], 16'd4096);
        wait_for_responses(10000);
        wait_for_idle(100);
        check(!error, "4 KiB split request reported an error");
        check(ar_seen == 5, "4 KiB split AR count mismatch");
        check(last_count == 1, "multi-burst request emitted an intermediate local last");

        reset_case();
        expected_burst_count = 1;
        expected_araddr[0] = 64'h0000_0004_2600_0000;
        expected_arbeats[0] = 256;
        expected_rsp_count = 256;
        expected_real_rsp_count = 256;
        expected_data_base = 32'h0256_0000;
        start_request(expected_araddr[0], 16'd1024);
        wait_for_responses(1000);
        wait_for_idle(50);
        check(!error, "exact 256-beat request reported an error");
        check(ar_seen == 1, "exact 256-beat request did not use one burst");

        reset_case();
        expected_burst_count = 2;
        expected_araddr[0] = 64'h0000_0004_2700_0000;
        expected_araddr[1] = 64'h0000_0004_2700_0400;
        expected_arbeats[0] = 256;
        expected_arbeats[1] = 1;
        expected_rsp_count = 257;
        expected_real_rsp_count = 257;
        expected_data_base = 32'h0257_0000;
        start_request(expected_araddr[0], 16'd1028);
        wait_for_responses(1000);
        wait_for_idle(50);
        check(!error, "257-beat request reported an error");
        check(ar_seen == 2, "257-beat request did not split 256+1");

        reset_case();
        expected_burst_count = 2;
        expected_araddr[0] = 64'h0000_0004_2800_0FFC;
        expected_araddr[1] = 64'h0000_0004_2800_1000;
        expected_arbeats[0] = 1;
        expected_arbeats[1] = 1;
        expected_rsp_count = 2;
        expected_real_rsp_count = 2;
        expected_data_base = 32'h4FFC_0000;
        start_request(expected_araddr[0], 16'd8);
        wait_for_responses(50);
        wait_for_idle(50);
        check(!error, "0xFFC boundary request reported an error");
        check(ar_seen == 2, "0xFFC request did not split at 4 KiB");

        reset_case();
        expected_burst_count = 1;
        expected_araddr[0] = 64'h0000_0004_2900_0000;
        expected_arbeats[0] = 2;
        expected_rsp_count = 2;
        expected_real_rsp_count = 2;
        expected_data_base = 32'h0006_0000;
        start_request(expected_araddr[0], 16'd6);
        wait_for_responses(50);
        wait_for_idle(50);
        check(!error, "partial final word request reported an error");
        check(last_count == 1, "partial final word did not terminate on its second beat");

        reset_case();
        expected_burst_count = 64;
        for (index = 0; index < 64; index = index + 1) begin
            expected_araddr[index] = 64'h0000_0004_6000_0000 + (index * 1024);
            expected_arbeats[index] = 256;
        end
        expected_rsp_count = 16384;
        expected_real_rsp_count = 16384;
        expected_data_base = 32'hFFFF_0000;
        enable_stalls = 1'b1;
        start_request(expected_araddr[0], 16'hFFFF);
        wait_for_responses(100000);
        wait_for_idle(100);
        check(!error, "maximum 65535-byte request reported an error");
        check(ar_seen == 64, "maximum request did not produce 64 legal bursts");

        reset_case();
        expected_burst_count = 2;
        expected_araddr[0] = 64'h0000_0004_2A00_0000;
        expected_araddr[1] = 64'h0000_0004_2B00_0000;
        expected_arbeats[0] = 4;
        expected_arbeats[1] = 4;
        expected_rsp_count = 4;
        expected_real_rsp_count = 4;
        expected_data_base = 32'hB2B0_0000;
        start_request(expected_araddr[0], 16'd16);
        wait_for_responses(50);
        wait_for_idle(50);
        check(last_count == 1, "first back-to-back request did not finish exactly once");
        expected_rsp_count = 8;
        expected_real_rsp_count = 8;
        start_request(expected_araddr[1], 16'd16);
        wait_for_responses(50);
        wait_for_idle(50);
        check(!error, "back-to-back request sequence reported an error");
        check(ar_seen == 2, "back-to-back requests did not issue two bursts");
        check(last_count == 2, "back-to-back requests did not each emit one local last");

        reset_case();
        expected_burst_count = 2;
        expected_error_burst = 1;
        expected_araddr[0] = 64'h0000_0004_3000_0000;
        expected_araddr[1] = 64'h0000_0004_3000_0400;
        expected_arbeats[0] = 256;
        expected_arbeats[1] = 256;
        expected_rsp_count = 512;
        expected_real_rsp_count = 512;
        expected_data_base = 32'hE000_0000;
        start_request(expected_araddr[0], 16'd2048);
        wait_for_responses(2000);
        wait_for_idle(100);
        check(error, "RRESP error was not propagated");
        check(ar_seen == 2, "RRESP error test AR count mismatch");

        reset_case();
        expected_burst_count = 0;
        expected_rsp_count = 4;
        expected_real_rsp_count = 0;
        start_request(64'h0000_0004_0000_0002, 16'd16);
        wait_for_responses(20);
        wait_for_idle(20);
        check(error, "unaligned request did not report an error");
        check(ar_seen == 0, "unaligned request issued an AXI burst");

        reset_case();
        expected_burst_count = 0;
        expected_rsp_count = 1;
        expected_real_rsp_count = 0;
        start_request(64'h0000_0004_0000_0000, 16'd0);
        wait_for_responses(20);
        wait_for_idle(20);
        check(error, "zero-length request did not report an error");
        check(ar_seen == 0, "zero-length request issued an AXI burst");

        reset_case();
        expected_burst_count = 1;
        expected_araddr[0] = 64'h0000_0004_4000_0000;
        expected_arbeats[0] = 4;
        expected_rsp_count = 4;
        expected_real_rsp_count = 2;
        expected_data_base = 32'hEA71_0000;
        malformed_mode = MALFORM_EARLY;
        malformed_burst = 0;
        start_request(expected_araddr[0], 16'd16);
        wait_for_responses(50);
        wait_for_idle(20);
        check(error, "early RLAST was not detected");
        check(last_count == 1, "early RLAST recovery did not close the local request once");

        reset_case();
        expected_burst_count = 1;
        expected_araddr[0] = 64'h0000_0004_5000_0000;
        expected_arbeats[0] = 4;
        expected_rsp_count = 4;
        expected_real_rsp_count = 4;
        expected_data_base = 32'h1A7E_0000;
        malformed_mode = MALFORM_LATE;
        malformed_burst = 0;
        start_request(expected_araddr[0], 16'd16);
        wait_for_responses(50);
        wait_for_idle(50);
        check(error, "late RLAST was not detected");
        check(received_count == 4, "late RLAST leaked an extra AXI beat locally");
        check(last_count == 1, "late RLAST request did not emit exactly one local last");

        reset_case();
        expected_burst_count = 2;
        expected_araddr[0] = 64'h0000_0004_5100_0000;
        expected_araddr[1] = 64'h0000_0004_5100_0400;
        expected_arbeats[0] = 256;
        expected_arbeats[1] = 256;
        expected_rsp_count = 512;
        expected_real_rsp_count = 512;
        expected_data_base = 32'h1A7E_1000;
        malformed_mode = MALFORM_LATE;
        malformed_burst = 0;
        start_request(expected_araddr[0], 16'd2048);
        wait_for_responses(2000);
        wait_for_idle(100);
        check(error, "intermediate late RLAST was not detected");
        check(ar_seen == 2, "intermediate late RLAST prevented the next legal burst");
        check(received_count == 512, "intermediate late RLAST changed the local response length");

        if (fail_count != 0) begin
            $display("FAIL: tb_axi4_read_master saw %0d failure(s).", fail_count);
            $finish(1);
        end

        $display("PASS: axi4_read_master split long reads at 256 beats and 4 KiB boundaries with stalls and errors covered.");
        $finish;
    end

    initial begin
        #10000000;
        $display("TIMEOUT: tb_axi4_read_master did not finish");
        $finish(1);
    end

endmodule

`default_nettype wire
