`timescale 1ns/1ps
`default_nettype none

module tb_axi4lite_to_mmio_regs;
    localparam int AXI_ADDR_WIDTH = 12;
    localparam int REG_ADDR_WIDTH = 12;
    localparam logic [1 : 0] AXI_RESP_OKAY   = 2'b00;
    localparam logic [1 : 0] AXI_RESP_SLVERR = 2'b10;
    localparam logic [11 : 0] ERR_ADDR = 12'h0FC;

    logic clk;
    logic rst_n;

    logic [AXI_ADDR_WIDTH-1 : 0] s_axi_awaddr;
    logic [2 : 0]                s_axi_awprot;
    logic                        s_axi_awvalid;
    logic                        s_axi_awready;
    logic [31 : 0]               s_axi_wdata;
    logic [3 : 0]                s_axi_wstrb;
    logic                        s_axi_wvalid;
    logic                        s_axi_wready;
    logic [1 : 0]                s_axi_bresp;
    logic                        s_axi_bvalid;
    logic                        s_axi_bready;
    logic [AXI_ADDR_WIDTH-1 : 0] s_axi_araddr;
    logic [2 : 0]                s_axi_arprot;
    logic                        s_axi_arvalid;
    logic                        s_axi_arready;
    logic [31 : 0]               s_axi_rdata;
    logic [1 : 0]                s_axi_rresp;
    logic                        s_axi_rvalid;
    logic                        s_axi_rready;

    logic                        reg_wr_valid;
    logic                        reg_wr_ready;
    logic                        reg_rd_valid;
    logic                        reg_rd_ready;
    logic [REG_ADDR_WIDTH-1 : 0] reg_addr;
    logic [31 : 0]               reg_wdata;
    logic [31 : 0]               reg_rdata;
    logic                        reg_error;
    logic                        busy;

    logic                        force_reg_error;
    integer                      fail_count;
    integer                      phase;
    integer                      wr_count;
    integer                      rd_count;
    logic [REG_ADDR_WIDTH-1 : 0] last_wr_addr;
    logic [31 : 0]               last_wr_data;
    logic [REG_ADDR_WIDTH-1 : 0] last_rd_addr;
    logic [31 : 0]               read_data;

    axi4lite_to_mmio_regs #(
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .REG_ADDR_WIDTH(REG_ADDR_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_s_axi_awaddr(s_axi_awaddr),
        .i_s_axi_awprot(s_axi_awprot),
        .i_s_axi_awvalid(s_axi_awvalid),
        .o_s_axi_awready(s_axi_awready),
        .i_s_axi_wdata(s_axi_wdata),
        .i_s_axi_wstrb(s_axi_wstrb),
        .i_s_axi_wvalid(s_axi_wvalid),
        .o_s_axi_wready(s_axi_wready),
        .o_s_axi_bresp(s_axi_bresp),
        .o_s_axi_bvalid(s_axi_bvalid),
        .i_s_axi_bready(s_axi_bready),
        .i_s_axi_araddr(s_axi_araddr),
        .i_s_axi_arprot(s_axi_arprot),
        .i_s_axi_arvalid(s_axi_arvalid),
        .o_s_axi_arready(s_axi_arready),
        .o_s_axi_rdata(s_axi_rdata),
        .o_s_axi_rresp(s_axi_rresp),
        .o_s_axi_rvalid(s_axi_rvalid),
        .i_s_axi_rready(s_axi_rready),
        .o_reg_wr_valid(reg_wr_valid),
        .i_reg_wr_ready(reg_wr_ready),
        .o_reg_rd_valid(reg_rd_valid),
        .i_reg_rd_ready(reg_rd_ready),
        .o_reg_addr(reg_addr),
        .o_reg_wdata(reg_wdata),
        .i_reg_rdata(reg_rdata),
        .i_reg_error(reg_error),
        .o_busy(busy)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @* begin
        reg_rdata = 32'hA500_0000 | {20'd0, reg_addr};
        reg_error = force_reg_error || (reg_addr == ERR_ADDR);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_count     <= 0;
            rd_count     <= 0;
            last_wr_addr <= '0;
            last_wr_data <= 32'd0;
            last_rd_addr <= '0;
        end else begin
            if (reg_wr_valid && reg_wr_ready) begin
                wr_count     <= wr_count + 1;
                last_wr_addr <= reg_addr;
                last_wr_data <= reg_wdata;
            end
            if (reg_rd_valid && reg_rd_ready) begin
                rd_count     <= rd_count + 1;
                last_rd_addr <= reg_addr;
            end
        end
    end

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

    task automatic clear_axi_master;
        begin
            s_axi_awaddr  = '0;
            s_axi_awprot  = 3'b000;
            s_axi_awvalid = 1'b0;
            s_axi_wdata   = 32'd0;
            s_axi_wstrb   = 4'hF;
            s_axi_wvalid  = 1'b0;
            s_axi_bready  = 1'b0;
            s_axi_araddr  = '0;
            s_axi_arprot  = 3'b000;
            s_axi_arvalid = 1'b0;
            s_axi_rready  = 1'b0;
        end
    endtask

    task automatic send_aw;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        begin
            @(negedge clk);
            s_axi_awaddr  = addr;
            s_axi_awvalid = 1'b1;
            #1;
            while (!s_axi_awready) begin
                @(negedge clk);
                #1;
            end
            wait_clk();
            @(negedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_awaddr  = '0;
        end
    endtask

    task automatic send_w;
        input logic [31 : 0] data;
        input logic [3 : 0]  strb;
        begin
            @(negedge clk);
            s_axi_wdata  = data;
            s_axi_wstrb  = strb;
            s_axi_wvalid = 1'b1;
            #1;
            while (!s_axi_wready) begin
                @(negedge clk);
                #1;
            end
            wait_clk();
            @(negedge clk);
            s_axi_wvalid = 1'b0;
            s_axi_wdata  = 32'd0;
            s_axi_wstrb  = 4'hF;
        end
    endtask

    task automatic send_ar;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        begin
            @(negedge clk);
            s_axi_araddr  = addr;
            s_axi_arvalid = 1'b1;
            #1;
            while (!s_axi_arready) begin
                @(negedge clk);
                #1;
            end
            wait_clk();
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            s_axi_araddr  = '0;
        end
    endtask

    task automatic accept_b;
        input logic [1 : 0] expected_resp;
        input integer       stall_cycles;
        integer             idx;
        begin
            @(negedge clk);
            s_axi_bready = 1'b0;
            while (!s_axi_bvalid) begin
                wait_clk();
            end
            for (idx = 0; idx < stall_cycles; idx = idx + 1) begin
                check(s_axi_bvalid, "BVALID held while BREADY is low");
                check(s_axi_bresp == expected_resp, "BRESP stable while BREADY is low");
                wait_clk();
            end
            check(s_axi_bvalid, "BVALID asserted before accepting response");
            check(s_axi_bresp == expected_resp, "BRESP matches expected response");
            @(negedge clk);
            s_axi_bready = 1'b1;
            wait_clk();
            check(!s_axi_bvalid, "BVALID clears after B handshake");
            @(negedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic accept_r;
        input logic [1 : 0] expected_resp;
        input integer       stall_cycles;
        output logic [31 : 0] data;
        integer             idx;
        begin
            @(negedge clk);
            s_axi_rready = 1'b0;
            while (!s_axi_rvalid) begin
                wait_clk();
            end
            data = s_axi_rdata;
            for (idx = 0; idx < stall_cycles; idx = idx + 1) begin
                check(s_axi_rvalid, "RVALID held while RREADY is low");
                check(s_axi_rresp == expected_resp, "RRESP stable while RREADY is low");
                check(s_axi_rdata == data, "RDATA stable while RREADY is low");
                wait_clk();
            end
            check(s_axi_rvalid, "RVALID asserted before accepting response");
            check(s_axi_rresp == expected_resp, "RRESP matches expected response");
            @(negedge clk);
            s_axi_rready = 1'b1;
            wait_clk();
            check(!s_axi_rvalid, "RVALID clears after R handshake");
            @(negedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    task automatic axi_write_aw_first;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        input logic [1 : 0] expected_resp;
        input integer       b_stall_cycles;
        begin
            send_aw(addr);
            send_w(data, 4'hF);
            accept_b(expected_resp, b_stall_cycles);
        end
    endtask

    task automatic axi_write_w_first;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        input logic [1 : 0] expected_resp;
        input integer       b_stall_cycles;
        begin
            send_w(data, 4'hF);
            send_aw(addr);
            accept_b(expected_resp, b_stall_cycles);
        end
    endtask

    initial begin
        #200000;
        $display("TIMEOUT: tb_axi4lite_to_mmio_regs stuck in phase %0d state=%0d awv=%b awr=%b wv=%b wr=%b bv=%b br=%b regwv=%b regwr=%b", phase, dut.state, s_axi_awvalid, s_axi_awready, s_axi_wvalid, s_axi_wready, s_axi_bvalid, s_axi_bready, reg_wr_valid, reg_wr_ready);
        $finish(1);
    end

    initial begin
        integer base_wr;
        integer base_rd;

        fail_count      = 0;
        phase           = 0;
        force_reg_error = 1'b0;
        reg_wr_ready    = 1'b1;
        reg_rd_ready    = 1'b1;
        clear_axi_master();
        rst_n           = 1'b0;

        repeat (5) wait_clk();
        rst_n = 1'b1;
        repeat (2) wait_clk();

        check(!busy, "adapter idle after reset");
        check(!s_axi_bvalid && !s_axi_rvalid, "response channels idle after reset");

        phase = 1;
        base_wr = wr_count;
        axi_write_aw_first(12'h020, 32'h1122_3344, AXI_RESP_OKAY, 0);
        wait_clk();
        check(wr_count == base_wr + 1, "AW-before-W write issues exactly one tiny-MMIO write");
        check(last_wr_addr == 12'h020, "AW-before-W write address forwarded");
        check(last_wr_data == 32'h1122_3344, "AW-before-W write data forwarded");

        phase = 2;
        base_wr = wr_count;
        axi_write_w_first(12'h024, 32'h5566_7788, AXI_RESP_OKAY, 0);
        wait_clk();
        check(wr_count == base_wr + 1, "W-before-AW write issues exactly one tiny-MMIO write");
        check(last_wr_addr == 12'h024, "W-before-AW write address forwarded");
        check(last_wr_data == 32'h5566_7788, "W-before-AW write data forwarded");

        phase = 3;
        base_wr = wr_count;
        axi_write_aw_first(12'h028, 32'hCAFE_BABE, AXI_RESP_OKAY, 3);
        wait_clk();
        check(wr_count == base_wr + 1, "B-channel stall write still completes once");
        check(last_wr_addr == 12'h028, "B-channel stall write address forwarded");

        phase = 4;
        base_wr = wr_count;
        reg_wr_ready = 1'b0;
        send_aw(12'h02C);
        send_w(32'hDEAD_BEEF, 4'hF);
        repeat (3) begin
            wait_clk();
            check(reg_wr_valid, "tiny-MMIO write valid held while reg_wr_ready is low");
            check(reg_addr == 12'h02C, "tiny-MMIO write address stable while stalled");
            check(reg_wdata == 32'hDEAD_BEEF, "tiny-MMIO write data stable while stalled");
            check(!s_axi_bvalid, "BRESP not returned before tiny-MMIO write handshake");
            check(wr_count == base_wr, "write count unchanged while reg_wr_ready is low");
        end
        @(negedge clk);
        reg_wr_ready = 1'b1;
        accept_b(AXI_RESP_OKAY, 1);
        wait_clk();
        check(wr_count == base_wr + 1, "stalled tiny-MMIO write completes exactly once");
        check(last_wr_addr == 12'h02C, "stalled tiny-MMIO write address forwarded");
        check(last_wr_data == 32'hDEAD_BEEF, "stalled tiny-MMIO write data forwarded");

        phase = 5;
        base_rd = rd_count;
        send_ar(12'h040);
        accept_r(AXI_RESP_OKAY, 3, read_data);
        wait_clk();
        check(rd_count == base_rd + 1, "read issues exactly one tiny-MMIO read");
        check(last_rd_addr == 12'h040, "read address forwarded");
        check(read_data == (32'hA500_0000 | 32'h040), "read data returned from tiny-MMIO target");

        phase = 6;
        base_rd = rd_count;
        reg_rd_ready = 1'b0;
        send_ar(12'h044);
        repeat (3) begin
            wait_clk();
            check(reg_rd_valid, "tiny-MMIO read valid held while reg_rd_ready is low");
            check(reg_addr == 12'h044, "tiny-MMIO read address stable while stalled");
            check(!s_axi_rvalid, "RRESP not returned before tiny-MMIO read handshake");
            check(rd_count == base_rd, "read count unchanged while reg_rd_ready is low");
        end
        @(negedge clk);
        reg_rd_ready = 1'b1;
        accept_r(AXI_RESP_OKAY, 0, read_data);
        wait_clk();
        check(rd_count == base_rd + 1, "stalled tiny-MMIO read completes exactly once");
        check(read_data == (32'hA500_0000 | 32'h044), "stalled read data returned from tiny-MMIO target");

        phase = 7;
        base_wr = wr_count;
        axi_write_aw_first(ERR_ADDR, 32'h1234_5678, AXI_RESP_SLVERR, 0);
        wait_clk();
        check(wr_count == base_wr + 1, "downstream-error write still performs tiny-MMIO handshake");
        check(last_wr_addr == ERR_ADDR, "downstream-error write address forwarded");

        phase = 8;
        base_rd = rd_count;
        send_ar(ERR_ADDR);
        accept_r(AXI_RESP_SLVERR, 0, read_data);
        wait_clk();
        check(rd_count == base_rd + 1, "downstream-error read still performs tiny-MMIO handshake");
        check(last_rd_addr == ERR_ADDR, "downstream-error read address forwarded");

        phase = 9;
        base_wr = wr_count;
        axi_write_aw_first(12'h021, 32'hFACE_CAFE, AXI_RESP_SLVERR, 0);
        wait_clk();
        check(wr_count == base_wr, "unaligned write rejected without tiny-MMIO side effect");

        phase = 10;
        base_rd = rd_count;
        send_ar(12'h023);
        accept_r(AXI_RESP_SLVERR, 0, read_data);
        wait_clk();
        check(rd_count == base_rd, "unaligned read rejected without tiny-MMIO side effect");
        check(read_data == 32'd0, "unaligned read returns zero data");

        phase = 11;
        base_wr = wr_count;
        send_aw(12'h050);
        send_w(32'hA5A5_5A5A, 4'h3);
        accept_b(AXI_RESP_SLVERR, 0);
        wait_clk();
        check(wr_count == base_wr, "partial write rejected without tiny-MMIO side effect");

        phase = 12;
        base_wr = wr_count;
        base_rd = rd_count;
        @(negedge clk);
        s_axi_awaddr  = 12'h060;
        s_axi_awvalid = 1'b1;
        s_axi_wdata   = 32'h0BAD_F00D;
        s_axi_wstrb   = 4'hF;
        s_axi_wvalid  = 1'b1;
        s_axi_araddr  = 12'h064;
        s_axi_arvalid = 1'b1;
        #1;
        check(s_axi_awready, "simultaneous AW/AR: AWREADY asserted for write priority");
        check(s_axi_wready, "simultaneous W/AR: WREADY asserted for write priority");
        check(!s_axi_arready, "simultaneous write/read: ARREADY deasserted while write wins priority");
        wait_clk();
        @(negedge clk);
        s_axi_awvalid = 1'b0;
        s_axi_wvalid  = 1'b0;
        s_axi_awaddr  = '0;
        s_axi_wdata   = 32'd0;
        accept_b(AXI_RESP_OKAY, 0);
        #1;
        check(s_axi_arvalid && s_axi_arready, "held AR accepted after prioritized write response");
        wait_clk();
        @(negedge clk);
        s_axi_arvalid = 1'b0;
        s_axi_araddr  = '0;
        accept_r(AXI_RESP_OKAY, 0, read_data);
        wait_clk();
        check(wr_count == base_wr + 1, "simultaneous transaction performs one write");
        check(rd_count == base_rd + 1, "held read completes after prioritized write");
        check(last_wr_addr == 12'h060, "simultaneous transaction write address forwarded");
        check(last_rd_addr == 12'h064, "simultaneous transaction read address forwarded");
        check(read_data == (32'hA500_0000 | 32'h064), "simultaneous transaction read data returned");

        clear_axi_master();
        repeat (4) wait_clk();

        if (fail_count == 0) begin
            $display("PASS: axi4lite_to_mmio_regs serialized AXI4-Lite reads/writes to tiny MMIO with stalls and errors covered.");
            $finish;
        end else begin
            $display("FAIL: tb_axi4lite_to_mmio_regs saw %0d failure(s).", fail_count);
            $finish(1);
        end
    end
endmodule

`default_nettype wire





