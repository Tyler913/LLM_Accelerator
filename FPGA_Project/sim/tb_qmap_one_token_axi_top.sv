`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_axi_top;
    localparam int ADDR_WIDTH = 64;
    localparam logic [1 : 0] AXI_RESP_OKAY = 2'b00;

    localparam logic [11 : 0] REG_CTRL               = 12'h000;
    localparam logic [11 : 0] REG_STATUS             = 12'h004;
    localparam logic [11 : 0] REG_LAYER_START        = 12'h008;
    localparam logic [11 : 0] REG_LAYER_COUNT        = 12'h00C;
    localparam logic [11 : 0] REG_POSITION           = 12'h010;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_LO    = 12'h020;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_HI    = 12'h024;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_LO   = 12'h028;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_HI   = 12'h02C;
    localparam logic [11 : 0] REG_KV_CACHE_LO        = 12'h030;
    localparam logic [11 : 0] REG_KV_CACHE_HI        = 12'h034;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_LO = 12'h038;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_HI = 12'h03C;
    localparam logic [11 : 0] REG_LAYERS             = 12'h06C;
    localparam logic [11 : 0] REG_LAYER_ERROR_MASK   = 12'h074;
    localparam logic [11 : 0] REG_MEM_RD_REQS        = 12'h090;
    localparam logic [11 : 0] REG_MEM_RD_WORDS       = 12'h094;
    localparam logic [11 : 0] REG_MEM_WR_REQS        = 12'h098;
    localparam logic [11 : 0] REG_MEM_WR_WORDS       = 12'h09C;

    logic clk;
    logic rst_n;

    logic [11 : 0] s_axi_awaddr;
    logic [2 : 0]  s_axi_awprot;
    logic          s_axi_awvalid;
    logic          s_axi_awready;
    logic [31 : 0] s_axi_wdata;
    logic [3 : 0]  s_axi_wstrb;
    logic          s_axi_wvalid;
    logic          s_axi_wready;
    logic [1 : 0]  s_axi_bresp;
    logic          s_axi_bvalid;
    logic          s_axi_bready;
    logic [11 : 0] s_axi_araddr;
    logic [2 : 0]  s_axi_arprot;
    logic          s_axi_arvalid;
    logic          s_axi_arready;
    logic [31 : 0] s_axi_rdata;
    logic [1 : 0]  s_axi_rresp;
    logic          s_axi_rvalid;
    logic          s_axi_rready;

    logic axil_busy;
    logic busy;
    logic done;
    logic error;
    logic mem_error;
    logic [31 : 0] status;
    logic [31 : 0] best_token_id;
    logic [31 : 0] best_score_low32;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;

    logic [ADDR_WIDTH-1 : 0] m_axi_awaddr;
    logic [7 : 0]            m_axi_awlen;
    logic [2 : 0]            m_axi_awsize;
    logic [1 : 0]            m_axi_awburst;
    logic [2 : 0]            m_axi_awprot;
    logic [3 : 0]            m_axi_awcache;
    logic                    m_axi_awvalid;
    logic                    m_axi_awready;
    logic [31 : 0]           m_axi_wdata;
    logic [3 : 0]            m_axi_wstrb;
    logic                    m_axi_wlast;
    logic                    m_axi_wvalid;
    logic                    m_axi_wready;
    logic [1 : 0]            m_axi_bresp;
    logic                    m_axi_bvalid;
    logic                    m_axi_bready;
    logic [ADDR_WIDTH-1 : 0] m_axi_araddr;
    logic [7 : 0]            m_axi_arlen;
    logic [2 : 0]            m_axi_arsize;
    logic [1 : 0]            m_axi_arburst;
    logic [2 : 0]            m_axi_arprot;
    logic [3 : 0]            m_axi_arcache;
    logic                    m_axi_arvalid;
    logic                    m_axi_arready;
    logic [31 : 0]           m_axi_rdata;
    logic [1 : 0]            m_axi_rresp;
    logic                    m_axi_rlast;
    logic                    m_axi_rvalid;
    logic                    m_axi_rready;

    integer fail_count;
    integer timeout_count;
    logic [31 : 0] read_data;

    always #5 clk = ~clk;

    qmap_one_token_axi_top dut (
        .aclk(clk),
        .aresetn(rst_n),
        .S_AXI_AWADDR(s_axi_awaddr),
        .S_AXI_AWPROT(s_axi_awprot),
        .S_AXI_AWVALID(s_axi_awvalid),
        .S_AXI_AWREADY(s_axi_awready),
        .S_AXI_WDATA(s_axi_wdata),
        .S_AXI_WSTRB(s_axi_wstrb),
        .S_AXI_WVALID(s_axi_wvalid),
        .S_AXI_WREADY(s_axi_wready),
        .S_AXI_BRESP(s_axi_bresp),
        .S_AXI_BVALID(s_axi_bvalid),
        .S_AXI_BREADY(s_axi_bready),
        .S_AXI_ARADDR(s_axi_araddr),
        .S_AXI_ARPROT(s_axi_arprot),
        .S_AXI_ARVALID(s_axi_arvalid),
        .S_AXI_ARREADY(s_axi_arready),
        .S_AXI_RDATA(s_axi_rdata),
        .S_AXI_RRESP(s_axi_rresp),
        .S_AXI_RVALID(s_axi_rvalid),
        .S_AXI_RREADY(s_axi_rready),
        .o_axil_busy(axil_busy),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_mem_error(mem_error),
        .o_status(status),
        .o_best_token_id(best_token_id),
        .o_best_score_low32(best_score_low32),
        .o_mem_read_burst_count(mem_read_burst_count),
        .o_mem_read_word_count(mem_read_word_count),
        .o_mem_write_req_count(mem_write_req_count),
        .o_mem_write_word_count(mem_write_word_count),
        .M_AXI_AWADDR(m_axi_awaddr),
        .M_AXI_AWLEN(m_axi_awlen),
        .M_AXI_AWSIZE(m_axi_awsize),
        .M_AXI_AWBURST(m_axi_awburst),
        .M_AXI_AWPROT(m_axi_awprot),
        .M_AXI_AWCACHE(m_axi_awcache),
        .M_AXI_AWVALID(m_axi_awvalid),
        .M_AXI_AWREADY(m_axi_awready),
        .M_AXI_WDATA(m_axi_wdata),
        .M_AXI_WSTRB(m_axi_wstrb),
        .M_AXI_WLAST(m_axi_wlast),
        .M_AXI_WVALID(m_axi_wvalid),
        .M_AXI_WREADY(m_axi_wready),
        .M_AXI_BRESP(m_axi_bresp),
        .M_AXI_BVALID(m_axi_bvalid),
        .M_AXI_BREADY(m_axi_bready),
        .M_AXI_ARADDR(m_axi_araddr),
        .M_AXI_ARLEN(m_axi_arlen),
        .M_AXI_ARSIZE(m_axi_arsize),
        .M_AXI_ARBURST(m_axi_arburst),
        .M_AXI_ARPROT(m_axi_arprot),
        .M_AXI_ARCACHE(m_axi_arcache),
        .M_AXI_ARVALID(m_axi_arvalid),
        .M_AXI_ARREADY(m_axi_arready),
        .M_AXI_RDATA(m_axi_rdata),
        .M_AXI_RRESP(m_axi_rresp),
        .M_AXI_RLAST(m_axi_rlast),
        .M_AXI_RVALID(m_axi_rvalid),
        .M_AXI_RREADY(m_axi_rready)
    );

    task check;
        input logic cond;
        input string msg;
        begin
            if (!cond) begin
                $display("FAIL: %s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task wait_clk;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task axil_write;
        input logic [11 : 0] addr;
        input logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata = data;
            s_axi_wstrb = 4'hF;
            s_axi_wvalid = 1'b1;
            s_axi_bready = 1'b1;

            while (!(s_axi_awready && s_axi_wready)) begin
                wait_clk();
            end
            wait_clk();
            @(negedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;

            while (!s_axi_bvalid) begin
                wait_clk();
            end
            check(s_axi_bresp == AXI_RESP_OKAY, "AXI write BRESP should be OKAY");
            wait_clk();
            @(negedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task axil_read;
        input logic [11 : 0] addr;
        output logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_araddr = addr;
            s_axi_arvalid = 1'b1;
            s_axi_rready = 1'b0;

            while (!s_axi_arready) begin
                wait_clk();
            end
            wait_clk();
            @(negedge clk);
            s_axi_arvalid = 1'b0;

            while (!s_axi_rvalid) begin
                wait_clk();
            end
            data = s_axi_rdata;
            check(s_axi_rresp == AXI_RESP_OKAY, "AXI read RRESP should be OKAY");
            @(negedge clk);
            s_axi_rready = 1'b1;
            wait_clk();
            @(negedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    task write_addr64;
        input logic [11 : 0] lo_addr;
        input logic [11 : 0] hi_addr;
        input logic [63 : 0] value;
        begin
            axil_write(lo_addr, value[31 : 0]);
            axil_write(hi_addr, value[63 : 32]);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        s_axi_awaddr = 12'd0;
        s_axi_awprot = 3'b000;
        s_axi_awvalid = 1'b0;
        s_axi_wdata = 32'd0;
        s_axi_wstrb = 4'hF;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b0;
        s_axi_araddr = 12'd0;
        s_axi_arprot = 3'b000;
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b0;
        m_axi_awready = 1'b1;
        m_axi_wready = 1'b1;
        m_axi_bresp = AXI_RESP_OKAY;
        m_axi_bvalid = 1'b0;
        m_axi_arready = 1'b1;
        m_axi_rdata = 32'd0;
        m_axi_rresp = AXI_RESP_OKAY;
        m_axi_rlast = 1'b0;
        m_axi_rvalid = 1'b0;
        fail_count = 0;
        timeout_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        axil_write(REG_LAYER_START, 32'd0);
        axil_write(REG_LAYER_COUNT, 32'd0);
        axil_write(REG_POSITION, 32'd4);
        write_addr64(REG_INPUT_HIDDEN_LO, REG_INPUT_HIDDEN_HI, 64'h0000_0004_0509_2540);
        write_addr64(REG_OUTPUT_HIDDEN_LO, REG_OUTPUT_HIDDEN_HI, 64'h0000_0004_1509_2540);
        write_addr64(REG_KV_CACHE_LO, REG_KV_CACHE_HI, 64'h0000_0004_1410_0000);
        write_addr64(REG_FINAL_TAIL_QMAP_LO, REG_FINAL_TAIL_QMAP_HI, 64'h0000_0004_0501_0000);

        axil_write(REG_CTRL, 32'd1);

        while (!done && timeout_count < 200) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
            check(!m_axi_arvalid, "invalid BD wrapper run must not issue AXI read address");
            check(!m_axi_awvalid, "invalid BD wrapper run must not issue AXI write address");
            check(!m_axi_wvalid, "invalid BD wrapper run must not issue AXI write data");
            check(!m_axi_bready, "invalid BD wrapper run must not wait for AXI write response");
            check(!m_axi_rready, "invalid BD wrapper run must not wait for AXI read data");
        end

        check(timeout_count < 200, "BD-facing wrapper invalid run should finish quickly");
        check(error, "BD-facing wrapper invalid run should assert top error");
        check(!mem_error, "BD-facing wrapper invalid run should not assert memory error");
        check(status[1], "BD-facing status should expose done");
        check(status[2], "BD-facing status should expose top error");
        check(mem_read_burst_count == 32'd0, "read request counter should stay zero");
        check(mem_read_word_count == 32'd0, "read word counter should stay zero");
        check(mem_write_req_count == 32'd0, "write request counter should stay zero");
        check(mem_write_word_count == 32'd0, "write word counter should stay zero");

        repeat (2) @(posedge clk);
        axil_read(REG_STATUS, read_data);
        check(read_data[1], "register status.done sticky should be set through BD wrapper");
        check(read_data[2], "register status.error sticky should be set through BD wrapper");
        axil_read(REG_LAYERS, read_data);
        check(read_data == 32'd0, "layers started/completed should be zero for validation exit");
        axil_read(REG_LAYER_ERROR_MASK, read_data);
        check(read_data[0], "layer error mask register should expose layer0 error through BD wrapper");
        axil_read(REG_MEM_RD_REQS, read_data);
        check(read_data == 32'd0, "register read request counter should be zero");
        axil_read(REG_MEM_RD_WORDS, read_data);
        check(read_data == 32'd0, "register read word counter should be zero");
        axil_read(REG_MEM_WR_REQS, read_data);
        check(read_data == 32'd0, "register write request counter should be zero");
        axil_read(REG_MEM_WR_WORDS, read_data);
        check(read_data == 32'd0, "register write word counter should be zero");

        axil_write(REG_CTRL, 32'd2);
        axil_read(REG_STATUS, read_data);
        check(!read_data[1], "CTRL.clear should clear done sticky through BD wrapper");
        check(!read_data[3], "CTRL.clear should clear command error sticky through BD wrapper");
        check(read_data[2], "error sticky may relatch while top_error remains asserted");

        if (fail_count != 0) begin
            $display("FAIL: qmap_one_token_axi_top BD-facing smoke failures=%0d", fail_count);
            $finish(1);
        end

        $display("PASS: qmap_one_token_axi_top exposes AXI-Lite control and does not issue M_AXI traffic on validation exit.");
        $finish;
    end
endmodule

`default_nettype wire
