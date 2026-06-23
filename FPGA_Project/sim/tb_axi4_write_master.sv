`timescale 1ns/1ps
`default_nettype none

// Smoke test for the minimal AXI4 write master.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_axi4_write_master.vvp \
//     FPGA_Project/sim/tb_axi4_write_master.sv \
//     FPGA_Project/rtl/axi4_write_master.sv
//   vvp FPGA_Project/sim/tb_axi4_write_master.vvp

module tb_axi4_write_master;

    localparam int ADDR_WIDTH = 64;
    localparam int DATA_WIDTH = 32;

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

    integer beat_count;
    integer mismatch_count;
    integer cycle_count;
    integer i;

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

    assign m_axi_awready = 1'b1;
    assign m_axi_wready  = 1'b1;
    assign m_axi_bresp   = 2'b00;

    always @(posedge clk) begin
        if (!rst_n) begin
            beat_count    <= 0;
            m_axi_bvalid  <= 1'b0;
        end else begin
            if (m_axi_wvalid && m_axi_wready) begin
                $display("write beat %0d data=0x%08h last=%0d", beat_count, m_axi_wdata, m_axi_wlast);
                if (m_axi_wdata !== (32'hCAFE_1000 + beat_count)) begin
                    mismatch_count = mismatch_count + 1;
                end
                if (m_axi_wstrb !== 4'hF) begin
                    mismatch_count = mismatch_count + 1;
                end
                if ((beat_count == 3) && (m_axi_wlast !== 1'b1)) begin
                    mismatch_count = mismatch_count + 1;
                end
                if ((beat_count != 3) && (m_axi_wlast !== 1'b0)) begin
                    mismatch_count = mismatch_count + 1;
                end
                beat_count <= beat_count + 1;
                if (m_axi_wlast) begin
                    m_axi_bvalid <= 1'b1;
                end
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        req_valid = 1'b0;
        req_addr = 64'h0000_0004_0008_8000;
        req_len_bytes = 16'd16;
        wdata = 32'd0;
        wdata_valid = 1'b0;
        wdata_last = 1'b0;
        mismatch_count = 0;
        cycle_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        req_valid = 1'b1;
        @(negedge clk);
        req_valid = 1'b0;

        for (i = 0 ; i < 4 ; i = i + 1) begin
            wait (wdata_ready == 1'b1);
            @(negedge clk);
            wdata = 32'hCAFE_1000 + i;
            wdata_last = (i == 3);
            wdata_valid = 1'b1;
            @(negedge clk);
            wdata_valid = 1'b0;
            wdata_last = 1'b0;
        end

        while ((done != 1'b1) && (cycle_count < 100)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for AXI write response");
            $finish(1);
        end

        if (m_axi_awaddr !== 64'h0000_0004_0008_8000) begin
            $display("FAIL: AWADDR mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_awlen !== 8'd3) begin
            $display("FAIL: AWLEN mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_awsize !== 3'd2) begin
            $display("FAIL: AWSIZE mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_awburst !== 2'b01) begin
            $display("FAIL: AWBURST mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (error) begin
            $display("FAIL: axi4_write_master reported error");
            mismatch_count = mismatch_count + 1;
        end
        if (beat_count != 4) begin
            $display("FAIL: beat_count mismatch actual=%0d expected=4", beat_count);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d AXI write master mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: axi4_write_master issued one 4-beat write burst and accepted the B response.");
        $finish;
    end

endmodule

`default_nettype wire
