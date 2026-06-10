`timescale 1ns/1ps
`default_nettype none

// Smoke test for the minimal read-only AXI4 master.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_axi4_read_master.vvp \
//     FPGA_Project/sim/tb_axi4_read_master.sv \
//     FPGA_Project/rtl/axi4_read_master.sv
//   vvp FPGA_Project/sim/tb_axi4_read_master.vvp

module tb_axi4_read_master;

    localparam int ADDR_WIDTH = 64;
    localparam int DATA_WIDTH = 32;

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

    logic slave_active;
    logic [8 : 0] slave_beats_left;
    logic [31 : 0] slave_data_word;

    integer received_count;
    integer mismatch_count;
    integer cycle_count;

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

    assign m_axi_arready = 1'b1;
    assign m_axi_rresp   = 2'b00;

    always @(posedge clk) begin
        if (!rst_n) begin
            slave_active     <= 1'b0;
            slave_beats_left <= 9'd0;
            slave_data_word  <= 32'hCAFE_0000;
            m_axi_rvalid     <= 1'b0;
            m_axi_rdata      <= 32'd0;
            m_axi_rlast      <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                slave_active     <= 1'b1;
                slave_beats_left <= {1'b0, m_axi_arlen} + 1'b1;
                slave_data_word  <= 32'hCAFE_0000;
                m_axi_rvalid     <= 1'b1;
                m_axi_rdata      <= 32'hCAFE_0000;
                m_axi_rlast      <= (m_axi_arlen == 8'd0);
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (slave_beats_left == 9'd1) begin
                    slave_active     <= 1'b0;
                    slave_beats_left <= 9'd0;
                    m_axi_rvalid     <= 1'b0;
                    m_axi_rlast      <= 1'b0;
                end else begin
                    slave_beats_left <= slave_beats_left - 1'b1;
                    slave_data_word  <= slave_data_word + 1'b1;
                    m_axi_rdata      <= slave_data_word + 1'b1;
                    m_axi_rlast      <= (slave_beats_left == 9'd2);
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            received_count <= 0;
        end else if (rsp_valid && rsp_ready) begin
            $display("response beat %0d data=0x%08h last=%0d",
                     received_count, rsp_data, rsp_last);
            if (rsp_data !== (32'hCAFE_0000 + received_count)) begin
                mismatch_count = mismatch_count + 1;
            end
            if ((received_count == 3) && (rsp_last !== 1'b1)) begin
                mismatch_count = mismatch_count + 1;
            end
            if ((received_count != 3) && (rsp_last !== 1'b0)) begin
                mismatch_count = mismatch_count + 1;
            end
            received_count <= received_count + 1;
        end
    end

    initial begin
        rst_n = 1'b0;
        req_valid = 1'b0;
        req_addr = 64'h0000_0004_1B10_0000;
        req_len_bytes = 16'd16;
        rsp_ready = 1'b1;
        mismatch_count = 0;
        cycle_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        req_valid = 1'b1;
        @(negedge clk);
        req_valid = 1'b0;

        while ((received_count < 4) && (cycle_count < 100)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (received_count != 4) begin
            $display("FAIL: timed out waiting for AXI read responses");
            $finish(1);
        end

        if (m_axi_araddr !== 64'h0000_0004_1B10_0000) begin
            $display("FAIL: ARADDR mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_arlen !== 8'd3) begin
            $display("FAIL: ARLEN mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_arsize !== 3'd2) begin
            $display("FAIL: ARSIZE mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (m_axi_arburst !== 2'b01) begin
            $display("FAIL: ARBURST mismatch");
            mismatch_count = mismatch_count + 1;
        end
        if (error) begin
            $display("FAIL: axi4_read_master reported error");
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d AXI read master mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: axi4_read_master issued one 4-beat read burst and returned all data.");
        $finish;
    end

endmodule

`default_nettype wire
