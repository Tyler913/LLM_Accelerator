`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// Smoke test for the Vivado-facing QMAP row1024 AXI top.
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_row1024_axi_smoke_top.vvp \
//     FPGA_Project/sim/tb_qmap_row1024_axi_smoke_top.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_row1024_payload_fetcher.sv \
//     FPGA_Project/rtl/qmap_row1024_compute_path.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv \
//     FPGA_Project/rtl/q4_gemv_row_1024.sv \
//     FPGA_Project/rtl/axi4_read_master.sv \
//     FPGA_Project/rtl/qmap_row1024_axi_smoke_top.sv
//   vvp FPGA_Project/sim/tb_qmap_row1024_axi_smoke_top.vvp

module tb_qmap_row1024_axi_smoke_top;

    localparam int ADDR_WIDTH  = 64;
    localparam int DATA_WIDTH  = 32;
    localparam int IMAGE_BYTES = 16'h1000;
    localparam int MEM_WORDS   = IMAGE_BYTES / 4;

    localparam logic signed [63 : 0] EXPECTED_ROW_SUM_Q26 = -64'sd3482169;
    localparam logic [31 : 0] EXPECTED_ROW_SUM_LOW32 = 32'hFFCA_DDC7;

    logic aclk;
    logic aresetn;
    logic start;
    logic clear;

    logic busy;
    logic done_sticky;
    logic error_sticky;
    logic compare_match_sticky;
    logic [3 : 0] status;
    logic [31 : 0] row_sum_q26_low32;
    logic [31 : 0] expected_row_sum_q26_low32;
    logic [63 : 0] row_sum_q26;
    logic [63 : 0] expected_row_sum_q26;

    logic [ADDR_WIDTH-1 : 0] M_AXI_AWADDR;
    logic [7 : 0] M_AXI_AWLEN;
    logic [2 : 0] M_AXI_AWSIZE;
    logic [1 : 0] M_AXI_AWBURST;
    logic [2 : 0] M_AXI_AWPROT;
    logic [3 : 0] M_AXI_AWCACHE;
    logic M_AXI_AWVALID;
    logic M_AXI_AWREADY;
    logic [DATA_WIDTH-1 : 0] M_AXI_WDATA;
    logic [(DATA_WIDTH/8)-1 : 0] M_AXI_WSTRB;
    logic M_AXI_WLAST;
    logic M_AXI_WVALID;
    logic M_AXI_WREADY;
    logic [1 : 0] M_AXI_BRESP;
    logic M_AXI_BVALID;
    logic M_AXI_BREADY;
    logic [ADDR_WIDTH-1 : 0] M_AXI_ARADDR;
    logic [7 : 0] M_AXI_ARLEN;
    logic [2 : 0] M_AXI_ARSIZE;
    logic [1 : 0] M_AXI_ARBURST;
    logic [2 : 0] M_AXI_ARPROT;
    logic [3 : 0] M_AXI_ARCACHE;
    logic M_AXI_ARVALID;
    logic M_AXI_ARREADY;
    logic [DATA_WIDTH-1 : 0] M_AXI_RDATA;
    logic [1 : 0] M_AXI_RRESP;
    logic M_AXI_RLAST;
    logic M_AXI_RVALID;
    logic M_AXI_RREADY;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic axi_read_active;
    logic [31 : 0] axi_read_index;
    logic [8 : 0] axi_beats_left;
    integer cycle_count;
    integer burst_count;

    qmap_row1024_axi_smoke_top dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .i_start(start),
        .i_clear(clear),
        .o_busy(busy),
        .o_done_sticky(done_sticky),
        .o_error_sticky(error_sticky),
        .o_compare_match_sticky(compare_match_sticky),
        .o_status(status),
        .o_row_sum_q26_low32(row_sum_q26_low32),
        .o_expected_row_sum_q26_low32(expected_row_sum_q26_low32),
        .o_row_sum_q26(row_sum_q26),
        .o_expected_row_sum_q26(expected_row_sum_q26),
        .M_AXI_AWADDR(M_AXI_AWADDR),
        .M_AXI_AWLEN(M_AXI_AWLEN),
        .M_AXI_AWSIZE(M_AXI_AWSIZE),
        .M_AXI_AWBURST(M_AXI_AWBURST),
        .M_AXI_AWPROT(M_AXI_AWPROT),
        .M_AXI_AWCACHE(M_AXI_AWCACHE),
        .M_AXI_AWVALID(M_AXI_AWVALID),
        .M_AXI_AWREADY(M_AXI_AWREADY),
        .M_AXI_WDATA(M_AXI_WDATA),
        .M_AXI_WSTRB(M_AXI_WSTRB),
        .M_AXI_WLAST(M_AXI_WLAST),
        .M_AXI_WVALID(M_AXI_WVALID),
        .M_AXI_WREADY(M_AXI_WREADY),
        .M_AXI_BRESP(M_AXI_BRESP),
        .M_AXI_BVALID(M_AXI_BVALID),
        .M_AXI_BREADY(M_AXI_BREADY),
        .M_AXI_ARADDR(M_AXI_ARADDR),
        .M_AXI_ARLEN(M_AXI_ARLEN),
        .M_AXI_ARSIZE(M_AXI_ARSIZE),
        .M_AXI_ARBURST(M_AXI_ARBURST),
        .M_AXI_ARPROT(M_AXI_ARPROT),
        .M_AXI_ARCACHE(M_AXI_ARCACHE),
        .M_AXI_ARVALID(M_AXI_ARVALID),
        .M_AXI_ARREADY(M_AXI_ARREADY),
        .M_AXI_RDATA(M_AXI_RDATA),
        .M_AXI_RRESP(M_AXI_RRESP),
        .M_AXI_RLAST(M_AXI_RLAST),
        .M_AXI_RVALID(M_AXI_RVALID),
        .M_AXI_RREADY(M_AXI_RREADY)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    initial begin
        $readmemh("FPGA_Project/sim/vectors/qmap_row1024_image_words32.hex", mem);
    end

    assign M_AXI_AWREADY = 1'b1;
    assign M_AXI_WREADY  = 1'b1;
    assign M_AXI_BRESP   = 2'b00;
    assign M_AXI_BVALID  = 1'b0;
    assign M_AXI_ARREADY = !axi_read_active;
    assign M_AXI_RRESP   = 2'b00;

    always @(posedge aclk) begin
        if (!aresetn) begin
            axi_read_active <= 1'b0;
            axi_read_index  <= 32'd0;
            axi_beats_left  <= 9'd0;
            M_AXI_RVALID    <= 1'b0;
            M_AXI_RDATA     <= 32'd0;
            M_AXI_RLAST     <= 1'b0;
            burst_count     <= 0;
        end else begin
            if (M_AXI_AWVALID || M_AXI_WVALID) begin
                $display("FAIL: QMAP row1024 smoke top must not issue AXI writes");
                $finish(1);
            end

            if (M_AXI_ARVALID && M_AXI_ARREADY) begin
                axi_read_active <= 1'b1;
                axi_read_index  <= (M_AXI_ARADDR - `QMAP_ROW1024_BASE_ADDR) >> 2;
                axi_beats_left  <= {1'b0, M_AXI_ARLEN} + 1'b1;
                M_AXI_RVALID    <= 1'b1;
                M_AXI_RDATA     <= mem[(M_AXI_ARADDR - `QMAP_ROW1024_BASE_ADDR) >> 2];
                M_AXI_RLAST     <= (M_AXI_ARLEN == 8'd0);
                burst_count     <= burst_count + 1;
            end else if (M_AXI_RVALID && M_AXI_RREADY) begin
                if (axi_beats_left == 9'd1) begin
                    axi_read_active <= 1'b0;
                    axi_beats_left  <= 9'd0;
                    M_AXI_RVALID    <= 1'b0;
                    M_AXI_RLAST     <= 1'b0;
                end else begin
                    axi_read_index  <= axi_read_index + 1'b1;
                    axi_beats_left  <= axi_beats_left - 1'b1;
                    M_AXI_RDATA     <= mem[axi_read_index + 1'b1];
                    M_AXI_RLAST     <= (axi_beats_left == 9'd2);
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        start = 1'b0;
        clear = 1'b0;
        cycle_count = 0;

        repeat (4) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;

        @(negedge aclk);
        start = 1'b1;
        @(negedge aclk);
        start = 1'b0;

        while ((done_sticky != 1'b1) && (cycle_count < 12000)) begin
            @(posedge aclk);
            cycle_count++;
        end

        if (done_sticky != 1'b1) begin
            $display("FAIL: timed out waiting for qmap_row1024_axi_smoke_top");
            $finish(1);
        end

        #1;
        $display("QMAP row1024 AXI smoke top checks");
        $display("  AXI read bursts   = %0d", burst_count);
        $display("  status            = 0x%0h", status);
        $display("  row_sum_q26       = %0d", $signed(row_sum_q26));
        $display("  row_sum_low32     = 0x%08h", row_sum_q26_low32);

        if (busy || error_sticky || !compare_match_sticky) begin
            $display("FAIL: unexpected status bits");
            $finish(1);
        end

        if (status !== 4'b1010) begin
            $display("FAIL: status expected 0b1010 {compare,error,done,busy}");
            $finish(1);
        end

        if (burst_count != 10) begin
            $display("FAIL: expected 10 AXI read bursts");
            $finish(1);
        end

        if ((row_sum_q26 !== EXPECTED_ROW_SUM_Q26) ||
            (expected_row_sum_q26 !== EXPECTED_ROW_SUM_Q26)) begin
            $display("FAIL: row_sum_q26 mismatch");
            $finish(1);
        end

        if ((row_sum_q26_low32 !== EXPECTED_ROW_SUM_LOW32) ||
            (expected_row_sum_q26_low32 !== EXPECTED_ROW_SUM_LOW32)) begin
            $display("FAIL: row_sum_q26 low32 mismatch");
            $finish(1);
        end

        @(negedge aclk);
        clear = 1'b1;
        @(negedge aclk);
        clear = 1'b0;
        #1;

        if (status !== 4'b0000) begin
            $display("FAIL: clear did not reset sticky status");
            $finish(1);
        end

        $display("PASS: qmap_row1024_axi_smoke_top completed AXI-backed row1024 smoke path.");
        $finish;
    end

endmodule

`default_nettype wire
