`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// AXI-backed QMAP dot64 compute-path smoke test.
//
// This test proves the first read-only AXI path:
//
//   qmap_dot64_compute_path
//     -> project-local memory request/response
//     -> axi4_read_master
//     -> AXI4 read address/data channels
//     -> AXI memory model loaded from qmap_dot64_image_words32.hex
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_dot64_compute_path_axi4.vvp \
//     FPGA_Project/sim/tb_qmap_dot64_compute_path_axi4.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_payload_fetcher.sv \
//     FPGA_Project/rtl/qmap_dot64_compute_path.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv \
//     FPGA_Project/rtl/axi4_read_master.sv
//   vvp FPGA_Project/sim/tb_qmap_dot64_compute_path_axi4.vvp

module tb_qmap_dot64_compute_path_axi4;

    localparam int ADDR_WIDTH    = 64;
    localparam int DATA_WIDTH    = 32;
    localparam int IMAGE_BYTES   = 16'h0600;
    localparam int MEM_WORDS     = IMAGE_BYTES / 4;
    localparam int GROUP_SIZE    = 64;
    localparam int ACT_WIDTH     = 16;
    localparam int ACT_FRAC      = 12;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int SCALE_FRAC    = 14;
    localparam int PRODUCT_WIDTH = ACT_WIDTH + WEIGHT_WIDTH;
    localparam int PARTIAL_WIDTH = PRODUCT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH;

    localparam logic signed [63 : 0] EXPECTED_PARTIAL_SUM = 64'sd24751;
    localparam logic signed [63 : 0] EXPECTED_SCALED_SUM_Q26 = 64'sd3019622;

    logic clk;
    logic rst_n;
    logic start;
    logic compute_busy;
    logic compute_done;
    logic compute_error;
    logic compare_match;

    logic local_req_valid;
    logic local_req_ready;
    logic [ADDR_WIDTH-1 : 0] local_req_addr;
    logic [15 : 0] local_req_len_bytes;
    logic local_rsp_valid;
    logic local_rsp_ready;
    logic [DATA_WIDTH-1 : 0] local_rsp_data;
    logic local_rsp_last;

    logic axi_adapter_busy;
    logic axi_adapter_error;
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

    logic signed [63 : 0] partial_sum;
    logic signed [63 : 0] scaled_sum_q26;
    logic signed [63 : 0] expected_partial_sum;
    logic signed [63 : 0] expected_scaled_sum_q26;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic axi_read_active;
    logic [31 : 0] axi_read_index;
    logic [8 : 0] axi_beats_left;
    integer cycle_count;
    integer burst_count;

    qmap_dot64_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .GROUP_SIZE(GROUP_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PRODUCT_WIDTH(PRODUCT_WIDTH),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH)
    ) compute_path (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(`QMAP_DOT64_BASE_ADDR),
        .o_busy(compute_busy),
        .o_done(compute_done),
        .o_error(compute_error),
        .o_compare_match(compare_match),
        .o_mem_req_valid(local_req_valid),
        .i_mem_req_ready(local_req_ready),
        .o_mem_req_addr(local_req_addr),
        .o_mem_req_len_bytes(local_req_len_bytes),
        .i_mem_rsp_valid(local_rsp_valid),
        .o_mem_rsp_ready(local_rsp_ready),
        .i_mem_rsp_data(local_rsp_data),
        .i_mem_rsp_last(local_rsp_last),
        .o_partial_sum(partial_sum),
        .o_scaled_sum_q26(scaled_sum_q26),
        .o_expected_partial_sum(expected_partial_sum),
        .o_expected_scaled_sum_q26(expected_scaled_sum_q26)
    );

    axi4_read_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) axi_reader (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_req_valid(local_req_valid),
        .o_req_ready(local_req_ready),
        .i_req_addr(local_req_addr),
        .i_req_len_bytes(local_req_len_bytes),
        .o_rsp_valid(local_rsp_valid),
        .i_rsp_ready(local_rsp_ready),
        .o_rsp_data(local_rsp_data),
        .o_rsp_last(local_rsp_last),
        .o_busy(axi_adapter_busy),
        .o_error(axi_adapter_error),
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

    initial begin
        $readmemh("FPGA_Project/sim/vectors/qmap_dot64_image_words32.hex", mem);
    end

    assign m_axi_arready = !axi_read_active;
    assign m_axi_rresp = 2'b00;

    always @(posedge clk) begin
        if (!rst_n) begin
            axi_read_active <= 1'b0;
            axi_read_index  <= 32'd0;
            axi_beats_left  <= 9'd0;
            m_axi_rvalid    <= 1'b0;
            m_axi_rdata     <= 32'd0;
            m_axi_rlast     <= 1'b0;
            burst_count     <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                if (m_axi_arburst !== 2'b01) begin
                    $display("FAIL: AXI memory model expected INCR burst");
                    $finish(1);
                end
                if (m_axi_arsize !== 3'd2) begin
                    $display("FAIL: AXI memory model expected 32-bit beats");
                    $finish(1);
                end

                axi_read_active <= 1'b1;
                axi_read_index  <= (m_axi_araddr - `QMAP_DOT64_BASE_ADDR) >> 2;
                axi_beats_left  <= {1'b0, m_axi_arlen} + 1'b1;
                m_axi_rvalid    <= 1'b1;
                m_axi_rdata     <= mem[(m_axi_araddr - `QMAP_DOT64_BASE_ADDR) >> 2];
                m_axi_rlast     <= (m_axi_arlen == 8'd0);
                burst_count     <= burst_count + 1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (axi_beats_left == 9'd1) begin
                    axi_read_active <= 1'b0;
                    axi_beats_left  <= 9'd0;
                    m_axi_rvalid    <= 1'b0;
                    m_axi_rlast     <= 1'b0;
                end else begin
                    axi_read_index  <= axi_read_index + 1'b1;
                    axi_beats_left  <= axi_beats_left - 1'b1;
                    m_axi_rdata     <= mem[axi_read_index + 1'b1];
                    m_axi_rlast     <= (axi_beats_left == 9'd2);
                end
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((compute_done != 1'b1) && (cycle_count < 6000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (compute_done != 1'b1) begin
            $display("FAIL: timed out waiting for AXI-backed QMAP compute path");
            $finish(1);
        end

        #1;
        $display("AXI-backed QMAP dot64 compute path checks");
        $display("  AXI read bursts     = %0d", burst_count);
        $display("  partial_sum         actual=%0d expected_payload=%0d expected_const=%0d",
                 partial_sum, expected_partial_sum, EXPECTED_PARTIAL_SUM);
        $display("  scaled_sum_q26      actual=%0d expected_payload=%0d expected_const=%0d",
                 scaled_sum_q26, expected_scaled_sum_q26, EXPECTED_SCALED_SUM_Q26);
        $display("  compare_match=%0d compute_error=%0d axi_error=%0d",
                 compare_match, compute_error, axi_adapter_error);

        if (axi_adapter_error) begin
            $display("FAIL: axi4_read_master reported error");
            $finish(1);
        end

        if (compute_error) begin
            $display("FAIL: qmap_dot64_compute_path reported error");
            $finish(1);
        end

        if (!compare_match) begin
            $display("FAIL: AXI-backed compute result did not match expected payload");
            $finish(1);
        end

        if ((partial_sum !== EXPECTED_PARTIAL_SUM) ||
            (expected_partial_sum !== EXPECTED_PARTIAL_SUM)) begin
            $display("FAIL: partial_sum mismatch");
            $finish(1);
        end

        if ((scaled_sum_q26 !== EXPECTED_SCALED_SUM_Q26) ||
            (expected_scaled_sum_q26 !== EXPECTED_SCALED_SUM_Q26)) begin
            $display("FAIL: scaled_sum_q26 mismatch");
            $finish(1);
        end

        if (burst_count == 0) begin
            $display("FAIL: no AXI read bursts were observed");
            $finish(1);
        end

        $display("PASS: qmap_dot64_compute_path read through axi4_read_master and matched expected dot64 result.");
        $finish;
    end

endmodule

`default_nettype wire
