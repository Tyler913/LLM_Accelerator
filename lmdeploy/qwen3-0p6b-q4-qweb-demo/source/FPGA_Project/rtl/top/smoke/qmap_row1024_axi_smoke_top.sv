`default_nettype none

`include "qmap_defs.svh"

// Vivado-facing top for the QMAP row1024 PL smoke path.
//
// This module proves the second-level PL DDR read/compute path:
//   QMAP row1024 image in PL DDR4 -> AXI4 reads -> q4_gemv_row_1024.
//
// It is still a bring-up micro-kernel. The final accelerator should replace
// these GPIO-style status pins with a real control/status register block.
module qmap_row1024_axi_smoke_top #(
    parameter int ADDR_WIDTH    = 64,
    parameter int DATA_WIDTH    = 32,
    parameter int INPUT_SIZE    = 1024,
    parameter int GROUP_SIZE    = 64,
    parameter int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH     = 16,
    parameter int ACT_FRAC      = 12,
    parameter int WEIGHT_WIDTH  = 4,
    parameter int SCALE_WIDTH   = 16,
    parameter int SCALE_FRAC    = 14,
    parameter int PARTIAL_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF M_AXI, ASSOCIATED_RESET aresetn" *)
    input  wire logic                    aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire logic                    aresetn,

    input  wire logic                    i_start,
    input  wire logic                    i_clear,

    output logic                    o_busy,
    output logic                    o_done_sticky,
    output logic                    o_error_sticky,
    output logic                    o_compare_match_sticky,
    output logic [3 : 0]            o_status,

    output logic [31 : 0]           o_row_sum_q26_low32,
    output logic [31 : 0]           o_expected_row_sum_q26_low32,

    output logic [63 : 0]           o_row_sum_q26,
    output logic [63 : 0]           o_expected_row_sum_q26,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    output logic [ADDR_WIDTH-1 : 0] M_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output logic [7 : 0]            M_AXI_AWLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output logic [2 : 0]            M_AXI_AWSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output logic [1 : 0]            M_AXI_AWBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output logic [2 : 0]            M_AXI_AWPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output logic [3 : 0]            M_AXI_AWCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output logic                    M_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  wire logic                    M_AXI_AWREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output logic [DATA_WIDTH-1 : 0] M_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output logic [(DATA_WIDTH/8)-1 : 0] M_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output logic                    M_AXI_WLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output logic                    M_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  wire logic                    M_AXI_WREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  wire logic [1 : 0]            M_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  wire logic                    M_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output logic                    M_AXI_BREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    (* X_INTERFACE_PARAMETER = "ADDR_WIDTH 64, DATA_WIDTH 32, PROTOCOL AXI4, FREQ_HZ 96968727, HAS_BURST 1, HAS_PROT 1, HAS_CACHE 1, MAX_BURST_LENGTH 256, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 0" *)
    output logic [ADDR_WIDTH-1 : 0] M_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output logic [7 : 0]            M_AXI_ARLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output logic [2 : 0]            M_AXI_ARSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output logic [1 : 0]            M_AXI_ARBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *)
    output logic [2 : 0]            M_AXI_ARPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *)
    output logic [3 : 0]            M_AXI_ARCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output logic                    M_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  wire logic                    M_AXI_ARREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  wire logic [DATA_WIDTH-1 : 0] M_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  wire logic [1 : 0]            M_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  wire logic                    M_AXI_RLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  wire logic                    M_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output logic                    M_AXI_RREADY
);

    logic start_d;
    logic start_pulse;

    logic local_req_valid;
    logic local_req_ready;
    logic [ADDR_WIDTH-1 : 0] local_req_addr;
    logic [15 : 0] local_req_len_bytes;
    logic local_rsp_valid;
    logic local_rsp_ready;
    logic [DATA_WIDTH-1 : 0] local_rsp_data;
    logic local_rsp_last;

    logic compute_busy;
    logic compute_done;
    logic compute_error;
    logic compute_compare_match;
    logic signed [63 : 0] compute_row_sum_q26;
    logic signed [63 : 0] compute_expected_row_sum_q26;

    logic axi_busy;
    logic axi_error;
    logic axi_error_sticky;

    assign start_pulse = i_start && !start_d && !compute_busy;

    assign o_busy = compute_busy || axi_busy;
    assign o_status = {
        o_compare_match_sticky,
        o_error_sticky,
        o_done_sticky,
        o_busy
    };

    assign o_row_sum_q26_low32 = o_row_sum_q26[31 : 0];
    assign o_expected_row_sum_q26_low32 = o_expected_row_sum_q26[31 : 0];

    assign M_AXI_AWADDR  = 'd0;
    assign M_AXI_AWLEN   = 8'd0;
    assign M_AXI_AWSIZE  = 3'd2;
    assign M_AXI_AWBURST = 2'b01;
    assign M_AXI_AWPROT  = 3'b000;
    assign M_AXI_AWCACHE = 4'b0011;
    assign M_AXI_AWVALID = 1'b0;
    assign M_AXI_WDATA   = 'd0;
    assign M_AXI_WSTRB   = 'd0;
    assign M_AXI_WLAST   = 1'b0;
    assign M_AXI_WVALID  = 1'b0;
    assign M_AXI_BREADY  = 1'b1;

    qmap_row1024_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH)
    ) compute_path (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_start(start_pulse),
        .i_qmap_base_addr(`QMAP_ROW1024_BASE_ADDR),
        .o_busy(compute_busy),
        .o_done(compute_done),
        .o_error(compute_error),
        .o_compare_match(compute_compare_match),
        .o_mem_req_valid(local_req_valid),
        .i_mem_req_ready(local_req_ready),
        .o_mem_req_addr(local_req_addr),
        .o_mem_req_len_bytes(local_req_len_bytes),
        .i_mem_rsp_valid(local_rsp_valid),
        .o_mem_rsp_ready(local_rsp_ready),
        .i_mem_rsp_data(local_rsp_data),
        .i_mem_rsp_last(local_rsp_last),
        .o_row_sum_q26(compute_row_sum_q26),
        .o_expected_row_sum_q26(compute_expected_row_sum_q26)
    );

    axi4_read_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) axi_reader (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_req_valid(local_req_valid),
        .o_req_ready(local_req_ready),
        .i_req_addr(local_req_addr),
        .i_req_len_bytes(local_req_len_bytes),
        .o_rsp_valid(local_rsp_valid),
        .i_rsp_ready(local_rsp_ready),
        .o_rsp_data(local_rsp_data),
        .o_rsp_last(local_rsp_last),
        .o_busy(axi_busy),
        .o_error(axi_error),
        .o_m_axi_araddr(M_AXI_ARADDR),
        .o_m_axi_arlen(M_AXI_ARLEN),
        .o_m_axi_arsize(M_AXI_ARSIZE),
        .o_m_axi_arburst(M_AXI_ARBURST),
        .o_m_axi_arprot(M_AXI_ARPROT),
        .o_m_axi_arcache(M_AXI_ARCACHE),
        .o_m_axi_arvalid(M_AXI_ARVALID),
        .i_m_axi_arready(M_AXI_ARREADY),
        .i_m_axi_rdata(M_AXI_RDATA),
        .i_m_axi_rresp(M_AXI_RRESP),
        .i_m_axi_rlast(M_AXI_RLAST),
        .i_m_axi_rvalid(M_AXI_RVALID),
        .o_m_axi_rready(M_AXI_RREADY)
    );

    always @(posedge aclk) begin
        if (!aresetn) begin
            start_d                    <= 1'b0;
            axi_error_sticky           <= 1'b0;
            o_done_sticky              <= 1'b0;
            o_error_sticky             <= 1'b0;
            o_compare_match_sticky     <= 1'b0;
            o_row_sum_q26              <= 64'd0;
            o_expected_row_sum_q26     <= 64'd0;
        end else begin
            start_d <= i_start;

            if (i_clear || start_pulse) begin
                axi_error_sticky           <= 1'b0;
                o_done_sticky              <= 1'b0;
                o_error_sticky             <= 1'b0;
                o_compare_match_sticky     <= 1'b0;
                o_row_sum_q26              <= 64'd0;
                o_expected_row_sum_q26     <= 64'd0;
            end else begin
                if (axi_error) begin
                    axi_error_sticky <= 1'b1;
                    o_error_sticky   <= 1'b1;
                end

                if (compute_done) begin
                    o_done_sticky          <= 1'b1;
                    o_error_sticky         <= compute_error || axi_error_sticky || axi_error;
                    o_compare_match_sticky <= compute_compare_match &&
                                              !(compute_error || axi_error_sticky || axi_error);
                    o_row_sum_q26          <= compute_row_sum_q26;
                    o_expected_row_sum_q26 <= compute_expected_row_sum_q26;
                end
            end
        end
    end

endmodule

`default_nettype wire
