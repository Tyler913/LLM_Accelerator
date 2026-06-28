`default_nettype none

`include "qmap_defs.svh"

// Vivado-facing top for the QMAP Layer 0 Q/K/V projection path.
//
// This module connects the descriptor-driven QKV compute path to the reusable
// AXI4 read and write masters. It is the first hardware-facing top that both
// reads a QMAP work packet from PL DDR4 and writes computed activation buffers
// back to PL DDR4.
//
// The success/status outputs only report controller completion with no detected
// read/write/compute errors. Golden-output comparison stays in simulation or PS
// software so this top does not pretend to do an in-PL compare.
module qmap_qkv_projection_axi_smoke_top #(
    parameter int ADDR_WIDTH     = 64,
    parameter int DATA_WIDTH     = 32,
    parameter int INPUT_SIZE     = 1024,
    parameter int GROUP_SIZE     = 64,
    parameter int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL = 4,
    parameter int ACT_WIDTH      = 24,
    parameter int ACT_FRAC       = 12,
    parameter int WEIGHT_WIDTH   = 4,
    parameter int SCALE_WIDTH    = 16,
    parameter int SCALE_FRAC     = 14,
    parameter int PARTIAL_WIDTH  = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH   = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH  = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
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
    output logic                    o_success_sticky,
    output logic [3 : 0]            o_status,

    output logic [31 : 0]           o_rows_done,
    output logic [31 : 0]           o_last_output_q12_12,
    output logic [31 : 0]           o_last_row_sum_q26_low32,
    output logic [63 : 0]           o_last_row_sum_q26,

    output logic                    o_axi_read_error_sticky,
    output logic                    o_axi_write_error_sticky,

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
    (* X_INTERFACE_PARAMETER = "ADDR_WIDTH 64, DATA_WIDTH 32, PROTOCOL AXI4, FREQ_HZ 96968727, HAS_BURST 1, HAS_PROT 1, HAS_CACHE 1, MAX_BURST_LENGTH 256, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
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

    logic rd_req_valid;
    logic rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] rd_req_addr;
    logic [15 : 0] rd_req_len_bytes;
    logic rd_rsp_valid;
    logic rd_rsp_ready;
    logic [DATA_WIDTH-1 : 0] rd_rsp_data;
    logic rd_rsp_last;

    logic wr_req_valid;
    logic wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] wr_req_addr;
    logic [15 : 0] wr_req_len_bytes;
    logic [DATA_WIDTH-1 : 0] wr_data;
    logic wr_data_valid;
    logic wr_data_ready;
    logic wr_data_last;
    logic wr_done;

    logic compute_busy;
    logic compute_done;
    logic compute_error;
    logic signed [ROW_ACC_WIDTH-1 : 0] compute_last_row_sum_q26;
    logic signed [31 : 0] compute_last_output_q12_12;
    logic [31 : 0] compute_rows_done;

    logic axi_read_busy;
    logic axi_read_error;
    logic axi_write_busy;
    logic axi_write_error;

    logic operation_error;
    logic signed [63 : 0] sign_extended_row_sum_q26;

    assign o_busy = compute_busy || axi_read_busy || axi_write_busy;
    assign start_pulse = i_start && !start_d && !o_busy;

    assign operation_error =
        compute_error ||
        axi_read_error ||
        axi_write_error ||
        o_axi_read_error_sticky ||
        o_axi_write_error_sticky;

    assign sign_extended_row_sum_q26 =
        {{(64-ROW_ACC_WIDTH){compute_last_row_sum_q26[ROW_ACC_WIDTH-1]}},
         compute_last_row_sum_q26};

    assign o_status = {
        o_success_sticky,
        o_error_sticky,
        o_done_sticky,
        o_busy
    };

    qmap_qkv_projection_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
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
        .i_qmap_base_addr(`QMAP_QKV_BASE_ADDR),
        .o_busy(compute_busy),
        .o_done(compute_done),
        .o_error(compute_error),
        .o_rows_done(compute_rows_done),
        .o_last_row_sum_q26(compute_last_row_sum_q26),
        .o_last_output_q12_12(compute_last_output_q12_12),
        .o_mem_rd_req_valid(rd_req_valid),
        .i_mem_rd_req_ready(rd_req_ready),
        .o_mem_rd_req_addr(rd_req_addr),
        .o_mem_rd_req_len_bytes(rd_req_len_bytes),
        .i_mem_rd_rsp_valid(rd_rsp_valid),
        .o_mem_rd_rsp_ready(rd_rsp_ready),
        .i_mem_rd_rsp_data(rd_rsp_data),
        .i_mem_rd_rsp_last(rd_rsp_last),
        .o_mem_wr_req_valid(wr_req_valid),
        .i_mem_wr_req_ready(wr_req_ready),
        .o_mem_wr_req_addr(wr_req_addr),
        .o_mem_wr_req_len_bytes(wr_req_len_bytes),
        .o_mem_wr_data(wr_data),
        .o_mem_wr_data_valid(wr_data_valid),
        .i_mem_wr_data_ready(wr_data_ready),
        .o_mem_wr_data_last(wr_data_last),
        .i_mem_wr_done(wr_done),
        .i_mem_wr_error(axi_write_error)
    );

    axi4_read_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) axi_reader (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_req_valid(rd_req_valid),
        .o_req_ready(rd_req_ready),
        .i_req_addr(rd_req_addr),
        .i_req_len_bytes(rd_req_len_bytes),
        .o_rsp_valid(rd_rsp_valid),
        .i_rsp_ready(rd_rsp_ready),
        .o_rsp_data(rd_rsp_data),
        .o_rsp_last(rd_rsp_last),
        .o_busy(axi_read_busy),
        .o_error(axi_read_error),
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

    axi4_write_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) axi_writer (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_req_valid(wr_req_valid),
        .o_req_ready(wr_req_ready),
        .i_req_addr(wr_req_addr),
        .i_req_len_bytes(wr_req_len_bytes),
        .i_wdata(wr_data),
        .i_wdata_valid(wr_data_valid),
        .o_wdata_ready(wr_data_ready),
        .i_wdata_last(wr_data_last),
        .o_done(wr_done),
        .o_busy(axi_write_busy),
        .o_error(axi_write_error),
        .o_m_axi_awaddr(M_AXI_AWADDR),
        .o_m_axi_awlen(M_AXI_AWLEN),
        .o_m_axi_awsize(M_AXI_AWSIZE),
        .o_m_axi_awburst(M_AXI_AWBURST),
        .o_m_axi_awprot(M_AXI_AWPROT),
        .o_m_axi_awcache(M_AXI_AWCACHE),
        .o_m_axi_awvalid(M_AXI_AWVALID),
        .i_m_axi_awready(M_AXI_AWREADY),
        .o_m_axi_wdata(M_AXI_WDATA),
        .o_m_axi_wstrb(M_AXI_WSTRB),
        .o_m_axi_wlast(M_AXI_WLAST),
        .o_m_axi_wvalid(M_AXI_WVALID),
        .i_m_axi_wready(M_AXI_WREADY),
        .i_m_axi_bresp(M_AXI_BRESP),
        .i_m_axi_bvalid(M_AXI_BVALID),
        .o_m_axi_bready(M_AXI_BREADY)
    );

    always @(posedge aclk) begin
        if (!aresetn) begin
            start_d                  <= 1'b0;
            o_done_sticky            <= 1'b0;
            o_error_sticky           <= 1'b0;
            o_success_sticky         <= 1'b0;
            o_rows_done              <= 32'd0;
            o_last_output_q12_12     <= 32'd0;
            o_last_row_sum_q26_low32 <= 32'd0;
            o_last_row_sum_q26       <= 64'd0;
            o_axi_read_error_sticky  <= 1'b0;
            o_axi_write_error_sticky <= 1'b0;
        end else begin
            start_d <= i_start;

            if (i_clear || start_pulse) begin
                o_done_sticky            <= 1'b0;
                o_error_sticky           <= 1'b0;
                o_success_sticky         <= 1'b0;
                o_rows_done              <= 32'd0;
                o_last_output_q12_12     <= 32'd0;
                o_last_row_sum_q26_low32 <= 32'd0;
                o_last_row_sum_q26       <= 64'd0;
                o_axi_read_error_sticky  <= 1'b0;
                o_axi_write_error_sticky <= 1'b0;
            end else begin
                if (axi_read_error) begin
                    o_axi_read_error_sticky <= 1'b1;
                    o_error_sticky          <= 1'b1;
                end

                if (axi_write_error) begin
                    o_axi_write_error_sticky <= 1'b1;
                    o_error_sticky           <= 1'b1;
                end

                if (compute_done) begin
                    o_done_sticky            <= 1'b1;
                    o_error_sticky           <= operation_error;
                    o_success_sticky         <= !operation_error;
                    o_rows_done              <= compute_rows_done;
                    o_last_output_q12_12     <= compute_last_output_q12_12;
                    o_last_row_sum_q26_low32 <= sign_extended_row_sum_q26[31 : 0];
                    o_last_row_sum_q26       <= sign_extended_row_sum_q26;
                end
            end
        end
    end

endmodule

`default_nettype wire
