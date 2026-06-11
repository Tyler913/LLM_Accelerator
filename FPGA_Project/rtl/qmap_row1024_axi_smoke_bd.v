`default_nettype none

// Block Design friendly wrapper for qmap_row1024_axi_smoke_top.
//
// Keep this BD-facing surface simple and fixed; the real SystemVerilog logic
// stays inside qmap_row1024_axi_smoke_top.sv.
module qmap_row1024_axi_smoke_bd (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF M_AXI, ASSOCIATED_RESET aresetn, FREQ_HZ 96968727" *)
    input  wire        aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aresetn, POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire        i_start,
    input  wire        i_clear,

    output wire        o_busy,
    output wire        o_done_sticky,
    output wire        o_error_sticky,
    output wire        o_compare_match_sticky,
    output wire [3:0]  o_status,

    output wire [31:0] o_row_sum_q26_low32,
    output wire [31:0] o_expected_row_sum_q26_low32,

    output wire [63:0] o_row_sum_q26,
    output wire [63:0] o_expected_row_sum_q26,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    output wire [63:0] M_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output wire [7:0]  M_AXI_AWLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output wire [2:0]  M_AXI_AWSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output wire [1:0]  M_AXI_AWBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output wire [2:0]  M_AXI_AWPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output wire [3:0]  M_AXI_AWCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output wire        M_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  wire        M_AXI_AWREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output wire [31:0] M_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output wire [3:0]  M_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output wire        M_AXI_WLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output wire        M_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  wire        M_AXI_WREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  wire [1:0]  M_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  wire        M_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output wire        M_AXI_BREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI, ADDR_WIDTH 64, DATA_WIDTH 32, PROTOCOL AXI4, FREQ_HZ 96968727, HAS_BURST 1, HAS_PROT 1, HAS_CACHE 1, MAX_BURST_LENGTH 256, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 0" *)
    output wire [63:0] M_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output wire [7:0]  M_AXI_ARLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output wire [2:0]  M_AXI_ARSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output wire [1:0]  M_AXI_ARBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *)
    output wire [2:0]  M_AXI_ARPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *)
    output wire [3:0]  M_AXI_ARCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output wire        M_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  wire        M_AXI_ARREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  wire [31:0] M_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  wire [1:0]  M_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  wire        M_AXI_RLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  wire        M_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output wire        M_AXI_RREADY
);

    qmap_row1024_axi_smoke_top u_core (
        .aclk(aclk),
        .aresetn(aresetn),
        .i_start(i_start),
        .i_clear(i_clear),
        .o_busy(o_busy),
        .o_done_sticky(o_done_sticky),
        .o_error_sticky(o_error_sticky),
        .o_compare_match_sticky(o_compare_match_sticky),
        .o_status(o_status),
        .o_row_sum_q26_low32(o_row_sum_q26_low32),
        .o_expected_row_sum_q26_low32(o_expected_row_sum_q26_low32),
        .o_row_sum_q26(o_row_sum_q26),
        .o_expected_row_sum_q26(o_expected_row_sum_q26),
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

endmodule

`default_nettype wire
