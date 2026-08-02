`default_nettype none

// Vivado/block-design facing wrapper for the one-token accelerator.
//
// qmap_one_token_axil_top already contains the AXI4-Lite register/control seam
// and exposes a simple single-port memory request/write interface. This module
// adds the same lightweight AXI4 read/write masters used by the earlier QMAP
// smoke tests so Vivado BD can see one AXI4-Lite slave (S_AXI) and one AXI4
// memory master (M_AXI).
//
// The RTL remains intentionally thin: register semantics stay in
// qmap_one_token_control_regs.sv, the local AXI-Lite seam stays in
// qmap_one_token_axil_top.sv, and this wrapper only performs protocol adaptation
// for the board-facing memory path.
module qmap_one_token_axi_top #(
    parameter int AXI_ADDR_WIDTH   = 12,
    parameter int ADDR_WIDTH       = 64,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_LAYERS       = 28,
    parameter int MAX_CONTEXT      = 256,
    parameter int LAYER_INDEX_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS),
    parameter int LAYER_COUNT_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS + 1),
    parameter int POSITION_WIDTH   = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int INPUT_SIZE       = 1024,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL   = 1,
    parameter int BODY_GROUP_PARALLEL = 1,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TAIL_MAX_TILES   = 9496,
    parameter int TAIL_TILE_ROWS   = 16,
    parameter int TAIL_ROW_PARALLEL = 1,
    parameter int TOKEN_ID_WIDTH   = 32,
    parameter int SCORE_WIDTH      = ROW_ACC_WIDTH
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI, ASSOCIATED_RESET aresetn" *)
    input  wire logic                         aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire logic                         aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "ADDR_WIDTH 12, DATA_WIDTH 32, PROTOCOL AXI4LITE, FREQ_HZ 96968727, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 1, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
    input  wire logic [AXI_ADDR_WIDTH-1 : 0]  S_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire logic [2 : 0]                 S_AXI_AWPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire logic                         S_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic                              S_AXI_AWREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire logic [31 : 0]                S_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire logic [3 : 0]                 S_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire logic                         S_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic                              S_AXI_WREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1 : 0]                      S_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic                              S_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire logic                         S_AXI_BREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire logic [AXI_ADDR_WIDTH-1 : 0]  S_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire logic [2 : 0]                 S_AXI_ARPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire logic                         S_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic                              S_AXI_ARREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [31 : 0]                     S_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1 : 0]                      S_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic                              S_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire logic                         S_AXI_RREADY,

    output logic                              o_axil_busy,
    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic                              o_mem_error,
    output logic [31 : 0]                     o_status,
    output logic [31 : 0]                     o_best_token_id,
    output logic [31 : 0]                     o_best_score_low32,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWADDR" *)
    output logic [ADDR_WIDTH-1 : 0]           M_AXI_AWADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWLEN" *)
    output logic [7 : 0]                      M_AXI_AWLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWSIZE" *)
    output logic [2 : 0]                      M_AXI_AWSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWBURST" *)
    output logic [1 : 0]                      M_AXI_AWBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWPROT" *)
    output logic [2 : 0]                      M_AXI_AWPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWCACHE" *)
    output logic [3 : 0]                      M_AXI_AWCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWVALID" *)
    output logic                              M_AXI_AWVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI AWREADY" *)
    input  wire logic                         M_AXI_AWREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WDATA" *)
    output logic [MEM_DATA_WIDTH-1 : 0]       M_AXI_WDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WSTRB" *)
    output logic [(MEM_DATA_WIDTH/8)-1 : 0]   M_AXI_WSTRB,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WLAST" *)
    output logic                              M_AXI_WLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WVALID" *)
    output logic                              M_AXI_WVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI WREADY" *)
    input  wire logic                         M_AXI_WREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BRESP" *)
    input  wire logic [1 : 0]                 M_AXI_BRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BVALID" *)
    input  wire logic                         M_AXI_BVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI BREADY" *)
    output logic                              M_AXI_BREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    (* X_INTERFACE_PARAMETER = "ADDR_WIDTH 64, DATA_WIDTH 32, PROTOCOL AXI4, FREQ_HZ 96968727, HAS_BURST 1, HAS_PROT 1, HAS_CACHE 1, MAX_BURST_LENGTH 256, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1" *)
    output logic [ADDR_WIDTH-1 : 0]           M_AXI_ARADDR,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output logic [7 : 0]                      M_AXI_ARLEN,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output logic [2 : 0]                      M_AXI_ARSIZE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output logic [1 : 0]                      M_AXI_ARBURST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARPROT" *)
    output logic [2 : 0]                      M_AXI_ARPROT,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARCACHE" *)
    output logic [3 : 0]                      M_AXI_ARCACHE,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output logic                              M_AXI_ARVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  wire logic                         M_AXI_ARREADY,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  wire logic [MEM_DATA_WIDTH-1 : 0]  M_AXI_RDATA,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  wire logic [1 : 0]                 M_AXI_RRESP,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  wire logic                         M_AXI_RLAST,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  wire logic                         M_AXI_RVALID,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output logic                              M_AXI_RREADY
);

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;

    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [MEM_DATA_WIDTH-1 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;
    logic mem_rd_error;
    logic axi_read_busy;
    logic axi_write_busy;
    logic axi_read_busy_d;
    logic axi_write_busy_d;
    logic run_busy_d;
    logic mem_error_sticky;

    logic [LAYER_INDEX_WIDTH-1:0] active_layer_index;
    logic [LAYER_COUNT_WIDTH-1:0] layers_started;
    logic [LAYER_COUNT_WIDTH-1:0] layers_completed;
    logic [MAX_LAYERS-1:0] layer_done_mask;
    logic [MAX_LAYERS-1:0] layer_error_mask;
    logic [ADDR_WIDTH-1 : 0] last_layer_output_base_addr;
    logic [1 : 0] layer0_active_stage_debug;
    logic [7 : 0] layer0_state_debug;
    logic [1 : 0] layer0_stage_done_mask;
    logic [1 : 0] layer0_stage_error_mask;
    logic [3 : 0] layer0_full_stage_done_mask;
    logic [3 : 0] layer0_full_stage_error_mask;
    logic [4 : 0] body_stage_done_mask;
    logic [4 : 0] body_stage_error_mask;
    logic [31 : 0] scheduler_mem_read_burst_count;
    logic [31 : 0] scheduler_mem_read_word_count;
    logic [31 : 0] scheduler_mem_write_req_count;
    logic [31 : 0] scheduler_mem_write_word_count;
    logic tail_error;
    logic tail_norm_saturation;
    logic [ADDR_WIDTH-1 : 0] tail_effective_final_hidden_base_addr;
    logic [TOKEN_ID_WIDTH-1 : 0] tail_best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] tail_best_score_q26;
    logic [31 : 0] tail_tiles_started;
    logic [31 : 0] tail_tiles_completed;
    logic [31 : 0] tail_norm_cycle_count;
    logic [31 : 0] tail_mem_read_burst_count;
    logic [31 : 0] tail_mem_read_word_count;
    logic [31 : 0] tail_mem_write_word_count;
    logic [7 : 0] state_debug;
    logic [7 : 0] phase_debug;
    logic scheduler_done_pulse;
    logic tail_start_pulse;
    logic tail_done_pulse;
    logic tail_active;

    // The per-request masters clear their error flags when they accept the
    // next request. Preserve any completed-request failure for the full run so
    // software cannot miss an early AXI error after later traffic succeeds.
    assign o_mem_error = mem_error_sticky ||
                         (axi_read_busy && mem_rd_error) ||
                         (axi_write_busy && mem_wr_error);
    assign o_status = {
        phase_debug,
        state_debug,
        8'd0,
        2'd0,
        tail_norm_saturation,
        tail_error,
        o_mem_error,
        o_error,
        o_done,
        o_busy
    };
    assign o_best_token_id = {{(32-TOKEN_ID_WIDTH){1'b0}}, tail_best_token_id};
    assign o_best_score_low32 = tail_best_score_q26[31 : 0];

    always @(posedge aclk) begin
        if (!aresetn) begin
            axi_read_busy_d  <= 1'b0;
            axi_write_busy_d <= 1'b0;
            run_busy_d       <= 1'b0;
            mem_error_sticky <= 1'b0;
        end else begin
            axi_read_busy_d  <= axi_read_busy;
            axi_write_busy_d <= axi_write_busy;
            run_busy_d       <= o_busy;

            if (!run_busy_d && o_busy) begin
                mem_error_sticky <= 1'b0;
            end else if ((axi_read_busy_d && !axi_read_busy && mem_rd_error) ||
                         (axi_write_busy_d && !axi_write_busy && mem_wr_error)) begin
                mem_error_sticky <= 1'b1;
            end
        end
    end

    qmap_one_token_axil_top #(
        .AXI_ADDR_WIDTH    (AXI_ADDR_WIDTH),
        .ADDR_WIDTH        (ADDR_WIDTH),
        .MEM_DATA_WIDTH    (MEM_DATA_WIDTH),
        .MAX_LAYERS        (MAX_LAYERS),
        .MAX_CONTEXT       (MAX_CONTEXT),
        .LAYER_INDEX_WIDTH (LAYER_INDEX_WIDTH),
        .LAYER_COUNT_WIDTH (LAYER_COUNT_WIDTH),
        .POSITION_WIDTH    (POSITION_WIDTH),
        .INPUT_SIZE        (INPUT_SIZE),
        .GROUP_SIZE        (GROUP_SIZE),
        .GROUP_COUNT       (GROUP_COUNT),
        .GROUP_PARALLEL    (GROUP_PARALLEL),
        .BODY_GROUP_PARALLEL(BODY_GROUP_PARALLEL),
        .ACT_WIDTH         (ACT_WIDTH),
        .ACT_FRAC          (ACT_FRAC),
        .WEIGHT_WIDTH      (WEIGHT_WIDTH),
        .SCALE_WIDTH       (SCALE_WIDTH),
        .SCALE_FRAC        (SCALE_FRAC),
        .PARTIAL_WIDTH     (PARTIAL_WIDTH),
        .SCALED_WIDTH      (SCALED_WIDTH),
        .ROW_ACC_WIDTH     (ROW_ACC_WIDTH),
        .TAIL_MAX_TILES    (TAIL_MAX_TILES),
        .TAIL_TILE_ROWS    (TAIL_TILE_ROWS),
        .TAIL_ROW_PARALLEL (TAIL_ROW_PARALLEL),
        .TOKEN_ID_WIDTH    (TOKEN_ID_WIDTH),
        .SCORE_WIDTH       (SCORE_WIDTH)
    ) control_top (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_s_axi_awaddr(S_AXI_AWADDR),
        .i_s_axi_awprot(S_AXI_AWPROT),
        .i_s_axi_awvalid(S_AXI_AWVALID),
        .o_s_axi_awready(S_AXI_AWREADY),
        .i_s_axi_wdata(S_AXI_WDATA),
        .i_s_axi_wstrb(S_AXI_WSTRB),
        .i_s_axi_wvalid(S_AXI_WVALID),
        .o_s_axi_wready(S_AXI_WREADY),
        .o_s_axi_bresp(S_AXI_BRESP),
        .o_s_axi_bvalid(S_AXI_BVALID),
        .i_s_axi_bready(S_AXI_BREADY),
        .i_s_axi_araddr(S_AXI_ARADDR),
        .i_s_axi_arprot(S_AXI_ARPROT),
        .i_s_axi_arvalid(S_AXI_ARVALID),
        .o_s_axi_arready(S_AXI_ARREADY),
        .o_s_axi_rdata(S_AXI_RDATA),
        .o_s_axi_rresp(S_AXI_RRESP),
        .o_s_axi_rvalid(S_AXI_RVALID),
        .i_s_axi_rready(S_AXI_RREADY),
        .o_axil_busy(o_axil_busy),
        .o_busy(o_busy),
        .o_done(o_done),
        .o_error(o_error),
        .o_state_debug(state_debug),
        .o_phase_debug(phase_debug),
        .o_scheduler_done_pulse(scheduler_done_pulse),
        .o_tail_start_pulse(tail_start_pulse),
        .o_tail_done_pulse(tail_done_pulse),
        .o_tail_active(tail_active),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_last_layer_output_base_addr(last_layer_output_base_addr),
        .o_layer0_active_stage_debug(layer0_active_stage_debug),
        .o_layer0_state_debug(layer0_state_debug),
        .o_layer0_stage_done_mask(layer0_stage_done_mask),
        .o_layer0_stage_error_mask(layer0_stage_error_mask),
        .o_layer0_full_stage_done_mask(layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(layer0_full_stage_error_mask),
        .o_body_stage_done_mask(body_stage_done_mask),
        .o_body_stage_error_mask(body_stage_error_mask),
        .o_scheduler_mem_read_burst_count(scheduler_mem_read_burst_count),
        .o_scheduler_mem_read_word_count(scheduler_mem_read_word_count),
        .o_scheduler_mem_write_req_count(scheduler_mem_write_req_count),
        .o_scheduler_mem_write_word_count(scheduler_mem_write_word_count),
        .o_tail_error(tail_error),
        .o_tail_norm_saturation(tail_norm_saturation),
        .o_tail_effective_final_hidden_base_addr(tail_effective_final_hidden_base_addr),
        .o_tail_best_token_id(tail_best_token_id),
        .o_tail_best_score_q26(tail_best_score_q26),
        .o_tail_tiles_started(tail_tiles_started),
        .o_tail_tiles_completed(tail_tiles_completed),
        .o_tail_norm_cycle_count(tail_norm_cycle_count),
        .o_tail_mem_read_burst_count(tail_mem_read_burst_count),
        .o_tail_mem_read_word_count(tail_mem_read_word_count),
        .o_tail_mem_write_word_count(tail_mem_write_word_count),
        .o_mem_read_burst_count(o_mem_read_burst_count),
        .o_mem_read_word_count(o_mem_read_word_count),
        .o_mem_write_req_count(o_mem_write_req_count),
        .o_mem_write_word_count(o_mem_write_word_count),
        .o_mem_rd_req_valid(mem_rd_req_valid),
        .i_mem_rd_req_ready(mem_rd_req_ready),
        .o_mem_rd_req_addr(mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(mem_rd_rsp_data),
        .i_mem_rd_rsp_last(mem_rd_rsp_last),
        .o_mem_wr_req_valid(mem_wr_req_valid),
        .i_mem_wr_req_ready(mem_wr_req_ready),
        .o_mem_wr_req_addr(mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(mem_wr_req_len_bytes),
        .o_mem_wr_data(mem_wr_data),
        .o_mem_wr_data_valid(mem_wr_data_valid),
        .i_mem_wr_data_ready(mem_wr_data_ready),
        .o_mem_wr_data_last(mem_wr_data_last),
        .i_mem_wr_done(mem_wr_done),
        .i_mem_wr_error(mem_wr_error)
    );

    axi4_read_master #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(MEM_DATA_WIDTH)
    ) axi_reader (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_req_valid(mem_rd_req_valid),
        .o_req_ready(mem_rd_req_ready),
        .i_req_addr(mem_rd_req_addr),
        .i_req_len_bytes(mem_rd_req_len_bytes),
        .o_rsp_valid(mem_rd_rsp_valid),
        .i_rsp_ready(mem_rd_rsp_ready),
        .o_rsp_data(mem_rd_rsp_data),
        .o_rsp_last(mem_rd_rsp_last),
        .o_busy(axi_read_busy),
        .o_error(mem_rd_error),
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
        .DATA_WIDTH(MEM_DATA_WIDTH)
    ) axi_writer (
        .i_clk(aclk),
        .i_rst_n(aresetn),
        .i_req_valid(mem_wr_req_valid),
        .o_req_ready(mem_wr_req_ready),
        .i_req_addr(mem_wr_req_addr),
        .i_req_len_bytes(mem_wr_req_len_bytes),
        .i_wdata(mem_wr_data),
        .i_wdata_valid(mem_wr_data_valid),
        .o_wdata_ready(mem_wr_data_ready),
        .i_wdata_last(mem_wr_data_last),
        .o_done(mem_wr_done),
        .o_busy(axi_write_busy),
        .o_error(mem_wr_error),
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

endmodule

`default_nettype wire
