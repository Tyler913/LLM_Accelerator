`default_nettype none

// Generic AXI4-Lite slave adapter for the project-local tiny MMIO register port.
//
// Contract:
//   - one outstanding AXI4-Lite transaction total (read or write)
//   - write priority when read and write address/data are presented together
//   - full 32-bit writes only (WSTRB must be 4'hF)
//   - aligned byte addresses only (ADDR[1:0] must be 0)
//   - BRESP/RRESP is SLVERR when the local register block reports an error
//     or when this adapter rejects an unaligned/partial write locally
//
// AXI_ADDR_WIDTH is the local AXI-Lite offset width exposed by the wrapper. It
// must be >= REG_ADDR_WIDTH; non-zero bits above REG_ADDR_WIDTH are rejected as
// out of range to avoid aliasing a wider AXI address space into the register map.
module axi4lite_to_mmio_regs #(
    parameter int AXI_ADDR_WIDTH = 12,
    parameter int REG_ADDR_WIDTH = 12
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic [AXI_ADDR_WIDTH-1 : 0]  i_s_axi_awaddr,
    input  wire logic [2 : 0]                 i_s_axi_awprot,
    input  wire logic                         i_s_axi_awvalid,
    output logic                              o_s_axi_awready,

    input  wire logic [31 : 0]                i_s_axi_wdata,
    input  wire logic [3 : 0]                 i_s_axi_wstrb,
    input  wire logic                         i_s_axi_wvalid,
    output logic                              o_s_axi_wready,

    output logic [1 : 0]                      o_s_axi_bresp,
    output logic                              o_s_axi_bvalid,
    input  wire logic                         i_s_axi_bready,

    input  wire logic [AXI_ADDR_WIDTH-1 : 0]  i_s_axi_araddr,
    input  wire logic [2 : 0]                 i_s_axi_arprot,
    input  wire logic                         i_s_axi_arvalid,
    output logic                              o_s_axi_arready,

    output logic [31 : 0]                     o_s_axi_rdata,
    output logic [1 : 0]                      o_s_axi_rresp,
    output logic                              o_s_axi_rvalid,
    input  wire logic                         i_s_axi_rready,

    output logic                              o_reg_wr_valid,
    input  wire logic                         i_reg_wr_ready,
    output logic                              o_reg_rd_valid,
    input  wire logic                         i_reg_rd_ready,
    output logic [REG_ADDR_WIDTH-1 : 0]       o_reg_addr,
    output logic [31 : 0]                     o_reg_wdata,
    input  wire logic [31 : 0]                i_reg_rdata,
    input  wire logic                         i_reg_error,

    output logic                              o_busy
);

    localparam logic [1 : 0] AXI_RESP_OKAY   = 2'b00;
    localparam logic [1 : 0] AXI_RESP_SLVERR = 2'b10;
    localparam logic [3 : 0] FULL_WSTRB      = 4'hF;

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_WR_COLLECT,
        S_WR_ISSUE,
        S_WR_RESP,
        S_RD_ISSUE,
        S_RD_RESP
    } state_t;

    state_t state;

    logic                              aw_seen_q;
    logic                              w_seen_q;
    logic [REG_ADDR_WIDTH-1 : 0]       awaddr_q;
    logic                              awaddr_error_q;
    logic [31 : 0]                     wdata_q;
    logic                              wstrb_error_q;

    wire logic                         write_priority;
    wire logic                         aw_handshake;
    wire logic                         w_handshake;
    wire logic                         ar_handshake;
    wire logic [REG_ADDR_WIDTH-1 : 0]  aw_reg_addr_now;
    wire logic [REG_ADDR_WIDTH-1 : 0]  ar_reg_addr_now;
    wire logic                         aw_addr_error_now;
    wire logic                         ar_addr_error_now;
    wire logic                         wstrb_error_now;
    wire logic                         have_aw_next;
    wire logic                         have_w_next;
    wire logic [REG_ADDR_WIDTH-1 : 0]  wr_addr_next;
    wire logic [31 : 0]                wr_data_next;
    wire logic                         wr_addr_error_next;
    wire logic                         wr_wstrb_error_next;
    wire logic                         wr_local_error_next;
    wire logic                         unused_prot_reduce;

    function automatic logic address_aligned;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        begin
            address_aligned = (addr[1 : 0] == 2'b00);
        end
    endfunction

    function automatic logic address_in_range;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        integer idx;
        begin
            address_in_range = 1'b1;
            for (idx = REG_ADDR_WIDTH; idx < AXI_ADDR_WIDTH; idx = idx + 1) begin
                if (addr[idx]) begin
                    address_in_range = 1'b0;
                end
            end
        end
    endfunction

    function automatic logic address_ok;
        input logic [AXI_ADDR_WIDTH-1 : 0] addr;
        begin
            address_ok = address_aligned(addr) && address_in_range(addr);
        end
    endfunction

    assign write_priority = i_s_axi_awvalid || i_s_axi_wvalid;

    assign o_s_axi_awready = ((state == S_IDLE) && (write_priority || !i_s_axi_arvalid)) ||
                             ((state == S_WR_COLLECT) && !aw_seen_q);
    assign o_s_axi_wready  = ((state == S_IDLE) && (write_priority || !i_s_axi_arvalid)) ||
                             ((state == S_WR_COLLECT) && !w_seen_q);
    assign o_s_axi_arready = (state == S_IDLE) && !write_priority;

    assign aw_handshake = o_s_axi_awready && i_s_axi_awvalid;
    assign w_handshake  = o_s_axi_wready  && i_s_axi_wvalid;
    assign ar_handshake = o_s_axi_arready && i_s_axi_arvalid;

    assign aw_reg_addr_now    = i_s_axi_awaddr[REG_ADDR_WIDTH-1 : 0];
    assign ar_reg_addr_now    = i_s_axi_araddr[REG_ADDR_WIDTH-1 : 0];
    assign aw_addr_error_now  = !address_ok(i_s_axi_awaddr);
    assign ar_addr_error_now  = !address_ok(i_s_axi_araddr);
    assign wstrb_error_now    = (i_s_axi_wstrb != FULL_WSTRB);

    assign have_aw_next       = aw_seen_q || aw_handshake;
    assign have_w_next        = w_seen_q || w_handshake;
    assign wr_addr_next       = aw_handshake ? aw_reg_addr_now   : awaddr_q;
    assign wr_data_next       = w_handshake  ? i_s_axi_wdata     : wdata_q;
    assign wr_addr_error_next = aw_handshake ? aw_addr_error_now : awaddr_error_q;
    assign wr_wstrb_error_next = w_handshake ? wstrb_error_now   : wstrb_error_q;
    assign wr_local_error_next = wr_addr_error_next || wr_wstrb_error_next;

    assign o_reg_wr_valid = (state == S_WR_ISSUE);
    assign o_reg_rd_valid = (state == S_RD_ISSUE);
    assign o_busy         = (state != S_IDLE);

    // Mark protection attributes as intentionally unused. The local register map
    // is privilege/security agnostic; any policy belongs in the upstream interconnect.
    assign unused_prot_reduce = (^i_s_axi_awprot) ^ (^i_s_axi_arprot);

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state          <= S_IDLE;
            aw_seen_q      <= 1'b0;
            w_seen_q       <= 1'b0;
            awaddr_q       <= '0;
            awaddr_error_q <= 1'b0;
            wdata_q        <= 32'd0;
            wstrb_error_q  <= 1'b0;
            o_reg_addr     <= '0;
            o_reg_wdata    <= 32'd0;
            o_s_axi_bresp  <= AXI_RESP_OKAY;
            o_s_axi_bvalid <= 1'b0;
            o_s_axi_rdata  <= 32'd0;
            o_s_axi_rresp  <= AXI_RESP_OKAY;
            o_s_axi_rvalid <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    aw_seen_q <= 1'b0;
                    w_seen_q  <= 1'b0;

                    if (aw_handshake || w_handshake) begin
                        aw_seen_q      <= have_aw_next;
                        w_seen_q       <= have_w_next;
                        awaddr_q       <= wr_addr_next;
                        awaddr_error_q <= wr_addr_error_next;
                        wdata_q        <= wr_data_next;
                        wstrb_error_q  <= wr_wstrb_error_next;

                        if (have_aw_next && have_w_next) begin
                            aw_seen_q   <= 1'b0;
                            w_seen_q    <= 1'b0;
                            o_reg_addr  <= wr_addr_next;
                            o_reg_wdata <= wr_data_next;
                            if (wr_local_error_next) begin
                                o_s_axi_bresp  <= AXI_RESP_SLVERR;
                                o_s_axi_bvalid <= 1'b1;
                                state          <= S_WR_RESP;
                            end else begin
                                state <= S_WR_ISSUE;
                            end
                        end else begin
                            state <= S_WR_COLLECT;
                        end
                    end else if (ar_handshake) begin
                        o_reg_addr <= ar_reg_addr_now;
                        if (ar_addr_error_now) begin
                            o_s_axi_rdata  <= 32'd0;
                            o_s_axi_rresp  <= AXI_RESP_SLVERR;
                            o_s_axi_rvalid <= 1'b1;
                            state          <= S_RD_RESP;
                        end else begin
                            state <= S_RD_ISSUE;
                        end
                    end
                end

                S_WR_COLLECT: begin
                    if (aw_handshake || w_handshake) begin
                        aw_seen_q      <= have_aw_next;
                        w_seen_q       <= have_w_next;
                        awaddr_q       <= wr_addr_next;
                        awaddr_error_q <= wr_addr_error_next;
                        wdata_q        <= wr_data_next;
                        wstrb_error_q  <= wr_wstrb_error_next;

                        if (have_aw_next && have_w_next) begin
                            aw_seen_q   <= 1'b0;
                            w_seen_q    <= 1'b0;
                            o_reg_addr  <= wr_addr_next;
                            o_reg_wdata <= wr_data_next;
                            if (wr_local_error_next) begin
                                o_s_axi_bresp  <= AXI_RESP_SLVERR;
                                o_s_axi_bvalid <= 1'b1;
                                state          <= S_WR_RESP;
                            end else begin
                                state <= S_WR_ISSUE;
                            end
                        end
                    end
                end

                S_WR_ISSUE: begin
                    if (i_reg_wr_ready) begin
                        o_s_axi_bresp  <= i_reg_error ? AXI_RESP_SLVERR : AXI_RESP_OKAY;
                        o_s_axi_bvalid <= 1'b1;
                        state          <= S_WR_RESP;
                    end
                end

                S_WR_RESP: begin
                    if (o_s_axi_bvalid && i_s_axi_bready) begin
                        o_s_axi_bvalid <= 1'b0;
                        o_s_axi_bresp  <= AXI_RESP_OKAY;
                        state          <= S_IDLE;
                    end
                end

                S_RD_ISSUE: begin
                    if (i_reg_rd_ready) begin
                        o_s_axi_rdata  <= i_reg_rdata;
                        o_s_axi_rresp  <= i_reg_error ? AXI_RESP_SLVERR : AXI_RESP_OKAY;
                        o_s_axi_rvalid <= 1'b1;
                        state          <= S_RD_RESP;
                    end
                end

                S_RD_RESP: begin
                    if (o_s_axi_rvalid && i_s_axi_rready) begin
                        o_s_axi_rvalid <= 1'b0;
                        o_s_axi_rresp  <= AXI_RESP_OKAY;
                        state          <= S_IDLE;
                    end
                end

                default: begin
                    state          <= S_IDLE;
                    aw_seen_q      <= 1'b0;
                    w_seen_q       <= 1'b0;
                    o_s_axi_bvalid <= 1'b0;
                    o_s_axi_rvalid <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire

