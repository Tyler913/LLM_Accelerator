`default_nettype none

// Minimal read-only AXI4 master for the project-local memory request/response
// interface used by the QMAP readers.
//
// First-version limits:
//   - read channel only
//   - one outstanding burst
//   - aligned requests
//   - request length must fit in one AXI burst, up to 256 beats
module axi4_read_master #(
    parameter int ADDR_WIDTH = 64,
    parameter int DATA_WIDTH = 32
)
(
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_req_valid,
    output logic                    o_req_ready,
    input  wire logic [ADDR_WIDTH-1 : 0] i_req_addr,
    input  wire logic [15 : 0]           i_req_len_bytes,

    output logic                    o_rsp_valid,
    input  wire logic                    i_rsp_ready,
    output logic [DATA_WIDTH-1 : 0] o_rsp_data,
    output logic                    o_rsp_last,

    output logic                    o_busy,
    output logic                    o_error,

    output logic [ADDR_WIDTH-1 : 0] o_m_axi_araddr,
    output logic [7 : 0]            o_m_axi_arlen,
    output logic [2 : 0]            o_m_axi_arsize,
    output logic [1 : 0]            o_m_axi_arburst,
    output logic [2 : 0]            o_m_axi_arprot,
    output logic [3 : 0]            o_m_axi_arcache,
    output logic                    o_m_axi_arvalid,
    input  wire logic                    i_m_axi_arready,

    input  wire logic [DATA_WIDTH-1 : 0] i_m_axi_rdata,
    input  wire logic [1 : 0]            i_m_axi_rresp,
    input  wire logic                    i_m_axi_rlast,
    input  wire logic                    i_m_axi_rvalid,
    output logic                    o_m_axi_rready
);

    localparam int DATA_BYTES = DATA_WIDTH / 8;
    localparam int ADDR_LSB   = $clog2(DATA_BYTES);

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_AR,
        S_R
    } state_t;

    state_t state;
    logic [8 : 0] beats_total;
    logic [8 : 0] beats_accepted;
    logic request_is_aligned;
    logic [15 : 0] rounded_len_bytes;

    assign request_is_aligned = (i_req_addr[ADDR_LSB-1 : 0] == 'd0);
    assign rounded_len_bytes  = i_req_len_bytes + DATA_BYTES - 1;

    assign o_req_ready     = (state == S_IDLE);
    assign o_busy          = (state != S_IDLE);
    assign o_rsp_valid     = (state == S_R) && i_m_axi_rvalid;
    assign o_rsp_data      = i_m_axi_rdata;
    assign o_rsp_last      = i_m_axi_rlast;
    assign o_m_axi_rready  = (state == S_R) && i_rsp_ready;
    assign o_m_axi_arsize  = ADDR_LSB[2:0];
    assign o_m_axi_arburst = 2'b01;
    assign o_m_axi_arprot  = 3'b000;
    assign o_m_axi_arcache = 4'b0011;

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state           <= S_IDLE;
            beats_total     <= 9'd0;
            beats_accepted  <= 9'd0;
            o_error         <= 1'b0;
            o_m_axi_araddr  <= 'd0;
            o_m_axi_arlen   <= 8'd0;
            o_m_axi_arvalid <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_m_axi_arvalid <= 1'b0;
                    if (i_req_valid) begin
                        o_error <= 1'b0;
                        if ((i_req_len_bytes == 16'd0) || !request_is_aligned ||
                            (((rounded_len_bytes >> ADDR_LSB) + 1'b0) > 16'd256)) begin
                            o_error <= 1'b1;
                        end else begin
                            beats_total     <= rounded_len_bytes >> ADDR_LSB;
                            beats_accepted  <= 9'd0;
                            o_m_axi_araddr  <= i_req_addr;
                            o_m_axi_arlen   <= (rounded_len_bytes >> ADDR_LSB) - 1'b1;
                            o_m_axi_arvalid <= 1'b1;
                            state           <= S_AR;
                        end
                    end
                end

                S_AR: begin
                    if (o_m_axi_arvalid && i_m_axi_arready) begin
                        o_m_axi_arvalid <= 1'b0;
                        state           <= S_R;
                    end
                end

                S_R: begin
                    if (i_m_axi_rvalid && i_rsp_ready) begin
                        if (i_m_axi_rresp != 2'b00) begin
                            o_error <= 1'b1;
                        end
                        beats_accepted <= beats_accepted + 1'b1;
                        if (i_m_axi_rlast) begin
                            if ((beats_accepted + 1'b1) != beats_total) begin
                                o_error <= 1'b1;
                            end
                            state <= S_IDLE;
                        end
                    end
                end

                default: begin
                    state           <= S_IDLE;
                    o_m_axi_arvalid <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
