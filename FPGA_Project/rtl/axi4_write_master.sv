`default_nettype none

// Minimal AXI4 write master for the project-local memory write stream.
//
// One project-local request may span multiple AXI bursts. Bursts are capped at
// 256 beats and never cross a 4 KiB AXI boundary.
//
// Limits:
//   - one outstanding AXI burst
//   - aligned requests
//   - full-word writes only
module axi4_write_master #(
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

    input  wire logic [DATA_WIDTH-1 : 0] i_wdata,
    input  wire logic                    i_wdata_valid,
    output logic                    o_wdata_ready,
    input  wire logic                    i_wdata_last,

    output logic                    o_done,
    output logic                    o_busy,
    output logic                    o_error,

    output logic [ADDR_WIDTH-1 : 0] o_m_axi_awaddr,
    output logic [7 : 0]            o_m_axi_awlen,
    output logic [2 : 0]            o_m_axi_awsize,
    output logic [1 : 0]            o_m_axi_awburst,
    output logic [2 : 0]            o_m_axi_awprot,
    output logic [3 : 0]            o_m_axi_awcache,
    output logic                    o_m_axi_awvalid,
    input  wire logic                    i_m_axi_awready,

    output logic [DATA_WIDTH-1 : 0] o_m_axi_wdata,
    output logic [DATA_WIDTH/8-1 : 0] o_m_axi_wstrb,
    output logic                    o_m_axi_wlast,
    output logic                    o_m_axi_wvalid,
    input  wire logic                    i_m_axi_wready,

    input  wire logic [1 : 0]       i_m_axi_bresp,
    input  wire logic                    i_m_axi_bvalid,
    output logic                    o_m_axi_bready
);

    localparam int DATA_BYTES = DATA_WIDTH / 8;
    localparam int ADDR_LSB   = $clog2(DATA_BYTES);

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_AW,
        S_W,
        S_B
    } state_t;

    state_t state;
    logic [ADDR_WIDTH-1 : 0] current_burst_addr;
    logic [15 : 0] beats_remaining;
    logic [8 : 0] burst_beats;
    logic [8 : 0] burst_beats_sent;
    logic request_is_aligned;
    logic [16 : 0] rounded_len_bytes;
    logic [15 : 0] request_beats;
    logic expected_axi_last;
    logic expected_request_last;
    logic [ADDR_WIDTH-1 : 0] next_burst_addr;
    logic [8 : 0] first_burst_beats;
    logic [8 : 0] next_burst_beats;

    function automatic logic [8 : 0] calculate_burst_beats(
        input logic [ADDR_WIDTH-1 : 0] address,
        input logic [15 : 0] remaining
    );
        logic [12 : 0] bytes_to_4k;
        logic [10 : 0] beats_to_4k;
        logic [15 : 0] limited_beats;
        begin
            bytes_to_4k = 13'd4096 - {1'b0, address[11 : 0]};
            beats_to_4k = bytes_to_4k >> ADDR_LSB;
            limited_beats = remaining;
            if (limited_beats > 16'd256) begin
                limited_beats = 16'd256;
            end
            if (limited_beats > beats_to_4k) begin
                limited_beats = beats_to_4k;
            end
            calculate_burst_beats = limited_beats[8 : 0];
        end
    endfunction

    assign request_is_aligned = (i_req_addr[ADDR_LSB-1 : 0] == 'd0);
    assign rounded_len_bytes  = {1'b0, i_req_len_bytes} + DATA_BYTES - 1;
    assign request_beats      = rounded_len_bytes >> ADDR_LSB;
    assign expected_axi_last  = (burst_beats_sent == (burst_beats - 1'b1));
    assign expected_request_last = (beats_remaining == 16'd1);
    assign first_burst_beats  = calculate_burst_beats(i_req_addr, request_beats);
    assign next_burst_addr    = current_burst_addr + (burst_beats * DATA_BYTES);
    assign next_burst_beats   = calculate_burst_beats(next_burst_addr, beats_remaining);

    assign o_req_ready      = (state == S_IDLE);
    assign o_busy           = (state != S_IDLE);
    assign o_wdata_ready    = (state == S_W) && i_m_axi_wready;
    assign o_m_axi_wvalid   = (state == S_W) && i_wdata_valid;
    assign o_m_axi_wdata    = i_wdata;
    assign o_m_axi_wstrb    = {DATA_BYTES{1'b1}};
    assign o_m_axi_wlast    = expected_axi_last;
    assign o_m_axi_bready   = (state == S_B);
    assign o_m_axi_awsize   = ADDR_LSB[2:0];
    assign o_m_axi_awburst  = 2'b01;
    assign o_m_axi_awprot   = 3'b000;
    assign o_m_axi_awcache  = 4'b0011;

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state           <= S_IDLE;
            current_burst_addr <= 'd0;
            beats_remaining <= 16'd0;
            burst_beats     <= 9'd0;
            burst_beats_sent <= 9'd0;
            o_done          <= 1'b0;
            o_error         <= 1'b0;
            o_m_axi_awaddr  <= 'd0;
            o_m_axi_awlen   <= 8'd0;
            o_m_axi_awvalid <= 1'b0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    o_m_axi_awvalid <= 1'b0;
                    if (i_req_valid) begin
                        o_error <= 1'b0;
                        if ((i_req_len_bytes == 16'd0) || !request_is_aligned) begin
                            o_error <= 1'b1;
                            o_done  <= 1'b1;
                        end else begin
                            current_burst_addr <= i_req_addr;
                            beats_remaining <= request_beats;
                            burst_beats     <= first_burst_beats;
                            burst_beats_sent <= 9'd0;
                            o_m_axi_awaddr  <= i_req_addr;
                            o_m_axi_awlen   <= first_burst_beats - 1'b1;
                            o_m_axi_awvalid <= 1'b1;
                            state           <= S_AW;
                        end
                    end
                end

                S_AW: begin
                    if (o_m_axi_awvalid && i_m_axi_awready) begin
                        o_m_axi_awvalid <= 1'b0;
                        state           <= S_W;
                    end
                end

                S_W: begin
                    if (i_wdata_valid && i_m_axi_wready) begin
                        beats_remaining <= beats_remaining - 1'b1;
                        if (i_wdata_last != expected_request_last) begin
                            o_error <= 1'b1;
                        end

                        if (expected_axi_last) begin
                            state <= S_B;
                        end else begin
                            burst_beats_sent <= burst_beats_sent + 1'b1;
                        end
                    end
                end

                S_B: begin
                    if (i_m_axi_bvalid) begin
                        if (i_m_axi_bresp != 2'b00) begin
                            o_error <= 1'b1;
                            o_done  <= 1'b1;
                            state   <= S_IDLE;
                        end else if (beats_remaining == 16'd0) begin
                            o_done <= 1'b1;
                            state  <= S_IDLE;
                        end else begin
                            current_burst_addr <= next_burst_addr;
                            burst_beats        <= next_burst_beats;
                            burst_beats_sent   <= 9'd0;
                            o_m_axi_awaddr     <= next_burst_addr;
                            o_m_axi_awlen      <= next_burst_beats - 1'b1;
                            o_m_axi_awvalid    <= 1'b1;
                            state              <= S_AW;
                        end
                    end
                end

                default: begin
                    state           <= S_IDLE;
                    o_m_axi_awvalid <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
