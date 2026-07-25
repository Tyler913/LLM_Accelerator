`default_nettype none

// AXI4 read master for the project-local memory request/response interface.
//
// One project-local request may span multiple AXI bursts. Bursts are capped at
// 256 beats and never cross a 4 KiB AXI boundary. The project-local `last`
// signal marks only the final word of the complete request, not each AXI burst.
//
// Limits:
//   - one outstanding AXI burst
//   - aligned request addresses
//   - one full data word is returned for a partial final word
module axi4_read_master #(
    parameter int ADDR_WIDTH = 64,
    parameter int DATA_WIDTH = 32
)
(
    input  wire logic                    i_clk,
    input  wire logic                    i_rst_n,

    input  wire logic                    i_req_valid,
    output logic                         o_req_ready,
    input  wire logic [ADDR_WIDTH-1 : 0] i_req_addr,
    input  wire logic [15 : 0]           i_req_len_bytes,

    output logic                         o_rsp_valid,
    input  wire logic                    i_rsp_ready,
    output logic [DATA_WIDTH-1 : 0]      o_rsp_data,
    output logic                         o_rsp_last,

    output logic                         o_busy,
    output logic                         o_error,

    output logic [ADDR_WIDTH-1 : 0]      o_m_axi_araddr,
    output logic [7 : 0]                 o_m_axi_arlen,
    output logic [2 : 0]                 o_m_axi_arsize,
    output logic [1 : 0]                 o_m_axi_arburst,
    output logic [2 : 0]                 o_m_axi_arprot,
    output logic [3 : 0]                 o_m_axi_arcache,
    output logic                         o_m_axi_arvalid,
    input  wire logic                    i_m_axi_arready,

    input  wire logic [DATA_WIDTH-1 : 0] i_m_axi_rdata,
    input  wire logic [1 : 0]            i_m_axi_rresp,
    input  wire logic                    i_m_axi_rlast,
    input  wire logic                    i_m_axi_rvalid,
    output logic                         o_m_axi_rready
);

    localparam int DATA_BYTES = DATA_WIDTH / 8;
    localparam int ADDR_LSB   = $clog2(DATA_BYTES);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_AR,
        S_R,
        S_DRAIN,
        S_ERROR_RSP
    } state_t;

    state_t state;
    logic [ADDR_WIDTH-1 : 0] current_burst_addr;
    logic [15 : 0] beats_remaining;
    logic [8 : 0] burst_beats;
    logic [8 : 0] burst_beats_received;
    logic [15 : 0] error_beats_remaining;
    logic request_is_aligned;
    logic [16 : 0] rounded_len_bytes;
    logic [15 : 0] request_beats;
    logic expected_axi_last;
    logic [15 : 0] remaining_after_beat;
    logic [ADDR_WIDTH-1 : 0] next_burst_addr;
    logic [8 : 0] first_burst_beats;
    logic [8 : 0] next_burst_beats;

    function automatic logic [8 : 0] calculate_burst_beats(
        input logic [ADDR_WIDTH-1 : 0] address,
        input logic [15 : 0] remaining
    );
        logic [12 : 0] bytes_to_4k;
        logic [12 : 0] beats_to_4k;
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
    assign expected_axi_last  = (burst_beats_received == (burst_beats - 1'b1));
    assign remaining_after_beat = beats_remaining - 1'b1;
    assign first_burst_beats  = calculate_burst_beats(i_req_addr, request_beats);
    assign next_burst_addr    = current_burst_addr + (burst_beats * DATA_BYTES);
    assign next_burst_beats   = calculate_burst_beats(next_burst_addr, remaining_after_beat);

    assign o_req_ready     = (state == S_IDLE);
    assign o_busy          = (state != S_IDLE);
    assign o_rsp_valid     = ((state == S_R) && i_m_axi_rvalid) || (state == S_ERROR_RSP);
    assign o_rsp_data      = (state == S_ERROR_RSP) ? 'd0 : i_m_axi_rdata;
    assign o_rsp_last      = ((state == S_ERROR_RSP) && (error_beats_remaining == 16'd1)) ||
                             ((state == S_R) && i_m_axi_rvalid && (beats_remaining == 16'd1));
    assign o_m_axi_rready  = ((state == S_R) && i_rsp_ready) || (state == S_DRAIN);
    assign o_m_axi_arsize  = ADDR_LSB[2:0];
    assign o_m_axi_arburst = 2'b01;
    assign o_m_axi_arprot  = 3'b000;
    assign o_m_axi_arcache = 4'b0011;

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                <= S_IDLE;
            current_burst_addr   <= 'd0;
            beats_remaining      <= 16'd0;
            burst_beats          <= 9'd0;
            burst_beats_received <= 9'd0;
            error_beats_remaining <= 16'd0;
            o_error              <= 1'b0;
            o_m_axi_araddr       <= 'd0;
            o_m_axi_arlen        <= 8'd0;
            o_m_axi_arvalid      <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_m_axi_arvalid <= 1'b0;
                    if (i_req_valid) begin
                        o_error <= 1'b0;
                        if ((i_req_len_bytes == 16'd0) || !request_is_aligned) begin
                            // The local read interface has no separate done
                            // signal, so return the promised number of zero
                            // beats instead of leaving a fixed-count caller
                            // waiting indefinitely.
                            o_error               <= 1'b1;
                            error_beats_remaining <= (i_req_len_bytes == 16'd0) ?
                                                     16'd1 : request_beats;
                            state                 <= S_ERROR_RSP;
                        end else begin
                            current_burst_addr   <= i_req_addr;
                            beats_remaining      <= request_beats;
                            burst_beats          <= first_burst_beats;
                            burst_beats_received <= 9'd0;
                            o_m_axi_araddr       <= i_req_addr;
                            o_m_axi_arlen        <= first_burst_beats - 1'b1;
                            o_m_axi_arvalid      <= 1'b1;
                            state                <= S_AR;
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
                        beats_remaining <= remaining_after_beat;
                        if (i_m_axi_rresp != 2'b00) begin
                            o_error <= 1'b1;
                        end
                        if (i_m_axi_rlast != expected_axi_last) begin
                            o_error <= 1'b1;
                        end

                        if (i_m_axi_rlast && !expected_axi_last) begin
                            // An AXI slave ended the burst early. The missing
                            // data cannot be reconstructed, so zero-fill the
                            // exact remainder promised to the local caller.
                            error_beats_remaining <= remaining_after_beat;
                            state                 <= S_ERROR_RSP;
                        end else if (expected_axi_last) begin
                            burst_beats_received <= 9'd0;
                            if (!i_m_axi_rlast) begin
                                // Do not mistake an extra AXI beat for data
                                // from the next project-local burst. Drain to
                                // the real RLAST and retain the sticky error.
                                state <= S_DRAIN;
                            end else if (beats_remaining == 16'd1) begin
                                state <= S_IDLE;
                            end else begin
                                current_burst_addr <= next_burst_addr;
                                burst_beats        <= next_burst_beats;
                                o_m_axi_araddr     <= next_burst_addr;
                                o_m_axi_arlen      <= next_burst_beats - 1'b1;
                                o_m_axi_arvalid    <= 1'b1;
                                state              <= S_AR;
                            end
                        end else begin
                            burst_beats_received <= burst_beats_received + 1'b1;
                        end
                    end
                end

                S_DRAIN: begin
                    if (i_m_axi_rvalid) begin
                        if (i_m_axi_rresp != 2'b00) begin
                            o_error <= 1'b1;
                        end
                        if (i_m_axi_rlast) begin
                            if (beats_remaining == 16'd0) begin
                                state <= S_IDLE;
                            end else begin
                                current_burst_addr <= next_burst_addr;
                                burst_beats        <= calculate_burst_beats(next_burst_addr, beats_remaining);
                                o_m_axi_araddr     <= next_burst_addr;
                                o_m_axi_arlen      <= calculate_burst_beats(next_burst_addr, beats_remaining) - 1'b1;
                                o_m_axi_arvalid    <= 1'b1;
                                state              <= S_AR;
                            end
                        end
                    end
                end

                S_ERROR_RSP: begin
                    if (i_rsp_ready) begin
                        if (error_beats_remaining == 16'd1) begin
                            error_beats_remaining <= 16'd0;
                            state                 <= S_IDLE;
                        end else begin
                            error_beats_remaining <= error_beats_remaining - 1'b1;
                        end
                    end
                end

                default: begin
                    state           <= S_IDLE;
                    o_m_axi_arvalid <= 1'b0;
                    o_error         <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
