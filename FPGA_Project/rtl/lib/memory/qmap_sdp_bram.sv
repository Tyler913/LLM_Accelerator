`default_nettype none

// Generic one-write/one-read synchronous block RAM.
//
// Keeping the RAM in a small hierarchy prevents downstream arithmetic from
// changing Vivado's memory-port inference.  The contents are deliberately not
// reset; every caller fills the valid address range before consuming it.
(* keep_hierarchy = "yes" *)
module qmap_sdp_bram #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH      = 1024,
    parameter int ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire logic                  i_clk,

    input  wire logic                  i_wr_en,
    input  wire logic [ADDR_WIDTH-1:0] i_wr_addr,
    input  wire logic [DATA_WIDTH-1:0] i_wr_data,

    input  wire logic                  i_rd_en,
    input  wire logic [ADDR_WIDTH-1:0] i_rd_addr,
    output logic [DATA_WIDTH-1:0]      o_rd_data
);

    (* ram_style = "block" *)
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge i_clk) begin
        if (i_wr_en) begin
            mem[i_wr_addr] <= i_wr_data;
        end
    end

    always_ff @(posedge i_clk) begin
        if (i_rd_en) begin
            o_rd_data <= mem[i_rd_addr];
        end
    end

endmodule

`default_nettype wire
