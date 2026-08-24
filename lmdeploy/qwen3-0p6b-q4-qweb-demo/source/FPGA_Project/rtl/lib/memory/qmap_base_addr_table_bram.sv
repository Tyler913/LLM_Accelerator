`default_nettype none

// One software-programmable per-layer base-address table.
//
// Port A is shared by the software selector/readback path and table commits.
// Port B continuously follows the scheduler's active layer.  Keeping both
// reads synchronous lets Vivado map the table into a true-dual-port block RAM
// instead of building a large register bank and variable-index mux.
module qmap_base_addr_table_bram #(
    parameter int ADDR_WIDTH  = 64,
    parameter int DEPTH       = 28,
    parameter int INDEX_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire logic                    i_clk,

    input  wire logic                    i_wr_en,
    input  wire logic [INDEX_WIDTH-1:0]  i_wr_index,
    input  wire logic [ADDR_WIDTH-1:0]   i_wr_data,

    input  wire logic [INDEX_WIDTH-1:0]  i_sw_rd_index,
    output logic [ADDR_WIDTH-1:0]        o_sw_rd_data,

    input  wire logic [INDEX_WIDTH-1:0]  i_runtime_rd_index,
    output logic [ADDR_WIDTH-1:0]        o_runtime_rd_data
);

    (* ram_style = "block" *)
    logic [ADDR_WIDTH-1:0] table_mem [0:DEPTH-1];

    integer init_index;
    initial begin
        for (init_index = 0; init_index < DEPTH; init_index = init_index + 1) begin
            table_mem[init_index] = '0;
        end
    end

    // Software port: a commit occupies the port for one cycle.  On the next
    // cycle, normal selector readback resumes and observes the committed word.
    always_ff @(posedge i_clk) begin
        if (i_wr_en) begin
            table_mem[i_wr_index] <= i_wr_data;
        end
        else begin
            o_sw_rd_data <= table_mem[i_sw_rd_index];
        end
    end

    // Scheduler port: one-cycle synchronous lookup of the active layer.
    always_ff @(posedge i_clk) begin
        o_runtime_rd_data <= table_mem[i_runtime_rd_index];
    end

endmodule

`default_nettype wire
