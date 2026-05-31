`default_nettype none

// Fixed-point unsigned square-root helper.
//
// Default RMSNorm use:
//
//   i_radicand: mean_square + epsilon, unsigned Q28.20-style value
//   o_root:     RMS value, unsigned 24-bit Q14.10
//
// The default fractional relationship is IN_FRAC = 2*OUT_FRAC.
module fixed_sqrt_u64 #(
    parameter int IN_WIDTH        = 64,
    parameter int IN_FRAC         = 20,
    parameter int OUT_WIDTH       = 24,
    parameter int OUT_FRAC        = 10,
    parameter int ITERATION_COUNT = OUT_WIDTH,
    parameter int ITERATION_W     = (ITERATION_COUNT <= 1) ? 1 : $clog2(ITERATION_COUNT)
)
(
    input  logic                         i_clk,
    input  logic                         i_rst_n,

    // Start a new square-root transaction when the module is not busy.
    // Keep i_radicand stable until o_done is asserted.
    input  logic                         i_start,

    // Unsigned fixed-point radicand.
    input  logic [IN_WIDTH-1 : 0]        i_radicand,

    // Busy is high after start is accepted and before the root is ready.
    output logic                         o_busy,

    // Done pulses for one cycle when o_root is valid.
    output logic                         o_done,

    // Unsigned fixed-point square-root result.
    output logic [OUT_WIDTH-1 : 0]       o_root
);

    // TODO: Implement a restoring or non-restoring square-root datapath.

endmodule

`default_nettype wire
