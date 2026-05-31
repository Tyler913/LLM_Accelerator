`default_nettype none

// Sequential unsigned divider helper.
//
// RMSNorm inv_rms use:
//
//   i_numerator   = 2^(RMS_FRAC + INV_RMS_FRAC) = 2^26
//   i_denominator = rms_q10
//   o_quotient    = inv_rms_q16, unsigned UQ8.16
module fixed_udiv #(
    parameter int NUMERATOR_WIDTH   = 48,
    parameter int DENOMINATOR_WIDTH = 24,
    parameter int QUOTIENT_WIDTH    = 24,
    parameter int REMAINDER_WIDTH   = DENOMINATOR_WIDTH,
    parameter int ITERATION_COUNT   = QUOTIENT_WIDTH,
    parameter int ITERATION_W       = (ITERATION_COUNT <= 1) ? 1 : $clog2(ITERATION_COUNT)
)
(
    input  logic                                  i_clk,
    input  logic                                  i_rst_n,

    // Start a new division transaction when the module is not busy.
    // Keep numerator and denominator stable until o_done is asserted.
    input  logic                                  i_start,

    input  logic [NUMERATOR_WIDTH-1 : 0]          i_numerator,
    input  logic [DENOMINATOR_WIDTH-1 : 0]        i_denominator,

    // Busy is high after start is accepted and before the quotient is ready.
    output logic                                  o_busy,

    // Done pulses for one cycle when quotient/remainder are valid.
    output logic                                  o_done,

    // High with o_done if i_denominator was zero.
    output logic                                  o_divide_by_zero,

    output logic [QUOTIENT_WIDTH-1 : 0]           o_quotient,
    output logic [REMAINDER_WIDTH-1 : 0]          o_remainder
);

    // TODO: Implement the sequential unsigned divide datapath.

endmodule

`default_nettype wire
