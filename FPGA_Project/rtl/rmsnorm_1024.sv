`default_nettype none

// RMSNorm top-level controller for one 1024-wide hidden vector.
//
// Math contract:
//
//   sum_squares = sum_i x[i]^2
//   mean_square = sum_squares / 1024
//   inv_rms     = 1 / sqrt(mean_square + epsilon)
//   y[i]        = x[i] * inv_rms * gamma[i]
//
// Default fixed-point formats:
//
//   input x[i]: signed 24-bit Q14.10
//   gamma[i]:   unsigned 16-bit UQ8.8
//   inv_rms:    unsigned 24-bit UQ8.16
//   output y[i]: signed 24-bit Q12.12
module rmsnorm_1024 #(
    parameter int INPUT_SIZE       = 1024,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 10,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 8,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int SUM_WIDTH        = 64,
    parameter int SUM_FRAC         = 2 * IN_FRAC,
    parameter int MEAN_SHIFT       = $clog2(INPUT_SIZE),
    parameter int RMS_WIDTH        = IN_WIDTH,
    parameter int RMS_FRAC         = IN_FRAC,
    parameter int DIV_NUM_WIDTH    = 48,
    parameter int DIV_NUM_SHIFT    = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q20          = 1
)
(
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    // Start a new RMSNorm transaction when the module is not busy.
    // Keep input and gamma buses stable until o_done is asserted.
    input  logic                                      i_start,

    // INPUT_SIZE signed fixed-point values, flattened little-element-endian.
    input  logic [INPUT_SIZE*IN_WIDTH-1 : 0]          i_input_flat,

    // INPUT_SIZE unsigned gamma values, flattened little-element-endian.
    input  logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]       i_gamma_flat,

    output logic                                      o_busy,
    output logic                                      o_done,

    // High with o_done if any output element saturated.
    output logic                                      o_saturation,

    // INPUT_SIZE signed fixed-point outputs, flattened little-element-endian.
    output logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         o_output_flat,

    // Debug/validation outputs for early RTL bring-up.
    output logic [SUM_WIDTH-1 : 0]                    o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                    o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]                o_inv_rms
);

    // TODO: Implement the RMSNorm controller and instantiate the submodules.

endmodule

`default_nettype wire
