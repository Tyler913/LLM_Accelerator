`default_nettype none

// RMSNorm apply stage for one 1024-wide hidden vector.
//
// Math contract:
//
//   y[i] = x[i] * inv_rms * gamma[i]
//
// Default fixed-point formats:
//
//   x[i]:    signed 24-bit Q14.10
//   inv_rms: unsigned 24-bit UQ8.16
//   gamma[i]: unsigned 16-bit UQ8.8
//   y[i]:    signed 24-bit Q12.12
//
// With these defaults, the raw product has IN_FRAC + INV_RMS_FRAC +
// GAMMA_FRAC fractional bits. OUTPUT_SHIFT converts it to OUT_FRAC.
module rmsnorm_apply_1024 #(
    parameter int INPUT_SIZE       = 1024,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 10,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 8,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int PRODUCT1_WIDTH   = IN_WIDTH + INV_RMS_WIDTH,
    parameter int PRODUCT2_WIDTH   = PRODUCT1_WIDTH + GAMMA_WIDTH,
    parameter int OUTPUT_SHIFT     = IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC,
    parameter int ELEMENT_INDEX_W  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE)
)
(
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    // Start a new apply transaction when the module is not busy.
    // Keep inputs stable until o_done is asserted.
    input  logic                                      i_start,

    // INPUT_SIZE signed fixed-point values, flattened little-element-endian.
    input  logic [INPUT_SIZE*IN_WIDTH-1 : 0]          i_input_flat,

    // INPUT_SIZE unsigned gamma values, flattened little-element-endian.
    input  logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]       i_gamma_flat,

    // Shared unsigned inv_rms for this vector.
    input  logic [INV_RMS_WIDTH-1 : 0]                i_inv_rms,

    output logic                                      o_busy,
    output logic                                      o_done,

    // High with o_done if any output element saturated.
    output logic                                      o_saturation,

    // INPUT_SIZE signed fixed-point outputs, flattened little-element-endian.
    output logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         o_output_flat
);

    // TODO: Implement the serial multiply-scale-saturate datapath.

endmodule

`default_nettype wire
