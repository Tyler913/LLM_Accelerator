`default_nettype none

// Q4 dot-product kernel for one quantization group.
//
// This module is the first Verilog target for the Q4 v0 format documented in
// Q4_FORMAT.md. It should match:
//
//   artifacts/test_vectors/qwen3_0p6b_q4_v0/q_proj_row0_group0_dot64.npz
//
// Math contract:
//
//   partial_sum = sum_j activation[j] * weight_q4[j], for j = 0..63
//   scaled_sum_q26 = partial_sum * scale_q2_14
//   float_value = scaled_sum_q26 / 2^(ACT_FRAC + SCALE_FRAC)
//
// The default ACT_WIDTH=24 matches the planned signed Q12.12 RMSNorm output
// path. The width parameters are derived so the same datapath can also be
// instantiated with ACT_WIDTH=16 for the original Q4.12 bring-up vectors.
//
// Packing contract:
//
//   activation_flat_i[ACT_WIDTH*j +: ACT_WIDTH] holds activation[j].
//   weight_packed_i[8*b +: 8] holds two Q4 weights:
//     low  nibble [3:0] = weight_q4[2*b]
//     high nibble [7:4] = weight_q4[2*b + 1]
//   Q4 weights are signed 4-bit two's-complement values in [-8, 7].
//
// Implementation is intentionally left for hand-written RTL practice.
module q4_dot_product_64 # (
    parameter int GROUP_SIZE    = 64,
    parameter int ACT_WIDTH     = 24,
    parameter int ACT_FRAC      = 12,
    parameter int WEIGHT_WIDTH  = 4,
    parameter int SCALE_WIDTH   = 16,
    parameter int SCALE_FRAC    = 14,
    parameter int PRODUCT_WIDTH = ACT_WIDTH + WEIGHT_WIDTH,
    parameter int PARTIAL_WIDTH = PRODUCT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH
)
(
    input  logic                                         i_clk,
    input  logic                                         i_rst_n,

    // Start a new dot-product transaction when the module is not busy.
    // Keep activation_flat_i, weight_packed_i, and scale_q2_14_i stable until
    // done_o is asserted.
    input  logic                                         i_start,

    // GROUP_SIZE signed fixed-point activations, flattened little-element-endian:
    // element j is activation_flat_i[ACT_WIDTH*j +: ACT_WIDTH].
    input  logic        [GROUP_SIZE*ACT_WIDTH-1 : 0]     i_activation_flat,

    // 64 signed int4 Q4 weights packed into 32 bytes.
    input  logic        [GROUP_SIZE*WEIGHT_WIDTH-1 : 0]  i_weight_packed,

    // Unsigned Q2.14 scale for this 64-weight group.
    input  logic        [SCALE_WIDTH-1 : 0]              i_scale_q2_14,

    // Busy is high after start is accepted and before the result is ready.
    output logic                                         o_busy,

    // Done should pulse for one cycle when partial_sum_o and scaled_sum_q26_o
    // are valid.
    output logic                                         o_done,

    // Raw integer sum before applying the Q2.14 scale.
    output logic signed [PARTIAL_WIDTH-1 : 0]            o_partial_sum,

    // Scaled integer result. The binary point is after ACT_FRAC + SCALE_FRAC
    // fractional bits, so the current Q4 v0 denominator is 2^26.
    output logic signed [SCALED_WIDTH-1 : 0]             o_scaled_sum_q26
);

    // TODO: Implement the sequential datapath.
    //
    // Suggested structure:
    // 1. IDLE: wait for start_i while busy_o is low.
    // 2. RUN: for index 0..63:
    //      - select the correct byte and nibble from weight_packed_i
    //      - sign-extend the 4-bit Q4 weight to WEIGHT_WIDTH or wider
    //      - sign-extend activation[j]
    //      - multiply activation[j] * weight_q4[j]
    //      - accumulate into partial_sum_o-width storage
    // 3. SCALE: multiply partial_sum by scale_q2_14_i.
    // 4. DONE: drive outputs and pulse done_o for one cycle.
    //
    // Expected first smoke-vector values from q_proj_row0_group0_dot64.npz:
    //   partial_sum_int64    = 24751
    //   scaled_sum_q26_int64 = 3019622

    localparam IDLE  = 2'd0;
    localparam RUN   = 2'd1;
    localparam SCALE = 2'd2;
    localparam DONE  = 2'd3;

    reg [1:0] current_state;
    reg [1:0] next_state;

    reg [5:0] index;


    assign o_busy = (current_state == RUN) || (current_state == SCALE);
    assign o_done = current_state == DONE;


    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            current_state <= IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end
    
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            index <= 6'd0;
        end
        else if (current_state == IDLE) begin
            index <= 6'd0;
        end
        else if (current_state == RUN) begin
            index <= index + 6'd1;
        end
        else if (current_state == SCALE) begin
            index <= 6'd0;
        end
        else if (current_state == DONE) begin
            index <= 6'd0;
        end
    end

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = RUN;
                end
            end

            RUN: begin
                if (index == 6'd63) begin
                    next_state = SCALE;
                end
            end

            SCALE: begin
                next_state = DONE;
            end

            DONE: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end


    logic signed [ACT_WIDTH-1 : 0]     current_activation;
    logic signed [WEIGHT_WIDTH-1 : 0]  current_weight_q4;
    logic signed [PRODUCT_WIDTH-1 : 0] current_product;
    logic signed [SCALE_WIDTH : 0]     signed_scale_q2_14;


    always_comb begin
        current_activation = i_activation_flat[index*ACT_WIDTH +: ACT_WIDTH];
        current_weight_q4  = i_weight_packed[index*WEIGHT_WIDTH +: WEIGHT_WIDTH];
        current_product    = current_activation * current_weight_q4;
        signed_scale_q2_14 = $signed({1'b0, i_scale_q2_14});
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_partial_sum    <= 'd0;
            o_scaled_sum_q26 <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    o_partial_sum    <= 'd0;
                    o_scaled_sum_q26 <= 'd0;
                end
                RUN: begin
                    o_partial_sum    <= o_partial_sum + current_product;
                    o_scaled_sum_q26 <= 'd0;
                end
                SCALE: begin
                    o_scaled_sum_q26 <= o_partial_sum * signed_scale_q2_14;
                end
                DONE: begin
                    o_partial_sum    <= o_partial_sum;
                    o_scaled_sum_q26 <= o_scaled_sum_q26;
                end
                default: begin
                    o_partial_sum    <= 'd0;
                    o_scaled_sum_q26 <= 'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
