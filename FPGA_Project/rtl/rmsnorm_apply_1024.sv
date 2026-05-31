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

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX =
        INPUT_SIZE - 1;

    localparam int INV_RMS_SIGNED_WIDTH = INV_RMS_WIDTH + 1;
    localparam int GAMMA_SIGNED_WIDTH   = GAMMA_WIDTH + 1;
    localparam int PRODUCT1_SIGNED_WIDTH = IN_WIDTH + INV_RMS_SIGNED_WIDTH;
    localparam int PRODUCT2_SIGNED_WIDTH = PRODUCT1_SIGNED_WIDTH + GAMMA_SIGNED_WIDTH;

    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {OUT_WIDTH-1{1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {OUT_WIDTH-1{1'b0}}};

    logic [1 : 0]                                current_state;
    logic [1 : 0]                                next_state;
    logic [ELEMENT_INDEX_W-1 : 0]                element_index;
    logic                                        saturation_reg;

    logic signed [IN_WIDTH-1 : 0]                current_input;
    logic signed [INV_RMS_SIGNED_WIDTH-1 : 0]    current_inv_rms;
    logic signed [GAMMA_SIGNED_WIDTH-1 : 0]      current_gamma;
    logic signed [PRODUCT1_SIGNED_WIDTH-1 : 0]   product_input_inv_rms;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0]   product_full;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0]   product_shifted;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0]   out_max_extended;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0]   out_min_extended;
    logic signed [OUT_WIDTH-1 : 0]               saturated_output;
    logic                                        current_saturates;

    assign o_busy       = (current_state == RUN);
    assign o_done       = (current_state == DONE);
    assign o_saturation = saturation_reg;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = RUN;
                end
            end

            RUN: begin
                if (element_index == LAST_ELEMENT_INDEX) begin
                    next_state = DONE;
                end
            end

            DONE: begin
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

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
            element_index <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    element_index <= 'd0;
                end

                RUN: begin
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= 'd0;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                DONE: begin
                    element_index <= 'd0;
                end

                default: begin
                    element_index <= 'd0;
                end
            endcase
        end
    end

    always_comb begin
        current_input =
            i_input_flat[element_index*IN_WIDTH +: IN_WIDTH];
        current_inv_rms = $signed({1'b0, i_inv_rms});
        current_gamma =
            $signed({1'b0, i_gamma_flat[element_index*GAMMA_WIDTH +: GAMMA_WIDTH]});

        product_input_inv_rms = current_input * current_inv_rms;
        product_full = product_input_inv_rms * current_gamma;
        product_shifted = product_full >>> OUTPUT_SHIFT;

        out_max_extended =
            {{(PRODUCT2_SIGNED_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
        out_min_extended =
            {{(PRODUCT2_SIGNED_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

        if (product_shifted > out_max_extended) begin
            saturated_output = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (product_shifted < out_min_extended) begin
            saturated_output = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_output = product_shifted;
            current_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_output_flat <= 'd0;
            saturation_reg <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_output_flat <= 'd0;
                        saturation_reg <= 1'b0;
                    end
                    else begin
                        o_output_flat <= o_output_flat;
                        saturation_reg <= saturation_reg;
                    end
                end

                RUN: begin
                    o_output_flat[element_index*OUT_WIDTH +: OUT_WIDTH] <=
                        saturated_output;
                    saturation_reg <= saturation_reg | current_saturates;
                end

                DONE: begin
                    o_output_flat <= o_output_flat;
                    saturation_reg <= saturation_reg;
                end

                default: begin
                    o_output_flat <= 'd0;
                    saturation_reg <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
