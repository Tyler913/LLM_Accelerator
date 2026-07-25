`default_nettype none

// Layer 0 MLP activation stage:
//
//   gate[3072], up[3072] -> silu(gate) * up -> mlp_hidden[3072]
//
// Gate and up inputs use signed Q12.12. The sigmoid part of SiLU is a fixed
// UQ0.16 LUT over [-8, 8] with 1/64 spacing. The stage streams one output for
// each accepted input pair and keeps the output stable under backpressure.
module mlp_silu_mul_stage #(
    parameter int FEATURES              = 3072,
    parameter int IN_WIDTH              = 24,
    parameter int IN_FRAC               = 12,
    parameter int SIGMOID_WIDTH         = 16,
    parameter int SIGMOID_FRAC          = 16,
    parameter int SIGMOID_LUT_INDEX_FRAC = 6,
    parameter int SIGMOID_LUT_MIN_INT   = -8,
    parameter int SIGMOID_LUT_MAX_INT   = 8,
    parameter int SIGMOID_LUT_SIZE      = ((SIGMOID_LUT_MAX_INT - SIGMOID_LUT_MIN_INT) << SIGMOID_LUT_INDEX_FRAC) + 1,
    parameter int OUT_WIDTH             = 24,
    parameter int OUT_FRAC              = 12,
    parameter int ROW_INDEX_W           = (FEATURES <= 1) ? 1 : $clog2(FEATURES),
    parameter int LUT_INDEX_W           = (SIGMOID_LUT_SIZE <= 1) ? 1 : $clog2(SIGMOID_LUT_SIZE),
    parameter int SILU_PRODUCT_WIDTH    = IN_WIDTH + SIGMOID_WIDTH + 1,
    parameter int HIDDEN_PRODUCT_WIDTH  = OUT_WIDTH + IN_WIDTH
)
(
    input  wire logic                                             i_clk,
    input  wire logic                                             i_rst_n,

    input  wire logic                                             i_start,
    input  wire logic [SIGMOID_LUT_SIZE*SIGMOID_WIDTH-1 : 0]      i_sigmoid_lut_flat,

    input  wire logic                                             i_in_valid,
    output logic                                             o_in_ready,
    input  wire logic [ROW_INDEX_W-1 : 0]                         i_in_index,
    input  wire logic signed [IN_WIDTH-1 : 0]                     i_gate_data,
    input  wire logic signed [IN_WIDTH-1 : 0]                     i_up_data,
    input  wire logic                                             i_in_last,

    output logic                                             o_busy,
    output logic                                             o_done,
    output logic                                             o_error,
    output logic                                             o_saturation,

    output logic                                             o_out_valid,
    input  wire logic                                             i_out_ready,
    output logic [ROW_INDEX_W-1 : 0]                         o_out_index,
    output logic signed [OUT_WIDTH-1 : 0]                    o_hidden_data,
    output logic                                             o_out_last,

    output logic [31 : 0]                                   o_input_count,
    output logic [31 : 0]                                   o_output_count
);

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam int LUT_SHIFT = IN_FRAC - SIGMOID_LUT_INDEX_FRAC;
    localparam logic signed [IN_WIDTH-1 : 0] LUT_MIN_Q =
        SIGMOID_LUT_MIN_INT <<< IN_FRAC;
    localparam logic signed [IN_WIDTH-1 : 0] LUT_MAX_Q =
        SIGMOID_LUT_MAX_INT <<< IN_FRAC;
    localparam logic [ROW_INDEX_W-1 : 0] LAST_ROW_INDEX = FEATURES - 1;

    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX = {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN = {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam logic signed [SILU_PRODUCT_WIDTH-1 : 0] OUT_MAX_SILU_EXT =
        {{(SILU_PRODUCT_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam logic signed [SILU_PRODUCT_WIDTH-1 : 0] OUT_MIN_SILU_EXT =
        {{(SILU_PRODUCT_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};
    localparam logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] OUT_MAX_HIDDEN_EXT =
        {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] OUT_MIN_HIDDEN_EXT =
        {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

    logic [1 : 0] current_state;
    logic output_valid_reg;
    logic saw_last_input;
    logic error_reg;
    logic saturation_reg;
    logic [ROW_INDEX_W-1 : 0] out_index_reg;
    logic signed [OUT_WIDTH-1 : 0] hidden_data_reg;
    logic out_last_reg;
    logic input_fire;
    logic output_fire;

    logic [LUT_INDEX_W-1 : 0] lut_index_comb;
    logic [SIGMOID_WIDTH-1 : 0] sigmoid_value_comb;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] gate_ext_comb;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] sigmoid_ext_comb;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] silu_product_comb;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] silu_shifted_comb;
    logic signed [OUT_WIDTH-1 : 0] silu_gate_comb;
    logic silu_saturation_comb;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] silu_gate_ext_comb;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] up_ext_comb;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] hidden_product_comb;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] hidden_shifted_comb;
    logic signed [OUT_WIDTH-1 : 0] hidden_data_comb;
    logic hidden_saturation_comb;

    assign o_busy = (current_state == RUN);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_saturation = saturation_reg;
    assign o_out_valid = output_valid_reg;
    assign o_out_index = out_index_reg;
    assign o_hidden_data = hidden_data_reg;
    assign o_out_last = output_valid_reg && out_last_reg;
    assign o_in_ready =
        (current_state == RUN) &&
        (saw_last_input == 1'b0) &&
        ((output_valid_reg == 1'b0) || (i_out_ready == 1'b1));
    assign input_fire = i_in_valid && o_in_ready;
    assign output_fire = output_valid_reg && i_out_ready;

    function automatic logic [LUT_INDEX_W-1 : 0] gate_to_lut_index(
        input logic signed [IN_WIDTH-1 : 0] gate_value
    );
        logic signed [IN_WIDTH : 0] clipped;
        logic signed [IN_WIDTH : 0] shifted_from_min;
        logic signed [IN_WIDTH : 0] rounded_steps;
        begin
            if (gate_value <= LUT_MIN_Q) begin
                gate_to_lut_index = 'd0;
            end
            else if (gate_value >= LUT_MAX_Q) begin
                gate_to_lut_index = SIGMOID_LUT_SIZE - 1;
            end
            else begin
                clipped = {gate_value[IN_WIDTH-1], gate_value};
                shifted_from_min = clipped - {LUT_MIN_Q[IN_WIDTH-1], LUT_MIN_Q};
                rounded_steps = (shifted_from_min + ({{IN_WIDTH{1'b0}}, 1'b1} << (LUT_SHIFT - 1))) >>> LUT_SHIFT;
                gate_to_lut_index = rounded_steps[LUT_INDEX_W-1 : 0];
            end
        end
    endfunction

    function automatic logic signed [OUT_WIDTH-1 : 0] saturate_silu(
        input logic signed [SILU_PRODUCT_WIDTH-1 : 0] value
    );
        begin
            if (value > OUT_MAX_SILU_EXT) begin
                saturate_silu = OUT_MAX;
            end
            else if (value < OUT_MIN_SILU_EXT) begin
                saturate_silu = OUT_MIN;
            end
            else begin
                saturate_silu = value[OUT_WIDTH-1 : 0];
            end
        end
    endfunction

    function automatic logic signed [OUT_WIDTH-1 : 0] saturate_hidden(
        input logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] value
    );
        begin
            if (value > OUT_MAX_HIDDEN_EXT) begin
                saturate_hidden = OUT_MAX;
            end
            else if (value < OUT_MIN_HIDDEN_EXT) begin
                saturate_hidden = OUT_MIN;
            end
            else begin
                saturate_hidden = value[OUT_WIDTH-1 : 0];
            end
        end
    endfunction

    always @* begin
        lut_index_comb = gate_to_lut_index(i_gate_data);
        sigmoid_value_comb = i_sigmoid_lut_flat[lut_index_comb*SIGMOID_WIDTH +: SIGMOID_WIDTH];

        gate_ext_comb = {{(SILU_PRODUCT_WIDTH-IN_WIDTH){i_gate_data[IN_WIDTH-1]}}, i_gate_data};
        sigmoid_ext_comb = {{(SILU_PRODUCT_WIDTH-SIGMOID_WIDTH){1'b0}}, sigmoid_value_comb};
        silu_product_comb = gate_ext_comb * sigmoid_ext_comb;
        silu_shifted_comb = silu_product_comb >>> SIGMOID_FRAC;
        silu_gate_comb = saturate_silu(silu_shifted_comb);
        silu_saturation_comb = (silu_shifted_comb > OUT_MAX_SILU_EXT) ||
                               (silu_shifted_comb < OUT_MIN_SILU_EXT);

        silu_gate_ext_comb = {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){silu_gate_comb[OUT_WIDTH-1]}}, silu_gate_comb};
        up_ext_comb = {{(HIDDEN_PRODUCT_WIDTH-IN_WIDTH){i_up_data[IN_WIDTH-1]}}, i_up_data};
        hidden_product_comb = silu_gate_ext_comb * up_ext_comb;
        hidden_shifted_comb = hidden_product_comb >>> IN_FRAC;
        hidden_data_comb = saturate_hidden(hidden_shifted_comb);
        hidden_saturation_comb = (hidden_shifted_comb > OUT_MAX_HIDDEN_EXT) ||
                                 (hidden_shifted_comb < OUT_MIN_HIDDEN_EXT);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            current_state <= IDLE;
            output_valid_reg <= 1'b0;
            saw_last_input <= 1'b0;
            error_reg <= 1'b0;
            saturation_reg <= 1'b0;
            out_index_reg <= 'd0;
            hidden_data_reg <= 'd0;
            out_last_reg <= 1'b0;
            o_input_count <= 32'd0;
            o_output_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    output_valid_reg <= 1'b0;
                    saw_last_input <= 1'b0;
                    out_index_reg <= 'd0;
                    hidden_data_reg <= 'd0;
                    out_last_reg <= 1'b0;
                    if (i_start == 1'b1) begin
                        current_state <= RUN;
                        error_reg <= 1'b0;
                        saturation_reg <= 1'b0;
                        o_input_count <= 32'd0;
                        o_output_count <= 32'd0;
                    end
                end

                RUN: begin
                    if (output_fire == 1'b1) begin
                        o_output_count <= o_output_count + 1'b1;
                        if (out_last_reg == 1'b1) begin
                            current_state <= DONE;
                            if (o_output_count != (FEATURES - 1)) begin
                                error_reg <= 1'b1;
                            end
                        end
                    end

                    if (input_fire == 1'b1) begin
                        output_valid_reg <= 1'b1;
                        out_index_reg <= i_in_index;
                        hidden_data_reg <= hidden_data_comb;
                        out_last_reg <= i_in_last;
                        o_input_count <= o_input_count + 1'b1;
                        saw_last_input <= i_in_last;
                        saturation_reg <= saturation_reg || silu_saturation_comb || hidden_saturation_comb;

                        if (i_in_index != o_input_count[ROW_INDEX_W-1 : 0]) begin
                            error_reg <= 1'b1;
                        end
                        if (i_in_last != (o_input_count == (FEATURES - 1))) begin
                            error_reg <= 1'b1;
                        end
                    end
                    else if (output_fire == 1'b1) begin
                        output_valid_reg <= 1'b0;
                    end
                end

                DONE: begin
                    current_state <= IDLE;
                    output_valid_reg <= 1'b0;
                    saw_last_input <= 1'b0;
                end

                default: begin
                    current_state <= IDLE;
                    output_valid_reg <= 1'b0;
                    saw_last_input <= 1'b0;
                    error_reg <= 1'b1;
                    saturation_reg <= 1'b0;
                    out_index_reg <= 'd0;
                    hidden_data_reg <= 'd0;
                    out_last_reg <= 1'b0;
                    o_input_count <= 32'd0;
                    o_output_count <= 32'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
