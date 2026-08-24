`default_nettype none

// Sequential, BRAM-backed RMSNorm engine.
//
// This module implements the same fixed-point contract as rmsnorm_1024, but
// keeps the input vector, gamma vector, and normalized output in inferred
// block RAMs.  The native write ports are intended to be driven directly by a
// descriptor-backed memory loader.  The output has both a synchronous read
// port and a write stream so that wrappers can either burst the completed
// vector later or consume each result as it is produced.
module rmsnorm_bram #(
    parameter int INPUT_SIZE        = 1024,
    parameter int IN_WIDTH          = 24,
    parameter int IN_FRAC           = 10,
    parameter int GAMMA_WIDTH       = 16,
    parameter int GAMMA_FRAC        = 8,
    parameter int GAMMA_SIGNED      = 0,
    parameter int INV_RMS_WIDTH     = 24,
    parameter int INV_RMS_FRAC      = 16,
    parameter int OUT_WIDTH         = 24,
    parameter int OUT_FRAC          = 12,
    parameter int SUM_WIDTH         = 64,
    parameter int SUM_FRAC          = 2 * IN_FRAC,
    parameter int MEAN_SHIFT        = $clog2(INPUT_SIZE),
    parameter int RMS_WIDTH         = IN_WIDTH,
    parameter int RMS_FRAC          = IN_FRAC,
    parameter int DIV_NUM_WIDTH     = 48,
    parameter int DIV_NUM_SHIFT     = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q20           = 1,
    parameter int ELEMENT_INDEX_W   = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE)
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_input_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_input_wr_addr,
    input  wire logic [IN_WIDTH-1 : 0]             i_input_wr_data,

    input  wire logic                              i_gamma_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_gamma_wr_addr,
    input  wire logic [GAMMA_WIDTH-1 : 0]          i_gamma_wr_data,

    input  wire logic                              i_start,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic                                   o_saturation,

    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_output_rd_addr,
    output logic [OUT_WIDTH-1 : 0]                 o_output_rd_data,

    output logic                                   o_output_wr_valid,
    output logic [ELEMENT_INDEX_W-1 : 0]           o_output_wr_addr,
    output logic [OUT_WIDTH-1 : 0]                 o_output_wr_data,

    output logic [SUM_WIDTH-1 : 0]                 o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                 o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]             o_inv_rms
);

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_SUM_PRIME,
        S_SUM_RUN,
        S_SUM_FINISH,
        S_SQRT_START,
        S_SQRT_WAIT,
        S_DIV_START,
        S_DIV_WAIT,
        S_APPLY_PRIME,
        S_APPLY_RUN,
        S_DONE
    } state_t;

    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX =
        INPUT_SIZE - 1;
    localparam int SQUARE_WIDTH = 2 * IN_WIDTH;
    localparam int INV_RMS_SIGNED_WIDTH = INV_RMS_WIDTH + 1;
    localparam int GAMMA_SIGNED_WIDTH = GAMMA_WIDTH + 1;
    localparam int PRODUCT1_SIGNED_WIDTH =
        IN_WIDTH + INV_RMS_SIGNED_WIDTH;
    localparam int PRODUCT2_SIGNED_WIDTH =
        PRODUCT1_SIGNED_WIDTH + GAMMA_SIGNED_WIDTH;
    localparam int OUTPUT_SHIFT =
        IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC;
    localparam logic [DIV_NUM_WIDTH-1 : 0] DIV_NUMERATOR_VALUE =
        ({{(DIV_NUM_WIDTH-1){1'b0}}, 1'b1} << DIV_NUM_SHIFT);
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {OUT_WIDTH-1{1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {OUT_WIDTH-1{1'b0}}};

    state_t state;
    logic [ELEMENT_INDEX_W-1 : 0] element_index;

    (* ram_style = "block" *)
    logic [GAMMA_WIDTH-1 : 0] gamma_mem [0 : INPUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0] output_mem [0 : INPUT_SIZE-1];

    logic [ELEMENT_INDEX_W-1 : 0] input_internal_rd_addr;
    logic [ELEMENT_INDEX_W-1 : 0] gamma_internal_rd_addr;
    logic signed [IN_WIDTH-1 : 0] input_internal_rd_data;
    logic [IN_WIDTH-1 : 0] input_internal_rd_bits;
    logic [GAMMA_WIDTH-1 : 0] gamma_internal_rd_data;

    logic [SUM_WIDTH-1 : 0] sum_squares_reg;
    logic signed [SQUARE_WIDTH-1 : 0] current_square_signed;
    logic [SQUARE_WIDTH-1 : 0] current_square;
    logic [SUM_WIDTH-1 : 0] current_square_extended;

    logic sqrt_start;
    logic sqrt_done;
    logic [SUM_WIDTH-1 : 0] sqrt_radicand;
    logic [RMS_WIDTH-1 : 0] rms_value;

    logic div_start;
    logic div_done;
    logic div_by_zero;
    logic [RMS_WIDTH-1 : 0] div_denominator;
    logic [INV_RMS_WIDTH-1 : 0] div_quotient;
    logic [RMS_WIDTH-1 : 0] div_remainder;

    logic signed [IN_WIDTH-1 : 0] current_input;
    logic signed [INV_RMS_SIGNED_WIDTH-1 : 0] current_inv_rms;
    logic signed [GAMMA_SIGNED_WIDTH-1 : 0] current_gamma;
    logic signed [PRODUCT1_SIGNED_WIDTH-1 : 0]
        product_input_inv_rms;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0] product_full;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0] product_shifted;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0] out_max_extended;
    logic signed [PRODUCT2_SIGNED_WIDTH-1 : 0] out_min_extended;
    logic signed [OUT_WIDTH-1 : 0] saturated_output;
    logic current_saturates;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);
    assign sqrt_start = (state == S_SQRT_START);
    assign div_start = (state == S_DIV_START);
    assign o_output_wr_valid = (state == S_APPLY_RUN);
    assign o_output_wr_addr = element_index;
    assign o_output_wr_data = saturated_output;

    always_comb begin
        input_internal_rd_addr = '0;
        gamma_internal_rd_addr = '0;

        case (state)
            S_SUM_PRIME: begin
                input_internal_rd_addr = '0;
            end

            S_SUM_RUN: begin
                if (element_index == LAST_ELEMENT_INDEX) begin
                    input_internal_rd_addr = element_index;
                end
                else begin
                    input_internal_rd_addr = element_index + 1'b1;
                end
            end

            S_APPLY_PRIME: begin
                input_internal_rd_addr = '0;
                gamma_internal_rd_addr = '0;
            end

            S_APPLY_RUN: begin
                if (element_index == LAST_ELEMENT_INDEX) begin
                    input_internal_rd_addr = element_index;
                    gamma_internal_rd_addr = element_index;
                end
                else begin
                    input_internal_rd_addr = element_index + 1'b1;
                    gamma_internal_rd_addr = element_index + 1'b1;
                end
            end

            default: begin
                input_internal_rd_addr = '0;
                gamma_internal_rd_addr = '0;
            end
        endcase
    end

    // No reset is intentionally applied to any vector RAM.  This preserves
    // block-RAM inference and is safe because wrappers load all elements before
    // asserting i_start.
    // Isolate the externally-written input store from the two downstream
    // arithmetic passes.  Without this boundary Vivado 2025.1 turns each
    // 1024x24 input buffer into 448 LUTRAMs despite the block-RAM attribute.
    qmap_sdp_bram #(
        .DATA_WIDTH(IN_WIDTH),
        .DEPTH(INPUT_SIZE),
        .ADDR_WIDTH(ELEMENT_INDEX_W)
    ) input_store (
        .i_clk(i_clk),
        .i_wr_en(i_input_wr_en),
        .i_wr_addr(i_input_wr_addr),
        .i_wr_data(i_input_wr_data),
        .i_rd_en(1'b1),
        .i_rd_addr(input_internal_rd_addr),
        .o_rd_data(input_internal_rd_bits)
    );

    always_comb begin
        input_internal_rd_data = $signed(input_internal_rd_bits);
    end

    always_ff @(posedge i_clk) begin
        if (i_gamma_wr_en) begin
            gamma_mem[i_gamma_wr_addr] <= i_gamma_wr_data;
        end
    end

    always_ff @(posedge i_clk) begin
        gamma_internal_rd_data <= gamma_mem[gamma_internal_rd_addr];
    end

    always_ff @(posedge i_clk) begin
        if (state == S_APPLY_RUN) begin
            output_mem[element_index] <= saturated_output;
        end
    end

    always_ff @(posedge i_clk) begin
        o_output_rd_data <= output_mem[i_output_rd_addr];
    end

    always_comb begin
        current_square_signed =
            input_internal_rd_data * input_internal_rd_data;
        current_square = current_square_signed;
        current_square_extended =
            {{(SUM_WIDTH-SQUARE_WIDTH){1'b0}}, current_square};

        current_input = input_internal_rd_data;
        current_inv_rms = $signed({1'b0, o_inv_rms});
        if (GAMMA_SIGNED != 0) begin
            current_gamma =
                $signed({
                    gamma_internal_rd_data[GAMMA_WIDTH-1],
                    gamma_internal_rd_data
                });
        end
        else begin
            current_gamma =
                $signed({1'b0, gamma_internal_rd_data});
        end

        product_input_inv_rms = current_input * current_inv_rms;
        product_full = product_input_inv_rms * current_gamma;
        product_shifted = product_full >>> OUTPUT_SHIFT;
        out_max_extended =
            {{(PRODUCT2_SIGNED_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}},
             OUT_MAX};
        out_min_extended =
            {{(PRODUCT2_SIGNED_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}},
             OUT_MIN};

        if (product_shifted > out_max_extended) begin
            saturated_output = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (product_shifted < out_min_extended) begin
            saturated_output = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_output = product_shifted[OUT_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    fixed_sqrt_u64 #(
        .IN_WIDTH(SUM_WIDTH),
        .IN_FRAC(SUM_FRAC),
        .OUT_WIDTH(RMS_WIDTH),
        .OUT_FRAC(RMS_FRAC),
        .ITERATION_COUNT(RMS_WIDTH),
        .ITERATION_W((RMS_WIDTH <= 1) ? 1 : $clog2(RMS_WIDTH))
    ) inst_fixed_sqrt_u64 (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(sqrt_start),
        .i_radicand(sqrt_radicand),
        .o_busy(),
        .o_done(sqrt_done),
        .o_root(rms_value)
    );

    fixed_udiv #(
        .NUMERATOR_WIDTH(DIV_NUM_WIDTH),
        .DENOMINATOR_WIDTH(RMS_WIDTH),
        .QUOTIENT_WIDTH(INV_RMS_WIDTH),
        .REMAINDER_WIDTH(RMS_WIDTH),
        .ITERATION_COUNT(INV_RMS_WIDTH),
        .ITERATION_W((INV_RMS_WIDTH <= 1) ? 1 : $clog2(INV_RMS_WIDTH))
    ) inst_fixed_udiv (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(div_start),
        .i_numerator(DIV_NUMERATOR_VALUE),
        .i_denominator(div_denominator),
        .o_busy(),
        .o_done(div_done),
        .o_divide_by_zero(div_by_zero),
        .o_quotient(div_quotient),
        .o_remainder(div_remainder)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            element_index <= '0;
            sum_squares_reg <= '0;
            sqrt_radicand <= '0;
            div_denominator <= '0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_sum_squares <= '0;
            o_mean_square <= '0;
            o_inv_rms <= '0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_SUM_PRIME;
                        element_index <= '0;
                        sum_squares_reg <= '0;
                        sqrt_radicand <= '0;
                        div_denominator <= '0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_sum_squares <= '0;
                        o_mean_square <= '0;
                        o_inv_rms <= '0;
                    end
                end

                S_SUM_PRIME: begin
                    element_index <= '0;
                    state <= S_SUM_RUN;
                end

                S_SUM_RUN: begin
                    sum_squares_reg <=
                        sum_squares_reg + current_square_extended;
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= '0;
                        state <= S_SUM_FINISH;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                S_SUM_FINISH: begin
                    o_sum_squares <= sum_squares_reg;
                    o_mean_square <= sum_squares_reg >> MEAN_SHIFT;
                    sqrt_radicand <=
                        (sum_squares_reg >> MEAN_SHIFT) + EPS_Q20;
                    state <= S_SQRT_START;
                end

                S_SQRT_START: begin
                    state <= S_SQRT_WAIT;
                end

                S_SQRT_WAIT: begin
                    if (sqrt_done) begin
                        div_denominator <= rms_value;
                        state <= S_DIV_START;
                    end
                end

                S_DIV_START: begin
                    state <= S_DIV_WAIT;
                end

                S_DIV_WAIT: begin
                    if (div_done) begin
                        o_inv_rms <= div_quotient;
                        if (div_by_zero) begin
                            o_error <= 1'b1;
                        end
                        element_index <= '0;
                        state <= S_APPLY_PRIME;
                    end
                end

                S_APPLY_PRIME: begin
                    element_index <= '0;
                    state <= S_APPLY_RUN;
                end

                S_APPLY_RUN: begin
                    o_saturation <=
                        o_saturation | current_saturates;
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= '0;
                        state <= S_DONE;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                    o_error <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
