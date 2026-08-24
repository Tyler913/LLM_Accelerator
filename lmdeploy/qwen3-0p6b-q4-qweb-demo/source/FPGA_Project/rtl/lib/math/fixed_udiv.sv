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
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,

    // Start a new division transaction when the module is not busy.
    // Keep numerator and denominator stable until o_done is asserted.
    input  wire logic                                  i_start,

    input  wire logic [NUMERATOR_WIDTH-1 : 0]          i_numerator,
    input  wire logic [DENOMINATOR_WIDTH-1 : 0]        i_denominator,

    // Busy is high after start is accepted and before the quotient is ready.
    output logic                                  o_busy,

    // Done pulses for one cycle when quotient/remainder are valid.
    output logic                                  o_done,

    // High with o_done if i_denominator was zero.
    output logic                                  o_divide_by_zero,

    output logic [QUOTIENT_WIDTH-1 : 0]           o_quotient,
    output logic [REMAINDER_WIDTH-1 : 0]          o_remainder
);

    // Shift-subtract restoring divider.
    //
    // For each quotient bit from MSB to LSB:
    //   shifted_denominator = denominator << bit_index
    //   if remaining >= shifted_denominator:
    //       remaining = remaining - shifted_denominator
    //       quotient[bit_index] = 1
    //
    // This computes the exact quotient/remainder when the quotient fits in
    // QUOTIENT_WIDTH bits. If the mathematical quotient is wider, the quotient
    // naturally saturates to all ones and the remainder is the post-subtraction
    // leftover after the maximum representable quotient.

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam int DEN_SHIFT_WIDTH = DENOMINATOR_WIDTH + QUOTIENT_WIDTH;
    localparam int BASE_WORK_WIDTH = (NUMERATOR_WIDTH > DEN_SHIFT_WIDTH) ?
                                     NUMERATOR_WIDTH : DEN_SHIFT_WIDTH;
    localparam int WORK_WIDTH      = (BASE_WORK_WIDTH > REMAINDER_WIDTH) ?
                                     BASE_WORK_WIDTH : REMAINDER_WIDTH;

    localparam logic [ITERATION_W-1 : 0] LAST_QUOTIENT_INDEX =
        QUOTIENT_WIDTH - 1;

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;

    logic [ITERATION_W-1 : 0]             bit_index;
    logic [DENOMINATOR_WIDTH-1 : 0]       denominator_reg;
    logic [QUOTIENT_WIDTH-1 : 0]          quotient_reg;
    logic [WORK_WIDTH-1 : 0]              remaining_reg;
    logic                                 divide_by_zero_reg;

    logic [WORK_WIDTH-1 : 0]              numerator_extended;
    logic [WORK_WIDTH-1 : 0]              denominator_extended;
    logic [WORK_WIDTH-1 : 0]              shifted_denominator;
    logic [WORK_WIDTH-1 : 0]              next_remaining;
    logic                                 subtract_this_bit;

    assign o_busy           = (current_state == RUN);
    assign o_done           = (current_state == DONE);
    assign o_divide_by_zero = divide_by_zero_reg;
    assign o_quotient       = quotient_reg;
    assign o_remainder      = remaining_reg[REMAINDER_WIDTH-1 : 0];

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    if (i_denominator == 'd0) begin
                        next_state = DONE;
                    end
                    else begin
                        next_state = RUN;
                    end
                end
            end

            RUN: begin
                if (bit_index == 'd0) begin
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

    always_comb begin
        numerator_extended = 'd0;
        numerator_extended[NUMERATOR_WIDTH-1 : 0] = i_numerator;

        denominator_extended = 'd0;
        denominator_extended[DENOMINATOR_WIDTH-1 : 0] = denominator_reg;

        shifted_denominator = denominator_extended << bit_index;
        subtract_this_bit = (remaining_reg >= shifted_denominator);
        next_remaining = remaining_reg - shifted_denominator;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            bit_index          <= 'd0;
            denominator_reg    <= 'd0;
            quotient_reg       <= 'd0;
            remaining_reg      <= 'd0;
            divide_by_zero_reg <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        bit_index       <= LAST_QUOTIENT_INDEX;
                        denominator_reg <= i_denominator;
                        remaining_reg   <= numerator_extended;

                        if (i_denominator == 'd0) begin
                            quotient_reg       <= '1;
                            divide_by_zero_reg <= 1'b1;
                        end
                        else begin
                            quotient_reg       <= 'd0;
                            divide_by_zero_reg <= 1'b0;
                        end
                    end
                end

                RUN: begin
                    if (subtract_this_bit == 1'b1) begin
                        remaining_reg            <= next_remaining;
                        quotient_reg[bit_index]  <= 1'b1;
                    end
                    else begin
                        remaining_reg            <= remaining_reg;
                        quotient_reg[bit_index]  <= 1'b0;
                    end

                    if (bit_index != 'd0) begin
                        bit_index <= bit_index - 1'b1;
                    end
                end

                DONE: begin
                    bit_index          <= bit_index;
                    denominator_reg    <= denominator_reg;
                    quotient_reg       <= quotient_reg;
                    remaining_reg      <= remaining_reg;
                    divide_by_zero_reg <= divide_by_zero_reg;
                end

                default: begin
                    bit_index          <= 'd0;
                    denominator_reg    <= 'd0;
                    quotient_reg       <= 'd0;
                    remaining_reg      <= 'd0;
                    divide_by_zero_reg <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
