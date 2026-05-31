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

    localparam IDLE        = 4'd0;
    localparam START_SUM   = 4'd1;
    localparam WAIT_SUM    = 4'd2;
    localparam START_SQRT  = 4'd3;
    localparam WAIT_SQRT   = 4'd4;
    localparam START_DIV   = 4'd5;
    localparam WAIT_DIV    = 4'd6;
    localparam START_APPLY = 4'd7;
    localparam WAIT_APPLY  = 4'd8;
    localparam DONE        = 4'd9;

    localparam logic [DIV_NUM_WIDTH-1 : 0] DIV_NUMERATOR_VALUE =
        ({{(DIV_NUM_WIDTH-1){1'b0}}, 1'b1} << DIV_NUM_SHIFT);

    logic [3 : 0]                         current_state;
    logic [3 : 0]                         next_state;

    logic                                 sum_start;
    logic                                 sum_busy;
    logic                                 sum_done;
    logic [SUM_WIDTH-1 : 0]               sum_squares;

    logic                                 sqrt_start;
    logic                                 sqrt_busy;
    logic                                 sqrt_done;
    logic [SUM_WIDTH-1 : 0]               sqrt_radicand;
    logic [RMS_WIDTH-1 : 0]               rms_q10;

    logic                                 div_start;
    logic                                 div_busy;
    logic                                 div_done;
    logic                                 div_by_zero;
    logic [DIV_NUM_WIDTH-1 : 0]           div_numerator;
    logic [RMS_WIDTH-1 : 0]               div_remainder;
    logic [RMS_WIDTH-1 : 0]               div_denominator;
    logic [INV_RMS_WIDTH-1 : 0]           inv_rms_q16;

    logic                                 apply_start;
    logic                                 apply_busy;
    logic                                 apply_done;
    logic                                 apply_saturation;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0]    apply_output_flat;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);

    assign o_saturation = apply_saturation;
    assign o_output_flat = apply_output_flat;

    assign sum_start   = (current_state == START_SUM);
    assign sqrt_start  = (current_state == START_SQRT);
    assign div_start   = (current_state == START_DIV);
    assign apply_start = (current_state == START_APPLY);

    assign div_numerator = DIV_NUMERATOR_VALUE;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_SUM;
                end
            end

            START_SUM: begin
                next_state = WAIT_SUM;
            end

            WAIT_SUM: begin
                if (sum_done == 1'b1) begin
                    next_state = START_SQRT;
                end
            end

            START_SQRT: begin
                next_state = WAIT_SQRT;
            end

            WAIT_SQRT: begin
                if (sqrt_done == 1'b1) begin
                    next_state = START_DIV;
                end
            end

            START_DIV: begin
                next_state = WAIT_DIV;
            end

            WAIT_DIV: begin
                if (div_done == 1'b1) begin
                    next_state = START_APPLY;
                end
            end

            START_APPLY: begin
                next_state = WAIT_APPLY;
            end

            WAIT_APPLY: begin
                if (apply_done == 1'b1) begin
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
            o_sum_squares  <= 'd0;
            o_mean_square  <= 'd0;
            sqrt_radicand  <= 'd0;
            div_denominator <= 'd0;
            o_inv_rms      <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_sum_squares  <= 'd0;
                        o_mean_square  <= 'd0;
                        sqrt_radicand  <= 'd0;
                        div_denominator <= 'd0;
                        o_inv_rms      <= 'd0;
                    end
                end

                WAIT_SUM: begin
                    if (sum_done == 1'b1) begin
                        o_sum_squares <= sum_squares;
                        o_mean_square <= sum_squares >> MEAN_SHIFT;
                        sqrt_radicand <= (sum_squares >> MEAN_SHIFT) + EPS_Q20;
                    end
                end

                WAIT_SQRT: begin
                    if (sqrt_done == 1'b1) begin
                        div_denominator <= rms_q10;
                    end
                end

                WAIT_DIV: begin
                    if (div_done == 1'b1) begin
                        o_inv_rms <= inv_rms_q16;
                    end
                end

                default: begin
                    o_sum_squares  <= o_sum_squares;
                    o_mean_square  <= o_mean_square;
                    sqrt_radicand  <= sqrt_radicand;
                    div_denominator <= div_denominator;
                    o_inv_rms      <= o_inv_rms;
                end
            endcase
        end
    end

    rmsnorm_sum_squares_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .PRODUCT_WIDTH  (IN_WIDTH * 2),
        .SUM_WIDTH      (SUM_WIDTH),
        .ELEMENT_INDEX_W((INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE))
    ) inst_rmsnorm_sum_squares_1024 (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (sum_start),
        .i_input_flat (i_input_flat),
        .o_busy       (sum_busy),
        .o_done       (sum_done),
        .o_sum_squares(sum_squares)
    );

    fixed_sqrt_u64 #(
        .IN_WIDTH       (SUM_WIDTH),
        .IN_FRAC        (SUM_FRAC),
        .OUT_WIDTH      (RMS_WIDTH),
        .OUT_FRAC       (RMS_FRAC),
        .ITERATION_COUNT(RMS_WIDTH),
        .ITERATION_W    ((RMS_WIDTH <= 1) ? 1 : $clog2(RMS_WIDTH))
    ) inst_fixed_sqrt_u64 (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_start   (sqrt_start),
        .i_radicand(sqrt_radicand),
        .o_busy    (sqrt_busy),
        .o_done    (sqrt_done),
        .o_root    (rms_q10)
    );

    fixed_udiv #(
        .NUMERATOR_WIDTH  (DIV_NUM_WIDTH),
        .DENOMINATOR_WIDTH(RMS_WIDTH),
        .QUOTIENT_WIDTH   (INV_RMS_WIDTH),
        .REMAINDER_WIDTH  (RMS_WIDTH),
        .ITERATION_COUNT  (INV_RMS_WIDTH),
        .ITERATION_W      ((INV_RMS_WIDTH <= 1) ? 1 : $clog2(INV_RMS_WIDTH))
    ) inst_fixed_udiv (
        .i_clk           (i_clk),
        .i_rst_n         (i_rst_n),
        .i_start         (div_start),
        .i_numerator     (div_numerator),
        .i_denominator   (div_denominator),
        .o_busy          (div_busy),
        .o_done          (div_done),
        .o_divide_by_zero(div_by_zero),
        .o_quotient      (inv_rms_q16),
        .o_remainder     (div_remainder)
    );

    rmsnorm_apply_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .INV_RMS_WIDTH  (INV_RMS_WIDTH),
        .INV_RMS_FRAC   (INV_RMS_FRAC),
        .GAMMA_WIDTH    (GAMMA_WIDTH),
        .GAMMA_FRAC     (GAMMA_FRAC),
        .OUT_WIDTH      (OUT_WIDTH),
        .OUT_FRAC       (OUT_FRAC),
        .PRODUCT1_WIDTH (IN_WIDTH + INV_RMS_WIDTH),
        .PRODUCT2_WIDTH (IN_WIDTH + INV_RMS_WIDTH + GAMMA_WIDTH),
        .OUTPUT_SHIFT   (IN_FRAC + INV_RMS_FRAC + GAMMA_FRAC - OUT_FRAC),
        .ELEMENT_INDEX_W((INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE))
    ) inst_rmsnorm_apply_1024 (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (apply_start),
        .i_input_flat (i_input_flat),
        .i_gamma_flat (i_gamma_flat),
        .i_inv_rms    (o_inv_rms),
        .o_busy       (apply_busy),
        .o_done       (apply_done),
        .o_saturation (apply_saturation),
        .o_output_flat(apply_output_flat)
    );

endmodule

`default_nettype wire
