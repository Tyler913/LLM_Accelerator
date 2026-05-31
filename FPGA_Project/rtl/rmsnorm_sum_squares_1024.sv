`default_nettype none

// RMSNorm sum-of-squares stage for one 1024-wide hidden vector.
//
// Math contract:
//
//   o_sum_squares = sum_i x[i] * x[i], for i = 0..INPUT_SIZE-1
//
// Default input format:
//
//   x[i]: signed 24-bit Q14.10
//
// Each square has 2*IN_FRAC fractional bits. With the default Q14.10 input,
// o_sum_squares is an unsigned integer carrying Q20 fractional scaling.
module rmsnorm_sum_squares_1024 #(
    parameter int INPUT_SIZE       = 1024,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 10,
    parameter int PRODUCT_WIDTH    = IN_WIDTH * 2,
    parameter int SUM_WIDTH        = 64,
    parameter int ELEMENT_INDEX_W  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE)
)
(
    input  logic                                      i_clk,
    input  logic                                      i_rst_n,

    // Start a new sum-of-squares transaction when the module is not busy.
    // Keep i_input_flat stable until o_done is asserted.
    input  logic                                      i_start,

    // INPUT_SIZE signed fixed-point values, flattened little-element-endian:
    // element i is i_input_flat[IN_WIDTH*i +: IN_WIDTH].
    input  logic [INPUT_SIZE*IN_WIDTH-1 : 0]          i_input_flat,

    // Busy is high after start is accepted and before the result is ready.
    output logic                                      o_busy,

    // Done pulses for one cycle when o_sum_squares is valid.
    output logic                                      o_done,

    // Unsigned sum of squares. Fractional scaling is 2*IN_FRAC bits.
    output logic [SUM_WIDTH-1 : 0]                    o_sum_squares
);

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX =
        INPUT_SIZE - 1;

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;
    logic [ELEMENT_INDEX_W-1 : 0]         element_index;

    logic signed [IN_WIDTH-1 : 0]         current_input;
    logic signed [PRODUCT_WIDTH-1 : 0]    current_square_signed;
    logic        [PRODUCT_WIDTH-1 : 0]    current_square;
    logic        [SUM_WIDTH-1 : 0]        current_square_extended;

    assign o_busy = (current_state == RUN);
    assign o_done = (current_state == DONE);

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
        current_square_signed = current_input * current_input;
        current_square = current_square_signed[PRODUCT_WIDTH-1 : 0];
        current_square_extended =
            {{(SUM_WIDTH-PRODUCT_WIDTH){1'b0}}, current_square};
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_sum_squares <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    o_sum_squares <= 'd0;
                end

                RUN: begin
                    o_sum_squares <= o_sum_squares + current_square_extended;
                end

                DONE: begin
                    o_sum_squares <= o_sum_squares;
                end

                default: begin
                    o_sum_squares <= 'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
