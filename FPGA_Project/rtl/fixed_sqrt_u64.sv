`default_nettype none

// Fixed-point unsigned square-root helper.
//
// Default RMSNorm use:
//
//   i_radicand: mean_square + epsilon, unsigned Q28.20-style value
//   o_root:     RMS value, unsigned 24-bit Q14.10
//
// The default fractional relationship is IN_FRAC = 2*OUT_FRAC.
module fixed_sqrt_u64 #(
    parameter int IN_WIDTH        = 64,
    parameter int IN_FRAC         = 20,
    parameter int OUT_WIDTH       = 24,
    parameter int OUT_FRAC        = 10,
    parameter int ITERATION_COUNT = OUT_WIDTH,
    parameter int ITERATION_W     = (ITERATION_COUNT <= 1) ? 1 : $clog2(ITERATION_COUNT)
)
(
    input  logic                         i_clk,
    input  logic                         i_rst_n,

    // Start a new square-root transaction when the module is not busy.
    // Keep i_radicand stable until o_done is asserted.
    input  logic                         i_start,

    // Unsigned fixed-point radicand.
    input  logic [IN_WIDTH-1 : 0]        i_radicand,

    // Busy is high after start is accepted and before the root is ready.
    output logic                         o_busy,

    // Done pulses for one cycle when o_root is valid.
    output logic                         o_done,

    // Unsigned fixed-point square-root result.
    output logic [OUT_WIDTH-1 : 0]       o_root
);

    // Trial-bit square-root datapath.
    //
    // For each output bit from MSB to LSB:
    //   candidate = root_so_far | (1 << bit_index)
    //   if candidate*candidate <= radicand, keep that bit.
    //
    // This uses one square multiply per cycle. It is not the most resource-
    // efficient sqrt implementation, but it is easy to inspect in simulation
    // and good enough while the serial RMSNorm sum/apply stages dominate
    // latency.

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam int SQUARE_WIDTH  = OUT_WIDTH * 2;
    localparam int COMPARE_WIDTH = (IN_WIDTH > SQUARE_WIDTH) ? IN_WIDTH : SQUARE_WIDTH;

    localparam logic [ITERATION_W-1 : 0] LAST_BIT_INDEX =
        OUT_WIDTH - 1;

    logic [1 : 0]                       current_state;
    logic [1 : 0]                       next_state;

    logic [ITERATION_W-1 : 0]           bit_index;
    logic [IN_WIDTH-1 : 0]              radicand_reg;
    logic [OUT_WIDTH-1 : 0]             root_reg;

    logic [OUT_WIDTH-1 : 0]             bit_mask;
    logic [OUT_WIDTH-1 : 0]             candidate_root;
    logic [SQUARE_WIDTH-1 : 0]          candidate_square;
    logic [COMPARE_WIDTH-1 : 0]         candidate_square_extended;
    logic [COMPARE_WIDTH-1 : 0]         radicand_extended;
    logic                               candidate_fits;

    assign o_busy = (current_state == RUN);
    assign o_done = (current_state == DONE);
    assign o_root = root_reg;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = RUN;
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
        bit_mask = 'd0;
        bit_mask[bit_index] = 1'b1;

        candidate_root = root_reg | bit_mask;
        candidate_square = candidate_root * candidate_root;

        candidate_square_extended = 'd0;
        candidate_square_extended[SQUARE_WIDTH-1 : 0] = candidate_square;

        radicand_extended = 'd0;
        radicand_extended[IN_WIDTH-1 : 0] = radicand_reg;

        candidate_fits = (candidate_square_extended <= radicand_extended);
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            bit_index    <= 'd0;
            radicand_reg <= 'd0;
            root_reg     <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        bit_index    <= LAST_BIT_INDEX;
                        radicand_reg <= i_radicand;
                        root_reg     <= 'd0;
                    end
                end

                RUN: begin
                    if (candidate_fits == 1'b1) begin
                        root_reg <= candidate_root;
                    end
                    else begin
                        root_reg <= root_reg;
                    end

                    if (bit_index != 'd0) begin
                        bit_index <= bit_index - 1'b1;
                    end
                end

                DONE: begin
                    bit_index    <= bit_index;
                    radicand_reg <= radicand_reg;
                    root_reg     <= root_reg;
                end

                default: begin
                    bit_index    <= 'd0;
                    radicand_reg <= 'd0;
                    root_reg     <= 'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
