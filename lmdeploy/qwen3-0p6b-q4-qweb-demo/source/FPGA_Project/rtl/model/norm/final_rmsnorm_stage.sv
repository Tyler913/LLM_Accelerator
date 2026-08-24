`default_nettype none

// Final model RMSNorm stage:
//
//   final_hidden[1024] -> final_norm[1024]
//
// Fixed-point contract:
//   input hidden: signed Q14.10
//   final norm gamma: signed Q8.7
//   output: signed Q12.12
module final_rmsnorm_stage #(
    parameter int INPUT_SIZE       = 1024,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 10,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
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
    input  wire logic                                      i_clk,
    input  wire logic                                      i_rst_n,

    input  wire logic                                      i_start,
    input  wire logic [INPUT_SIZE*IN_WIDTH-1 : 0]          i_input_flat,
    input  wire logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]       i_gamma_flat,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_error,
    output logic                                      o_saturation,
    output logic [31 : 0]                             o_cycle_count,
    output logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         o_output_flat,
    output logic [SUM_WIDTH-1 : 0]                    o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                    o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]                o_inv_rms
);

    localparam IDLE       = 2'd0;
    localparam START_NORM = 2'd1;
    localparam WAIT_NORM  = 2'd2;
    localparam DONE       = 2'd3;

    logic [1 : 0] current_state;
    logic [1 : 0] next_state;
    logic         norm_start;
    logic         norm_busy;
    logic         norm_done;
    logic         norm_saturation;
    logic         error_reg;

    assign norm_start = (current_state == START_NORM);
    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_saturation = norm_saturation;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_NORM;
                end
            end

            START_NORM: begin
                next_state = WAIT_NORM;
            end

            WAIT_NORM: begin
                if (norm_done == 1'b1) begin
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
            error_reg <= 1'b0;
            o_cycle_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        error_reg <= 1'b0;
                        o_cycle_count <= 32'd0;
                    end
                end

                START_NORM,
                WAIT_NORM: begin
                    o_cycle_count <= o_cycle_count + 1'b1;
                end

                DONE: begin
                    error_reg <= error_reg || norm_saturation;
                end

                default: begin
                    error_reg <= 1'b1;
                    o_cycle_count <= 32'd0;
                end
            endcase
        end
    end

    rmsnorm_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (IN_WIDTH),
        .IN_FRAC        (IN_FRAC),
        .GAMMA_WIDTH    (GAMMA_WIDTH),
        .GAMMA_FRAC     (GAMMA_FRAC),
        .GAMMA_SIGNED   (1),
        .INV_RMS_WIDTH  (INV_RMS_WIDTH),
        .INV_RMS_FRAC   (INV_RMS_FRAC),
        .OUT_WIDTH      (OUT_WIDTH),
        .OUT_FRAC       (OUT_FRAC),
        .SUM_WIDTH      (SUM_WIDTH),
        .SUM_FRAC       (SUM_FRAC),
        .MEAN_SHIFT     (MEAN_SHIFT),
        .RMS_WIDTH      (RMS_WIDTH),
        .RMS_FRAC       (RMS_FRAC),
        .DIV_NUM_WIDTH  (DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT  (DIV_NUM_SHIFT),
        .EPS_Q20        (EPS_Q20)
    ) inst_final_rmsnorm (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (norm_start),
        .i_input_flat (i_input_flat),
        .i_gamma_flat (i_gamma_flat),
        .o_busy       (norm_busy),
        .o_done       (norm_done),
        .o_saturation (norm_saturation),
        .o_output_flat(o_output_flat),
        .o_sum_squares(o_sum_squares),
        .o_mean_square(o_mean_square),
        .o_inv_rms    (o_inv_rms)
    );

endmodule

`default_nettype wire
