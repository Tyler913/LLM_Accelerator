`default_nettype none

// Layer 0 post-attention residual and RMSNorm stage:
//
//   residual_in[1024] + o_proj_out[1024] -> post_attention_hidden[1024]
//   post_attention_hidden[1024] -> post_attention_layernorm -> post_norm[1024]
//
// Fixed-point contract:
//   residual_in: signed Q14.10
//   o_proj_out:  signed Q12.12
//   residual sum / RMSNorm input: signed Q14.10
//   post_attention_layernorm gamma: signed Q8.7 for Layer 0
//   RMSNorm output: signed Q12.12
module post_attention_residual_norm_stage #(
    parameter int INPUT_SIZE       = 1024,
    parameter int RESIDUAL_WIDTH   = 24,
    parameter int RESIDUAL_FRAC    = 10,
    parameter int O_PROJ_WIDTH     = 24,
    parameter int O_PROJ_FRAC      = 12,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int NORM_OUT_WIDTH   = 24,
    parameter int NORM_OUT_FRAC    = 12,
    parameter int SUM_WIDTH        = 64,
    parameter int SUM_FRAC         = 2 * RESIDUAL_FRAC,
    parameter int MEAN_SHIFT       = $clog2(INPUT_SIZE),
    parameter int RMS_WIDTH        = RESIDUAL_WIDTH,
    parameter int RMS_FRAC         = RESIDUAL_FRAC,
    parameter int DIV_NUM_WIDTH    = 48,
    parameter int DIV_NUM_SHIFT    = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q20          = 1
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,
    input  wire logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0]         i_residual_flat,
    input  wire logic [INPUT_SIZE*O_PROJ_WIDTH-1 : 0]           i_o_proj_flat,
    input  wire logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0]            i_gamma_flat,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_error,
    output logic                                           o_residual_saturation,
    output logic                                           o_norm_saturation,
    output logic [31 : 0]                                  o_residual_count,
    output logic [31 : 0]                                  o_cycle_count,

    output logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0]         o_post_attention_hidden_flat,
    output logic [INPUT_SIZE*NORM_OUT_WIDTH-1 : 0]         o_post_norm_flat,

    output logic [SUM_WIDTH-1 : 0]                         o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                         o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]                     o_inv_rms
);

    localparam IDLE           = 3'd0;
    localparam START_RESIDUAL = 3'd1;
    localparam WAIT_RESIDUAL  = 3'd2;
    localparam START_NORM     = 3'd3;
    localparam WAIT_NORM      = 3'd4;
    localparam DONE           = 3'd5;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;
    logic         residual_start;
    logic         residual_busy;
    logic         residual_done;
    logic         residual_saturation;
    logic [31:0]  residual_count;
    logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0] residual_flat;
    logic         norm_start;
    logic         norm_busy;
    logic         norm_done;
    logic         norm_saturation;
    logic [INPUT_SIZE*NORM_OUT_WIDTH-1 : 0] norm_output_flat;
    logic         error_reg;

    assign residual_start = (current_state == START_RESIDUAL);
    assign norm_start = (current_state == START_NORM);

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_residual_saturation = residual_saturation;
    assign o_norm_saturation = norm_saturation;
    assign o_residual_count = residual_count;
    assign o_post_attention_hidden_flat = residual_flat;
    assign o_post_norm_flat = norm_output_flat;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_RESIDUAL;
                end
            end

            START_RESIDUAL: begin
                next_state = WAIT_RESIDUAL;
            end

            WAIT_RESIDUAL: begin
                if (residual_done == 1'b1) begin
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

                START_RESIDUAL,
                WAIT_RESIDUAL,
                START_NORM,
                WAIT_NORM: begin
                    o_cycle_count <= o_cycle_count + 1'b1;
                end

                DONE: begin
                    error_reg <= error_reg || (residual_count != INPUT_SIZE);
                end

                default: begin
                    error_reg <= 1'b1;
                    o_cycle_count <= 32'd0;
                end
            endcase
        end
    end

    residual_add_1024 #(
        .INPUT_SIZE      (INPUT_SIZE),
        .RESIDUAL_WIDTH  (RESIDUAL_WIDTH),
        .RESIDUAL_FRAC   (RESIDUAL_FRAC),
        .O_PROJ_WIDTH    (O_PROJ_WIDTH),
        .O_PROJ_FRAC     (O_PROJ_FRAC),
        .OUT_WIDTH       (RESIDUAL_WIDTH),
        .OUT_FRAC        (RESIDUAL_FRAC),
        .ELEMENT_INDEX_W ((INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE)),
        .ACC_WIDTH       (RESIDUAL_WIDTH + 2)
    ) inst_residual_add_1024 (
        .i_clk          (i_clk),
        .i_rst_n        (i_rst_n),
        .i_start        (residual_start),
        .i_residual_flat(i_residual_flat),
        .i_o_proj_flat  (i_o_proj_flat),
        .o_busy         (residual_busy),
        .o_done         (residual_done),
        .o_saturation   (residual_saturation),
        .o_output_count (residual_count),
        .o_output_flat  (residual_flat)
    );

    rmsnorm_1024 #(
        .INPUT_SIZE     (INPUT_SIZE),
        .IN_WIDTH       (RESIDUAL_WIDTH),
        .IN_FRAC        (RESIDUAL_FRAC),
        .GAMMA_WIDTH    (GAMMA_WIDTH),
        .GAMMA_FRAC     (GAMMA_FRAC),
        .GAMMA_SIGNED   (1),
        .INV_RMS_WIDTH  (INV_RMS_WIDTH),
        .INV_RMS_FRAC   (INV_RMS_FRAC),
        .OUT_WIDTH      (NORM_OUT_WIDTH),
        .OUT_FRAC       (NORM_OUT_FRAC),
        .SUM_WIDTH      (SUM_WIDTH),
        .SUM_FRAC       (SUM_FRAC),
        .MEAN_SHIFT     (MEAN_SHIFT),
        .RMS_WIDTH      (RMS_WIDTH),
        .RMS_FRAC       (RMS_FRAC),
        .DIV_NUM_WIDTH  (DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT  (DIV_NUM_SHIFT),
        .EPS_Q20        (EPS_Q20)
    ) inst_post_attention_rmsnorm (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (norm_start),
        .i_input_flat (residual_flat),
        .i_gamma_flat (i_gamma_flat),
        .o_busy       (norm_busy),
        .o_done       (norm_done),
        .o_saturation (norm_saturation),
        .o_output_flat(norm_output_flat),
        .o_sum_squares(o_sum_squares),
        .o_mean_square(o_mean_square),
        .o_inv_rms    (o_inv_rms)
    );

endmodule

`default_nettype wire
