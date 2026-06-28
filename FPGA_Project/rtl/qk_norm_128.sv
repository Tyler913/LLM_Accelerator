`default_nettype none

// Qwen3 attention Q/K head RMSNorm wrapper.
//
// This block consumes one token's Q and K projection outputs after the QKV
// projection stage. Each attention head is normalized independently over its
// 128-dimensional vector. A single parameterized rmsnorm_1024 instance is
// reused across all heads to keep this first integration stage small and easy
// to inspect in simulation.
//
// Shape contract:
//
//   Q input/output: [16, 128], flattened as q[head][dim]
//   K input/output: [8, 128], flattened as k[head][dim]
//   q_gamma:        [128], shared across all Q heads
//   k_gamma:        [128], shared across all K heads
//
// Default fixed-point formats:
//
//   q/k input:  signed 24-bit Q12.12
//   gamma:      signed 16-bit Q8.7
//   q/k output: signed 24-bit Q12.12
//   inv_rms:    unsigned 24-bit UQ8.16
module qk_norm_128 #(
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_K_HEADS      = 8,
    parameter int HEAD_DIM         = 128,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 12,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
    parameter int GAMMA_SIGNED     = 1,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int SUM_WIDTH        = 64,
    parameter int SUM_FRAC         = 2 * IN_FRAC,
    parameter int MEAN_SHIFT       = $clog2(HEAD_DIM),
    parameter int RMS_WIDTH        = IN_WIDTH,
    parameter int RMS_FRAC         = IN_FRAC,
    parameter int DIV_NUM_WIDTH    = 48,
    parameter int DIV_NUM_SHIFT    = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q24          = 17,
    parameter int TOTAL_HEADS      = NUM_Q_HEADS + NUM_K_HEADS,
    parameter int TOTAL_HEAD_INDEX_W = (TOTAL_HEADS <= 1) ? 1 : $clog2(TOTAL_HEADS)
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,

    input  wire logic [NUM_Q_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_q_flat,
    input  wire logic [NUM_K_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_k_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_q_gamma_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_k_gamma_flat,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_saturation,

    output logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_q_norm_flat,
    output logic [NUM_K_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_k_norm_flat,

    output logic [31 : 0]                                  o_heads_done,
    output logic [TOTAL_HEAD_INDEX_W-1 : 0]                o_current_head_index,
    output logic [SUM_WIDTH-1 : 0]                         o_last_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                         o_last_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]                     o_last_inv_rms
);

    localparam IDLE       = 2'd0;
    localparam START_HEAD = 2'd1;
    localparam WAIT_HEAD  = 2'd2;
    localparam DONE       = 2'd3;

    localparam logic [TOTAL_HEAD_INDEX_W-1 : 0] LAST_HEAD_INDEX = TOTAL_HEADS - 1;

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;
    logic [TOTAL_HEAD_INDEX_W-1 : 0]      head_index;
    logic                                 norm_start;
    logic                                 norm_busy;
    logic                                 norm_done;
    logic                                 norm_saturation;
    logic [HEAD_DIM*IN_WIDTH-1 : 0]       norm_input_flat;
    logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]    norm_gamma_flat;
    logic [HEAD_DIM*OUT_WIDTH-1 : 0]      norm_output_flat;
    logic [SUM_WIDTH-1 : 0]               norm_sum_squares;
    logic [SUM_WIDTH-1 : 0]               norm_mean_square;
    logic [INV_RMS_WIDTH-1 : 0]           norm_inv_rms;
    logic                                 saturation_reg;

    integer dim_index;
    integer source_element_index;
    integer output_element_index;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_saturation = saturation_reg;
    assign o_current_head_index = head_index;
    assign norm_start = (current_state == START_HEAD);

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_HEAD;
                end
            end

            START_HEAD: begin
                next_state = WAIT_HEAD;
            end

            WAIT_HEAD: begin
                if (norm_done == 1'b1) begin
                    if (head_index == LAST_HEAD_INDEX) begin
                        next_state = DONE;
                    end
                    else begin
                        next_state = START_HEAD;
                    end
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
            head_index <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    head_index <= 'd0;
                end

                WAIT_HEAD: begin
                    if ((norm_done == 1'b1) && (head_index != LAST_HEAD_INDEX)) begin
                        head_index <= head_index + 1'b1;
                    end
                end

                DONE: begin
                    head_index <= 'd0;
                end

                default: begin
                    head_index <= head_index;
                end
            endcase
        end
    end

    always_comb begin
        norm_input_flat = 'd0;
        norm_gamma_flat = i_q_gamma_flat;

        if (head_index < NUM_Q_HEADS) begin
            norm_gamma_flat = i_q_gamma_flat;
            for (dim_index = 0; dim_index < HEAD_DIM; dim_index = dim_index + 1) begin
                source_element_index = (head_index * HEAD_DIM) + dim_index;
                norm_input_flat[dim_index*IN_WIDTH +: IN_WIDTH] =
                    i_q_flat[source_element_index*IN_WIDTH +: IN_WIDTH];
            end
        end
        else begin
            norm_gamma_flat = i_k_gamma_flat;
            for (dim_index = 0; dim_index < HEAD_DIM; dim_index = dim_index + 1) begin
                source_element_index = ((head_index - NUM_Q_HEADS) * HEAD_DIM) + dim_index;
                norm_input_flat[dim_index*IN_WIDTH +: IN_WIDTH] =
                    i_k_flat[source_element_index*IN_WIDTH +: IN_WIDTH];
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_q_norm_flat       <= 'd0;
            o_k_norm_flat       <= 'd0;
            saturation_reg      <= 1'b0;
            o_heads_done        <= 32'd0;
            o_last_sum_squares  <= 'd0;
            o_last_mean_square  <= 'd0;
            o_last_inv_rms      <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_q_norm_flat       <= 'd0;
                        o_k_norm_flat       <= 'd0;
                        saturation_reg      <= 1'b0;
                        o_heads_done        <= 32'd0;
                        o_last_sum_squares  <= 'd0;
                        o_last_mean_square  <= 'd0;
                        o_last_inv_rms      <= 'd0;
                    end
                end

                WAIT_HEAD: begin
                    if (norm_done == 1'b1) begin
                        if (head_index < NUM_Q_HEADS) begin
                            for (dim_index = 0; dim_index < HEAD_DIM; dim_index = dim_index + 1) begin
                                output_element_index = (head_index * HEAD_DIM) + dim_index;
                                o_q_norm_flat[output_element_index*OUT_WIDTH +: OUT_WIDTH] <=
                                    norm_output_flat[dim_index*OUT_WIDTH +: OUT_WIDTH];
                            end
                        end
                        else begin
                            for (dim_index = 0; dim_index < HEAD_DIM; dim_index = dim_index + 1) begin
                                output_element_index =
                                    ((head_index - NUM_Q_HEADS) * HEAD_DIM) + dim_index;
                                o_k_norm_flat[output_element_index*OUT_WIDTH +: OUT_WIDTH] <=
                                    norm_output_flat[dim_index*OUT_WIDTH +: OUT_WIDTH];
                            end
                        end

                        saturation_reg      <= saturation_reg | norm_saturation;
                        o_heads_done        <= o_heads_done + 1'b1;
                        o_last_sum_squares  <= norm_sum_squares;
                        o_last_mean_square  <= norm_mean_square;
                        o_last_inv_rms      <= norm_inv_rms;
                    end
                end

                default: begin
                    o_q_norm_flat       <= o_q_norm_flat;
                    o_k_norm_flat       <= o_k_norm_flat;
                    saturation_reg      <= saturation_reg;
                    o_heads_done        <= o_heads_done;
                    o_last_sum_squares  <= o_last_sum_squares;
                    o_last_mean_square  <= o_last_mean_square;
                    o_last_inv_rms      <= o_last_inv_rms;
                end
            endcase
        end
    end

    rmsnorm_1024 #(
        .INPUT_SIZE    (HEAD_DIM),
        .IN_WIDTH      (IN_WIDTH),
        .IN_FRAC       (IN_FRAC),
        .GAMMA_WIDTH   (GAMMA_WIDTH),
        .GAMMA_FRAC    (GAMMA_FRAC),
        .GAMMA_SIGNED  (GAMMA_SIGNED),
        .INV_RMS_WIDTH (INV_RMS_WIDTH),
        .INV_RMS_FRAC  (INV_RMS_FRAC),
        .OUT_WIDTH     (OUT_WIDTH),
        .OUT_FRAC      (OUT_FRAC),
        .SUM_WIDTH     (SUM_WIDTH),
        .SUM_FRAC      (SUM_FRAC),
        .MEAN_SHIFT    (MEAN_SHIFT),
        .RMS_WIDTH     (RMS_WIDTH),
        .RMS_FRAC      (RMS_FRAC),
        .DIV_NUM_WIDTH (DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT (DIV_NUM_SHIFT),
        .EPS_Q20       (EPS_Q24)
    ) norm_core (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (norm_start),
        .i_input_flat (norm_input_flat),
        .i_gamma_flat (norm_gamma_flat),
        .o_busy       (norm_busy),
        .o_done       (norm_done),
        .o_saturation (norm_saturation),
        .o_output_flat(norm_output_flat),
        .o_sum_squares(norm_sum_squares),
        .o_mean_square(norm_mean_square),
        .o_inv_rms    (norm_inv_rms)
    );

endmodule

`default_nettype wire
