`default_nettype none

// First downstream attention-front-end stage after QKV projection.
//
// This top connects:
//
//   Q/K projection outputs -> per-head q_norm/k_norm -> Q/K RoPE
//
// V does not pass through q_norm/k_norm or RoPE; it is handled by the later
// cache-write stage. This module is intentionally memoryless and bus-flattened
// so the fixed-point math contract can be proven locally before attaching a
// descriptor/AXI wrapper.
module qk_norm_rope_stage_128 #(
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_K_HEADS      = 8,
    parameter int HEAD_DIM         = 128,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 12,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
    parameter int TRIG_WIDTH       = 16,
    parameter int TRIG_FRAC        = 15,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int SUM_WIDTH        = 64,
    parameter int EPS_Q24          = 17
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,

    input  wire logic [NUM_Q_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_q_flat,
    input  wire logic [NUM_K_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_k_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_q_gamma_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_k_gamma_flat,
    input  wire logic [HEAD_DIM*TRIG_WIDTH-1 : 0]               i_cos_flat,
    input  wire logic [HEAD_DIM*TRIG_WIDTH-1 : 0]               i_sin_flat,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_saturation,
    output logic                                           o_norm_saturation,
    output logic                                           o_rope_saturation,

    output logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_q_norm_flat,
    output logic [NUM_K_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_k_norm_flat,
    output logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_q_rope_flat,
    output logic [NUM_K_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_k_rope_flat,

    output logic [31 : 0]                                  o_norm_heads_done
);

    localparam IDLE       = 3'd0;
    localparam START_NORM = 3'd1;
    localparam WAIT_NORM  = 3'd2;
    localparam START_ROPE = 3'd3;
    localparam WAIT_ROPE  = 3'd4;
    localparam DONE       = 3'd5;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;

    logic norm_start;
    logic norm_busy;
    logic norm_done;
    logic norm_saturation;
    logic [31 : 0] norm_heads_done;
    logic [SUM_WIDTH-1 : 0] norm_last_sum_squares;
    logic [SUM_WIDTH-1 : 0] norm_last_mean_square;
    logic [INV_RMS_WIDTH-1 : 0] norm_last_inv_rms;

    logic rope_start;
    logic rope_busy;
    logic rope_done;
    logic rope_saturation;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_norm_saturation = norm_saturation;
    assign o_rope_saturation = rope_saturation;
    assign o_saturation = norm_saturation | rope_saturation;
    assign o_norm_heads_done = norm_heads_done;

    assign norm_start = (current_state == START_NORM);
    assign rope_start = (current_state == START_ROPE);

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
                    next_state = START_ROPE;
                end
            end

            START_ROPE: begin
                next_state = WAIT_ROPE;
            end

            WAIT_ROPE: begin
                if (rope_done == 1'b1) begin
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

    qk_norm_128 #(
        .NUM_Q_HEADS   (NUM_Q_HEADS),
        .NUM_K_HEADS   (NUM_K_HEADS),
        .HEAD_DIM      (HEAD_DIM),
        .IN_WIDTH      (IN_WIDTH),
        .IN_FRAC       (IN_FRAC),
        .GAMMA_WIDTH   (GAMMA_WIDTH),
        .GAMMA_FRAC    (GAMMA_FRAC),
        .INV_RMS_WIDTH (INV_RMS_WIDTH),
        .INV_RMS_FRAC  (INV_RMS_FRAC),
        .OUT_WIDTH     (OUT_WIDTH),
        .OUT_FRAC      (OUT_FRAC),
        .SUM_WIDTH     (SUM_WIDTH),
        .SUM_FRAC      (2 * IN_FRAC),
        .MEAN_SHIFT    ($clog2(HEAD_DIM)),
        .RMS_WIDTH     (IN_WIDTH),
        .RMS_FRAC      (IN_FRAC),
        .DIV_NUM_WIDTH (48),
        .DIV_NUM_SHIFT (IN_FRAC + INV_RMS_FRAC),
        .EPS_Q24       (EPS_Q24)
    ) qk_norm (
        .i_clk               (i_clk),
        .i_rst_n             (i_rst_n),
        .i_start             (norm_start),
        .i_q_flat            (i_q_flat),
        .i_k_flat            (i_k_flat),
        .i_q_gamma_flat      (i_q_gamma_flat),
        .i_k_gamma_flat      (i_k_gamma_flat),
        .o_busy              (norm_busy),
        .o_done              (norm_done),
        .o_saturation        (norm_saturation),
        .o_q_norm_flat       (o_q_norm_flat),
        .o_k_norm_flat       (o_k_norm_flat),
        .o_heads_done        (norm_heads_done),
        .o_current_head_index(),
        .o_last_sum_squares  (norm_last_sum_squares),
        .o_last_mean_square  (norm_last_mean_square),
        .o_last_inv_rms      (norm_last_inv_rms)
    );

    rope_qk_layer_128 #(
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_K_HEADS(NUM_K_HEADS),
        .HEAD_DIM   (HEAD_DIM),
        .IN_WIDTH   (OUT_WIDTH),
        .IN_FRAC    (OUT_FRAC),
        .TRIG_WIDTH (TRIG_WIDTH),
        .TRIG_FRAC  (TRIG_FRAC),
        .OUT_WIDTH  (OUT_WIDTH),
        .OUT_FRAC   (OUT_FRAC)
    ) rope (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (rope_start),
        .i_q_flat     (o_q_norm_flat),
        .i_k_flat     (o_k_norm_flat),
        .i_cos_flat   (i_cos_flat),
        .i_sin_flat   (i_sin_flat),
        .o_busy       (rope_busy),
        .o_done       (rope_done),
        .o_saturation (rope_saturation),
        .o_q_rope_flat(o_q_rope_flat),
        .o_k_rope_flat(o_k_rope_flat)
    );

endmodule

`default_nettype wire
