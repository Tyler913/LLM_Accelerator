`default_nettype none

// Attention front-end stage through KV-cache append.
//
// This module connects:
//
//   Q/K projection outputs -> q_norm/k_norm -> Q/K RoPE
//   K RoPE + V projection output -> KV-cache append write stream
//
// Q RoPE remains an output for the later attention-score stage. K/V cache
// writes are exposed as a simple ready/valid stream so the stage can be proven
// locally before attaching an AXI write adapter.
module qk_norm_rope_kv_cache_stage #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DATA_WIDTH       = 32,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
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
    parameter int EPS_Q24          = 17,
    parameter int LAYER_INDEX_W    = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int HEAD_INDEX_W     = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]                        i_cache_base_addr,
    input  wire logic [LAYER_INDEX_W-1 : 0]                     i_layer_id,
    input  wire logic [POSITION_INDEX_W-1 : 0]                  i_position,

    input  wire logic [NUM_Q_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_q_flat,
    input  wire logic [NUM_KV_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]    i_k_flat,
    input  wire logic [NUM_KV_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]    i_v_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_q_gamma_flat,
    input  wire logic [HEAD_DIM*GAMMA_WIDTH-1 : 0]              i_k_gamma_flat,
    input  wire logic [HEAD_DIM*TRIG_WIDTH-1 : 0]               i_cos_flat,
    input  wire logic [HEAD_DIM*TRIG_WIDTH-1 : 0]               i_sin_flat,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_error,
    output logic                                           o_saturation,
    output logic                                           o_norm_saturation,
    output logic                                           o_rope_saturation,

    output logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]    o_q_rope_flat,
    output logic [NUM_KV_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]   o_k_rope_flat,

    output logic                                           o_cache_wr_valid,
    input  wire logic                                           i_cache_wr_ready,
    output logic [ADDR_WIDTH-1 : 0]                       o_cache_wr_addr,
    output logic [DATA_WIDTH-1 : 0]                       o_cache_wr_data,
    output logic                                           o_cache_wr_last,
    output logic                                           o_cache_wr_kind,
    output logic [HEAD_INDEX_W-1 : 0]                     o_cache_wr_head,
    output logic [DIM_INDEX_W-1 : 0]                      o_cache_wr_dim,
    output logic [31 : 0]                                 o_cache_write_count
);

    localparam IDLE         = 3'd0;
    localparam START_QK     = 3'd1;
    localparam WAIT_QK      = 3'd2;
    localparam START_APPEND = 3'd3;
    localparam WAIT_APPEND  = 3'd4;
    localparam DONE         = 3'd5;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;

    logic qk_start;
    logic qk_busy;
    logic qk_done;
    logic qk_saturation;
    logic qk_norm_saturation;
    logic qk_rope_saturation;
    logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0] q_norm_flat;
    logic [NUM_KV_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0] k_norm_flat;
    logic [31 : 0] qk_norm_heads_done;

    logic append_start;
    logic append_busy;
    logic append_done;
    logic append_error;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = append_error;
    assign o_saturation = qk_saturation;
    assign o_norm_saturation = qk_norm_saturation;
    assign o_rope_saturation = qk_rope_saturation;

    assign qk_start = (current_state == START_QK);
    assign append_start = (current_state == START_APPEND);

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_QK;
                end
            end

            START_QK: begin
                next_state = WAIT_QK;
            end

            WAIT_QK: begin
                if (qk_done == 1'b1) begin
                    next_state = START_APPEND;
                end
            end

            START_APPEND: begin
                next_state = WAIT_APPEND;
            end

            WAIT_APPEND: begin
                if (append_done == 1'b1) begin
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

    qk_norm_rope_stage_128 #(
        .NUM_Q_HEADS  (NUM_Q_HEADS),
        .NUM_K_HEADS  (NUM_KV_HEADS),
        .HEAD_DIM     (HEAD_DIM),
        .IN_WIDTH     (IN_WIDTH),
        .IN_FRAC      (IN_FRAC),
        .GAMMA_WIDTH  (GAMMA_WIDTH),
        .GAMMA_FRAC   (GAMMA_FRAC),
        .TRIG_WIDTH   (TRIG_WIDTH),
        .TRIG_FRAC    (TRIG_FRAC),
        .OUT_WIDTH    (OUT_WIDTH),
        .OUT_FRAC     (OUT_FRAC),
        .INV_RMS_WIDTH(INV_RMS_WIDTH),
        .INV_RMS_FRAC (INV_RMS_FRAC),
        .SUM_WIDTH    (SUM_WIDTH),
        .EPS_Q24      (EPS_Q24)
    ) qk_stage (
        .i_clk             (i_clk),
        .i_rst_n           (i_rst_n),
        .i_start           (qk_start),
        .i_q_flat          (i_q_flat),
        .i_k_flat          (i_k_flat),
        .i_q_gamma_flat    (i_q_gamma_flat),
        .i_k_gamma_flat    (i_k_gamma_flat),
        .i_cos_flat        (i_cos_flat),
        .i_sin_flat        (i_sin_flat),
        .o_busy            (qk_busy),
        .o_done            (qk_done),
        .o_saturation      (qk_saturation),
        .o_norm_saturation (qk_norm_saturation),
        .o_rope_saturation (qk_rope_saturation),
        .o_q_norm_flat     (q_norm_flat),
        .o_k_norm_flat     (k_norm_flat),
        .o_q_rope_flat     (o_q_rope_flat),
        .o_k_rope_flat     (o_k_rope_flat),
        .o_norm_heads_done (qk_norm_heads_done)
    );

    kv_cache_append #(
        .ADDR_WIDTH   (ADDR_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .NUM_LAYERS   (NUM_LAYERS),
        .NUM_KV_HEADS (NUM_KV_HEADS),
        .HEAD_DIM     (HEAD_DIM),
        .MAX_CONTEXT  (MAX_CONTEXT),
        .ELEMENT_WIDTH(OUT_WIDTH),
        .ELEMENT_BYTES(4)
    ) append_stage (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (append_start),
        .i_base_addr  (i_cache_base_addr),
        .i_layer_id   (i_layer_id),
        .i_position   (i_position),
        .i_k_flat     (o_k_rope_flat),
        .i_v_flat     (i_v_flat),
        .o_busy       (append_busy),
        .o_done       (append_done),
        .o_error      (append_error),
        .o_wr_valid   (o_cache_wr_valid),
        .i_wr_ready   (i_cache_wr_ready),
        .o_wr_addr    (o_cache_wr_addr),
        .o_wr_data    (o_cache_wr_data),
        .o_wr_last    (o_cache_wr_last),
        .o_current_kind(o_cache_wr_kind),
        .o_current_head(o_cache_wr_head),
        .o_current_dim (o_cache_wr_dim),
        .o_write_count(o_cache_write_count)
    );

endmodule

`default_nettype wire
