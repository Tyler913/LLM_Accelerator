`default_nettype none

`include "qmap_defs.svh"

// Local Layer 0 body scheduler:
//
//   post-attention residual/RMSNorm
//     -> MLP gate/up
//     -> MLP SiLU/multiply
//     -> MLP down
//     -> final MLP residual add
//
// Each step is an already validated QMAP wrapper. This scheduler provides the
// first single-control, single-memory-port composition boundary for the local
// per-layer body simulation.
module qmap_layer0_body_scheduler #(
    parameter int ADDR_WIDTH     = 64,
    parameter int MEM_DATA_WIDTH = 32
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic                         i_residual_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_residual_base_override_addr,
    input  wire logic                         i_output_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_base_override_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_post_attn_norm_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_gate_up_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_silu_mul_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_down_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_residual_add_qmap_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [2 : 0]                      o_active_stage_debug,
    output logic [7 : 0]                      o_state_debug,
    output logic [4 : 0]                      o_stage_done_mask,
    output logic [4 : 0]                      o_stage_error_mask,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,

    output logic                              o_mem_rd_req_valid,
    input  wire logic                         i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_rd_req_addr,
    output logic [15 : 0]                     o_mem_rd_req_len_bytes,

    input  wire logic                         i_mem_rd_rsp_valid,
    output logic                              o_mem_rd_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]  i_mem_rd_rsp_data,
    input  wire logic                         i_mem_rd_rsp_last,

    output logic                              o_mem_wr_req_valid,
    input  wire logic                         i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_wr_req_addr,
    output logic [15 : 0]                     o_mem_wr_req_len_bytes,

    output logic [31 : 0]                     o_mem_wr_data,
    output logic                              o_mem_wr_data_valid,
    input  wire logic                         i_mem_wr_data_ready,
    output logic                              o_mem_wr_data_last,
    input  wire logic                         i_mem_wr_done,
    input  wire logic                         i_mem_wr_error
);

    localparam logic [2 : 0] STAGE_POST_ATTN_NORM = 3'd0;
    localparam logic [2 : 0] STAGE_MLP_GATE_UP    = 3'd1;
    localparam logic [2 : 0] STAGE_MLP_SILU_MUL   = 3'd2;
    localparam logic [2 : 0] STAGE_MLP_DOWN       = 3'd3;
    localparam logic [2 : 0] STAGE_MLP_RESIDUAL   = 3'd4;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_STAGE_START,
        S_STAGE_WAIT,
        S_DONE
    } state_t;

    state_t state;
    logic [2 : 0] active_stage;

    logic post_start;
    logic gate_start;
    logic silu_start;
    logic down_start;
    logic residual_start;

    logic post_done;
    logic gate_done;
    logic silu_done;
    logic down_done;
    logic residual_done;

    logic post_error;
    logic gate_error;
    logic silu_error;
    logic down_error;
    logic residual_error;

    logic post_rd_req_valid;
    logic gate_rd_req_valid;
    logic silu_rd_req_valid;
    logic down_rd_req_valid;
    logic residual_rd_req_valid;
    logic post_rd_req_ready;
    logic gate_rd_req_ready;
    logic silu_rd_req_ready;
    logic down_rd_req_ready;
    logic residual_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] post_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] gate_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] silu_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] down_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] residual_rd_req_addr;
    logic [15 : 0] post_rd_req_len;
    logic [15 : 0] gate_rd_req_len;
    logic [15 : 0] silu_rd_req_len;
    logic [15 : 0] down_rd_req_len;
    logic [15 : 0] residual_rd_req_len;
    logic post_rd_rsp_ready;
    logic gate_rd_rsp_ready;
    logic silu_rd_rsp_ready;
    logic down_rd_rsp_ready;
    logic residual_rd_rsp_ready;

    logic post_wr_req_valid;
    logic gate_wr_req_valid;
    logic silu_wr_req_valid;
    logic down_wr_req_valid;
    logic residual_wr_req_valid;
    logic post_wr_req_ready;
    logic gate_wr_req_ready;
    logic silu_wr_req_ready;
    logic down_wr_req_ready;
    logic residual_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] post_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] gate_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] silu_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] down_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] residual_wr_req_addr;
    logic [15 : 0] post_wr_req_len;
    logic [15 : 0] gate_wr_req_len;
    logic [15 : 0] silu_wr_req_len;
    logic [15 : 0] down_wr_req_len;
    logic [15 : 0] residual_wr_req_len;
    logic [31 : 0] post_wr_data;
    logic [31 : 0] gate_wr_data;
    logic [31 : 0] silu_wr_data;
    logic [31 : 0] down_wr_data;
    logic [31 : 0] residual_wr_data;
    logic post_wr_data_valid;
    logic gate_wr_data_valid;
    logic silu_wr_data_valid;
    logic down_wr_data_valid;
    logic residual_wr_data_valid;
    logic post_wr_data_ready;
    logic gate_wr_data_ready;
    logic silu_wr_data_ready;
    logic down_wr_data_ready;
    logic residual_wr_data_ready;
    logic post_wr_data_last;
    logic gate_wr_data_last;
    logic silu_wr_data_last;
    logic down_wr_data_last;
    logic residual_wr_data_last;

    logic post_active;
    logic gate_active;
    logic silu_active;
    logic down_active;
    logic residual_active;
    logic active_done;
    logic active_error;

    assign post_active = (state != S_IDLE) && (active_stage == STAGE_POST_ATTN_NORM);
    assign gate_active = (state != S_IDLE) && (active_stage == STAGE_MLP_GATE_UP);
    assign silu_active = (state != S_IDLE) && (active_stage == STAGE_MLP_SILU_MUL);
    assign down_active = (state != S_IDLE) && (active_stage == STAGE_MLP_DOWN);
    assign residual_active = (state != S_IDLE) && (active_stage == STAGE_MLP_RESIDUAL);

    assign post_start = (state == S_STAGE_START) && (active_stage == STAGE_POST_ATTN_NORM);
    assign gate_start = (state == S_STAGE_START) && (active_stage == STAGE_MLP_GATE_UP);
    assign silu_start = (state == S_STAGE_START) && (active_stage == STAGE_MLP_SILU_MUL);
    assign down_start = (state == S_STAGE_START) && (active_stage == STAGE_MLP_DOWN);
    assign residual_start = (state == S_STAGE_START) && (active_stage == STAGE_MLP_RESIDUAL);

    assign post_rd_req_ready = post_active ? i_mem_rd_req_ready : 1'b0;
    assign gate_rd_req_ready = gate_active ? i_mem_rd_req_ready : 1'b0;
    assign silu_rd_req_ready = silu_active ? i_mem_rd_req_ready : 1'b0;
    assign down_rd_req_ready = down_active ? i_mem_rd_req_ready : 1'b0;
    assign residual_rd_req_ready = residual_active ? i_mem_rd_req_ready : 1'b0;

    assign post_wr_req_ready = post_active ? i_mem_wr_req_ready : 1'b0;
    assign gate_wr_req_ready = gate_active ? i_mem_wr_req_ready : 1'b0;
    assign silu_wr_req_ready = silu_active ? i_mem_wr_req_ready : 1'b0;
    assign down_wr_req_ready = down_active ? i_mem_wr_req_ready : 1'b0;
    assign residual_wr_req_ready = residual_active ? i_mem_wr_req_ready : 1'b0;

    assign post_wr_data_ready = post_active ? i_mem_wr_data_ready : 1'b0;
    assign gate_wr_data_ready = gate_active ? i_mem_wr_data_ready : 1'b0;
    assign silu_wr_data_ready = silu_active ? i_mem_wr_data_ready : 1'b0;
    assign down_wr_data_ready = down_active ? i_mem_wr_data_ready : 1'b0;
    assign residual_wr_data_ready = residual_active ? i_mem_wr_data_ready : 1'b0;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_active_stage_debug = active_stage;
    assign o_state_debug = {6'd0, state};

    always @* begin
        o_mem_rd_req_valid = 1'b0;
        o_mem_rd_req_addr = '0;
        o_mem_rd_req_len_bytes = 16'd0;
        o_mem_rd_rsp_ready = 1'b0;
        o_mem_wr_req_valid = 1'b0;
        o_mem_wr_req_addr = '0;
        o_mem_wr_req_len_bytes = 16'd0;
        o_mem_wr_data = 32'd0;
        o_mem_wr_data_valid = 1'b0;
        o_mem_wr_data_last = 1'b0;
        active_done = 1'b0;
        active_error = 1'b0;

        case (active_stage)
            STAGE_POST_ATTN_NORM: begin
                o_mem_rd_req_valid = post_rd_req_valid;
                o_mem_rd_req_addr = post_rd_req_addr;
                o_mem_rd_req_len_bytes = post_rd_req_len;
                o_mem_rd_rsp_ready = post_rd_rsp_ready;
                o_mem_wr_req_valid = post_wr_req_valid;
                o_mem_wr_req_addr = post_wr_req_addr;
                o_mem_wr_req_len_bytes = post_wr_req_len;
                o_mem_wr_data = post_wr_data;
                o_mem_wr_data_valid = post_wr_data_valid;
                o_mem_wr_data_last = post_wr_data_last;
                active_done = post_done;
                active_error = post_error;
            end

            STAGE_MLP_GATE_UP: begin
                o_mem_rd_req_valid = gate_rd_req_valid;
                o_mem_rd_req_addr = gate_rd_req_addr;
                o_mem_rd_req_len_bytes = gate_rd_req_len;
                o_mem_rd_rsp_ready = gate_rd_rsp_ready;
                o_mem_wr_req_valid = gate_wr_req_valid;
                o_mem_wr_req_addr = gate_wr_req_addr;
                o_mem_wr_req_len_bytes = gate_wr_req_len;
                o_mem_wr_data = gate_wr_data;
                o_mem_wr_data_valid = gate_wr_data_valid;
                o_mem_wr_data_last = gate_wr_data_last;
                active_done = gate_done;
                active_error = gate_error;
            end

            STAGE_MLP_SILU_MUL: begin
                o_mem_rd_req_valid = silu_rd_req_valid;
                o_mem_rd_req_addr = silu_rd_req_addr;
                o_mem_rd_req_len_bytes = silu_rd_req_len;
                o_mem_rd_rsp_ready = silu_rd_rsp_ready;
                o_mem_wr_req_valid = silu_wr_req_valid;
                o_mem_wr_req_addr = silu_wr_req_addr;
                o_mem_wr_req_len_bytes = silu_wr_req_len;
                o_mem_wr_data = silu_wr_data;
                o_mem_wr_data_valid = silu_wr_data_valid;
                o_mem_wr_data_last = silu_wr_data_last;
                active_done = silu_done;
                active_error = silu_error;
            end

            STAGE_MLP_DOWN: begin
                o_mem_rd_req_valid = down_rd_req_valid;
                o_mem_rd_req_addr = down_rd_req_addr;
                o_mem_rd_req_len_bytes = down_rd_req_len;
                o_mem_rd_rsp_ready = down_rd_rsp_ready;
                o_mem_wr_req_valid = down_wr_req_valid;
                o_mem_wr_req_addr = down_wr_req_addr;
                o_mem_wr_req_len_bytes = down_wr_req_len;
                o_mem_wr_data = down_wr_data;
                o_mem_wr_data_valid = down_wr_data_valid;
                o_mem_wr_data_last = down_wr_data_last;
                active_done = down_done;
                active_error = down_error;
            end

            STAGE_MLP_RESIDUAL: begin
                o_mem_rd_req_valid = residual_rd_req_valid;
                o_mem_rd_req_addr = residual_rd_req_addr;
                o_mem_rd_req_len_bytes = residual_rd_req_len;
                o_mem_rd_rsp_ready = residual_rd_rsp_ready;
                o_mem_wr_req_valid = residual_wr_req_valid;
                o_mem_wr_req_addr = residual_wr_req_addr;
                o_mem_wr_req_len_bytes = residual_wr_req_len;
                o_mem_wr_data = residual_wr_data;
                o_mem_wr_data_valid = residual_wr_data_valid;
                o_mem_wr_data_last = residual_wr_data_last;
                active_done = residual_done;
                active_error = residual_error;
            end

            default: begin
            end
        endcase
    end

    qmap_post_attention_residual_norm_compute_path post_attention_norm (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(post_start),
        .i_qmap_base_addr(i_post_attn_norm_qmap_base_addr),
        .i_residual_base_override_valid(i_residual_base_override_valid),
        .i_residual_base_override_addr(i_residual_base_override_addr),
        .o_busy(),
        .o_done(post_done),
        .o_error(post_error),
        .o_effective_residual_base_addr(),
        .o_residual_saturation(),
        .o_norm_saturation(),
        .o_residual_count(),
        .o_stage_cycle_count(),
        .o_sum_squares(),
        .o_mean_square(),
        .o_inv_rms(),
        .o_post_hidden_write_word_count(),
        .o_post_norm_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_write_slot_debug(),
        .o_mem_rd_req_valid(post_rd_req_valid),
        .i_mem_rd_req_ready(post_rd_req_ready),
        .o_mem_rd_req_addr(post_rd_req_addr),
        .o_mem_rd_req_len_bytes(post_rd_req_len),
        .i_mem_rd_rsp_valid(post_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(post_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(post_wr_req_valid),
        .i_mem_wr_req_ready(post_wr_req_ready),
        .o_mem_wr_req_addr(post_wr_req_addr),
        .o_mem_wr_req_len_bytes(post_wr_req_len),
        .o_mem_wr_data(post_wr_data),
        .o_mem_wr_data_valid(post_wr_data_valid),
        .i_mem_wr_data_ready(post_wr_data_ready),
        .o_mem_wr_data_last(post_wr_data_last),
        .i_mem_wr_done(post_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(post_active ? i_mem_wr_error : 1'b0)
    );

    qmap_mlp_gate_up_compute_path mlp_gate_up (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(gate_start),
        .i_qmap_base_addr(i_mlp_gate_up_qmap_base_addr),
        .o_busy(),
        .o_done(gate_done),
        .o_error(gate_error),
        .o_saturation(),
        .o_rows_done(),
        .o_last_gate_row_sum_q26(),
        .o_last_up_row_sum_q26(),
        .o_last_gate_output_q12_12(),
        .o_last_up_output_q12_12(),
        .o_gate_write_word_count(),
        .o_up_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_row_index_debug(),
        .o_write_slot_debug(),
        .o_mem_rd_req_valid(gate_rd_req_valid),
        .i_mem_rd_req_ready(gate_rd_req_ready),
        .o_mem_rd_req_addr(gate_rd_req_addr),
        .o_mem_rd_req_len_bytes(gate_rd_req_len),
        .i_mem_rd_rsp_valid(gate_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(gate_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(gate_wr_req_valid),
        .i_mem_wr_req_ready(gate_wr_req_ready),
        .o_mem_wr_req_addr(gate_wr_req_addr),
        .o_mem_wr_req_len_bytes(gate_wr_req_len),
        .o_mem_wr_data(gate_wr_data),
        .o_mem_wr_data_valid(gate_wr_data_valid),
        .i_mem_wr_data_ready(gate_wr_data_ready),
        .o_mem_wr_data_last(gate_wr_data_last),
        .i_mem_wr_done(gate_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(gate_active ? i_mem_wr_error : 1'b0)
    );

    qmap_mlp_silu_mul_compute_path mlp_silu_mul (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(silu_start),
        .i_qmap_base_addr(i_mlp_silu_mul_qmap_base_addr),
        .o_busy(),
        .o_done(silu_done),
        .o_error(silu_error),
        .o_saturation(),
        .o_stage_input_count(),
        .o_stage_output_count(),
        .o_stage_cycle_count(),
        .o_output_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(silu_rd_req_valid),
        .i_mem_rd_req_ready(silu_rd_req_ready),
        .o_mem_rd_req_addr(silu_rd_req_addr),
        .o_mem_rd_req_len_bytes(silu_rd_req_len),
        .i_mem_rd_rsp_valid(silu_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(silu_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(silu_wr_req_valid),
        .i_mem_wr_req_ready(silu_wr_req_ready),
        .o_mem_wr_req_addr(silu_wr_req_addr),
        .o_mem_wr_req_len_bytes(silu_wr_req_len),
        .o_mem_wr_data(silu_wr_data),
        .o_mem_wr_data_valid(silu_wr_data_valid),
        .i_mem_wr_data_ready(silu_wr_data_ready),
        .o_mem_wr_data_last(silu_wr_data_last),
        .i_mem_wr_done(silu_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(silu_active ? i_mem_wr_error : 1'b0)
    );

    qmap_mlp_down_compute_path mlp_down (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(down_start),
        .i_qmap_base_addr(i_mlp_down_qmap_base_addr),
        .o_busy(),
        .o_done(down_done),
        .o_error(down_error),
        .o_saturation(),
        .o_rows_done(),
        .o_last_row_sum_q26(),
        .o_last_output_q12_12(),
        .o_output_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_row_index_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(down_rd_req_valid),
        .i_mem_rd_req_ready(down_rd_req_ready),
        .o_mem_rd_req_addr(down_rd_req_addr),
        .o_mem_rd_req_len_bytes(down_rd_req_len),
        .i_mem_rd_rsp_valid(down_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(down_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(down_wr_req_valid),
        .i_mem_wr_req_ready(down_wr_req_ready),
        .o_mem_wr_req_addr(down_wr_req_addr),
        .o_mem_wr_req_len_bytes(down_wr_req_len),
        .o_mem_wr_data(down_wr_data),
        .o_mem_wr_data_valid(down_wr_data_valid),
        .i_mem_wr_data_ready(down_wr_data_ready),
        .o_mem_wr_data_last(down_wr_data_last),
        .i_mem_wr_done(down_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(down_active ? i_mem_wr_error : 1'b0)
    );

    qmap_mlp_residual_add_compute_path mlp_residual_add (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(residual_start),
        .i_qmap_base_addr(i_mlp_residual_add_qmap_base_addr),
        .i_output_base_override_valid(i_output_base_override_valid),
        .i_output_base_override_addr(i_output_base_override_addr),
        .o_busy(),
        .o_done(residual_done),
        .o_error(residual_error),
        .o_effective_output_base_addr(),
        .o_saturation(),
        .o_output_count(),
        .o_stage_cycle_count(),
        .o_output_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(residual_rd_req_valid),
        .i_mem_rd_req_ready(residual_rd_req_ready),
        .o_mem_rd_req_addr(residual_rd_req_addr),
        .o_mem_rd_req_len_bytes(residual_rd_req_len),
        .i_mem_rd_rsp_valid(residual_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(residual_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(residual_wr_req_valid),
        .i_mem_wr_req_ready(residual_wr_req_ready),
        .o_mem_wr_req_addr(residual_wr_req_addr),
        .o_mem_wr_req_len_bytes(residual_wr_req_len),
        .o_mem_wr_data(residual_wr_data),
        .o_mem_wr_data_valid(residual_wr_data_valid),
        .i_mem_wr_data_ready(residual_wr_data_ready),
        .o_mem_wr_data_last(residual_wr_data_last),
        .i_mem_wr_done(residual_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(residual_active ? i_mem_wr_error : 1'b0)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_stage <= STAGE_POST_ATTN_NORM;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_stage_done_mask <= 5'd0;
            o_stage_error_mask <= 5'd0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count <= 32'd0;
            o_mem_write_req_count <= 32'd0;
            o_mem_write_word_count <= 32'd0;
        end
        else begin
            o_done <= 1'b0;

            if (o_mem_rd_req_valid && i_mem_rd_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end
            if (i_mem_rd_rsp_valid && o_mem_rd_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end
            if (o_mem_wr_req_valid && i_mem_wr_req_ready) begin
                o_mem_write_req_count <= o_mem_write_req_count + 1'b1;
            end
            if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                o_mem_write_word_count <= o_mem_write_word_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        active_stage <= STAGE_POST_ATTN_NORM;
                        o_error <= 1'b0;
                        o_stage_done_mask <= 5'd0;
                        o_stage_error_mask <= 5'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state <= S_STAGE_START;
                    end
                end

                S_STAGE_START: begin
                    state <= S_STAGE_WAIT;
                end

                S_STAGE_WAIT: begin
                    if (active_done) begin
                        if (active_error) begin
                            o_error <= 1'b1;
                            o_stage_error_mask[active_stage] <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            o_stage_done_mask[active_stage] <= 1'b1;
                            if (active_stage == STAGE_MLP_RESIDUAL) begin
                                state <= S_DONE;
                            end
                            else begin
                                active_stage <= active_stage + 1'b1;
                                state <= S_STAGE_START;
                            end
                        end
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    o_error <= 1'b1;
                    state <= S_DONE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
