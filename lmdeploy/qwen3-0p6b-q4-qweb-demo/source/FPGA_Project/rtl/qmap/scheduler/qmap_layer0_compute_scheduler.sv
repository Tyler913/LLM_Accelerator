`default_nettype none

`include "qmap_defs.svh"

// Local Layer 0 compute scheduler:
//
//   QKV projection
//     -> attention front-end
//     -> attention score/softmax/value
//     -> attention output projection
//     -> post-attention/MLP body scheduler
//
// This is the first local simulation boundary that starts from the QKV runtime
// packet and ends at the Layer 0 `layer_out[1024]` write-back. It still exposes
// the project-local memory request/write streams so the testbench can audit
// ordering, backpressure, and chained buffer dependencies directly.
module qmap_layer0_compute_scheduler #(
    parameter int ADDR_WIDTH     = 64,
    parameter int MEM_DATA_WIDTH = 32,
    parameter int NUM_LAYERS     = 28,
    parameter int MAX_CONTEXT    = 256,
    parameter int LAYER_INDEX_WIDTH = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int POSITION_WIDTH = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int INPUT_SIZE     = 1024,
    parameter int GROUP_SIZE     = 64,
    parameter int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL = 1,
    parameter int BODY_GROUP_PARALLEL = 1,
    parameter int ACT_WIDTH      = 24,
    parameter int ACT_FRAC       = 12,
    parameter int WEIGHT_WIDTH   = 4,
    parameter int SCALE_WIDTH    = 16,
    parameter int SCALE_FRAC     = 14,
    parameter int PARTIAL_WIDTH  = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH   = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH  = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic                         i_runtime_context_valid,
    input  wire logic [LAYER_INDEX_WIDTH-1:0] i_runtime_layer_id,
    input  wire logic [POSITION_WIDTH-1:0]    i_runtime_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_runtime_kv_cache_base_addr,
    input  wire logic                         i_hidden_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_hidden_base_override_addr,
    input  wire logic                         i_residual_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_residual_base_override_addr,
    input  wire logic                         i_output_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_base_override_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_input_norm_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qkv_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_attn_frontend_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_attn_score_value_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_o_proj_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_post_attn_norm_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_gate_up_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_silu_mul_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_down_qmap_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_mlp_residual_add_qmap_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [1 : 0]                      o_active_stage_debug,
    output logic [7 : 0]                      o_state_debug,
    output logic [1 : 0]                      o_stage_done_mask,
    output logic [1 : 0]                      o_stage_error_mask,
    output logic [3 : 0]                      o_layer0_full_stage_done_mask,
    output logic [3 : 0]                      o_layer0_full_stage_error_mask,
    output logic [4 : 0]                      o_body_stage_done_mask,
    output logic [4 : 0]                      o_body_stage_error_mask,
    output logic                              o_input_norm_done,
    output logic                              o_input_norm_error,
    output logic                              o_input_norm_saturation,
    output logic [31 : 0]                     o_input_norm_write_word_count,
    output logic [31 : 0]                     o_qkv_rows_done,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_qkv_last_row_sum_q26,
    output logic signed [31 : 0]              o_qkv_last_output_q12_12,
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

    localparam logic [1 : 0] STAGE_INPUT_NORM = 2'd0;
    localparam logic [1 : 0] STAGE_QKV        = 2'd1;
    localparam logic [1 : 0] STAGE_LAYER      = 2'd2;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_STAGE_START,
        S_STAGE_WAIT,
        S_DONE
    } state_t;

    state_t state;
    logic [1 : 0] active_stage;

    logic input_norm_start;
    logic qkv_start;
    logic layer_start;

    logic input_norm_done;
    logic input_norm_error;
    logic input_norm_saturation;
    logic [31 : 0] input_norm_write_word_count;
    logic [31 : 0] input_norm_read_burst_count;
    logic [31 : 0] input_norm_read_word_count;
    logic [31 : 0] input_norm_write_req_count;
    logic [31 : 0] input_norm_mem_write_word_count;
    logic qkv_done;
    logic qkv_error;
    logic layer_done;
    logic layer_error;

    logic input_norm_rd_req_valid;
    logic input_norm_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] input_norm_rd_req_addr;
    logic [15 : 0] input_norm_rd_req_len;
    logic input_norm_rd_rsp_ready;

    logic input_norm_wr_req_valid;
    logic input_norm_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] input_norm_wr_req_addr;
    logic [15 : 0] input_norm_wr_req_len;
    logic [31 : 0] input_norm_wr_data;
    logic input_norm_wr_data_valid;
    logic input_norm_wr_data_ready;
    logic input_norm_wr_data_last;

    logic qkv_rd_req_valid;
    logic qkv_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] qkv_rd_req_addr;
    logic [15 : 0] qkv_rd_req_len;
    logic qkv_rd_rsp_ready;

    logic qkv_wr_req_valid;
    logic qkv_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] qkv_wr_req_addr;
    logic [15 : 0] qkv_wr_req_len;
    logic [31 : 0] qkv_wr_data;
    logic qkv_wr_data_valid;
    logic qkv_wr_data_ready;
    logic qkv_wr_data_last;

    logic layer_rd_req_valid;
    logic layer_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] layer_rd_req_addr;
    logic [15 : 0] layer_rd_req_len;
    logic layer_rd_rsp_ready;

    logic layer_wr_req_valid;
    logic layer_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] layer_wr_req_addr;
    logic [15 : 0] layer_wr_req_len;
    logic [31 : 0] layer_wr_data;
    logic layer_wr_data_valid;
    logic layer_wr_data_ready;
    logic layer_wr_data_last;

    logic [3 : 0] layer_stage_done_mask;
    logic [3 : 0] layer_stage_error_mask;
    logic [4 : 0] layer_body_stage_done_mask;
    logic [4 : 0] layer_body_stage_error_mask;
    logic [31 : 0] layer_read_burst_count;
    logic [31 : 0] layer_read_word_count;
    logic [31 : 0] layer_write_req_count;
    logic [31 : 0] layer_write_word_count;

    logic input_norm_active;
    logic qkv_active;
    logic layer_active;
    logic active_done;
    logic active_error;

    assign input_norm_active = (state != S_IDLE) && (active_stage == STAGE_INPUT_NORM);
    assign qkv_active = (state != S_IDLE) && (active_stage == STAGE_QKV);
    assign layer_active = (state != S_IDLE) && (active_stage == STAGE_LAYER);

    assign input_norm_start = (state == S_STAGE_START) && (active_stage == STAGE_INPUT_NORM);
    assign qkv_start = (state == S_STAGE_START) && (active_stage == STAGE_QKV);
    assign layer_start = (state == S_STAGE_START) && (active_stage == STAGE_LAYER);

    assign input_norm_rd_req_ready = input_norm_active ? i_mem_rd_req_ready : 1'b0;
    assign qkv_rd_req_ready = qkv_active ? i_mem_rd_req_ready : 1'b0;
    assign layer_rd_req_ready = layer_active ? i_mem_rd_req_ready : 1'b0;

    assign input_norm_wr_req_ready = input_norm_active ? i_mem_wr_req_ready : 1'b0;
    assign qkv_wr_req_ready = qkv_active ? i_mem_wr_req_ready : 1'b0;
    assign layer_wr_req_ready = layer_active ? i_mem_wr_req_ready : 1'b0;

    assign input_norm_wr_data_ready = input_norm_active ? i_mem_wr_data_ready : 1'b0;
    assign qkv_wr_data_ready = qkv_active ? i_mem_wr_data_ready : 1'b0;
    assign layer_wr_data_ready = layer_active ? i_mem_wr_data_ready : 1'b0;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_active_stage_debug = active_stage;
    assign o_state_debug = {6'd0, state};
    assign o_layer0_full_stage_done_mask = layer_stage_done_mask;
    assign o_layer0_full_stage_error_mask = layer_stage_error_mask;
    assign o_body_stage_done_mask = layer_body_stage_done_mask;
    assign o_body_stage_error_mask = layer_body_stage_error_mask;

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
            STAGE_INPUT_NORM: begin
                o_mem_rd_req_valid = input_norm_rd_req_valid;
                o_mem_rd_req_addr = input_norm_rd_req_addr;
                o_mem_rd_req_len_bytes = input_norm_rd_req_len;
                o_mem_rd_rsp_ready = input_norm_rd_rsp_ready;
                o_mem_wr_req_valid = input_norm_wr_req_valid;
                o_mem_wr_req_addr = input_norm_wr_req_addr;
                o_mem_wr_req_len_bytes = input_norm_wr_req_len;
                o_mem_wr_data = input_norm_wr_data;
                o_mem_wr_data_valid = input_norm_wr_data_valid;
                o_mem_wr_data_last = input_norm_wr_data_last;
                active_done = input_norm_done;
                active_error = input_norm_error;
            end

            STAGE_QKV: begin
                o_mem_rd_req_valid = qkv_rd_req_valid;
                o_mem_rd_req_addr = qkv_rd_req_addr;
                o_mem_rd_req_len_bytes = qkv_rd_req_len;
                o_mem_rd_rsp_ready = qkv_rd_rsp_ready;
                o_mem_wr_req_valid = qkv_wr_req_valid;
                o_mem_wr_req_addr = qkv_wr_req_addr;
                o_mem_wr_req_len_bytes = qkv_wr_req_len;
                o_mem_wr_data = qkv_wr_data;
                o_mem_wr_data_valid = qkv_wr_data_valid;
                o_mem_wr_data_last = qkv_wr_data_last;
                active_done = qkv_done;
                active_error = qkv_error;
            end

            STAGE_LAYER: begin
                o_mem_rd_req_valid = layer_rd_req_valid;
                o_mem_rd_req_addr = layer_rd_req_addr;
                o_mem_rd_req_len_bytes = layer_rd_req_len;
                o_mem_rd_rsp_ready = layer_rd_rsp_ready;
                o_mem_wr_req_valid = layer_wr_req_valid;
                o_mem_wr_req_addr = layer_wr_req_addr;
                o_mem_wr_req_len_bytes = layer_wr_req_len;
                o_mem_wr_data = layer_wr_data;
                o_mem_wr_data_valid = layer_wr_data_valid;
                o_mem_wr_data_last = layer_wr_data_last;
                active_done = layer_done;
                active_error = layer_error;
            end

            default: begin
            end
        endcase
    end

    qmap_input_rmsnorm_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(`QMAP_INPUT_NORM_DESCRIPTOR_COUNT),
        .INPUT_SIZE(INPUT_SIZE),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .MAX_READ_BYTES(`QMAP_INPUT_NORM_MAX_READ_BYTES)
    ) input_rmsnorm (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(input_norm_start),
        .i_qmap_base_addr(i_input_norm_qmap_base_addr),
        .i_hidden_base_override_valid(i_hidden_base_override_valid),
        .i_hidden_base_override_addr(i_hidden_base_override_addr),
        .o_busy(),
        .o_done(input_norm_done),
        .o_error(input_norm_error),
        .o_effective_hidden_base_addr(),
        .o_norm_saturation(input_norm_saturation),
        .o_norm_cycle_count(),
        .o_sum_squares(),
        .o_mean_square(),
        .o_inv_rms(),
        .o_norm_write_word_count(input_norm_write_word_count),
        .o_mem_read_burst_count(input_norm_read_burst_count),
        .o_mem_read_word_count(input_norm_read_word_count),
        .o_mem_write_req_count(input_norm_write_req_count),
        .o_mem_write_word_count(input_norm_mem_write_word_count),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(input_norm_rd_req_valid),
        .i_mem_rd_req_ready(input_norm_rd_req_ready),
        .o_mem_rd_req_addr(input_norm_rd_req_addr),
        .o_mem_rd_req_len_bytes(input_norm_rd_req_len),
        .i_mem_rd_rsp_valid(input_norm_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(input_norm_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(input_norm_active ? i_mem_rd_rsp_last : 1'b0),
        .o_mem_wr_req_valid(input_norm_wr_req_valid),
        .i_mem_wr_req_ready(input_norm_wr_req_ready),
        .o_mem_wr_req_addr(input_norm_wr_req_addr),
        .o_mem_wr_req_len_bytes(input_norm_wr_req_len),
        .o_mem_wr_data(input_norm_wr_data),
        .o_mem_wr_data_valid(input_norm_wr_data_valid),
        .i_mem_wr_data_ready(input_norm_wr_data_ready),
        .o_mem_wr_data_last(input_norm_wr_data_last),
        .i_mem_wr_done(input_norm_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(input_norm_active ? i_mem_wr_error : 1'b0)
    );

    qmap_qkv_projection_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH)
    ) qkv_projection (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(qkv_start),
        .i_qmap_base_addr(i_qkv_qmap_base_addr),
        .o_busy(),
        .o_done(qkv_done),
        .o_error(qkv_error),
        .o_rows_done(o_qkv_rows_done),
        .o_last_row_sum_q26(o_qkv_last_row_sum_q26),
        .o_last_output_q12_12(o_qkv_last_output_q12_12),
        .o_mem_rd_req_valid(qkv_rd_req_valid),
        .i_mem_rd_req_ready(qkv_rd_req_ready),
        .o_mem_rd_req_addr(qkv_rd_req_addr),
        .o_mem_rd_req_len_bytes(qkv_rd_req_len),
        .i_mem_rd_rsp_valid(qkv_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(qkv_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(qkv_active ? i_mem_rd_rsp_last : 1'b0),
        .o_mem_wr_req_valid(qkv_wr_req_valid),
        .i_mem_wr_req_ready(qkv_wr_req_ready),
        .o_mem_wr_req_addr(qkv_wr_req_addr),
        .o_mem_wr_req_len_bytes(qkv_wr_req_len),
        .o_mem_wr_data(qkv_wr_data),
        .o_mem_wr_data_valid(qkv_wr_data_valid),
        .i_mem_wr_data_ready(qkv_wr_data_ready),
        .o_mem_wr_data_last(qkv_wr_data_last),
        .i_mem_wr_done(qkv_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(qkv_active ? i_mem_wr_error : 1'b0)
    );

    qmap_layer0_full_scheduler #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .NUM_LAYERS(NUM_LAYERS),
        .MAX_CONTEXT(MAX_CONTEXT),
        .LAYER_INDEX_WIDTH(LAYER_INDEX_WIDTH),
        .POSITION_WIDTH(POSITION_WIDTH),
        .GROUP_PARALLEL(BODY_GROUP_PARALLEL)
    ) layer0_full (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(layer_start),
        .i_runtime_context_valid(i_runtime_context_valid),
        .i_runtime_layer_id(i_runtime_layer_id),
        .i_runtime_position(i_runtime_position),
        .i_runtime_kv_cache_base_addr(i_runtime_kv_cache_base_addr),
        .i_residual_base_override_valid(i_residual_base_override_valid),
        .i_residual_base_override_addr(i_residual_base_override_addr),
        .i_output_base_override_valid(i_output_base_override_valid),
        .i_output_base_override_addr(i_output_base_override_addr),
        .i_attn_frontend_qmap_base_addr(i_attn_frontend_qmap_base_addr),
        .i_attn_score_value_qmap_base_addr(i_attn_score_value_qmap_base_addr),
        .i_o_proj_qmap_base_addr(i_o_proj_qmap_base_addr),
        .i_post_attn_norm_qmap_base_addr(i_post_attn_norm_qmap_base_addr),
        .i_mlp_gate_up_qmap_base_addr(i_mlp_gate_up_qmap_base_addr),
        .i_mlp_silu_mul_qmap_base_addr(i_mlp_silu_mul_qmap_base_addr),
        .i_mlp_down_qmap_base_addr(i_mlp_down_qmap_base_addr),
        .i_mlp_residual_add_qmap_base_addr(i_mlp_residual_add_qmap_base_addr),
        .o_busy(),
        .o_done(layer_done),
        .o_error(layer_error),
        .o_active_stage_debug(),
        .o_state_debug(),
        .o_stage_done_mask(layer_stage_done_mask),
        .o_stage_error_mask(layer_stage_error_mask),
        .o_body_stage_done_mask(layer_body_stage_done_mask),
        .o_body_stage_error_mask(layer_body_stage_error_mask),
        .o_mem_read_burst_count(layer_read_burst_count),
        .o_mem_read_word_count(layer_read_word_count),
        .o_mem_write_req_count(layer_write_req_count),
        .o_mem_write_word_count(layer_write_word_count),
        .o_mem_rd_req_valid(layer_rd_req_valid),
        .i_mem_rd_req_ready(layer_rd_req_ready),
        .o_mem_rd_req_addr(layer_rd_req_addr),
        .o_mem_rd_req_len_bytes(layer_rd_req_len),
        .i_mem_rd_rsp_valid(layer_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(layer_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(layer_active ? i_mem_rd_rsp_last : 1'b0),
        .o_mem_wr_req_valid(layer_wr_req_valid),
        .i_mem_wr_req_ready(layer_wr_req_ready),
        .o_mem_wr_req_addr(layer_wr_req_addr),
        .o_mem_wr_req_len_bytes(layer_wr_req_len),
        .o_mem_wr_data(layer_wr_data),
        .o_mem_wr_data_valid(layer_wr_data_valid),
        .i_mem_wr_data_ready(layer_wr_data_ready),
        .o_mem_wr_data_last(layer_wr_data_last),
        .i_mem_wr_done(layer_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(layer_active ? i_mem_wr_error : 1'b0)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_stage <= STAGE_QKV;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_input_norm_done <= 1'b0;
            o_input_norm_error <= 1'b0;
            o_input_norm_saturation <= 1'b0;
            o_input_norm_write_word_count <= 32'd0;
            o_stage_done_mask <= 2'd0;
            o_stage_error_mask <= 2'd0;
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
                        active_stage <= (i_input_norm_qmap_base_addr != '0) ? STAGE_INPUT_NORM : STAGE_QKV;
                        o_error <= 1'b0;
                        o_input_norm_done <= 1'b0;
                        o_input_norm_error <= 1'b0;
                        o_input_norm_saturation <= 1'b0;
                        o_input_norm_write_word_count <= 32'd0;
                        o_stage_done_mask <= 2'd0;
                        o_stage_error_mask <= 2'd0;
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
                            if (active_stage == STAGE_INPUT_NORM) begin
                                o_input_norm_error <= 1'b1;
                            end
                            else if (active_stage == STAGE_QKV) begin
                                o_stage_error_mask[0] <= 1'b1;
                            end
                            else begin
                                o_stage_error_mask[1] <= 1'b1;
                            end
                            state <= S_DONE;
                        end
                        else begin
                            if (active_stage == STAGE_INPUT_NORM) begin
                                o_input_norm_done <= 1'b1;
                                o_input_norm_saturation <= input_norm_saturation;
                                o_input_norm_write_word_count <= input_norm_write_word_count;
                                active_stage <= STAGE_QKV;
                                state <= S_STAGE_START;
                            end
                            else if (active_stage == STAGE_QKV) begin
                                o_stage_done_mask[0] <= 1'b1;
                                active_stage <= STAGE_LAYER;
                                state <= S_STAGE_START;
                            end
                            else begin
                                o_stage_done_mask[1] <= 1'b1;
                                state <= S_DONE;
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
