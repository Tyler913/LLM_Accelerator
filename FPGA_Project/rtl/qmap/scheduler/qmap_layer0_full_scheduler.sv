`default_nettype none

`include "qmap_defs.svh"

// Local Layer 0 full scheduler:
//
//   attention front-end
//     -> attention score/softmax/value
//     -> attention output projection
//     -> post-attention/MLP body scheduler
//
// The child blocks are already QMAP-backed wrappers. This top provides the
// first wider single-control, single-memory-port composition boundary for a
// locally simulated Layer 0 path.
module qmap_layer0_full_scheduler #(
    parameter int ADDR_WIDTH     = 64,
    parameter int MEM_DATA_WIDTH = 32,
    parameter int NUM_LAYERS     = 28,
    parameter int MAX_CONTEXT    = 256,
    parameter int LAYER_INDEX_WIDTH = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int POSITION_WIDTH = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT)
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic                         i_runtime_context_valid,
    input  wire logic [LAYER_INDEX_WIDTH-1:0] i_runtime_layer_id,
    input  wire logic [POSITION_WIDTH-1:0]    i_runtime_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_runtime_kv_cache_base_addr,
    input  wire logic                         i_residual_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_residual_base_override_addr,
    input  wire logic                         i_output_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_base_override_addr,
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
    output logic [2 : 0]                      o_active_stage_debug,
    output logic [7 : 0]                      o_state_debug,
    output logic [3 : 0]                      o_stage_done_mask,
    output logic [3 : 0]                      o_stage_error_mask,
    output logic [4 : 0]                      o_body_stage_done_mask,
    output logic [4 : 0]                      o_body_stage_error_mask,
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

    localparam logic [2 : 0] STAGE_ATTN_FRONTEND    = 3'd0;
    localparam logic [2 : 0] STAGE_ATTN_SCORE_VALUE = 3'd1;
    localparam logic [2 : 0] STAGE_O_PROJ           = 3'd2;
    localparam logic [2 : 0] STAGE_BODY             = 3'd3;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_STAGE_START,
        S_STAGE_WAIT,
        S_DONE
    } state_t;

    state_t state;
    logic [2 : 0] active_stage;

    logic frontend_start;
    logic score_value_start;
    logic o_proj_start;
    logic body_start;

    logic frontend_done;
    logic score_value_done;
    logic o_proj_done;
    logic body_done;

    logic frontend_error;
    logic score_value_error;
    logic o_proj_error;
    logic body_error;

    logic frontend_rd_req_valid;
    logic score_value_rd_req_valid;
    logic o_proj_rd_req_valid;
    logic body_rd_req_valid;
    logic frontend_rd_req_ready;
    logic score_value_rd_req_ready;
    logic o_proj_rd_req_ready;
    logic body_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] frontend_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] score_value_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] o_proj_rd_req_addr;
    logic [ADDR_WIDTH-1 : 0] body_rd_req_addr;
    logic [15 : 0] frontend_rd_req_len;
    logic [15 : 0] score_value_rd_req_len;
    logic [15 : 0] o_proj_rd_req_len;
    logic [15 : 0] body_rd_req_len;
    logic frontend_rd_rsp_ready;
    logic score_value_rd_rsp_ready;
    logic o_proj_rd_rsp_ready;
    logic body_rd_rsp_ready;

    logic frontend_wr_req_valid;
    logic score_value_wr_req_valid;
    logic o_proj_wr_req_valid;
    logic body_wr_req_valid;
    logic frontend_wr_req_ready;
    logic score_value_wr_req_ready;
    logic o_proj_wr_req_ready;
    logic body_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] frontend_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] score_value_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] o_proj_wr_req_addr;
    logic [ADDR_WIDTH-1 : 0] body_wr_req_addr;
    logic [15 : 0] frontend_wr_req_len;
    logic [15 : 0] score_value_wr_req_len;
    logic [15 : 0] o_proj_wr_req_len;
    logic [15 : 0] body_wr_req_len;
    logic [31 : 0] frontend_wr_data;
    logic [31 : 0] score_value_wr_data;
    logic [31 : 0] o_proj_wr_data;
    logic [31 : 0] body_wr_data;
    logic frontend_wr_data_valid;
    logic score_value_wr_data_valid;
    logic o_proj_wr_data_valid;
    logic body_wr_data_valid;
    logic frontend_wr_data_ready;
    logic score_value_wr_data_ready;
    logic o_proj_wr_data_ready;
    logic body_wr_data_ready;
    logic frontend_wr_data_last;
    logic score_value_wr_data_last;
    logic o_proj_wr_data_last;
    logic body_wr_data_last;

    logic frontend_active;
    logic score_value_active;
    logic o_proj_active;
    logic body_active;
    logic active_done;
    logic active_error;

    assign frontend_active = (state != S_IDLE) && (active_stage == STAGE_ATTN_FRONTEND);
    assign score_value_active = (state != S_IDLE) && (active_stage == STAGE_ATTN_SCORE_VALUE);
    assign o_proj_active = (state != S_IDLE) && (active_stage == STAGE_O_PROJ);
    assign body_active = (state != S_IDLE) && (active_stage == STAGE_BODY);

    assign frontend_start = (state == S_STAGE_START) && (active_stage == STAGE_ATTN_FRONTEND);
    assign score_value_start = (state == S_STAGE_START) && (active_stage == STAGE_ATTN_SCORE_VALUE);
    assign o_proj_start = (state == S_STAGE_START) && (active_stage == STAGE_O_PROJ);
    assign body_start = (state == S_STAGE_START) && (active_stage == STAGE_BODY);

    assign frontend_rd_req_ready = frontend_active ? i_mem_rd_req_ready : 1'b0;
    assign score_value_rd_req_ready = score_value_active ? i_mem_rd_req_ready : 1'b0;
    assign o_proj_rd_req_ready = o_proj_active ? i_mem_rd_req_ready : 1'b0;
    assign body_rd_req_ready = body_active ? i_mem_rd_req_ready : 1'b0;

    assign frontend_wr_req_ready = frontend_active ? i_mem_wr_req_ready : 1'b0;
    assign score_value_wr_req_ready = score_value_active ? i_mem_wr_req_ready : 1'b0;
    assign o_proj_wr_req_ready = o_proj_active ? i_mem_wr_req_ready : 1'b0;
    assign body_wr_req_ready = body_active ? i_mem_wr_req_ready : 1'b0;

    assign frontend_wr_data_ready = frontend_active ? i_mem_wr_data_ready : 1'b0;
    assign score_value_wr_data_ready = score_value_active ? i_mem_wr_data_ready : 1'b0;
    assign o_proj_wr_data_ready = o_proj_active ? i_mem_wr_data_ready : 1'b0;
    assign body_wr_data_ready = body_active ? i_mem_wr_data_ready : 1'b0;

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
            STAGE_ATTN_FRONTEND: begin
                o_mem_rd_req_valid = frontend_rd_req_valid;
                o_mem_rd_req_addr = frontend_rd_req_addr;
                o_mem_rd_req_len_bytes = frontend_rd_req_len;
                o_mem_rd_rsp_ready = frontend_rd_rsp_ready;
                o_mem_wr_req_valid = frontend_wr_req_valid;
                o_mem_wr_req_addr = frontend_wr_req_addr;
                o_mem_wr_req_len_bytes = frontend_wr_req_len;
                o_mem_wr_data = frontend_wr_data;
                o_mem_wr_data_valid = frontend_wr_data_valid;
                o_mem_wr_data_last = frontend_wr_data_last;
                active_done = frontend_done;
                active_error = frontend_error;
            end

            STAGE_ATTN_SCORE_VALUE: begin
                o_mem_rd_req_valid = score_value_rd_req_valid;
                o_mem_rd_req_addr = score_value_rd_req_addr;
                o_mem_rd_req_len_bytes = score_value_rd_req_len;
                o_mem_rd_rsp_ready = score_value_rd_rsp_ready;
                o_mem_wr_req_valid = score_value_wr_req_valid;
                o_mem_wr_req_addr = score_value_wr_req_addr;
                o_mem_wr_req_len_bytes = score_value_wr_req_len;
                o_mem_wr_data = score_value_wr_data;
                o_mem_wr_data_valid = score_value_wr_data_valid;
                o_mem_wr_data_last = score_value_wr_data_last;
                active_done = score_value_done;
                active_error = score_value_error;
            end

            STAGE_O_PROJ: begin
                o_mem_rd_req_valid = o_proj_rd_req_valid;
                o_mem_rd_req_addr = o_proj_rd_req_addr;
                o_mem_rd_req_len_bytes = o_proj_rd_req_len;
                o_mem_rd_rsp_ready = o_proj_rd_rsp_ready;
                o_mem_wr_req_valid = o_proj_wr_req_valid;
                o_mem_wr_req_addr = o_proj_wr_req_addr;
                o_mem_wr_req_len_bytes = o_proj_wr_req_len;
                o_mem_wr_data = o_proj_wr_data;
                o_mem_wr_data_valid = o_proj_wr_data_valid;
                o_mem_wr_data_last = o_proj_wr_data_last;
                active_done = o_proj_done;
                active_error = o_proj_error;
            end

            STAGE_BODY: begin
                o_mem_rd_req_valid = body_rd_req_valid;
                o_mem_rd_req_addr = body_rd_req_addr;
                o_mem_rd_req_len_bytes = body_rd_req_len;
                o_mem_rd_rsp_ready = body_rd_rsp_ready;
                o_mem_wr_req_valid = body_wr_req_valid;
                o_mem_wr_req_addr = body_wr_req_addr;
                o_mem_wr_req_len_bytes = body_wr_req_len;
                o_mem_wr_data = body_wr_data;
                o_mem_wr_data_valid = body_wr_data_valid;
                o_mem_wr_data_last = body_wr_data_last;
                active_done = body_done;
                active_error = body_error;
            end

            default: begin
            end
        endcase
    end

    qmap_attention_frontend_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .NUM_LAYERS(NUM_LAYERS),
        .MAX_CONTEXT(MAX_CONTEXT),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .LAYER_INDEX_W(LAYER_INDEX_WIDTH),
        .POSITION_INDEX_W(POSITION_WIDTH)
    ) attention_frontend (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(frontend_start),
        .i_qmap_base_addr(i_attn_frontend_qmap_base_addr),
        .i_runtime_context_valid(i_runtime_context_valid),
        .i_runtime_layer_id(i_runtime_layer_id),
        .i_runtime_position(i_runtime_position),
        .i_runtime_kv_cache_base_addr(i_runtime_kv_cache_base_addr),
        .o_busy(),
        .o_done(frontend_done),
        .o_error(frontend_error),
        .o_saturation(),
        .o_norm_saturation(),
        .o_rope_saturation(),
        .o_cache_write_count(),
        .o_q_rope_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(frontend_rd_req_valid),
        .i_mem_rd_req_ready(frontend_rd_req_ready),
        .o_mem_rd_req_addr(frontend_rd_req_addr),
        .o_mem_rd_req_len_bytes(frontend_rd_req_len),
        .i_mem_rd_rsp_valid(frontend_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(frontend_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(frontend_wr_req_valid),
        .i_mem_wr_req_ready(frontend_wr_req_ready),
        .o_mem_wr_req_addr(frontend_wr_req_addr),
        .o_mem_wr_req_len_bytes(frontend_wr_req_len),
        .o_mem_wr_data(frontend_wr_data),
        .o_mem_wr_data_valid(frontend_wr_data_valid),
        .i_mem_wr_data_ready(frontend_wr_data_ready),
        .o_mem_wr_data_last(frontend_wr_data_last),
        .i_mem_wr_done(frontend_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(frontend_active ? i_mem_wr_error : 1'b0)
    );

    qmap_attention_score_value_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .NUM_LAYERS(NUM_LAYERS),
        .MAX_CONTEXT(MAX_CONTEXT),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .LAYER_INDEX_W(LAYER_INDEX_WIDTH),
        .POSITION_INDEX_W(POSITION_WIDTH)
    ) attention_score_value (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(score_value_start),
        .i_qmap_base_addr(i_attn_score_value_qmap_base_addr),
        .i_runtime_context_valid(i_runtime_context_valid),
        .i_runtime_layer_id(i_runtime_layer_id),
        .i_runtime_position(i_runtime_position),
        .i_runtime_kv_cache_base_addr(i_runtime_kv_cache_base_addr),
        .o_busy(),
        .o_done(score_value_done),
        .o_error(score_value_error),
        .o_saturation(),
        .o_score_count(),
        .o_k_read_count(),
        .o_v_read_count(),
        .o_attn_out_capture_count(),
        .o_attn_out_write_word_count(),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_state_debug(),
        .o_read_slot_debug(),
        .o_mem_rd_req_valid(score_value_rd_req_valid),
        .i_mem_rd_req_ready(score_value_rd_req_ready),
        .o_mem_rd_req_addr(score_value_rd_req_addr),
        .o_mem_rd_req_len_bytes(score_value_rd_req_len),
        .i_mem_rd_rsp_valid(score_value_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(score_value_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(score_value_wr_req_valid),
        .i_mem_wr_req_ready(score_value_wr_req_ready),
        .o_mem_wr_req_addr(score_value_wr_req_addr),
        .o_mem_wr_req_len_bytes(score_value_wr_req_len),
        .o_mem_wr_data(score_value_wr_data),
        .o_mem_wr_data_valid(score_value_wr_data_valid),
        .i_mem_wr_data_ready(score_value_wr_data_ready),
        .o_mem_wr_data_last(score_value_wr_data_last),
        .i_mem_wr_done(score_value_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(score_value_active ? i_mem_wr_error : 1'b0)
    );

    qmap_o_proj_compute_path o_proj (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(o_proj_start),
        .i_qmap_base_addr(i_o_proj_qmap_base_addr),
        .o_busy(),
        .o_done(o_proj_done),
        .o_error(o_proj_error),
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
        .o_mem_rd_req_valid(o_proj_rd_req_valid),
        .i_mem_rd_req_ready(o_proj_rd_req_ready),
        .o_mem_rd_req_addr(o_proj_rd_req_addr),
        .o_mem_rd_req_len_bytes(o_proj_rd_req_len),
        .i_mem_rd_rsp_valid(o_proj_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(o_proj_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(o_proj_wr_req_valid),
        .i_mem_wr_req_ready(o_proj_wr_req_ready),
        .o_mem_wr_req_addr(o_proj_wr_req_addr),
        .o_mem_wr_req_len_bytes(o_proj_wr_req_len),
        .o_mem_wr_data(o_proj_wr_data),
        .o_mem_wr_data_valid(o_proj_wr_data_valid),
        .i_mem_wr_data_ready(o_proj_wr_data_ready),
        .o_mem_wr_data_last(o_proj_wr_data_last),
        .i_mem_wr_done(o_proj_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(o_proj_active ? i_mem_wr_error : 1'b0)
    );

    qmap_layer0_body_scheduler body_scheduler (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(body_start),
        .i_residual_base_override_valid(i_residual_base_override_valid),
        .i_residual_base_override_addr(i_residual_base_override_addr),
        .i_output_base_override_valid(i_output_base_override_valid),
        .i_output_base_override_addr(i_output_base_override_addr),
        .i_post_attn_norm_qmap_base_addr(i_post_attn_norm_qmap_base_addr),
        .i_mlp_gate_up_qmap_base_addr(i_mlp_gate_up_qmap_base_addr),
        .i_mlp_silu_mul_qmap_base_addr(i_mlp_silu_mul_qmap_base_addr),
        .i_mlp_down_qmap_base_addr(i_mlp_down_qmap_base_addr),
        .i_mlp_residual_add_qmap_base_addr(i_mlp_residual_add_qmap_base_addr),
        .o_busy(),
        .o_done(body_done),
        .o_error(body_error),
        .o_active_stage_debug(),
        .o_state_debug(),
        .o_stage_done_mask(o_body_stage_done_mask),
        .o_stage_error_mask(o_body_stage_error_mask),
        .o_mem_read_burst_count(),
        .o_mem_read_word_count(),
        .o_mem_write_req_count(),
        .o_mem_write_word_count(),
        .o_mem_rd_req_valid(body_rd_req_valid),
        .i_mem_rd_req_ready(body_rd_req_ready),
        .o_mem_rd_req_addr(body_rd_req_addr),
        .o_mem_rd_req_len_bytes(body_rd_req_len),
        .i_mem_rd_rsp_valid(body_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(body_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(body_wr_req_valid),
        .i_mem_wr_req_ready(body_wr_req_ready),
        .o_mem_wr_req_addr(body_wr_req_addr),
        .o_mem_wr_req_len_bytes(body_wr_req_len),
        .o_mem_wr_data(body_wr_data),
        .o_mem_wr_data_valid(body_wr_data_valid),
        .i_mem_wr_data_ready(body_wr_data_ready),
        .o_mem_wr_data_last(body_wr_data_last),
        .i_mem_wr_done(body_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(body_active ? i_mem_wr_error : 1'b0)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_stage <= STAGE_ATTN_FRONTEND;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_stage_done_mask <= 4'd0;
            o_stage_error_mask <= 4'd0;
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
                        active_stage <= STAGE_ATTN_FRONTEND;
                        o_error <= 1'b0;
                        o_stage_done_mask <= 4'd0;
                        o_stage_error_mask <= 4'd0;
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
                            if (active_stage == STAGE_BODY) begin
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
