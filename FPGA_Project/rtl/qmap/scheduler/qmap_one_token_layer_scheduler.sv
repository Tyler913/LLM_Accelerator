`default_nettype none

`include "qmap_defs.svh"

// First local one-token/layer-loop boundary.
//
// The only implemented compute primitive today is the already validated Layer 0
// compute scheduler:
//
//   QKV projection -> Layer 0 attention/body scheduler -> layer_out[1024]
//
// This wrapper deliberately exposes the eventual reusable loop contract
// (layer index/count, hidden buffer bases, KV cache base, token position, and
// one shared memory port) while selecting per-layer QMAP packet bases from
// flattened base-address tables. Multi-layer local simulations can now advance
// through the requested layer range by reusing that compute primitive with the
// table entry selected for the active layer.
module qmap_one_token_layer_scheduler #(
    parameter int ADDR_WIDTH        = 64,
    parameter int MEM_DATA_WIDTH    = 32,
    parameter int MAX_LAYERS        = 28,
    parameter int MAX_CONTEXT       = 256,
    parameter int LAYER_INDEX_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS),
    parameter int LAYER_COUNT_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS + 1),
    parameter int POSITION_WIDTH    = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int INPUT_SIZE        = 1024,
    parameter int GROUP_SIZE        = 64,
    parameter int GROUP_COUNT       = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL    = 4,
    parameter int ACT_WIDTH         = 24,
    parameter int ACT_FRAC          = 12,
    parameter int WEIGHT_WIDTH      = 4,
    parameter int SCALE_WIDTH       = 16,
    parameter int SCALE_FRAC        = 14,
    parameter int PARTIAL_WIDTH     = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH      = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH     = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [LAYER_INDEX_WIDTH-1:0] i_layer_start_index,
    input  wire logic [LAYER_COUNT_WIDTH-1:0] i_layer_count,
    input  wire logic [POSITION_WIDTH-1:0]    i_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_input_hidden_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_hidden_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_kv_cache_base_addr,

    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_qkv_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_input_norm_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_attn_frontend_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_attn_score_value_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_o_proj_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_post_attn_norm_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_mlp_gate_up_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_mlp_silu_mul_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_mlp_down_qmap_base_addr_table,
    input  wire logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] i_mlp_residual_add_qmap_base_addr_table,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [7 : 0]                      o_state_debug,
    output logic [LAYER_INDEX_WIDTH-1:0]      o_active_layer_index,
    output logic [LAYER_COUNT_WIDTH-1:0]      o_layers_started,
    output logic [LAYER_COUNT_WIDTH-1:0]      o_layers_completed,
    output logic [MAX_LAYERS-1:0]             o_layer_done_mask,
    output logic [MAX_LAYERS-1:0]             o_layer_error_mask,
    output logic [ADDR_WIDTH-1:0]             o_last_layer_output_base_addr,

    output logic [1 : 0]                      o_layer0_active_stage_debug,
    output logic [7 : 0]                      o_layer0_state_debug,
    output logic [1 : 0]                      o_layer0_stage_done_mask,
    output logic [1 : 0]                      o_layer0_stage_error_mask,
    output logic [3 : 0]                      o_layer0_full_stage_done_mask,
    output logic [3 : 0]                      o_layer0_full_stage_error_mask,
    output logic [4 : 0]                      o_body_stage_done_mask,
    output logic [4 : 0]                      o_body_stage_error_mask,
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

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_VALIDATE,
        S_LAYER_START,
        S_LAYER_WAIT,
        S_DONE
    } state_t;

    localparam logic [LAYER_COUNT_WIDTH-1:0] ONE_LAYER_COUNT =
        {{(LAYER_COUNT_WIDTH-1){1'b0}}, 1'b1};
    localparam logic [LAYER_INDEX_WIDTH-1:0] ONE_LAYER_INDEX =
        {{(LAYER_INDEX_WIDTH-1){1'b0}}, 1'b1};
    localparam logic [POSITION_WIDTH-1:0] MAX_POSITION_INDEX =
        MAX_CONTEXT[POSITION_WIDTH-1:0] - {{(POSITION_WIDTH-1){1'b0}}, 1'b1};
    localparam logic [LAYER_INDEX_WIDTH-1:0] MAX_LAYER_INDEX =
        MAX_LAYERS[LAYER_INDEX_WIDTH-1:0] - {{(LAYER_INDEX_WIDTH-1){1'b0}}, 1'b1};
    localparam logic [ADDR_WIDTH-1:0] MLP_RESIDUAL_OUTPUT_OFFSET =
        `QMAP_MLP_RESIDUAL_ADD_OUTPUT_OFFSET;

    state_t state;

    function automatic logic [ADDR_WIDTH-1:0] select_layer_base_addr;
        input logic [MAX_LAYERS*ADDR_WIDTH-1:0] base_table;
        input logic [LAYER_INDEX_WIDTH-1:0] layer_index;
        integer idx;
        begin
            select_layer_base_addr = '0;
            for (idx = 0 ; idx < MAX_LAYERS ; idx = idx + 1) begin
                if (layer_index == idx[LAYER_INDEX_WIDTH-1:0]) begin
                    select_layer_base_addr = base_table[idx*ADDR_WIDTH +: ADDR_WIDTH];
                end
            end
        end
    endfunction

    function automatic logic layer_base_has_zero;
        input logic [LAYER_INDEX_WIDTH-1:0] layer_index;
        begin
            layer_base_has_zero =
                (select_layer_base_addr(i_qkv_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_attn_frontend_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_attn_score_value_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_o_proj_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_post_attn_norm_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_mlp_gate_up_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_mlp_silu_mul_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_mlp_down_qmap_base_addr_table, layer_index) == '0) ||
                (select_layer_base_addr(i_mlp_residual_add_qmap_base_addr_table, layer_index) == '0);
        end
    endfunction

    logic layer0_start;
    logic layer0_done;
    logic layer0_error;
    logic layer0_busy;
    logic [1 : 0] layer0_active_stage_debug;
    logic [7 : 0] layer0_state_debug;
    logic [31 : 0] layer0_read_burst_count;
    logic [31 : 0] layer0_read_word_count;
    logic [31 : 0] layer0_write_req_count;
    logic [31 : 0] layer0_write_word_count;

    logic layer0_rd_req_valid;
    logic layer0_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] layer0_rd_req_addr;
    logic [15 : 0] layer0_rd_req_len_bytes;
    logic layer0_rd_rsp_ready;

    logic layer0_wr_req_valid;
    logic layer0_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] layer0_wr_req_addr;
    logic [15 : 0] layer0_wr_req_len_bytes;
    logic [31 : 0] layer0_wr_data;
    logic layer0_wr_data_valid;
    logic layer0_wr_data_ready;
    logic layer0_wr_data_last;

    logic validate_error;
    logic selected_base_error;
    logic validation_range_error;
    logic validation_base_error;
    logic validation_found_base_error;
    logic layer0_active;
    logic [LAYER_INDEX_WIDTH-1:0] active_layer_index_reg;
    logic [LAYER_INDEX_WIDTH-1:0] validation_error_layer_index;
    logic [ADDR_WIDTH-1:0] selected_input_norm_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_qkv_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_attn_frontend_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_attn_score_value_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_o_proj_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_post_attn_norm_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_mlp_gate_up_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_mlp_silu_mul_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_mlp_down_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_mlp_residual_add_qmap_base_addr;
    logic [ADDR_WIDTH-1:0] selected_layer_output_base_addr;
    integer validation_idx;
    integer validation_start_int;
    integer validation_stop_int;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {5'd0, state};
    assign o_active_layer_index = active_layer_index_reg;

    assign layer0_start = (state == S_LAYER_START);
    assign layer0_active = (state == S_LAYER_START) || (state == S_LAYER_WAIT);

    assign selected_qkv_qmap_base_addr =
        select_layer_base_addr(i_qkv_qmap_base_addr_table, active_layer_index_reg);
    assign selected_input_norm_qmap_base_addr =
        select_layer_base_addr(i_input_norm_qmap_base_addr_table, active_layer_index_reg);
    assign selected_attn_frontend_qmap_base_addr =
        select_layer_base_addr(i_attn_frontend_qmap_base_addr_table, active_layer_index_reg);
    assign selected_attn_score_value_qmap_base_addr =
        select_layer_base_addr(i_attn_score_value_qmap_base_addr_table, active_layer_index_reg);
    assign selected_o_proj_qmap_base_addr =
        select_layer_base_addr(i_o_proj_qmap_base_addr_table, active_layer_index_reg);
    assign selected_post_attn_norm_qmap_base_addr =
        select_layer_base_addr(i_post_attn_norm_qmap_base_addr_table, active_layer_index_reg);
    assign selected_mlp_gate_up_qmap_base_addr =
        select_layer_base_addr(i_mlp_gate_up_qmap_base_addr_table, active_layer_index_reg);
    assign selected_mlp_silu_mul_qmap_base_addr =
        select_layer_base_addr(i_mlp_silu_mul_qmap_base_addr_table, active_layer_index_reg);
    assign selected_mlp_down_qmap_base_addr =
        select_layer_base_addr(i_mlp_down_qmap_base_addr_table, active_layer_index_reg);
    assign selected_mlp_residual_add_qmap_base_addr =
        select_layer_base_addr(i_mlp_residual_add_qmap_base_addr_table, active_layer_index_reg);
    assign selected_layer_output_base_addr =
        selected_mlp_residual_add_qmap_base_addr + MLP_RESIDUAL_OUTPUT_OFFSET;

    assign selected_base_error =
        (selected_qkv_qmap_base_addr == '0) ||
        (selected_attn_frontend_qmap_base_addr == '0) ||
        (selected_attn_score_value_qmap_base_addr == '0) ||
        (selected_o_proj_qmap_base_addr == '0) ||
        (selected_post_attn_norm_qmap_base_addr == '0) ||
        (selected_mlp_gate_up_qmap_base_addr == '0) ||
        (selected_mlp_silu_mul_qmap_base_addr == '0) ||
        (selected_mlp_down_qmap_base_addr == '0) ||
        (selected_mlp_residual_add_qmap_base_addr == '0);

    always_comb begin
        validation_start_int = i_layer_start_index;
        validation_stop_int = i_layer_start_index + i_layer_count;
        validation_range_error =
            (i_layer_count == '0) ||
            (validation_start_int >= MAX_LAYERS) ||
            (validation_stop_int > MAX_LAYERS);
        validation_base_error = 1'b0;
        validation_found_base_error = 1'b0;
        validation_error_layer_index =
            (validation_start_int >= MAX_LAYERS) ? MAX_LAYER_INDEX : i_layer_start_index;

        for (validation_idx = 0 ; validation_idx < MAX_LAYERS ; validation_idx = validation_idx + 1) begin
            if (!validation_range_error &&
                (validation_idx >= validation_start_int) &&
                (validation_idx < validation_stop_int) &&
                layer_base_has_zero(validation_idx[LAYER_INDEX_WIDTH-1:0])) begin
                validation_base_error = 1'b1;
                if (!validation_found_base_error) begin
                    validation_found_base_error = 1'b1;
                    validation_error_layer_index = validation_idx[LAYER_INDEX_WIDTH-1:0];
                end
            end
        end
    end

    assign validate_error =
        validation_range_error ||
        (i_position > MAX_POSITION_INDEX) ||
        (i_input_hidden_base_addr == '0) ||
        (i_output_hidden_base_addr == '0) ||
        (i_kv_cache_base_addr == '0) ||
        selected_base_error ||
        validation_base_error;

    assign layer0_rd_req_ready = layer0_active ? i_mem_rd_req_ready : 1'b0;
    assign layer0_wr_req_ready = layer0_active ? i_mem_wr_req_ready : 1'b0;
    assign layer0_wr_data_ready = layer0_active ? i_mem_wr_data_ready : 1'b0;

    assign o_mem_rd_req_valid = layer0_active ? layer0_rd_req_valid : 1'b0;
    assign o_mem_rd_req_addr = layer0_active ? layer0_rd_req_addr : '0;
    assign o_mem_rd_req_len_bytes = layer0_active ? layer0_rd_req_len_bytes : 16'd0;
    assign o_mem_rd_rsp_ready = layer0_active ? layer0_rd_rsp_ready : 1'b0;

    assign o_mem_wr_req_valid = layer0_active ? layer0_wr_req_valid : 1'b0;
    assign o_mem_wr_req_addr = layer0_active ? layer0_wr_req_addr : '0;
    assign o_mem_wr_req_len_bytes = layer0_active ? layer0_wr_req_len_bytes : 16'd0;
    assign o_mem_wr_data = layer0_active ? layer0_wr_data : 32'd0;
    assign o_mem_wr_data_valid = layer0_active ? layer0_wr_data_valid : 1'b0;
    assign o_mem_wr_data_last = layer0_active ? layer0_wr_data_last : 1'b0;

    assign o_layer0_active_stage_debug = layer0_active_stage_debug;
    assign o_layer0_state_debug = layer0_state_debug;

    qmap_layer0_compute_scheduler #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
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
    ) layer0_compute (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(layer0_start),
        .i_input_norm_qmap_base_addr(selected_input_norm_qmap_base_addr),
        .i_qkv_qmap_base_addr(selected_qkv_qmap_base_addr),
        .i_attn_frontend_qmap_base_addr(selected_attn_frontend_qmap_base_addr),
        .i_attn_score_value_qmap_base_addr(selected_attn_score_value_qmap_base_addr),
        .i_o_proj_qmap_base_addr(selected_o_proj_qmap_base_addr),
        .i_post_attn_norm_qmap_base_addr(selected_post_attn_norm_qmap_base_addr),
        .i_mlp_gate_up_qmap_base_addr(selected_mlp_gate_up_qmap_base_addr),
        .i_mlp_silu_mul_qmap_base_addr(selected_mlp_silu_mul_qmap_base_addr),
        .i_mlp_down_qmap_base_addr(selected_mlp_down_qmap_base_addr),
        .i_mlp_residual_add_qmap_base_addr(selected_mlp_residual_add_qmap_base_addr),
        .o_busy(layer0_busy),
        .o_done(layer0_done),
        .o_error(layer0_error),
        .o_active_stage_debug(layer0_active_stage_debug),
        .o_state_debug(layer0_state_debug),
        .o_stage_done_mask(o_layer0_stage_done_mask),
        .o_stage_error_mask(o_layer0_stage_error_mask),
        .o_layer0_full_stage_done_mask(o_layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(o_layer0_full_stage_error_mask),
        .o_body_stage_done_mask(o_body_stage_done_mask),
        .o_body_stage_error_mask(o_body_stage_error_mask),
        .o_input_norm_done(),
        .o_input_norm_error(),
        .o_input_norm_saturation(),
        .o_input_norm_write_word_count(),
        .o_qkv_rows_done(o_qkv_rows_done),
        .o_qkv_last_row_sum_q26(o_qkv_last_row_sum_q26),
        .o_qkv_last_output_q12_12(o_qkv_last_output_q12_12),
        .o_mem_read_burst_count(layer0_read_burst_count),
        .o_mem_read_word_count(layer0_read_word_count),
        .o_mem_write_req_count(layer0_write_req_count),
        .o_mem_write_word_count(layer0_write_word_count),
        .o_mem_rd_req_valid(layer0_rd_req_valid),
        .i_mem_rd_req_ready(layer0_rd_req_ready),
        .o_mem_rd_req_addr(layer0_rd_req_addr),
        .o_mem_rd_req_len_bytes(layer0_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(layer0_active ? i_mem_rd_rsp_valid : 1'b0),
        .o_mem_rd_rsp_ready(layer0_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(layer0_active ? i_mem_rd_rsp_last : 1'b0),
        .o_mem_wr_req_valid(layer0_wr_req_valid),
        .i_mem_wr_req_ready(layer0_wr_req_ready),
        .o_mem_wr_req_addr(layer0_wr_req_addr),
        .o_mem_wr_req_len_bytes(layer0_wr_req_len_bytes),
        .o_mem_wr_data(layer0_wr_data),
        .o_mem_wr_data_valid(layer0_wr_data_valid),
        .i_mem_wr_data_ready(layer0_wr_data_ready),
        .o_mem_wr_data_last(layer0_wr_data_last),
        .i_mem_wr_done(layer0_active ? i_mem_wr_done : 1'b0),
        .i_mem_wr_error(layer0_active ? i_mem_wr_error : 1'b0)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            o_done <= 1'b0;
            o_error <= 1'b0;
            active_layer_index_reg <= '0;
            o_layers_started <= '0;
            o_layers_completed <= '0;
            o_layer_done_mask <= '0;
            o_layer_error_mask <= '0;
            o_last_layer_output_base_addr <= '0;
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
                        o_error <= 1'b0;
                        active_layer_index_reg <= i_layer_start_index;
                        o_layers_started <= '0;
                        o_layers_completed <= '0;
                        o_layer_done_mask <= '0;
                        o_layer_error_mask <= '0;
                        o_last_layer_output_base_addr <= '0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state <= S_VALIDATE;
                    end
                end

                S_VALIDATE: begin
                    if (validate_error) begin
                        o_error <= 1'b1;
                        if (validation_error_layer_index <= MAX_LAYER_INDEX) begin
                            o_layer_error_mask[validation_error_layer_index] <= 1'b1;
                        end
                        state <= S_DONE;
                    end
                    else begin
                        active_layer_index_reg <= i_layer_start_index;
                        state <= S_LAYER_START;
                    end
                end

                S_LAYER_START: begin
                    o_layers_started <= o_layers_started + ONE_LAYER_COUNT;
                    state <= S_LAYER_WAIT;
                end

                S_LAYER_WAIT: begin
                    if (layer0_done) begin
                        if (layer0_error) begin
                            o_error <= 1'b1;
                            o_layer_error_mask[active_layer_index_reg] <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            o_layers_completed <= o_layers_completed + ONE_LAYER_COUNT;
                            o_layer_done_mask[active_layer_index_reg] <= 1'b1;
                            o_last_layer_output_base_addr <= selected_layer_output_base_addr;
                            if ((o_layers_completed + ONE_LAYER_COUNT) == i_layer_count) begin
                                state <= S_DONE;
                            end
                            else begin
                                active_layer_index_reg <= active_layer_index_reg + ONE_LAYER_INDEX;
                                state <= S_LAYER_START;
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
