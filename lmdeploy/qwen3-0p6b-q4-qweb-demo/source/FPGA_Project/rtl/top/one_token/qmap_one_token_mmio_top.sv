`default_nettype none

// Productized local MMIO-style wrapper for the one-token RTL boundary.
//
// This module intentionally keeps the tiny valid/ready register interface used
// by the local testbenches. It is the stable seam for the next AXI4-Lite
// adapter: software-visible register semantics live in
// qmap_one_token_control_regs.sv, while qmap_one_token_top.sv owns the compute
// sequence and shared memory mux.
module qmap_one_token_mmio_top #(
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
    parameter int GROUP_PARALLEL    = 1,
    parameter int BODY_GROUP_PARALLEL = 1,
    parameter int ACT_WIDTH         = 24,
    parameter int ACT_FRAC          = 12,
    parameter int WEIGHT_WIDTH      = 4,
    parameter int SCALE_WIDTH       = 16,
    parameter int SCALE_FRAC        = 14,
    parameter int PARTIAL_WIDTH     = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH      = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH     = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TAIL_MAX_TILES    = 9496,
    parameter int TAIL_TILE_ROWS    = 16,
    parameter int TAIL_ROW_PARALLEL = 1,
    parameter int TOKEN_ID_WIDTH    = 32,
    parameter int SCORE_WIDTH       = ROW_ACC_WIDTH
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_reg_wr_valid,
    output logic                              o_reg_wr_ready,
    input  wire logic                         i_reg_rd_valid,
    output logic                              o_reg_rd_ready,
    input  wire logic [11 : 0]                i_reg_addr,
    input  wire logic [31 : 0]                i_reg_wdata,
    output logic [31 : 0]                     o_reg_rdata,
    output logic                              o_reg_error,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [7 : 0]                      o_state_debug,
    output logic [7 : 0]                      o_phase_debug,
    output logic                              o_scheduler_done_pulse,
    output logic                              o_tail_start_pulse,
    output logic                              o_tail_done_pulse,
    output logic                              o_tail_active,
    output logic [LAYER_INDEX_WIDTH-1:0]      o_active_layer_index,
    output logic [LAYER_COUNT_WIDTH-1:0]      o_layers_started,
    output logic [LAYER_COUNT_WIDTH-1:0]      o_layers_completed,
    output logic [MAX_LAYERS-1:0]             o_layer_done_mask,
    output logic [MAX_LAYERS-1:0]             o_layer_error_mask,
    output logic [ADDR_WIDTH-1 : 0]           o_last_layer_output_base_addr,
    output logic [1 : 0]                      o_layer0_active_stage_debug,
    output logic [7 : 0]                      o_layer0_state_debug,
    output logic [1 : 0]                      o_layer0_stage_done_mask,
    output logic [1 : 0]                      o_layer0_stage_error_mask,
    output logic [3 : 0]                      o_layer0_full_stage_done_mask,
    output logic [3 : 0]                      o_layer0_full_stage_error_mask,
    output logic [4 : 0]                      o_body_stage_done_mask,
    output logic [4 : 0]                      o_body_stage_error_mask,
    output logic [31 : 0]                     o_scheduler_mem_read_burst_count,
    output logic [31 : 0]                     o_scheduler_mem_read_word_count,
    output logic [31 : 0]                     o_scheduler_mem_write_req_count,
    output logic [31 : 0]                     o_scheduler_mem_write_word_count,
    output logic                              o_tail_error,
    output logic                              o_tail_norm_saturation,
    output logic [ADDR_WIDTH-1 : 0]           o_tail_effective_final_hidden_base_addr,
    output logic [TOKEN_ID_WIDTH-1 : 0]       o_tail_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_tail_best_score_q26,
    output logic [31 : 0]                     o_tail_tiles_started,
    output logic [31 : 0]                     o_tail_tiles_completed,
    output logic [31 : 0]                     o_tail_norm_cycle_count,
    output logic [31 : 0]                     o_tail_mem_read_burst_count,
    output logic [31 : 0]                     o_tail_mem_read_word_count,
    output logic [31 : 0]                     o_tail_mem_write_word_count,
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

    logic start_pulse;
    logic [31 : 0] input_token_id;
    logic embedding_enable;
    logic [ADDR_WIDTH-1 : 0] embedding_weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] embedding_scale_base_addr;
    logic [LAYER_INDEX_WIDTH-1 : 0] layer_start_index;
    logic [LAYER_COUNT_WIDTH-1 : 0] layer_count;
    logic [POSITION_WIDTH-1 : 0] position;
    logic runtime_context_enable;
    logic [ADDR_WIDTH-1 : 0] input_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] kv_cache_base_addr;
    logic [ADDR_WIDTH-1 : 0] final_tail_qmap_base_addr;
    logic final_hidden_base_override_valid;
    logic [ADDR_WIDTH-1 : 0] final_hidden_base_override_addr;

    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] qkv_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] input_norm_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] attn_frontend_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] attn_score_value_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_proj_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] post_attn_norm_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_gate_up_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_silu_mul_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_down_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_residual_add_qmap_base_addr_table;
    logic [ADDR_WIDTH-1 : 0] qkv_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] input_norm_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] attn_frontend_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] attn_score_value_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] o_proj_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] post_attn_norm_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] mlp_gate_up_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] mlp_silu_mul_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] mlp_down_qmap_base_addr;
    logic [ADDR_WIDTH-1 : 0] mlp_residual_add_qmap_base_addr;
    logic runtime_base_addr_valid;
    logic [LAYER_INDEX_WIDTH-1:0] runtime_base_addr_layer;

    qmap_one_token_control_regs #(
        .ADDR_WIDTH        (ADDR_WIDTH),
        .MAX_LAYERS        (MAX_LAYERS),
        .MAX_CONTEXT       (MAX_CONTEXT),
        .TOKEN_ID_WIDTH    (TOKEN_ID_WIDTH),
        .SCORE_WIDTH       (SCORE_WIDTH),
        .LAYER_INDEX_WIDTH (LAYER_INDEX_WIDTH),
        .LAYER_COUNT_WIDTH (LAYER_COUNT_WIDTH),
        .POSITION_WIDTH    (POSITION_WIDTH),
        .USE_BRAM_BASE_TABLES(1'b1)
    ) control_regs (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_reg_wr_valid(i_reg_wr_valid),
        .o_reg_wr_ready(o_reg_wr_ready),
        .i_reg_rd_valid(i_reg_rd_valid),
        .o_reg_rd_ready(o_reg_rd_ready),
        .i_reg_addr(i_reg_addr),
        .i_reg_wdata(i_reg_wdata),
        .o_reg_rdata(o_reg_rdata),
        .o_reg_error(o_reg_error),
        .o_start_pulse(start_pulse),
        .o_input_token_id(input_token_id),
        .o_embedding_enable(embedding_enable),
        .o_embedding_weight_base_addr(embedding_weight_base_addr),
        .o_embedding_scale_base_addr(embedding_scale_base_addr),
        .o_layer_start_index(layer_start_index),
        .o_layer_count(layer_count),
        .o_position(position),
        .o_runtime_context_enable(runtime_context_enable),
        .o_input_hidden_base_addr(input_hidden_base_addr),
        .o_output_hidden_base_addr(output_hidden_base_addr),
        .o_kv_cache_base_addr(kv_cache_base_addr),
        .o_final_tail_qmap_base_addr(final_tail_qmap_base_addr),
        .o_final_hidden_base_override_valid(final_hidden_base_override_valid),
        .o_final_hidden_base_override_addr(final_hidden_base_override_addr),
        .o_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .o_input_norm_qmap_base_addr_table(input_norm_qmap_base_addr_table),
        .o_attn_frontend_qmap_base_addr_table(attn_frontend_qmap_base_addr_table),
        .o_attn_score_value_qmap_base_addr_table(attn_score_value_qmap_base_addr_table),
        .o_o_proj_qmap_base_addr_table(o_proj_qmap_base_addr_table),
        .o_post_attn_norm_qmap_base_addr_table(post_attn_norm_qmap_base_addr_table),
        .o_mlp_gate_up_qmap_base_addr_table(mlp_gate_up_qmap_base_addr_table),
        .o_mlp_silu_mul_qmap_base_addr_table(mlp_silu_mul_qmap_base_addr_table),
        .o_mlp_down_qmap_base_addr_table(mlp_down_qmap_base_addr_table),
        .o_mlp_residual_add_qmap_base_addr_table(mlp_residual_add_qmap_base_addr_table),
        .i_runtime_table_layer(o_active_layer_index),
        .o_qkv_qmap_base_addr(qkv_qmap_base_addr),
        .o_input_norm_qmap_base_addr(input_norm_qmap_base_addr),
        .o_attn_frontend_qmap_base_addr(attn_frontend_qmap_base_addr),
        .o_attn_score_value_qmap_base_addr(attn_score_value_qmap_base_addr),
        .o_o_proj_qmap_base_addr(o_proj_qmap_base_addr),
        .o_post_attn_norm_qmap_base_addr(post_attn_norm_qmap_base_addr),
        .o_mlp_gate_up_qmap_base_addr(mlp_gate_up_qmap_base_addr),
        .o_mlp_silu_mul_qmap_base_addr(mlp_silu_mul_qmap_base_addr),
        .o_mlp_down_qmap_base_addr(mlp_down_qmap_base_addr),
        .o_mlp_residual_add_qmap_base_addr(mlp_residual_add_qmap_base_addr),
        .o_runtime_table_valid(runtime_base_addr_valid),
        .o_runtime_table_layer(runtime_base_addr_layer),
        .i_top_busy(o_busy),
        .i_top_done(o_done),
        .i_top_error(o_error),
        .i_top_state_debug(o_state_debug),
        .i_top_phase_debug(o_phase_debug),
        .i_layers_started(o_layers_started),
        .i_layers_completed(o_layers_completed),
        .i_layer_done_mask(o_layer_done_mask),
        .i_layer_error_mask(o_layer_error_mask),
        .i_last_layer_output_base_addr(o_last_layer_output_base_addr),
        .i_tail_error(o_tail_error),
        .i_tail_norm_saturation(o_tail_norm_saturation),
        .i_tail_effective_final_hidden_base_addr(o_tail_effective_final_hidden_base_addr),
        .i_tail_best_token_id(o_tail_best_token_id),
        .i_tail_best_score_q26(o_tail_best_score_q26),
        .i_tail_tiles_started(o_tail_tiles_started),
        .i_tail_tiles_completed(o_tail_tiles_completed),
        .i_mem_read_burst_count(o_mem_read_burst_count),
        .i_mem_read_word_count(o_mem_read_word_count),
        .i_mem_write_req_count(o_mem_write_req_count),
        .i_mem_write_word_count(o_mem_write_word_count)
    );

    qmap_one_token_top #(
        .ADDR_WIDTH        (ADDR_WIDTH),
        .MEM_DATA_WIDTH    (MEM_DATA_WIDTH),
        .MAX_LAYERS        (MAX_LAYERS),
        .MAX_CONTEXT       (MAX_CONTEXT),
        .LAYER_INDEX_WIDTH (LAYER_INDEX_WIDTH),
        .LAYER_COUNT_WIDTH (LAYER_COUNT_WIDTH),
        .POSITION_WIDTH    (POSITION_WIDTH),
        .INPUT_SIZE        (INPUT_SIZE),
        .GROUP_SIZE        (GROUP_SIZE),
        .GROUP_COUNT       (GROUP_COUNT),
        .GROUP_PARALLEL    (GROUP_PARALLEL),
        .BODY_GROUP_PARALLEL(BODY_GROUP_PARALLEL),
        .ACT_WIDTH         (ACT_WIDTH),
        .ACT_FRAC          (ACT_FRAC),
        .WEIGHT_WIDTH      (WEIGHT_WIDTH),
        .SCALE_WIDTH       (SCALE_WIDTH),
        .SCALE_FRAC        (SCALE_FRAC),
        .PARTIAL_WIDTH     (PARTIAL_WIDTH),
        .SCALED_WIDTH      (SCALED_WIDTH),
        .ROW_ACC_WIDTH     (ROW_ACC_WIDTH),
        .TAIL_MAX_TILES    (TAIL_MAX_TILES),
        .TAIL_TILE_ROWS    (TAIL_TILE_ROWS),
        .TAIL_ROW_PARALLEL (TAIL_ROW_PARALLEL),
        .TOKEN_ID_WIDTH    (TOKEN_ID_WIDTH),
        .USE_RUNTIME_BASE_ADDR_PORTS(1'b1)
    ) top (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(start_pulse),
        .i_embedding_enable(embedding_enable),
        .i_input_token_id(input_token_id),
        .i_embedding_weight_base_addr(embedding_weight_base_addr),
        .i_embedding_scale_base_addr(embedding_scale_base_addr),
        .i_layer_start_index(layer_start_index),
        .i_layer_count(layer_count),
        .i_position(position),
        .i_runtime_context_enable(runtime_context_enable),
        .i_input_hidden_base_addr(input_hidden_base_addr),
        .i_output_hidden_base_addr(output_hidden_base_addr),
        .i_kv_cache_base_addr(kv_cache_base_addr),
        .i_final_tail_qmap_base_addr(final_tail_qmap_base_addr),
        .i_final_hidden_base_override_valid(final_hidden_base_override_valid),
        .i_final_hidden_base_override_addr(final_hidden_base_override_addr),
        .i_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .i_input_norm_qmap_base_addr_table(input_norm_qmap_base_addr_table),
        .i_attn_frontend_qmap_base_addr_table(attn_frontend_qmap_base_addr_table),
        .i_attn_score_value_qmap_base_addr_table(attn_score_value_qmap_base_addr_table),
        .i_o_proj_qmap_base_addr_table(o_proj_qmap_base_addr_table),
        .i_post_attn_norm_qmap_base_addr_table(post_attn_norm_qmap_base_addr_table),
        .i_mlp_gate_up_qmap_base_addr_table(mlp_gate_up_qmap_base_addr_table),
        .i_mlp_silu_mul_qmap_base_addr_table(mlp_silu_mul_qmap_base_addr_table),
        .i_mlp_down_qmap_base_addr_table(mlp_down_qmap_base_addr_table),
        .i_mlp_residual_add_qmap_base_addr_table(mlp_residual_add_qmap_base_addr_table),
        .i_qkv_qmap_base_addr(qkv_qmap_base_addr),
        .i_input_norm_qmap_base_addr(input_norm_qmap_base_addr),
        .i_attn_frontend_qmap_base_addr(attn_frontend_qmap_base_addr),
        .i_attn_score_value_qmap_base_addr(attn_score_value_qmap_base_addr),
        .i_o_proj_qmap_base_addr(o_proj_qmap_base_addr),
        .i_post_attn_norm_qmap_base_addr(post_attn_norm_qmap_base_addr),
        .i_mlp_gate_up_qmap_base_addr(mlp_gate_up_qmap_base_addr),
        .i_mlp_silu_mul_qmap_base_addr(mlp_silu_mul_qmap_base_addr),
        .i_mlp_down_qmap_base_addr(mlp_down_qmap_base_addr),
        .i_mlp_residual_add_qmap_base_addr(mlp_residual_add_qmap_base_addr),
        .i_runtime_base_addr_valid(runtime_base_addr_valid),
        .i_runtime_base_addr_layer(runtime_base_addr_layer),
        .o_busy(o_busy),
        .o_done(o_done),
        .o_error(o_error),
        .o_state_debug(o_state_debug),
        .o_phase_debug(o_phase_debug),
        .o_scheduler_done_pulse(o_scheduler_done_pulse),
        .o_tail_start_pulse(o_tail_start_pulse),
        .o_tail_done_pulse(o_tail_done_pulse),
        .o_tail_active(o_tail_active),
        .o_active_layer_index(o_active_layer_index),
        .o_layers_started(o_layers_started),
        .o_layers_completed(o_layers_completed),
        .o_layer_done_mask(o_layer_done_mask),
        .o_layer_error_mask(o_layer_error_mask),
        .o_last_layer_output_base_addr(o_last_layer_output_base_addr),
        .o_layer0_active_stage_debug(o_layer0_active_stage_debug),
        .o_layer0_state_debug(o_layer0_state_debug),
        .o_layer0_stage_done_mask(o_layer0_stage_done_mask),
        .o_layer0_stage_error_mask(o_layer0_stage_error_mask),
        .o_layer0_full_stage_done_mask(o_layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(o_layer0_full_stage_error_mask),
        .o_body_stage_done_mask(o_body_stage_done_mask),
        .o_body_stage_error_mask(o_body_stage_error_mask),
        .o_qkv_rows_done(),
        .o_qkv_last_row_sum_q26(),
        .o_qkv_last_output_q12_12(),
        .o_scheduler_mem_read_burst_count(o_scheduler_mem_read_burst_count),
        .o_scheduler_mem_read_word_count(o_scheduler_mem_read_word_count),
        .o_scheduler_mem_write_req_count(o_scheduler_mem_write_req_count),
        .o_scheduler_mem_write_word_count(o_scheduler_mem_write_word_count),
        .o_tail_error(o_tail_error),
        .o_tail_norm_saturation(o_tail_norm_saturation),
        .o_tail_effective_final_hidden_base_addr(o_tail_effective_final_hidden_base_addr),
        .o_tail_best_token_id(o_tail_best_token_id),
        .o_tail_best_score_q26(o_tail_best_score_q26),
        .o_tail_tiles_started(o_tail_tiles_started),
        .o_tail_tiles_completed(o_tail_tiles_completed),
        .o_tail_norm_cycle_count(o_tail_norm_cycle_count),
        .o_tail_mem_read_burst_count(o_tail_mem_read_burst_count),
        .o_tail_mem_read_word_count(o_tail_mem_read_word_count),
        .o_tail_mem_write_word_count(o_tail_mem_write_word_count),
        .o_mem_read_burst_count(o_mem_read_burst_count),
        .o_mem_read_word_count(o_mem_read_word_count),
        .o_mem_write_req_count(o_mem_write_req_count),
        .o_mem_write_word_count(o_mem_write_word_count),
        .o_mem_rd_req_valid(o_mem_rd_req_valid),
        .i_mem_rd_req_ready(i_mem_rd_req_ready),
        .o_mem_rd_req_addr(o_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(o_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(i_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(o_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(i_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(i_mem_rd_rsp_last),
        .o_mem_wr_req_valid(o_mem_wr_req_valid),
        .i_mem_wr_req_ready(i_mem_wr_req_ready),
        .o_mem_wr_req_addr(o_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(o_mem_wr_req_len_bytes),
        .o_mem_wr_data(o_mem_wr_data),
        .o_mem_wr_data_valid(o_mem_wr_data_valid),
        .i_mem_wr_data_ready(i_mem_wr_data_ready),
        .o_mem_wr_data_last(o_mem_wr_data_last),
        .i_mem_wr_done(i_mem_wr_done),
        .i_mem_wr_error(i_mem_wr_error)
    );

endmodule

`default_nettype wire
