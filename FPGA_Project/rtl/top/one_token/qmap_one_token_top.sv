`default_nettype none

`include "qmap_defs.svh"

// Reusable local one-token control boundary.
//
// This module owns the high-level sequence:
//
//   optional tied-Q4 embedding -> layer scheduler -> final-token tail
//
// It deliberately keeps both existing compute wrappers intact and only adds the
// control/memory-mux boundary that the eventual PS-visible run_one_token block
// needs. The tail consumes the last layer output reported by the scheduler by
// default; an explicit override remains available for focused debug runs.
module qmap_one_token_top #(
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
    parameter int ROW_ACC_WIDTH     = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TAIL_MAX_TILES    = 9496,
    parameter int TAIL_TILE_ROWS    = 16,
    parameter int TOKEN_ID_WIDTH    = 32,
    parameter int VOCAB_SIZE        = 151936
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic                         i_embedding_enable,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]  i_input_token_id,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_embedding_weight_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_embedding_scale_base_addr,
    input  wire logic [LAYER_INDEX_WIDTH-1:0] i_layer_start_index,
    input  wire logic [LAYER_COUNT_WIDTH-1:0] i_layer_count,
    input  wire logic [POSITION_WIDTH-1:0]    i_position,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_input_hidden_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_hidden_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_kv_cache_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_final_tail_qmap_base_addr,
    input  wire logic                         i_final_hidden_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_final_hidden_base_override_addr,

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
    output logic [31 : 0]                     o_qkv_rows_done,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_qkv_last_row_sum_q26,
    output logic signed [31 : 0]              o_qkv_last_output_q12_12,
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

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_VALIDATE,
        S_EMBED_START,
        S_EMBED_WAIT,
        S_LAYER_START,
        S_LAYER_WAIT,
        S_TAIL_START,
        S_TAIL_WAIT,
        S_DONE
    } state_t;

    localparam logic [7 : 0] PHASE_IDLE   = 8'd0;
    localparam logic [7 : 0] PHASE_CONFIG = 8'd1;
    localparam logic [7 : 0] PHASE_EMBED  = 8'd5;
    localparam logic [7 : 0] PHASE_LAYERS = 8'd2;
    localparam logic [7 : 0] PHASE_TAIL   = 8'd3;
    localparam logic [7 : 0] PHASE_DONE   = 8'd4;

    state_t state;

    logic embedding_start;
    logic embedding_active;
    logic embedding_done;
    logic embedding_error;
    logic embed_mem_rd_req_valid;
    logic embed_mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] embed_mem_rd_req_addr;
    logic [15 : 0] embed_mem_rd_req_len_bytes;
    logic embed_mem_rd_rsp_valid;
    logic embed_mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] embed_mem_rd_rsp_data;
    logic embed_mem_rd_rsp_last;
    logic embed_mem_wr_req_valid;
    logic embed_mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] embed_mem_wr_req_addr;
    logic [15 : 0] embed_mem_wr_req_len_bytes;
    logic [MEM_DATA_WIDTH-1 : 0] embed_mem_wr_data;
    logic embed_mem_wr_data_valid;
    logic embed_mem_wr_data_ready;
    logic embed_mem_wr_data_last;
    logic embed_mem_wr_done;
    logic embed_mem_wr_error;

    logic scheduler_start;
    logic scheduler_busy;
    logic scheduler_done;
    logic scheduler_error;
    logic [7 : 0] scheduler_state_debug;
    logic scheduler_active;
    logic [ADDR_WIDTH-1 : 0] scheduler_last_layer_output_base_addr;

    logic sched_mem_rd_req_valid;
    logic sched_mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] sched_mem_rd_req_addr;
    logic [15 : 0] sched_mem_rd_req_len_bytes;
    logic sched_mem_rd_rsp_valid;
    logic sched_mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] sched_mem_rd_rsp_data;
    logic sched_mem_rd_rsp_last;

    logic sched_mem_wr_req_valid;
    logic sched_mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] sched_mem_wr_req_addr;
    logic [15 : 0] sched_mem_wr_req_len_bytes;
    logic [31 : 0] sched_mem_wr_data;
    logic sched_mem_wr_data_valid;
    logic sched_mem_wr_data_ready;
    logic sched_mem_wr_data_last;
    logic sched_mem_wr_done;
    logic sched_mem_wr_error;

    logic tail_start;
    logic tail_busy;
    logic tail_done;
    logic tail_mem_rd_req_valid;
    logic tail_mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] tail_mem_rd_req_addr;
    logic [15 : 0] tail_mem_rd_req_len_bytes;
    logic tail_mem_rd_rsp_valid;
    logic tail_mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] tail_mem_rd_rsp_data;
    logic tail_mem_rd_rsp_last;

    logic tail_mem_wr_req_valid;
    logic tail_mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] tail_mem_wr_req_addr;
    logic [15 : 0] tail_mem_wr_req_len_bytes;
    logic [31 : 0] tail_mem_wr_data;
    logic tail_mem_wr_data_valid;
    logic tail_mem_wr_data_ready;
    logic tail_mem_wr_data_last;
    logic tail_mem_wr_done;
    logic tail_mem_wr_error;

    logic top_validate_error;
    logic [ADDR_WIDTH-1 : 0] selected_final_hidden_base_addr;

    assign embedding_start = (state == S_EMBED_START);
    assign embedding_active = (state == S_EMBED_START) || (state == S_EMBED_WAIT);
    assign scheduler_start = (state == S_LAYER_START);
    assign tail_start = (state == S_TAIL_START);
    assign scheduler_active = (state == S_LAYER_START) || (state == S_LAYER_WAIT);
    assign o_tail_active = (state == S_TAIL_START) || (state == S_TAIL_WAIT);
    assign selected_final_hidden_base_addr =
        i_final_hidden_base_override_valid ?
        i_final_hidden_base_override_addr :
        scheduler_last_layer_output_base_addr;
    assign o_last_layer_output_base_addr = scheduler_last_layer_output_base_addr;
    assign top_validate_error =
        (i_final_tail_qmap_base_addr == '0) ||
        (i_embedding_enable &&
         ((i_input_token_id >= VOCAB_SIZE) ||
          (i_embedding_weight_base_addr == '0) ||
          (i_embedding_scale_base_addr == '0) ||
          (i_input_hidden_base_addr == '0))) ||
        (i_final_hidden_base_override_valid && (i_final_hidden_base_override_addr == '0));

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {5'd0, state};
    assign o_scheduler_done_pulse = (state == S_LAYER_WAIT) && scheduler_done;
    assign o_tail_start_pulse = tail_start;
    assign o_tail_done_pulse = (state == S_TAIL_WAIT) && tail_done;

    always @* begin
        case (state)
            S_IDLE:        o_phase_debug = PHASE_IDLE;
            S_VALIDATE:    o_phase_debug = PHASE_CONFIG;
            S_EMBED_START: o_phase_debug = PHASE_EMBED;
            S_EMBED_WAIT:  o_phase_debug = PHASE_EMBED;
            S_LAYER_START: o_phase_debug = PHASE_LAYERS;
            S_LAYER_WAIT:  o_phase_debug = PHASE_LAYERS;
            S_TAIL_START:  o_phase_debug = PHASE_TAIL;
            S_TAIL_WAIT:   o_phase_debug = PHASE_TAIL;
            S_DONE:        o_phase_debug = PHASE_DONE;
            default:       o_phase_debug = 8'hFF;
        endcase
    end

    assign o_mem_rd_req_valid =
        embedding_active ? embed_mem_rd_req_valid :
        scheduler_active ? sched_mem_rd_req_valid :
        o_tail_active    ? tail_mem_rd_req_valid :
        1'b0;
    assign o_mem_rd_req_addr =
        embedding_active ? embed_mem_rd_req_addr :
        scheduler_active ? sched_mem_rd_req_addr :
        o_tail_active    ? tail_mem_rd_req_addr :
        '0;
    assign o_mem_rd_req_len_bytes =
        embedding_active ? embed_mem_rd_req_len_bytes :
        scheduler_active ? sched_mem_rd_req_len_bytes :
        o_tail_active    ? tail_mem_rd_req_len_bytes :
        16'd0;

    assign embed_mem_rd_req_ready = embedding_active ? i_mem_rd_req_ready : 1'b0;
    assign sched_mem_rd_req_ready = scheduler_active ? i_mem_rd_req_ready : 1'b0;
    assign tail_mem_rd_req_ready = o_tail_active ? i_mem_rd_req_ready : 1'b0;
    assign embed_mem_rd_rsp_valid = embedding_active ? i_mem_rd_rsp_valid : 1'b0;
    assign sched_mem_rd_rsp_valid = scheduler_active ? i_mem_rd_rsp_valid : 1'b0;
    assign tail_mem_rd_rsp_valid = o_tail_active ? i_mem_rd_rsp_valid : 1'b0;
    assign embed_mem_rd_rsp_data = i_mem_rd_rsp_data;
    assign sched_mem_rd_rsp_data = i_mem_rd_rsp_data;
    assign tail_mem_rd_rsp_data = i_mem_rd_rsp_data;
    assign embed_mem_rd_rsp_last = embedding_active ? i_mem_rd_rsp_last : 1'b0;
    assign sched_mem_rd_rsp_last = scheduler_active ? i_mem_rd_rsp_last : 1'b0;
    assign tail_mem_rd_rsp_last = o_tail_active ? i_mem_rd_rsp_last : 1'b0;
    assign o_mem_rd_rsp_ready =
        embedding_active ? embed_mem_rd_rsp_ready :
        scheduler_active ? sched_mem_rd_rsp_ready :
        o_tail_active    ? tail_mem_rd_rsp_ready :
        1'b0;

    assign o_mem_wr_req_valid =
        embedding_active ? embed_mem_wr_req_valid :
        scheduler_active ? sched_mem_wr_req_valid :
        o_tail_active    ? tail_mem_wr_req_valid :
        1'b0;
    assign o_mem_wr_req_addr =
        embedding_active ? embed_mem_wr_req_addr :
        scheduler_active ? sched_mem_wr_req_addr :
        o_tail_active    ? tail_mem_wr_req_addr :
        '0;
    assign o_mem_wr_req_len_bytes =
        embedding_active ? embed_mem_wr_req_len_bytes :
        scheduler_active ? sched_mem_wr_req_len_bytes :
        o_tail_active    ? tail_mem_wr_req_len_bytes :
        16'd0;
    assign o_mem_wr_data =
        embedding_active ? embed_mem_wr_data :
        scheduler_active ? sched_mem_wr_data :
        o_tail_active    ? tail_mem_wr_data :
        32'd0;
    assign o_mem_wr_data_valid =
        embedding_active ? embed_mem_wr_data_valid :
        scheduler_active ? sched_mem_wr_data_valid :
        o_tail_active    ? tail_mem_wr_data_valid :
        1'b0;
    assign o_mem_wr_data_last =
        embedding_active ? embed_mem_wr_data_last :
        scheduler_active ? sched_mem_wr_data_last :
        o_tail_active    ? tail_mem_wr_data_last :
        1'b0;

    assign embed_mem_wr_req_ready = embedding_active ? i_mem_wr_req_ready : 1'b0;
    assign sched_mem_wr_req_ready = scheduler_active ? i_mem_wr_req_ready : 1'b0;
    assign tail_mem_wr_req_ready = o_tail_active ? i_mem_wr_req_ready : 1'b0;
    assign embed_mem_wr_data_ready = embedding_active ? i_mem_wr_data_ready : 1'b0;
    assign sched_mem_wr_data_ready = scheduler_active ? i_mem_wr_data_ready : 1'b0;
    assign tail_mem_wr_data_ready = o_tail_active ? i_mem_wr_data_ready : 1'b0;
    assign embed_mem_wr_done = embedding_active ? i_mem_wr_done : 1'b0;
    assign sched_mem_wr_done = scheduler_active ? i_mem_wr_done : 1'b0;
    assign tail_mem_wr_done = o_tail_active ? i_mem_wr_done : 1'b0;
    assign embed_mem_wr_error = embedding_active ? i_mem_wr_error : 1'b0;
    assign sched_mem_wr_error = scheduler_active ? i_mem_wr_error : 1'b0;
    assign tail_mem_wr_error = o_tail_active ? i_mem_wr_error : 1'b0;

    q4_embedding_lookup #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH),
        .VOCAB_SIZE     (VOCAB_SIZE),
        .INPUT_SIZE     (INPUT_SIZE),
        .GROUP_SIZE     (GROUP_SIZE),
        .GROUP_COUNT    (GROUP_COUNT),
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .SCALE_WIDTH    (SCALE_WIDTH),
        .SCALE_FRAC     (SCALE_FRAC),
        .OUTPUT_FRAC    (10)
    ) embedding (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(embedding_start),
        .i_token_id(i_input_token_id),
        .i_weight_base_addr(i_embedding_weight_base_addr),
        .i_scale_base_addr(i_embedding_scale_base_addr),
        .i_output_base_addr(i_input_hidden_base_addr),
        .o_busy(),
        .o_done(embedding_done),
        .o_error(embedding_error),
        .o_mem_rd_req_valid(embed_mem_rd_req_valid),
        .i_mem_rd_req_ready(embed_mem_rd_req_ready),
        .o_mem_rd_req_addr(embed_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(embed_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(embed_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(embed_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(embed_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(embed_mem_rd_rsp_last),
        .o_mem_wr_req_valid(embed_mem_wr_req_valid),
        .i_mem_wr_req_ready(embed_mem_wr_req_ready),
        .o_mem_wr_req_addr(embed_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(embed_mem_wr_req_len_bytes),
        .o_mem_wr_data(embed_mem_wr_data),
        .o_mem_wr_data_valid(embed_mem_wr_data_valid),
        .i_mem_wr_data_ready(embed_mem_wr_data_ready),
        .o_mem_wr_data_last(embed_mem_wr_data_last),
        .i_mem_wr_done(embed_mem_wr_done),
        .i_mem_wr_error(embed_mem_wr_error),
        .o_read_burst_count(),
        .o_read_word_count(),
        .o_write_req_count(),
        .o_write_word_count()
    );

    qmap_one_token_layer_scheduler #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .MEM_DATA_WIDTH   (MEM_DATA_WIDTH),
        .MAX_LAYERS       (MAX_LAYERS),
        .MAX_CONTEXT      (MAX_CONTEXT),
        .LAYER_INDEX_WIDTH(LAYER_INDEX_WIDTH),
        .LAYER_COUNT_WIDTH(LAYER_COUNT_WIDTH),
        .POSITION_WIDTH   (POSITION_WIDTH),
        .INPUT_SIZE       (INPUT_SIZE),
        .GROUP_SIZE       (GROUP_SIZE),
        .GROUP_COUNT      (GROUP_COUNT),
        .GROUP_PARALLEL   (GROUP_PARALLEL),
        .ACT_WIDTH        (ACT_WIDTH),
        .ACT_FRAC         (ACT_FRAC),
        .WEIGHT_WIDTH     (WEIGHT_WIDTH),
        .SCALE_WIDTH      (SCALE_WIDTH),
        .SCALE_FRAC       (SCALE_FRAC),
        .PARTIAL_WIDTH    (PARTIAL_WIDTH),
        .SCALED_WIDTH     (SCALED_WIDTH),
        .ROW_ACC_WIDTH    (ROW_ACC_WIDTH)
    ) scheduler (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(scheduler_start),
        .i_layer_start_index(i_layer_start_index),
        .i_layer_count(i_layer_count),
        .i_position(i_position),
        .i_input_hidden_base_addr(i_input_hidden_base_addr),
        .i_output_hidden_base_addr(i_output_hidden_base_addr),
        .i_kv_cache_base_addr(i_kv_cache_base_addr),
        .i_qkv_qmap_base_addr_table(i_qkv_qmap_base_addr_table),
        .i_input_norm_qmap_base_addr_table(i_input_norm_qmap_base_addr_table),
        .i_attn_frontend_qmap_base_addr_table(i_attn_frontend_qmap_base_addr_table),
        .i_attn_score_value_qmap_base_addr_table(i_attn_score_value_qmap_base_addr_table),
        .i_o_proj_qmap_base_addr_table(i_o_proj_qmap_base_addr_table),
        .i_post_attn_norm_qmap_base_addr_table(i_post_attn_norm_qmap_base_addr_table),
        .i_mlp_gate_up_qmap_base_addr_table(i_mlp_gate_up_qmap_base_addr_table),
        .i_mlp_silu_mul_qmap_base_addr_table(i_mlp_silu_mul_qmap_base_addr_table),
        .i_mlp_down_qmap_base_addr_table(i_mlp_down_qmap_base_addr_table),
        .i_mlp_residual_add_qmap_base_addr_table(i_mlp_residual_add_qmap_base_addr_table),
        .o_busy(scheduler_busy),
        .o_done(scheduler_done),
        .o_error(scheduler_error),
        .o_state_debug(scheduler_state_debug),
        .o_active_layer_index(o_active_layer_index),
        .o_layers_started(o_layers_started),
        .o_layers_completed(o_layers_completed),
        .o_layer_done_mask(o_layer_done_mask),
        .o_layer_error_mask(o_layer_error_mask),
        .o_last_layer_output_base_addr(scheduler_last_layer_output_base_addr),
        .o_layer0_active_stage_debug(o_layer0_active_stage_debug),
        .o_layer0_state_debug(o_layer0_state_debug),
        .o_layer0_stage_done_mask(o_layer0_stage_done_mask),
        .o_layer0_stage_error_mask(o_layer0_stage_error_mask),
        .o_layer0_full_stage_done_mask(o_layer0_full_stage_done_mask),
        .o_layer0_full_stage_error_mask(o_layer0_full_stage_error_mask),
        .o_body_stage_done_mask(o_body_stage_done_mask),
        .o_body_stage_error_mask(o_body_stage_error_mask),
        .o_qkv_rows_done(o_qkv_rows_done),
        .o_qkv_last_row_sum_q26(o_qkv_last_row_sum_q26),
        .o_qkv_last_output_q12_12(o_qkv_last_output_q12_12),
        .o_mem_read_burst_count(o_scheduler_mem_read_burst_count),
        .o_mem_read_word_count(o_scheduler_mem_read_word_count),
        .o_mem_write_req_count(o_scheduler_mem_write_req_count),
        .o_mem_write_word_count(o_scheduler_mem_write_word_count),
        .o_mem_rd_req_valid(sched_mem_rd_req_valid),
        .i_mem_rd_req_ready(sched_mem_rd_req_ready),
        .o_mem_rd_req_addr(sched_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(sched_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(sched_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(sched_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(sched_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(sched_mem_rd_rsp_last),
        .o_mem_wr_req_valid(sched_mem_wr_req_valid),
        .i_mem_wr_req_ready(sched_mem_wr_req_ready),
        .o_mem_wr_req_addr(sched_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(sched_mem_wr_req_len_bytes),
        .o_mem_wr_data(sched_mem_wr_data),
        .o_mem_wr_data_valid(sched_mem_wr_data_valid),
        .i_mem_wr_data_ready(sched_mem_wr_data_ready),
        .o_mem_wr_data_last(sched_mem_wr_data_last),
        .i_mem_wr_done(sched_mem_wr_done),
        .i_mem_wr_error(sched_mem_wr_error)
    );

    qmap_final_token_tail_compute_path #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .MAX_TILES      (TAIL_MAX_TILES),
        .TILE_ROWS      (TAIL_TILE_ROWS),
        .INPUT_SIZE     (INPUT_SIZE),
        .GROUP_SIZE     (GROUP_SIZE),
        .GROUP_COUNT    (GROUP_COUNT),
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .SCALE_WIDTH    (SCALE_WIDTH),
        .PARTIAL_WIDTH  (PARTIAL_WIDTH),
        .SCALED_WIDTH   (SCALED_WIDTH),
        .ROW_ACC_WIDTH  (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH (TOKEN_ID_WIDTH),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH)
    ) tail (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(tail_start),
        .i_qmap_base_addr(i_final_tail_qmap_base_addr),
        .i_final_hidden_base_override_valid(1'b1),
        .i_final_hidden_base_override_addr(selected_final_hidden_base_addr),
        .o_busy(tail_busy),
        .o_done(tail_done),
        .o_error(o_tail_error),
        .o_norm_saturation(o_tail_norm_saturation),
        .o_effective_final_hidden_base_addr(o_tail_effective_final_hidden_base_addr),
        .o_best_token_id(o_tail_best_token_id),
        .o_best_score_q26(o_tail_best_score_q26),
        .o_tiles_started(o_tail_tiles_started),
        .o_tiles_completed(o_tail_tiles_completed),
        .o_norm_cycle_count(o_tail_norm_cycle_count),
        .o_mem_read_burst_count(o_tail_mem_read_burst_count),
        .o_mem_read_word_count(o_tail_mem_read_word_count),
        .o_mem_write_word_count(o_tail_mem_write_word_count),
        .o_mem_rd_req_valid(tail_mem_rd_req_valid),
        .i_mem_rd_req_ready(tail_mem_rd_req_ready),
        .o_mem_rd_req_addr(tail_mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(tail_mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(tail_mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(tail_mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(tail_mem_rd_rsp_data),
        .i_mem_rd_rsp_last(tail_mem_rd_rsp_last),
        .o_mem_wr_req_valid(tail_mem_wr_req_valid),
        .i_mem_wr_req_ready(tail_mem_wr_req_ready),
        .o_mem_wr_req_addr(tail_mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(tail_mem_wr_req_len_bytes),
        .o_mem_wr_data(tail_mem_wr_data),
        .o_mem_wr_data_valid(tail_mem_wr_data_valid),
        .i_mem_wr_data_ready(tail_mem_wr_data_ready),
        .o_mem_wr_data_last(tail_mem_wr_data_last),
        .i_mem_wr_done(tail_mem_wr_done),
        .i_mem_wr_error(tail_mem_wr_error)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            o_done <= 1'b0;
            o_error <= 1'b0;
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
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state <= S_VALIDATE;
                    end
                end

                S_VALIDATE: begin
                    if (top_validate_error) begin
                        o_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else if (i_embedding_enable) begin
                        state <= S_EMBED_START;
                    end
                    else begin
                        state <= S_LAYER_START;
                    end
                end

                S_EMBED_START: begin
                    state <= S_EMBED_WAIT;
                end

                S_EMBED_WAIT: begin
                    if (embedding_done) begin
                        if (embedding_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            state <= S_LAYER_START;
                        end
                    end
                end

                S_LAYER_START: begin
                    state <= S_LAYER_WAIT;
                end

                S_LAYER_WAIT: begin
                    if (scheduler_done) begin
                        if (scheduler_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else if (selected_final_hidden_base_addr == '0) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_TAIL_START;
                        end
                    end
                end

                S_TAIL_START: begin
                    state <= S_TAIL_WAIT;
                end

                S_TAIL_WAIT: begin
                    if (tail_done) begin
                        if (o_tail_error) begin
                            o_error <= 1'b1;
                        end
                        state <= S_DONE;
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
