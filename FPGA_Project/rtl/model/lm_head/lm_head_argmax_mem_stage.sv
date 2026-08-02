`default_nettype none

// Memory-backed wrapper for the tiled Q4 LM-head argmax core.
//
// The argmax core still owns the tile scan/order and greedy best-score update.
// This wrapper answers each tile request by reading the corresponding packed
// Q4 weight rows and Q2.14 scales from a 32-bit project-local memory interface.
module lm_head_argmax_mem_stage #(
    parameter int ADDR_WIDTH      = 64,
    parameter int SCAN_ROWS       = 1024,
    parameter int TILE_ROWS       = 16,
    parameter int ROW_PARALLEL    = 1,
    parameter int INPUT_SIZE      = 1024,
    parameter int GROUP_SIZE      = 64,
    parameter int GROUP_COUNT     = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL  = 1,
    parameter int ACT_WIDTH       = 24,
    parameter int ACT_FRAC        = 12,
    parameter int WEIGHT_WIDTH    = 4,
    parameter int SCALE_WIDTH     = 16,
    parameter int SCALE_FRAC      = 14,
    parameter int PARTIAL_WIDTH   = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH    = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH   = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TOKEN_ID_WIDTH  = 32,
    parameter int TILE_COUNT      = SCAN_ROWS / TILE_ROWS,
    parameter int TILE_INDEX_W    = (TILE_COUNT <= 1) ? 1 : $clog2(TILE_COUNT),
    parameter int MEM_DATA_WIDTH  = 32,
    parameter int MAX_READ_BYTES  = 1024
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_start,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]       i_token_base,
    input  wire logic [INPUT_SIZE*ACT_WIDTH-1 : 0] i_activation_flat,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_weight_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_scale_base_addr,

    output logic                                   o_mem_req_valid,
    input  wire logic                              i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                o_mem_req_addr,
    output logic [15 : 0]                          o_mem_req_len_bytes,

    input  wire logic                              i_mem_rsp_valid,
    output logic                                   o_mem_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]       i_mem_rsp_data,
    input  wire logic                              i_mem_rsp_last,

    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic [TOKEN_ID_WIDTH-1 : 0]            o_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0]      o_best_score_q26,
    output logic [31 : 0]                          o_tiles_requested,
    output logic [31 : 0]                          o_tiles_completed,
    output logic [31 : 0]                          o_compute_cycle_count,
    output logic [31 : 0]                          o_mem_read_burst_count,
    output logic [31 : 0]                          o_mem_read_word_count
);

    localparam int TILE_WEIGHT_BITS = TILE_ROWS * INPUT_SIZE * WEIGHT_WIDTH;
    localparam int TILE_SCALE_BITS  = TILE_ROWS * GROUP_COUNT * SCALE_WIDTH;

    logic core_tile_req_valid;
    logic core_tile_req_ready;
    logic [TILE_INDEX_W-1 : 0] core_tile_index;
    logic [TOKEN_ID_WIDTH-1 : 0] core_tile_token_base;
    logic core_tile_valid;
    logic core_tile_ready;
    logic [TILE_WEIGHT_BITS-1 : 0] core_tile_weight_flat;
    logic [TILE_SCALE_BITS-1 : 0] core_tile_scale_flat;
    logic core_busy;
    logic core_done;
    logic core_error;

    logic reader_start;
    logic reader_busy;
    logic reader_done;
    logic reader_error;
    logic [TILE_WEIGHT_BITS-1 : 0] reader_tile_weight_flat;
    logic [TILE_SCALE_BITS-1 : 0] reader_tile_scale_flat;
    logic [31 : 0] reader_burst_count;
    logic [31 : 0] reader_word_count;
    logic error_sticky;

    assign core_tile_req_ready = (reader_busy == 1'b0) && (core_tile_valid == 1'b0);
    assign reader_start = core_tile_req_valid && core_tile_req_ready;

    assign core_tile_weight_flat = reader_tile_weight_flat;
    assign core_tile_scale_flat = reader_tile_scale_flat;

    assign o_busy = core_busy || reader_busy || core_tile_valid;
    assign o_done = core_done;
    assign o_error = core_error || reader_error || error_sticky;

    lm_head_argmax_stage #(
        .SCAN_ROWS      (SCAN_ROWS),
        .TILE_ROWS      (TILE_ROWS),
        .ROW_PARALLEL   (ROW_PARALLEL),
        .INPUT_SIZE     (INPUT_SIZE),
        .GROUP_SIZE     (GROUP_SIZE),
        .GROUP_COUNT    (GROUP_COUNT),
        .GROUP_PARALLEL (GROUP_PARALLEL),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACT_FRAC       (ACT_FRAC),
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .SCALE_WIDTH    (SCALE_WIDTH),
        .SCALE_FRAC     (SCALE_FRAC),
        .PARTIAL_WIDTH  (PARTIAL_WIDTH),
        .SCALED_WIDTH   (SCALED_WIDTH),
        .ROW_ACC_WIDTH  (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH (TOKEN_ID_WIDTH),
        .TILE_COUNT     (TILE_COUNT),
        .TILE_INDEX_W   (TILE_INDEX_W)
    ) argmax_core (
        .i_clk                (i_clk),
        .i_rst_n              (i_rst_n),
        .i_start              (i_start),
        .i_token_base         (i_token_base),
        .i_activation_flat    (i_activation_flat),
        .o_tile_req_valid     (core_tile_req_valid),
        .i_tile_req_ready     (core_tile_req_ready),
        .o_tile_index         (core_tile_index),
        .o_tile_token_base    (core_tile_token_base),
        .i_tile_valid         (core_tile_valid),
        .o_tile_ready         (core_tile_ready),
        .i_tile_weight_flat   (core_tile_weight_flat),
        .i_tile_scale_flat    (core_tile_scale_flat),
        .o_busy               (core_busy),
        .o_done               (core_done),
        .o_error              (core_error),
        .o_best_token_id      (o_best_token_id),
        .o_best_score_q26     (o_best_score_q26),
        .o_tiles_requested    (o_tiles_requested),
        .o_tiles_completed    (o_tiles_completed),
        .o_compute_cycle_count(o_compute_cycle_count)
    );

    lm_head_tile_mem_reader #(
        .ADDR_WIDTH    (ADDR_WIDTH),
        .TILE_ROWS     (TILE_ROWS),
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .MAX_READ_BYTES(MAX_READ_BYTES)
    ) tile_reader (
        .i_clk             (i_clk),
        .i_rst_n           (i_rst_n),
        .i_start           (reader_start),
        .i_tile_token_base (core_tile_token_base),
        .i_weight_base_addr(i_weight_base_addr),
        .i_scale_base_addr (i_scale_base_addr),
        .o_busy            (reader_busy),
        .o_done            (reader_done),
        .o_error           (reader_error),
        .o_mem_req_valid   (o_mem_req_valid),
        .i_mem_req_ready   (i_mem_req_ready),
        .o_mem_req_addr    (o_mem_req_addr),
        .o_mem_req_len_bytes(o_mem_req_len_bytes),
        .i_mem_rsp_valid   (i_mem_rsp_valid),
        .o_mem_rsp_ready   (o_mem_rsp_ready),
        .i_mem_rsp_data    (i_mem_rsp_data),
        .i_mem_rsp_last    (i_mem_rsp_last),
        .o_tile_weight_flat(reader_tile_weight_flat),
        .o_tile_scale_flat (reader_tile_scale_flat),
        .o_read_burst_count(reader_burst_count),
        .o_read_word_count (reader_word_count)
    );

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            core_tile_valid        <= 1'b0;
            error_sticky           <= 1'b0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count  <= 32'd0;
        end
        else begin
            if ((i_start == 1'b1) && (o_busy == 1'b0)) begin
                error_sticky           <= 1'b0;
                o_mem_read_burst_count <= 32'd0;
                o_mem_read_word_count  <= 32'd0;
            end

            if (o_mem_req_valid && i_mem_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end

            if (i_mem_rsp_valid && o_mem_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end

            if (reader_done == 1'b1) begin
                core_tile_valid <= 1'b1;
                if (reader_error == 1'b1) begin
                    error_sticky <= 1'b1;
                end
            end
            else if ((core_tile_valid == 1'b1) && (core_tile_ready == 1'b1)) begin
                core_tile_valid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
