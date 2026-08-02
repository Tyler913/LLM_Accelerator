`default_nettype none

// Tiled Q4 LM-head scan plus greedy argmax.
//
// The stage consumes one final RMSNorm vector and scans a contiguous range of
// tied LM-head rows. A future memory reader supplies one TILE_ROWS-row Q4
// weight/scale tile per request; this block computes tile logits and keeps only
// the running best token/score.
module lm_head_argmax_stage #(
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
    parameter int TILE_INDEX_W    = (TILE_COUNT <= 1) ? 1 : $clog2(TILE_COUNT)
)
(
    input  wire logic                                                   i_clk,
    input  wire logic                                                   i_rst_n,

    input  wire logic                                                   i_start,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]                            i_token_base,
    input  wire logic [INPUT_SIZE*ACT_WIDTH-1 : 0]                      i_activation_flat,

    output logic                                                   o_tile_req_valid,
    input  wire logic                                                   i_tile_req_ready,
    output logic [TILE_INDEX_W-1 : 0]                              o_tile_index,
    output logic [TOKEN_ID_WIDTH-1 : 0]                            o_tile_token_base,

    input  wire logic                                                   i_tile_valid,
    output logic                                                   o_tile_ready,
    input  wire logic [TILE_ROWS*INPUT_SIZE*WEIGHT_WIDTH-1 : 0]         i_tile_weight_flat,
    input  wire logic [TILE_ROWS*GROUP_COUNT*SCALE_WIDTH-1 : 0]         i_tile_scale_flat,

    output logic                                                   o_busy,
    output logic                                                   o_done,
    output logic                                                   o_error,
    output logic [TOKEN_ID_WIDTH-1 : 0]                            o_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0]                      o_best_score_q26,
    output logic [31 : 0]                                         o_tiles_requested,
    output logic [31 : 0]                                         o_tiles_completed,
    output logic [31 : 0]                                         o_compute_cycle_count
);

    localparam int TILE_WEIGHT_BITS = TILE_ROWS * INPUT_SIZE * WEIGHT_WIDTH;
    localparam int TILE_SCALE_BITS  = TILE_ROWS * GROUP_COUNT * SCALE_WIDTH;

    localparam IDLE          = 3'd0;
    localparam REQUEST_TILE  = 3'd1;
    localparam WAIT_RESPONSE = 3'd2;
    localparam START_TILE    = 3'd3;
    localparam WAIT_TILE     = 3'd4;
    localparam UPDATE_BEST   = 3'd5;
    localparam DONE          = 3'd6;

    localparam logic [TILE_INDEX_W-1 : 0] LAST_TILE_INDEX = TILE_COUNT - 1;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;
    logic [TILE_INDEX_W-1 : 0] tile_index_reg;
    logic [TOKEN_ID_WIDTH-1 : 0] token_base_reg;
    logic tile_req_fire;
    logic tile_resp_fire;
    logic tile_start;
    logic tile_busy;
    logic tile_done;
    logic last_tile;
    logic best_valid_reg;
    logic error_reg;

    logic [TILE_WEIGHT_BITS-1 : 0] tile_weight_reg;
    logic [TILE_SCALE_BITS-1 : 0] tile_scale_reg;
    logic [TILE_ROWS*ROW_ACC_WIDTH-1 : 0] tile_output_flat;

    logic signed [ROW_ACC_WIDTH-1 : 0] tile_best_score;
    logic signed [ROW_ACC_WIDTH-1 : 0] scan_score;
    logic [$clog2(TILE_ROWS)-1 : 0] tile_best_row;
    logic [TOKEN_ID_WIDTH-1 : 0] tile_best_token_id;

    integer row_index;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;

    assign o_tile_req_valid = (current_state == REQUEST_TILE);
    assign o_tile_ready = (current_state == WAIT_RESPONSE);
    assign o_tile_index = tile_index_reg;
    assign o_tile_token_base = token_base_reg + (tile_index_reg * TILE_ROWS);

    assign tile_req_fire = o_tile_req_valid && i_tile_req_ready;
    assign tile_resp_fire = i_tile_valid && o_tile_ready;
    assign tile_start = (current_state == START_TILE);
    assign last_tile = (tile_index_reg == LAST_TILE_INDEX);
    assign tile_best_token_id = o_tile_token_base + tile_best_row;

    always @* begin
        tile_best_score = $signed(tile_output_flat[0 +: ROW_ACC_WIDTH]);
        tile_best_row = 'd0;

        for (row_index = 1; row_index < TILE_ROWS; row_index = row_index + 1) begin
            scan_score = $signed(tile_output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH]);
            if (scan_score > tile_best_score) begin
                tile_best_score = scan_score;
                tile_best_row = row_index;
            end
        end
    end

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = REQUEST_TILE;
                end
            end

            REQUEST_TILE: begin
                if (tile_req_fire == 1'b1) begin
                    next_state = WAIT_RESPONSE;
                end
            end

            WAIT_RESPONSE: begin
                if (tile_resp_fire == 1'b1) begin
                    next_state = START_TILE;
                end
            end

            START_TILE: begin
                next_state = WAIT_TILE;
            end

            WAIT_TILE: begin
                if (tile_done == 1'b1) begin
                    next_state = UPDATE_BEST;
                end
            end

            UPDATE_BEST: begin
                if (last_tile == 1'b1) begin
                    next_state = DONE;
                end
                else begin
                    next_state = REQUEST_TILE;
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
            tile_index_reg <= 'd0;
            token_base_reg <= 'd0;
            tile_weight_reg <= 'd0;
            tile_scale_reg <= 'd0;
            best_valid_reg <= 1'b0;
            o_best_token_id <= 'd0;
            o_best_score_q26 <= 'd0;
            o_tiles_requested <= 32'd0;
            o_tiles_completed <= 32'd0;
            o_compute_cycle_count <= 32'd0;
            error_reg <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        tile_index_reg <= 'd0;
                        token_base_reg <= i_token_base;
                        tile_weight_reg <= 'd0;
                        tile_scale_reg <= 'd0;
                        best_valid_reg <= 1'b0;
                        o_best_token_id <= 'd0;
                        o_best_score_q26 <= 'd0;
                        o_tiles_requested <= 32'd0;
                        o_tiles_completed <= 32'd0;
                        o_compute_cycle_count <= 32'd0;
                        error_reg <= 1'b0;
                    end
                end

                REQUEST_TILE: begin
                    if (tile_req_fire == 1'b1) begin
                        o_tiles_requested <= o_tiles_requested + 1'b1;
                    end
                end

                WAIT_RESPONSE: begin
                    if (tile_resp_fire == 1'b1) begin
                        tile_weight_reg <= i_tile_weight_flat;
                        tile_scale_reg <= i_tile_scale_flat;
                    end
                end

                START_TILE: begin
                    tile_weight_reg <= tile_weight_reg;
                    tile_scale_reg <= tile_scale_reg;
                end

                WAIT_TILE: begin
                    o_compute_cycle_count <= o_compute_cycle_count + 1'b1;
                end

                UPDATE_BEST: begin
                    if ((best_valid_reg == 1'b0) || (tile_best_score > o_best_score_q26)) begin
                        best_valid_reg <= 1'b1;
                        o_best_score_q26 <= tile_best_score;
                        o_best_token_id <= tile_best_token_id;
                    end
                    o_tiles_completed <= o_tiles_completed + 1'b1;
                    if (last_tile == 1'b0) begin
                        tile_index_reg <= tile_index_reg + 1'b1;
                    end
                end

                DONE: begin
                    error_reg <=
                        error_reg ||
                        (best_valid_reg == 1'b0) ||
                        (o_tiles_requested != TILE_COUNT) ||
                        (o_tiles_completed != TILE_COUNT);
                    tile_index_reg <= 'd0;
                end

                default: begin
                    error_reg <= 1'b1;
                    tile_index_reg <= 'd0;
                    token_base_reg <= 'd0;
                    tile_weight_reg <= 'd0;
                    tile_scale_reg <= 'd0;
                    best_valid_reg <= 1'b0;
                    o_best_token_id <= 'd0;
                    o_best_score_q26 <= 'd0;
                    o_tiles_requested <= 32'd0;
                    o_tiles_completed <= 32'd0;
                    o_compute_cycle_count <= 32'd0;
                end
            endcase
        end
    end

    q4_gemv_tile_1024 #(
        .OUT_ROWS      (TILE_ROWS),
        .ROW_PARALLEL  (ROW_PARALLEL),
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH     (ACT_WIDTH),
        .ACT_FRAC      (ACT_FRAC),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .SCALE_FRAC    (SCALE_FRAC),
        .PARTIAL_WIDTH (PARTIAL_WIDTH),
        .SCALED_WIDTH  (SCALED_WIDTH),
        .ROW_ACC_WIDTH (ROW_ACC_WIDTH)
    ) lm_head_tile (
        .i_clk               (i_clk),
        .i_rst_n             (i_rst_n),
        .i_start             (tile_start),
        .i_activation_flat   (i_activation_flat),
        .i_weight_packed_flat(tile_weight_reg),
        .i_scale_flat        (tile_scale_reg),
        .o_busy              (tile_busy),
        .o_done              (tile_done),
        .o_output_flat       (tile_output_flat)
    );

endmodule

`default_nettype wire
