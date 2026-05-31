`default_nettype none

// Q4 GEMV projection controller for a 1024-wide input vector.
//
// This module is the next level above q4_gemv_tile_1024. It computes a full
// projection matrix by reusing one tile engine across multiple output-row
// tiles.
//
// Shape contract:
//
//   activation: [1024]
//   weights:    [OUT_FEATURES, 1024]
//   scales:     [OUT_FEATURES, 16]
//   outputs:    [OUT_FEATURES]
//
// OUT_FEATURES is the projection output size:
//
//   q_proj: OUT_FEATURES = 2048
//   k_proj: OUT_FEATURES = 1024
//   v_proj: OUT_FEATURES = 1024
//
// TILE_ROWS controls how many rows the reused tile computes in parallel.
// The first bring-up target uses TILE_ROWS=4, matching q4_gemv_tile_1024's
// default architecture. OUT_FEATURES must be divisible by TILE_ROWS. The
// default ACT_WIDTH=24 matches the planned signed Q12.12 RMSNorm output path;
// instantiate with ACT_WIDTH=16 for the original Q4.12 bring-up vectors.
module q4_gemv_projection_1024 # (
    parameter int OUT_FEATURES  = 2048,
    parameter int TILE_ROWS     = 4,
    parameter int INPUT_SIZE    = 1024,
    parameter int GROUP_SIZE    = 64,
    parameter int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH     = 24,
    parameter int ACT_FRAC      = 12,
    parameter int WEIGHT_WIDTH  = 4,
    parameter int SCALE_WIDTH   = 16,
    parameter int SCALE_FRAC    = 14,
    parameter int PARTIAL_WIDTH = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2
)
(
    input  logic                                              i_clk,
    input  logic                                              i_rst_n,

    // Start a new projection transaction when the controller is not busy.
    // Keep activation, weights, and scales stable until o_done is asserted.
    input  logic                                              i_start,

    // Shared INPUT_SIZE signed fixed-point activation vector.
    input  logic [INPUT_SIZE*ACT_WIDTH-1 : 0]                 i_activation_flat,

    // Full packed Q4 projection matrix. Row r is:
    // i_weight_packed_flat[INPUT_SIZE*WEIGHT_WIDTH*r +:
    //                      INPUT_SIZE*WEIGHT_WIDTH].
    input  logic [OUT_FEATURES*INPUT_SIZE*WEIGHT_WIDTH-1 : 0] i_weight_packed_flat,

    // Full scale matrix. Row r contains GROUP_COUNT unsigned Q2.14 scales:
    // i_scale_flat[GROUP_COUNT*SCALE_WIDTH*r +: GROUP_COUNT*SCALE_WIDTH].
    input  logic [OUT_FEATURES*GROUP_COUNT*SCALE_WIDTH-1 : 0] i_scale_flat,

    // Busy is high after start is accepted and before the projection is ready.
    output logic                                              o_busy,

    // Done pulses for one cycle when all OUT_FEATURES output lanes are valid.
    output logic                                              o_done,

    // OUT_FEATURES signed Q26 row results, flattened little-row-endian.
    output logic [OUT_FEATURES*ROW_ACC_WIDTH-1 : 0]           o_output_flat
);

    // Controller structure:
    //
    // 1. IDLE: wait for i_start.
    // 2. START_TILE: pulse the q4_gemv_tile_1024 instance for the current tile.
    // 3. WAIT_TILE: wait for the tile's done pulse.
    // 4. STORE_TILE: advance to the next tile after the just-finished tile's
    //    result has been captured.
    // 5. DONE: pulse o_done after the final tile has been stored.

    localparam int                            TILE_COUNT       = OUT_FEATURES / TILE_ROWS;
    localparam int                            TILE_INDEX_WIDTH = (TILE_COUNT <= 1) ? 1 : $clog2(TILE_COUNT);
    localparam int                            TILE_WEIGHT_BITS = TILE_ROWS * INPUT_SIZE * WEIGHT_WIDTH;
    localparam int                            TILE_SCALE_BITS  = TILE_ROWS * GROUP_COUNT * SCALE_WIDTH;
    localparam int                            TILE_OUTPUT_BITS = TILE_ROWS * ROW_ACC_WIDTH;
    localparam logic [TILE_INDEX_WIDTH-1 : 0] LAST_TILE_INDEX  = TILE_COUNT - 1;

    localparam IDLE       = 3'd0;
    localparam START_TILE = 3'd1;
    localparam WAIT_TILE  = 3'd2;
    localparam STORE_TILE = 3'd3;
    localparam DONE       = 3'd4;

    logic [2:0]                    current_state;
    logic [2:0]                    next_state;
    logic [TILE_INDEX_WIDTH-1 : 0] tile_index;
    logic                          last_tile;

    logic                          tile_start;
    logic                          tile_busy;
    logic                          tile_done;
    logic [TILE_WEIGHT_BITS-1 : 0] tile_weight_packed_flat;
    logic [TILE_SCALE_BITS-1 : 0]  tile_scale_flat;
    logic [TILE_OUTPUT_BITS-1 : 0] tile_output_flat;

    assign last_tile               = (tile_index == LAST_TILE_INDEX);
    assign tile_start              = (current_state == START_TILE);
    assign tile_weight_packed_flat = i_weight_packed_flat[tile_index*TILE_WEIGHT_BITS +: TILE_WEIGHT_BITS];
    assign tile_scale_flat         = i_scale_flat[tile_index*TILE_SCALE_BITS +: TILE_SCALE_BITS];

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            current_state <= IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end

    always_comb begin
        next_state = current_state;
        o_busy = 1'b0;
        o_done = 1'b0;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_TILE;
                    o_busy     = 1'b1;
                end
            end

            START_TILE: begin
                next_state = WAIT_TILE;
                o_busy     = 1'b1;
            end

            WAIT_TILE: begin
                o_busy = 1'b1;
                if (tile_done == 1'b1) begin
                    next_state = STORE_TILE;
                end
            end

            STORE_TILE: begin
                o_busy = 1'b1;
                if (last_tile == 1'b1) begin
                    next_state = DONE;
                end
                else begin
                    next_state = START_TILE;
                end
            end

            DONE: begin
                o_done     = 1'b1;
                next_state = IDLE;
            end

            default: begin
                next_state = IDLE;
            end
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            tile_index <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    tile_index <= 'd0;
                end

                STORE_TILE: begin
                    if (last_tile == 1'b0) begin
                        tile_index <= tile_index + 1'b1;
                    end
                end

                DONE: begin
                    tile_index <= 'd0;
                end

                default: begin
                    tile_index <= tile_index;
                end
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_output_flat <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_output_flat <= 'd0;
                    end
                end

                WAIT_TILE: begin
                    if (tile_done == 1'b1) begin
                        o_output_flat[tile_index*TILE_OUTPUT_BITS +: TILE_OUTPUT_BITS] <= tile_output_flat;
                    end
                end

                default: begin
                    o_output_flat <= o_output_flat;
                end
            endcase
        end
    end

    q4_gemv_tile_1024 #(
        .OUT_ROWS      (TILE_ROWS),
        .INPUT_SIZE    (INPUT_SIZE),
        .GROUP_SIZE    (GROUP_SIZE),
        .GROUP_COUNT   (GROUP_COUNT),
        .ACT_WIDTH     (ACT_WIDTH),
        .ACT_FRAC      (ACT_FRAC),
        .WEIGHT_WIDTH  (WEIGHT_WIDTH),
        .SCALE_WIDTH   (SCALE_WIDTH),
        .SCALE_FRAC    (SCALE_FRAC),
        .PARTIAL_WIDTH (PARTIAL_WIDTH),
        .SCALED_WIDTH  (SCALED_WIDTH),
        .ROW_ACC_WIDTH (ROW_ACC_WIDTH)
    ) inst_q4_gemv_tile_1024 (
        .i_clk                (i_clk),
        .i_rst_n              (i_rst_n),
        .i_start              (tile_start),
        .i_activation_flat    (i_activation_flat),
        .i_weight_packed_flat (tile_weight_packed_flat),
        .i_scale_flat         (tile_scale_flat),
        .o_busy               (tile_busy),
        .o_done               (tile_done),
        .o_output_flat        (tile_output_flat)
    );

endmodule

`default_nettype wire
