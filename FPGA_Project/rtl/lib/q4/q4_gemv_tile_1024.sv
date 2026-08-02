`default_nettype none

// Q4 GEMV tile kernel for multiple 1024-wide output rows.
//
// This module is the next level above q4_gemv_row_1024. It computes OUT_ROWS
// independent output rows in parallel. Every row shares the same 1024-value
// activation vector and owns one 1024-wide packed Q4 weight row plus 16 group
// scales.
//
// Shape contract for the first bring-up target:
//
//   activation: [1024]
//   weights:    [OUT_ROWS, 1024]
//   scales:     [OUT_ROWS, 16]
//   outputs:    [OUT_ROWS]
//
// Each output lane is a signed Q26 integer:
//
//   output_float[row] =
//     signed(o_output_flat[ROW_ACC_WIDTH*row +: ROW_ACC_WIDTH]) /
//     2^(ACT_FRAC + SCALE_FRAC)
//
// OUT_ROWS is intentionally parameterized so Vivado can be tried with
// OUT_ROWS=1, 2, and 4 on the XCZU2EG target. The default ACT_WIDTH=24 matches
// the planned signed Q12.12 RMSNorm output interface; instantiate with
// ACT_WIDTH=16 for the original Q4.12 bring-up vectors.
module q4_gemv_tile_1024 # (
    parameter int OUT_ROWS      = 4,
    parameter int ROW_PARALLEL  = OUT_ROWS,
    parameter int INPUT_SIZE    = 1024,
    parameter int GROUP_SIZE    = 64,
    parameter int GROUP_COUNT   = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL = 4,
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
    input  wire logic                                          i_clk,
    input  wire logic                                          i_rst_n,

    // Start a new OUT_ROWS-row GEMV tile transaction when the module is not
    // busy. Keep all input buses stable until o_done is asserted.
    input  wire logic                                          i_start,

    // Shared INPUT_SIZE signed fixed-point activation vector. Element j is:
    // i_activation_flat[ACT_WIDTH*j +: ACT_WIDTH].
    input  wire logic [INPUT_SIZE*ACT_WIDTH-1 : 0]             i_activation_flat,

    // OUT_ROWS packed Q4 weight rows. Row r is:
    // i_weight_packed_flat[INPUT_SIZE*WEIGHT_WIDTH*r +:
    //                      INPUT_SIZE*WEIGHT_WIDTH].
    input  wire logic [OUT_ROWS*INPUT_SIZE*WEIGHT_WIDTH-1 : 0] i_weight_packed_flat,

    // OUT_ROWS scale rows. Row r contains GROUP_COUNT unsigned Q2.14 scales:
    // i_scale_flat[GROUP_COUNT*SCALE_WIDTH*r +: GROUP_COUNT*SCALE_WIDTH].
    input  wire logic [OUT_ROWS*GROUP_COUNT*SCALE_WIDTH-1 : 0] i_scale_flat,

    // Busy is high after start is accepted and before the tile result is ready.
    output logic                                          o_busy,

    // Done should pulse for one cycle when all OUT_ROWS output lanes are valid.
    output logic                                          o_done,

    // OUT_ROWS signed Q26 row results, flattened little-row-endian.
    // Row r is o_output_flat[ROW_ACC_WIDTH*r +: ROW_ACC_WIDTH].
    output logic [OUT_ROWS*ROW_ACC_WIDTH-1 : 0]           o_output_flat
);

    // Batched row controller.
    //
    // ROW_PARALLEL=OUT_ROWS retains the original all-row parallel behavior.
    // Board-facing builds may set ROW_PARALLEL=1 to reuse one row GEMV across
    // every row in the externally unchanged OUT_ROWS-row tile.

    localparam int ROW_PARALLEL_LOCAL =
        (ROW_PARALLEL < 1) ? 1 :
        ((ROW_PARALLEL > OUT_ROWS) ? OUT_ROWS : ROW_PARALLEL);
    localparam int ROW_BATCH_COUNT =
        (OUT_ROWS + ROW_PARALLEL_LOCAL - 1) / ROW_PARALLEL_LOCAL;
    localparam int ROW_BATCH_INDEX_W =
        (ROW_BATCH_COUNT <= 1) ? 1 : $clog2(ROW_BATCH_COUNT);
    localparam int ROW_INDEX_W =
        (OUT_ROWS <= 1) ? 1 : $clog2(OUT_ROWS + 1);

    localparam logic [2 : 0] IDLE          = 3'd0;
    localparam logic [2 : 0] START_BATCH   = 3'd1;
    localparam logic [2 : 0] WAIT_BATCH    = 3'd2;
    localparam logic [2 : 0] CAPTURE_BATCH = 3'd3;
    localparam logic [2 : 0] DONE          = 3'd4;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;
    logic [ROW_BATCH_INDEX_W-1 : 0] row_batch_index;
    logic row_start;
    logic last_row_batch;
    logic [ROW_PARALLEL_LOCAL-1 : 0] row_busy;
    logic [ROW_PARALLEL_LOCAL-1 : 0] row_done;
    logic [ROW_PARALLEL_LOCAL-1 : 0] row_lane_valid;
    logic [ROW_PARALLEL_LOCAL-1 : 0] row_done_effective;
    logic all_rows_done;
    logic signed [ROW_ACC_WIDTH-1 : 0] row_output [ROW_PARALLEL_LOCAL];

    assign row_start = (current_state == START_BATCH);
    assign all_rows_done = &row_done_effective;
    assign last_row_batch = (row_batch_index == (ROW_BATCH_COUNT - 1));

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
                    next_state = START_BATCH;
                    o_busy     = 1'b1;
                end
            end

            START_BATCH: begin
                next_state = WAIT_BATCH;
                o_busy = 1'b1;
            end

            WAIT_BATCH: begin
                o_busy = 1'b1;
                if (all_rows_done == 1'b1) begin
                    next_state = CAPTURE_BATCH;
                end
            end

            CAPTURE_BATCH: begin
                o_busy = 1'b1;
                if (last_row_batch == 1'b1) begin
                    next_state = DONE;
                end
                else begin
                    next_state = START_BATCH;
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

    genvar row_lane;

    generate
        for (row_lane = 0; row_lane < ROW_PARALLEL_LOCAL; row_lane = row_lane + 1) begin : gen_q4_gemv_row_1024
            localparam int LANE_INDEX = row_lane;
            logic [ROW_INDEX_W-1 : 0] active_row_index;
            logic [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] lane_weight_packed;
            logic [GROUP_COUNT*SCALE_WIDTH-1 : 0] lane_scale_flat;

            assign active_row_index =
                (row_batch_index * ROW_PARALLEL_LOCAL) + LANE_INDEX;
            assign row_lane_valid[row_lane] = (active_row_index < OUT_ROWS);
            assign row_done_effective[row_lane] =
                row_lane_valid[row_lane] ? row_done[row_lane] : 1'b1;
            assign lane_weight_packed =
                row_lane_valid[row_lane] ?
                i_weight_packed_flat[
                    active_row_index*INPUT_SIZE*WEIGHT_WIDTH +:
                    INPUT_SIZE*WEIGHT_WIDTH
                ] :
                'd0;
            assign lane_scale_flat =
                row_lane_valid[row_lane] ?
                i_scale_flat[
                    active_row_index*GROUP_COUNT*SCALE_WIDTH +:
                    GROUP_COUNT*SCALE_WIDTH
                ] :
                'd0;

            q4_gemv_row_1024 #(
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
            ) inst_q4_gemv_row_1024 (
                .i_clk             (i_clk),
                .i_rst_n           (i_rst_n),
                .i_start           (row_start && row_lane_valid[row_lane]),
                .i_activation_flat (i_activation_flat),
                .i_weight_packed   (lane_weight_packed),
                .i_scale_flat      (lane_scale_flat),
                .o_busy            (row_busy[row_lane]),
                .o_done            (row_done[row_lane]),
                .o_row_sum_q26     (row_output[row_lane])
            );
        end
    endgenerate

    integer capture_lane;

    // Capture each completed row batch into the externally unchanged tile.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_output_flat <= 'd0;
            row_batch_index <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_output_flat <= 'd0;
                    end
                    row_batch_index <= 'd0;
                end

                CAPTURE_BATCH: begin
                    for (
                        capture_lane = 0;
                        capture_lane < ROW_PARALLEL_LOCAL;
                        capture_lane = capture_lane + 1
                    ) begin
                        if (
                            ((row_batch_index * ROW_PARALLEL_LOCAL) + capture_lane)
                            < OUT_ROWS
                        ) begin
                            o_output_flat[
                                ((row_batch_index * ROW_PARALLEL_LOCAL) + capture_lane)
                                * ROW_ACC_WIDTH +: ROW_ACC_WIDTH
                            ] <= row_output[capture_lane];
                        end
                    end
                    if (last_row_batch == 1'b0) begin
                        row_batch_index <= row_batch_index + 1'b1;
                    end
                end

                default: begin
                end
            endcase
        end
    end

endmodule

`default_nettype wire
