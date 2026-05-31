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
    input  logic                                          i_clk,
    input  logic                                          i_rst_n,

    // Start a new OUT_ROWS-row GEMV tile transaction when the module is not
    // busy. Keep all input buses stable until o_done is asserted.
    input  logic                                          i_start,

    // Shared INPUT_SIZE signed fixed-point activation vector. Element j is:
    // i_activation_flat[ACT_WIDTH*j +: ACT_WIDTH].
    input  logic [INPUT_SIZE*ACT_WIDTH-1 : 0]             i_activation_flat,

    // OUT_ROWS packed Q4 weight rows. Row r is:
    // i_weight_packed_flat[INPUT_SIZE*WEIGHT_WIDTH*r +:
    //                      INPUT_SIZE*WEIGHT_WIDTH].
    input  logic [OUT_ROWS*INPUT_SIZE*WEIGHT_WIDTH-1 : 0] i_weight_packed_flat,

    // OUT_ROWS scale rows. Row r contains GROUP_COUNT unsigned Q2.14 scales:
    // i_scale_flat[GROUP_COUNT*SCALE_WIDTH*r +: GROUP_COUNT*SCALE_WIDTH].
    input  logic [OUT_ROWS*GROUP_COUNT*SCALE_WIDTH-1 : 0] i_scale_flat,

    // Busy is high after start is accepted and before the tile result is ready.
    output logic                                          o_busy,

    // Done should pulse for one cycle when all OUT_ROWS output lanes are valid.
    output logic                                          o_done,

    // OUT_ROWS signed Q26 row results, flattened little-row-endian.
    // Row r is o_output_flat[ROW_ACC_WIDTH*r +: ROW_ACC_WIDTH].
    output logic [OUT_ROWS*ROW_ACC_WIDTH-1 : 0]           o_output_flat
);

    // OUT_ROWS parallel row controller.
    //
    // 1. IDLE: wait for i_start.
    // 2. PARALLEL_ROWS: pulse all OUT_ROWS q4_gemv_row_1024 instances and wait
    //    until every row instance asserts done.
    // 3. DONE: o_output_flat contains all OUT_ROWS signed Q26 row results.

    localparam IDLE          = 2'd0;
    localparam PARALLEL_ROWS = 2'd1;
    localparam DONE          = 2'd2;

    logic        [1 : 0]               current_state;
    logic        [1 : 0]               next_state;

    logic                              row_start;
    logic        [OUT_ROWS-1 : 0]      row_busy;
    logic        [OUT_ROWS-1 : 0]      row_done;
    logic                              all_rows_done;
    logic signed [ROW_ACC_WIDTH-1 : 0] row_output [OUT_ROWS];

    assign row_start     = (current_state == IDLE) && (i_start == 1'b1);
    assign all_rows_done = &row_done;

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
                    next_state = PARALLEL_ROWS;
                    o_busy     = 1'b1;
                end
            end

            PARALLEL_ROWS: begin
                o_busy = 1'b1;
                if (all_rows_done == 1'b1) begin
                    next_state = DONE;
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

    genvar row_index;

    generate
        for (row_index = 0; row_index < OUT_ROWS; row_index = row_index + 1) begin : gen_q4_gemv_row_1024
            q4_gemv_row_1024 #(
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
            ) inst_q4_gemv_row_1024 (
                .i_clk             (i_clk),
                .i_rst_n           (i_rst_n),
                .i_start           (row_start),
                .i_activation_flat (i_activation_flat),
                .i_weight_packed   (i_weight_packed_flat[row_index*INPUT_SIZE*WEIGHT_WIDTH +: INPUT_SIZE*WEIGHT_WIDTH]),
                .i_scale_flat      (i_scale_flat[row_index*GROUP_COUNT*SCALE_WIDTH +: GROUP_COUNT*SCALE_WIDTH]),
                .o_busy            (row_busy[row_index]),
                .o_done            (row_done[row_index]),
                .o_row_sum_q26     (row_output[row_index])
            );

            assign o_output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH] = row_output[row_index];
        end
    endgenerate

endmodule

`default_nettype wire
