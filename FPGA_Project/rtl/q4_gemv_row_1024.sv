`default_nettype none

// Q4 GEMV kernel for one 1024-wide output row.
//
// This module is the next level above q4_dot_product_64. A 1024-wide row is
// split into 16 groups of 64 values. Each group produces one signed Q26 scaled
// sum, and this module will accumulate those 16 group sums into one row result.
//
// Math contract:
//
//   for group = 0..15:
//     partial[group] =
//       sum_j activation[group*64 + j] * weight_q4[group*64 + j]
//     scaled[group] = partial[group] * scale_q2_14[group]
//
//   row_sum_q26 = sum_group scaled[group]
//
// The binary point of row_sum_q26 is after ACT_FRAC + SCALE_FRAC bits. The
// default ACT_WIDTH=24 matches the planned signed Q12.12 RMSNorm output path;
// instantiate with ACT_WIDTH=16 for the original Q4.12 bring-up vectors.
module q4_gemv_row_1024 # (
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
    input  wire logic                                        i_clk,
    input  wire logic                                        i_rst_n,

    // Start a new one-row GEMV transaction when the module is not busy.
    // Keep all input buses stable until o_done is asserted.
    input  wire logic                                        i_start,

    // INPUT_SIZE signed fixed-point activations, flattened little-element-endian:
    // element j is i_activation_flat[ACT_WIDTH*j +: ACT_WIDTH].
    input  wire logic        [INPUT_SIZE*ACT_WIDTH-1 : 0]    i_activation_flat,

    // 1024 signed int4 Q4 weights for one output row. Two weights are packed
    // per byte using the same low-nibble/even-index rule as q4_dot_product_64.
    input  wire logic        [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] i_weight_packed,

    // One unsigned Q2.14 scale per 64-weight group:
    // group g is i_scale_flat[SCALE_WIDTH*g +: SCALE_WIDTH].
    input  wire logic        [GROUP_COUNT*SCALE_WIDTH-1 : 0] i_scale_flat,

    // Busy is high after start is accepted and before the row result is ready.
    output logic                                        o_busy,

    // Done should pulse for one cycle when o_row_sum_q26 is valid.
    output logic                                        o_done,

    // Accumulated row result in Q26 integer form. Convert to float with:
    // row_float = o_row_sum_q26 / 2^(ACT_FRAC + SCALE_FRAC).
    output logic signed [ROW_ACC_WIDTH-1 : 0]           o_row_sum_q26
);

    // Batched group controller.
    //
    // Current structure:
    // 1. IDLE: wait for i_start.
    // 2. START_BATCH: pulse up to GROUP_PARALLEL q4_dot_product_64 instances.
    // 3. WAIT_BATCH: wait until the active group lanes complete.
    // 4. ACCUMULATE: add that batch's scaled Q26 sum into o_row_sum_q26.
    // 5. DONE: o_row_sum_q26 is valid for the row.

    localparam int GROUP_PARALLEL_LOCAL = (GROUP_PARALLEL > GROUP_COUNT) ? GROUP_COUNT : GROUP_PARALLEL;
    localparam int BATCH_COUNT          = (GROUP_COUNT + GROUP_PARALLEL_LOCAL - 1) / GROUP_PARALLEL_LOCAL;
    localparam int BATCH_INDEX_WIDTH    = (BATCH_COUNT <= 1) ? 1 : $clog2(BATCH_COUNT);
    localparam int GROUP_INDEX_WIDTH    = (GROUP_COUNT <= 1) ? 1 : $clog2(GROUP_COUNT);

    localparam IDLE        = 3'd0;
    localparam START_BATCH = 3'd1;
    localparam WAIT_BATCH  = 3'd2;
    localparam ACCUMULATE  = 3'd3;
    localparam DONE        = 3'd4;

    logic [2:0] current_state;
    logic [2:0] next_state;
    logic       batch_compute_done;
    logic       last_batch;
    logic [BATCH_INDEX_WIDTH-1 : 0] batch_index;

    
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            current_state <= IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end

    
    always_comb begin
        if (i_rst_n == 1'b0) begin
            next_state = IDLE;
            o_busy     = 1'b0;
            o_done     = 1'b0;
        end
        else begin
            next_state = current_state;
            o_busy     = 1'b0;
            o_done     = 1'b0;

            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        next_state = START_BATCH;
                        o_busy     = 1'b1;
                        o_done     = 1'b0;
                    end
                end

                START_BATCH: begin
                    next_state = WAIT_BATCH;
                    o_busy     = 1'b1;
                    o_done     = 1'b0;
                end

                WAIT_BATCH: begin
                    if (batch_compute_done == 1'b1) begin
                        next_state = ACCUMULATE;
                    end
                    o_busy     = 1'b1;
                    o_done     = 1'b0;
                end

                ACCUMULATE: begin
                    if (last_batch == 1'b1) begin
                        next_state = DONE;
                    end
                    else begin
                        next_state = START_BATCH;
                    end
                    o_busy     = 1'b1;
                    o_done     = 1'b0;
                end

                DONE: begin
                    next_state = IDLE;
                    o_busy     = 1'b0;
                    o_done     = 1'b1;
                end

                default: begin
                    next_state = IDLE;
                    o_busy     = 1'b0;
                    o_done     = 1'b0;
                end
            endcase
        end
    end


    logic                              instant_start;
    logic        [GROUP_PARALLEL_LOCAL-1 : 0]   instant_busy;
    logic        [GROUP_PARALLEL_LOCAL-1 : 0]   instant_done;
    logic        [GROUP_PARALLEL_LOCAL-1 : 0]   instant_lane_valid;
    logic        [GROUP_PARALLEL_LOCAL-1 : 0]   instant_done_effective;
    logic signed [PARTIAL_WIDTH-1 : 0]          instant_partial_sum    [GROUP_PARALLEL_LOCAL];
    logic signed [SCALED_WIDTH-1 : 0]           instant_scaled_sum_q26 [GROUP_PARALLEL_LOCAL];
    logic signed [ROW_ACC_WIDTH-1 : 0]          batch_scaled_sum;

    assign batch_compute_done = &instant_done_effective;
    assign instant_start      = (current_state == START_BATCH);
    assign last_batch         = (batch_index == (BATCH_COUNT - 1));

    genvar group_index;

    generate
        for (group_index = 0 ; group_index < GROUP_PARALLEL_LOCAL ; group_index = group_index + 1) begin : gen_q4_dot_product_64
            localparam int LANE_INDEX = group_index;
            logic [GROUP_INDEX_WIDTH : 0] active_group_index;
            logic [GROUP_SIZE*ACT_WIDTH-1 : 0] lane_activation_flat;
            logic [GROUP_SIZE*WEIGHT_WIDTH-1 : 0] lane_weight_packed;
            logic [SCALE_WIDTH-1 : 0] lane_scale_q2_14;

            assign active_group_index =
                (batch_index * GROUP_PARALLEL_LOCAL) + LANE_INDEX;
            assign instant_lane_valid[group_index] = (active_group_index < GROUP_COUNT);
            assign instant_done_effective[group_index] =
                instant_lane_valid[group_index] ? instant_done[group_index] : 1'b1;
            assign lane_activation_flat =
                instant_lane_valid[group_index] ?
                i_activation_flat[active_group_index*GROUP_SIZE*ACT_WIDTH +: GROUP_SIZE*ACT_WIDTH] :
                'd0;
            assign lane_weight_packed =
                instant_lane_valid[group_index] ?
                i_weight_packed[active_group_index*GROUP_SIZE*WEIGHT_WIDTH +: GROUP_SIZE*WEIGHT_WIDTH] :
                'd0;
            assign lane_scale_q2_14 =
                instant_lane_valid[group_index] ?
                i_scale_flat[active_group_index*SCALE_WIDTH +: SCALE_WIDTH] :
                'd0;

            q4_dot_product_64 #(
                .GROUP_SIZE    (GROUP_SIZE),
                .ACT_WIDTH     (ACT_WIDTH),
                .ACT_FRAC      (ACT_FRAC),
                .WEIGHT_WIDTH  (WEIGHT_WIDTH),
                .SCALE_WIDTH   (SCALE_WIDTH),
                .SCALE_FRAC    (SCALE_FRAC),
                .PRODUCT_WIDTH (ACT_WIDTH + WEIGHT_WIDTH),
                .PARTIAL_WIDTH (PARTIAL_WIDTH),
                .SCALED_WIDTH  (SCALED_WIDTH)
            ) inst_q4_dot_product_64 (
                .i_clk             (i_clk),
                .i_rst_n           (i_rst_n),

                .i_start           (instant_start && instant_lane_valid[group_index]),
                .i_activation_flat (lane_activation_flat),
                .i_weight_packed   (lane_weight_packed),
                .i_scale_q2_14     (lane_scale_q2_14),

                .o_busy            (instant_busy[group_index]),
                .o_done            (instant_done[group_index]),
                .o_partial_sum     (instant_partial_sum[group_index]),
                .o_scaled_sum_q26  (instant_scaled_sum_q26[group_index])
            );
        end
    endgenerate

    integer i;

    always @* begin
        batch_scaled_sum = 'd0;

        for (i = 0 ; i < GROUP_PARALLEL_LOCAL ; i = i + 1) begin
            if (instant_lane_valid[i] == 1'b1) begin
                batch_scaled_sum = batch_scaled_sum + $signed({{(ROW_ACC_WIDTH-SCALED_WIDTH){instant_scaled_sum_q26[i][SCALED_WIDTH-1]}}, instant_scaled_sum_q26[i]});
            end
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_row_sum_q26 <= 'd0;
            batch_index   <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    o_row_sum_q26 <= 'd0;
                    batch_index   <= 'd0;
                end
                START_BATCH: begin
                    o_row_sum_q26 <= o_row_sum_q26;
                    batch_index   <= batch_index;
                end
                WAIT_BATCH: begin
                    o_row_sum_q26 <= o_row_sum_q26;
                    batch_index   <= batch_index;
                end
                ACCUMULATE: begin
                    o_row_sum_q26 <= o_row_sum_q26 + batch_scaled_sum;
                    if (last_batch == 1'b0) begin
                        batch_index <= batch_index + 1'b1;
                    end
                end
                DONE: begin
                    o_row_sum_q26 <= o_row_sum_q26;
                    batch_index   <= batch_index;
                end
                default: begin
                    o_row_sum_q26 <= 'd0;
                    batch_index   <= 'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
