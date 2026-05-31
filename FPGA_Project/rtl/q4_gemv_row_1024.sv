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
    input  logic                                        i_clk,
    input  logic                                        i_rst_n,

    // Start a new one-row GEMV transaction when the module is not busy.
    // Keep all input buses stable until o_done is asserted.
    input  logic                                        i_start,

    // INPUT_SIZE signed fixed-point activations, flattened little-element-endian:
    // element j is i_activation_flat[ACT_WIDTH*j +: ACT_WIDTH].
    input  logic        [INPUT_SIZE*ACT_WIDTH-1 : 0]    i_activation_flat,

    // 1024 signed int4 Q4 weights for one output row. Two weights are packed
    // per byte using the same low-nibble/even-index rule as q4_dot_product_64.
    input  logic        [INPUT_SIZE*WEIGHT_WIDTH-1 : 0] i_weight_packed,

    // One unsigned Q2.14 scale per 64-weight group:
    // group g is i_scale_flat[SCALE_WIDTH*g +: SCALE_WIDTH].
    input  logic        [GROUP_COUNT*SCALE_WIDTH-1 : 0] i_scale_flat,

    // Busy is high after start is accepted and before the row result is ready.
    output logic                                        o_busy,

    // Done should pulse for one cycle when o_row_sum_q26 is valid.
    output logic                                        o_done,

    // Accumulated row result in Q26 integer form. Convert to float with:
    // row_float = o_row_sum_q26 / 2^(ACT_FRAC + SCALE_FRAC).
    output logic signed [ROW_ACC_WIDTH-1 : 0]           o_row_sum_q26
);

    // Parallel 16-group controller.
    //
    // Current structure:
    // 1. IDLE: wait for i_start.
    // 2. PARALLEL_COMPUTE: pulse all 16 q4_dot_product_64 instances and wait
    //    until every instance asserts done.
    // 3. ACCUMULATE: register the combinational sum of the 16 scaled Q26 group
    //    results into o_row_sum_q26.
    // 4. DONE: o_row_sum_q26 is valid for the row.

    localparam IDLE             = 2'd0;
    localparam PARALLEL_COMPUTE = 2'd1;
    localparam ACCUMULATE       = 2'd2;
    localparam DONE             = 2'd3;

    logic [1:0] current_state;
    logic [1:0] next_state;
    logic       parallel_compute_done;

    
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
                        next_state = PARALLEL_COMPUTE;
                        o_busy     = 1'b1;
                        o_done     = 1'b0;
                    end
                end

                PARALLEL_COMPUTE: begin
                    if (parallel_compute_done == 1'b1) begin
                        next_state = ACCUMULATE;
                    end
                    o_busy     = 1'b1;
                    o_done     = 1'b0;
                end

                ACCUMULATE: begin
                    next_state = DONE;
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
    logic        [GROUP_COUNT-1 : 0]   instant_busy;
    logic        [GROUP_COUNT-1 : 0]   instant_done;
    logic signed [PARTIAL_WIDTH-1 : 0] instant_partial_sum    [GROUP_COUNT];
    logic signed [SCALED_WIDTH-1 : 0]  instant_scaled_sum_q26 [GROUP_COUNT];
    logic signed [ROW_ACC_WIDTH-1 : 0] row_instant_scaled_sum;

    assign parallel_compute_done = &instant_done;
    assign instant_start         = (current_state == IDLE) && (i_start == 1'b1);

    genvar group_index;

    generate
        for (group_index = 0 ; group_index < GROUP_COUNT ; group_index = group_index + 1) begin : gen_q4_dot_product_64
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

                .i_start           (instant_start),
                .i_activation_flat (i_activation_flat[group_index*GROUP_SIZE*ACT_WIDTH +: GROUP_SIZE*ACT_WIDTH]),
                .i_weight_packed   (i_weight_packed[group_index*GROUP_SIZE*WEIGHT_WIDTH +: GROUP_SIZE*WEIGHT_WIDTH]),
                .i_scale_q2_14     (i_scale_flat[group_index*SCALE_WIDTH +: SCALE_WIDTH]),

                .o_busy            (instant_busy[group_index]),
                .o_done            (instant_done[group_index]),
                .o_partial_sum     (instant_partial_sum[group_index]),
                .o_scaled_sum_q26  (instant_scaled_sum_q26[group_index])
            );
        end
    endgenerate

    integer i;

    always @* begin
        row_instant_scaled_sum = 'd0;

        for (i = 0 ; i < GROUP_COUNT ; i = i + 1) begin
            row_instant_scaled_sum = row_instant_scaled_sum + $signed({{(ROW_ACC_WIDTH-SCALED_WIDTH){instant_scaled_sum_q26[i][SCALED_WIDTH-1]}}, instant_scaled_sum_q26[i]});
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_row_sum_q26 <= 'd0;
        end
        else begin
            case (current_state)
                IDLE:             o_row_sum_q26 <= 'd0;
                PARALLEL_COMPUTE: o_row_sum_q26 <= 'd0;
                ACCUMULATE:       o_row_sum_q26 <= row_instant_scaled_sum;
                DONE:             o_row_sum_q26 <= o_row_sum_q26;
                default:          o_row_sum_q26 <= 'd0;
            endcase
        end
    end

endmodule

`default_nettype wire
