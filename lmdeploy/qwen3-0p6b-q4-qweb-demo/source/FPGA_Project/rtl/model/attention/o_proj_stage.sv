`default_nettype none

// Layer attention output projection:
//
//   attn_out[2048] -> o_proj_out[1024]
//
// This local stage reuses the existing parameterized Q4 GEMV projection
// controller with INPUT_SIZE=2048 and OUT_FEATURES=1024. The projection core
// produces Q26 row sums; this wrapper converts each row to signed Q12.12 and
// emits an output ready/valid stream.
module o_proj_stage #(
    parameter int INPUT_SIZE       = 2048,
    parameter int OUT_FEATURES     = 1024,
    parameter int TILE_ROWS        = 4,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int ROW_INDEX_W      = (OUT_FEATURES <= 1) ? 1 : $clog2(OUT_FEATURES)
)
(
    input  wire logic                                                  i_clk,
    input  wire logic                                                  i_rst_n,

    input  wire logic                                                  i_start,
    input  wire logic [INPUT_SIZE*ACT_WIDTH-1 : 0]                    i_activation_flat,
    input  wire logic [OUT_FEATURES*INPUT_SIZE*WEIGHT_WIDTH-1 : 0]    i_weight_flat,
    input  wire logic [OUT_FEATURES*GROUP_COUNT*SCALE_WIDTH-1 : 0]    i_scale_flat,

    output logic                                                  o_busy,
    output logic                                                  o_done,
    output logic                                                  o_error,
    output logic                                                  o_saturation,

    output logic                                                  o_out_valid,
    input  wire logic                                                  i_out_ready,
    output logic [ROW_INDEX_W-1 : 0]                              o_out_row,
    output logic signed [OUT_WIDTH-1 : 0]                         o_out_data,
    output logic                                                  o_out_last,

    output logic [31 : 0]                                        o_output_count,
    output logic [31 : 0]                                        o_compute_cycle_count
);

    localparam IDLE          = 3'd0;
    localparam START_COMPUTE = 3'd1;
    localparam WAIT_COMPUTE  = 3'd2;
    localparam PREP_OUTPUT   = 3'd3;
    localparam EMIT_OUTPUT   = 3'd4;
    localparam DONE          = 3'd5;

    localparam logic [ROW_INDEX_W-1 : 0] LAST_ROW_INDEX = OUT_FEATURES - 1;
    localparam signed [OUT_WIDTH-1 : 0] OUT_MAX = {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam signed [OUT_WIDTH-1 : 0] OUT_MIN = {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam signed [ROW_ACC_WIDTH-1 : 0] OUT_MAX_EXT =
        {{(ROW_ACC_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam signed [ROW_ACC_WIDTH-1 : 0] OUT_MIN_EXT =
        {{(ROW_ACC_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;
    logic compute_start;
    logic compute_busy;
    logic compute_done;
    logic [OUT_FEATURES*ROW_ACC_WIDTH-1 : 0] q26_output_flat;
    logic [ROW_INDEX_W-1 : 0] row_index;
    logic [ROW_INDEX_W-1 : 0] out_row_reg;
    logic signed [OUT_WIDTH-1 : 0] out_data_reg;
    logic out_last_reg;
    logic error_reg;
    logic saturation_reg;
    logic signed [ROW_ACC_WIDTH-1 : 0] current_q26;
    logic signed [ROW_ACC_WIDTH-1 : 0] shifted_q12_12;

    assign compute_start = (current_state == START_COMPUTE);
    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_saturation = saturation_reg;
    assign o_out_valid = (current_state == EMIT_OUTPUT);
    assign o_out_row = out_row_reg;
    assign o_out_data = out_data_reg;
    assign o_out_last = o_out_valid && out_last_reg;

    assign current_q26 = q26_output_flat[row_index*ROW_ACC_WIDTH +: ROW_ACC_WIDTH];
    assign shifted_q12_12 = current_q26 >>> SCALE_FRAC;

    function automatic logic signed [OUT_WIDTH-1 : 0] saturate_to_out(
        input logic signed [ROW_ACC_WIDTH-1 : 0] value
    );
        begin
            if (value > OUT_MAX_EXT) begin
                saturate_to_out = OUT_MAX;
            end
            else if (value < OUT_MIN_EXT) begin
                saturate_to_out = OUT_MIN;
            end
            else begin
                saturate_to_out = value[OUT_WIDTH-1 : 0];
            end
        end
    endfunction

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_COMPUTE;
                end
            end

            START_COMPUTE: begin
                next_state = WAIT_COMPUTE;
            end

            WAIT_COMPUTE: begin
                if (compute_done == 1'b1) begin
                    next_state = PREP_OUTPUT;
                end
            end

            PREP_OUTPUT: begin
                next_state = EMIT_OUTPUT;
            end

            EMIT_OUTPUT: begin
                if (o_out_valid && i_out_ready) begin
                    if (out_last_reg == 1'b1) begin
                        next_state = DONE;
                    end
                    else begin
                        next_state = PREP_OUTPUT;
                    end
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
            row_index <= 'd0;
            out_row_reg <= 'd0;
            out_data_reg <= 'd0;
            out_last_reg <= 1'b0;
            error_reg <= 1'b0;
            saturation_reg <= 1'b0;
            o_output_count <= 32'd0;
            o_compute_cycle_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        row_index <= 'd0;
                        out_row_reg <= 'd0;
                        out_data_reg <= 'd0;
                        out_last_reg <= 1'b0;
                        error_reg <= 1'b0;
                        saturation_reg <= 1'b0;
                        o_output_count <= 32'd0;
                        o_compute_cycle_count <= 32'd0;
                    end
                end

                START_COMPUTE: begin
                    row_index <= 'd0;
                    out_row_reg <= 'd0;
                    out_data_reg <= 'd0;
                    out_last_reg <= 1'b0;
                end

                WAIT_COMPUTE: begin
                    o_compute_cycle_count <= o_compute_cycle_count + 1'b1;
                end

                PREP_OUTPUT: begin
                    out_row_reg <= row_index;
                    out_data_reg <= saturate_to_out(shifted_q12_12);
                    out_last_reg <= (row_index == LAST_ROW_INDEX);
                    saturation_reg <=
                        saturation_reg ||
                        (shifted_q12_12 > OUT_MAX_EXT) ||
                        (shifted_q12_12 < OUT_MIN_EXT);
                end

                EMIT_OUTPUT: begin
                    if (o_out_valid && i_out_ready) begin
                        o_output_count <= o_output_count + 1'b1;
                        if (out_last_reg == 1'b0) begin
                            row_index <= row_index + 1'b1;
                        end
                    end
                end

                DONE: begin
                    error_reg <= error_reg || (o_output_count != OUT_FEATURES);
                end

                default: begin
                    row_index <= 'd0;
                    out_row_reg <= 'd0;
                    out_data_reg <= 'd0;
                    out_last_reg <= 1'b0;
                    error_reg <= 1'b1;
                    saturation_reg <= 1'b0;
                    o_output_count <= 32'd0;
                    o_compute_cycle_count <= 32'd0;
                end
            endcase
        end
    end

    q4_gemv_projection_1024 #(
        .OUT_FEATURES  (OUT_FEATURES),
        .TILE_ROWS     (TILE_ROWS),
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
    ) projection_core (
        .i_clk               (i_clk),
        .i_rst_n             (i_rst_n),
        .i_start             (compute_start),
        .i_activation_flat   (i_activation_flat),
        .i_weight_packed_flat(i_weight_flat),
        .i_scale_flat        (i_scale_flat),
        .o_busy              (compute_busy),
        .o_done              (compute_done),
        .o_output_flat       (q26_output_flat)
    );

endmodule

`default_nettype wire
