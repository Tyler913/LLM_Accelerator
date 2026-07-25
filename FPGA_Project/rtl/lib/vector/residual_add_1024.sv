`default_nettype none

// Sequential residual add for a 1024-wide hidden vector.
//
// Contract:
//
//   residual_q14_10[1024] + (o_proj_q12_12[1024] >>> 2)
//     -> post_attention_hidden_q14_10[1024]
//
// The output saturates to the signed OUT_WIDTH range. This block is
// intentionally sequential so the stage has a clear cycle-level trace and does
// not require a 1024-lane combinational adder in early bring-up.
module residual_add_1024 #(
    parameter int INPUT_SIZE       = 1024,
    parameter int RESIDUAL_WIDTH   = 24,
    parameter int RESIDUAL_FRAC    = 10,
    parameter int O_PROJ_WIDTH     = 24,
    parameter int O_PROJ_FRAC      = 12,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 10,
    parameter int ELEMENT_INDEX_W  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE),
    parameter int ACC_WIDTH        = OUT_WIDTH + 2
)
(
    input  wire logic                                      i_clk,
    input  wire logic                                      i_rst_n,

    input  wire logic                                      i_start,
    input  wire logic [INPUT_SIZE*RESIDUAL_WIDTH-1 : 0]    i_residual_flat,
    input  wire logic [INPUT_SIZE*O_PROJ_WIDTH-1 : 0]      i_o_proj_flat,

    output logic                                      o_busy,
    output logic                                      o_done,
    output logic                                      o_saturation,
    output logic [31 : 0]                             o_output_count,
    output logic [INPUT_SIZE*OUT_WIDTH-1 : 0]         o_output_flat
);

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam int O_PROJ_TO_OUT_SHIFT = O_PROJ_FRAC - OUT_FRAC;
    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX = INPUT_SIZE - 1;
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX = {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN = {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MAX_EXT =
        {{(ACC_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MIN_EXT =
        {{(ACC_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;
    logic [ELEMENT_INDEX_W-1 : 0]         element_index;
    logic                                 saturation_reg;
    logic signed [RESIDUAL_WIDTH-1 : 0]   current_residual;
    logic signed [O_PROJ_WIDTH-1 : 0]     current_o_proj_q12_12;
    logic signed [OUT_WIDTH-1 : 0]        current_o_proj_q14_10;
    logic signed [ACC_WIDTH-1 : 0]        residual_ext;
    logic signed [ACC_WIDTH-1 : 0]        o_proj_ext;
    logic signed [ACC_WIDTH-1 : 0]        sum_ext;
    logic signed [OUT_WIDTH-1 : 0]        saturated_sum;
    logic                                 current_saturates;

    assign o_busy = (current_state == RUN);
    assign o_done = (current_state == DONE);
    assign o_saturation = saturation_reg;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = RUN;
                end
            end

            RUN: begin
                if (element_index == LAST_ELEMENT_INDEX) begin
                    next_state = DONE;
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
            element_index <= 'd0;
            o_output_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    element_index <= 'd0;
                    if (i_start == 1'b1) begin
                        o_output_count <= 32'd0;
                    end
                end

                RUN: begin
                    o_output_count <= o_output_count + 1'b1;
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= 'd0;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                DONE: begin
                    element_index <= 'd0;
                end

                default: begin
                    element_index <= 'd0;
                    o_output_count <= 32'd0;
                end
            endcase
        end
    end

    always @* begin
        current_residual =
            i_residual_flat[element_index*RESIDUAL_WIDTH +: RESIDUAL_WIDTH];
        current_o_proj_q12_12 =
            i_o_proj_flat[element_index*O_PROJ_WIDTH +: O_PROJ_WIDTH];
        current_o_proj_q14_10 =
            current_o_proj_q12_12 >>> O_PROJ_TO_OUT_SHIFT;
        residual_ext =
            {{(ACC_WIDTH-RESIDUAL_WIDTH){current_residual[RESIDUAL_WIDTH-1]}}, current_residual};
        o_proj_ext =
            {{(ACC_WIDTH-OUT_WIDTH){current_o_proj_q14_10[OUT_WIDTH-1]}}, current_o_proj_q14_10};
        sum_ext = residual_ext + o_proj_ext;

        if (sum_ext > OUT_MAX_EXT) begin
            saturated_sum = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (sum_ext < OUT_MIN_EXT) begin
            saturated_sum = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_sum = sum_ext[OUT_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_output_flat <= 'd0;
            saturation_reg <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_output_flat <= 'd0;
                        saturation_reg <= 1'b0;
                    end
                end

                RUN: begin
                    o_output_flat[element_index*OUT_WIDTH +: OUT_WIDTH] <= saturated_sum;
                    saturation_reg <= saturation_reg | current_saturates;
                end

                DONE: begin
                    o_output_flat <= o_output_flat;
                    saturation_reg <= saturation_reg;
                end

                default: begin
                    o_output_flat <= 'd0;
                    saturation_reg <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
