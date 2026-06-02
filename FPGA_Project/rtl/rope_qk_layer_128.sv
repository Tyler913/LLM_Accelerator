`default_nettype none

// RoPE stage for one token's Q/K attention heads.
//
// Shape contract for Qwen3-0.6B:
//
//   Q input:  [16, 128], flattened as q[head][dim]
//   K input:  [8, 128], flattened as k[head][dim]
//   cos/sin:  [128] for the current token position
//   Q output: [16, 128], same flattening
//   K output: [8, 128], same flattening
//
// Math contract:
//
//   rotate_half(x) = concat(-x[64:127], x[0:63])
//   rope(x)[i]     = x[i] * cos[i] + rotate_half(x)[i] * sin[i]
//
// Default fixed-point formats:
//
//   q/k input: signed 24-bit Q12.12
//   cos/sin:   signed 16-bit Q1.15
//   q/k output:signed 24-bit Q12.12, saturated
//
// This first version processes one scalar lane per cycle. It is intended for standalone RTL bring-up and can be wrapped later with BRAM/streaming access.
module rope_qk_layer_128 #(
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_K_HEADS      = 8,
    parameter int HEAD_DIM         = 128,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 12,
    parameter int TRIG_WIDTH       = 16,
    parameter int TRIG_FRAC        = 15,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int Q_HEAD_INDEX_W   = (NUM_Q_HEADS <= 1) ? 1 : $clog2(NUM_Q_HEADS),
    parameter int K_HEAD_INDEX_W   = (NUM_K_HEADS <= 1) ? 1 : $clog2(NUM_K_HEADS),
    parameter int Q_ELEMENT_INDEX_W = ((NUM_Q_HEADS*HEAD_DIM) <= 1) ? 1 : $clog2(NUM_Q_HEADS*HEAD_DIM),
    parameter int K_ELEMENT_INDEX_W = ((NUM_K_HEADS*HEAD_DIM) <= 1) ? 1 : $clog2(NUM_K_HEADS*HEAD_DIM),
    parameter int INPUT_SIGNED_WIDTH = IN_WIDTH + 1,
    parameter int PRODUCT_WIDTH    = INPUT_SIGNED_WIDTH + TRIG_WIDTH,
    parameter int SUM_WIDTH        = PRODUCT_WIDTH + 1,
    parameter int OUTPUT_SHIFT     = IN_FRAC + TRIG_FRAC - OUT_FRAC
)
(
    input  logic                                            i_clk,
    input  logic                                            i_rst_n,

    // Start a new RoPE transaction when the module is not busy.
    // Keep q/k/cos/sin inputs stable until o_done is asserted.
    input  logic                                            i_start,

    // Flattened little-element-endian Q and K heads:
    // q[head][dim] is i_q_flat[(head*HEAD_DIM + dim)*IN_WIDTH +: IN_WIDTH].
    input  logic [NUM_Q_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]      i_q_flat,
    input  logic [NUM_K_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]      i_k_flat,

    // Full 128-lane cos/sin vectors for the current position.
    input  logic [HEAD_DIM*TRIG_WIDTH-1 : 0]                i_cos_flat,
    input  logic [HEAD_DIM*TRIG_WIDTH-1 : 0]                i_sin_flat,

    output logic                                            o_busy,
    output logic                                            o_done,

    // High with o_done if any Q or K output element saturated.
    output logic                                            o_saturation,

    output logic [NUM_Q_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]     o_q_rope_flat,
    output logic [NUM_K_HEADS*HEAD_DIM*OUT_WIDTH-1 : 0]     o_k_rope_flat
);

    localparam int HALF_DIM = HEAD_DIM / 2;

    localparam IDLE  = 2'd0;
    localparam RUN_Q = 2'd1;
    localparam RUN_K = 2'd2;
    localparam DONE  = 2'd3;

    localparam logic [DIM_INDEX_W-1 : 0]    LAST_DIM_INDEX    = HEAD_DIM - 1;
    localparam logic [DIM_INDEX_W-1 : 0]    HALF_DIM_INDEX    = HALF_DIM;
    localparam logic [Q_HEAD_INDEX_W-1 : 0] LAST_Q_HEAD_INDEX = NUM_Q_HEADS - 1;
    localparam logic [K_HEAD_INDEX_W-1 : 0] LAST_K_HEAD_INDEX = NUM_K_HEADS - 1;

    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX = {1'b0, {OUT_WIDTH-1{1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN = {1'b1, {OUT_WIDTH-1{1'b0}}};

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;

    logic [Q_HEAD_INDEX_W-1 : 0]          q_head_index;
    logic [K_HEAD_INDEX_W-1 : 0]          k_head_index;
    logic [DIM_INDEX_W-1 : 0]             q_dim_index;
    logic [DIM_INDEX_W-1 : 0]             k_dim_index;

    logic [DIM_INDEX_W-1 : 0]             current_dim_index;
    logic [DIM_INDEX_W-1 : 0]             rotate_dim_index;
    logic [Q_ELEMENT_INDEX_W-1 : 0]       q_element_index;
    logic [Q_ELEMENT_INDEX_W-1 : 0]       q_rotate_element_index;
    logic [K_ELEMENT_INDEX_W-1 : 0]       k_element_index;
    logic [K_ELEMENT_INDEX_W-1 : 0]       k_rotate_element_index;

    logic signed [IN_WIDTH-1 : 0]         current_input;
    logic signed [IN_WIDTH-1 : 0]         rotate_source;
    logic signed [INPUT_SIGNED_WIDTH-1:0] current_input_extended;
    logic signed [INPUT_SIGNED_WIDTH-1:0] rotate_source_extended;
    logic signed [INPUT_SIGNED_WIDTH-1:0] rotated_input;
    logic signed [TRIG_WIDTH-1 : 0]       current_cos;
    logic signed [TRIG_WIDTH-1 : 0]       current_sin;

    logic signed [PRODUCT_WIDTH-1 : 0]    product_x_cos;
    logic signed [PRODUCT_WIDTH-1 : 0]    product_rot_sin;
    logic signed [SUM_WIDTH-1 : 0]        product_x_cos_extended;
    logic signed [SUM_WIDTH-1 : 0]        product_rot_sin_extended;
    logic signed [SUM_WIDTH-1 : 0]        rope_sum;
    logic signed [SUM_WIDTH-1 : 0]        shifted_sum;
    logic signed [SUM_WIDTH-1 : 0]        out_max_extended;
    logic signed [SUM_WIDTH-1 : 0]        out_min_extended;
    logic signed [OUT_WIDTH-1 : 0]        saturated_output;
    logic                                 current_saturates;
    logic                                 saturation_reg;

    assign o_busy       = (current_state == RUN_Q) || (current_state == RUN_K);
    assign o_done       = (current_state == DONE);
    assign o_saturation = saturation_reg;

    always @* begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = RUN_Q;
                end
            end

            RUN_Q: begin
                if ((q_head_index == LAST_Q_HEAD_INDEX) && (q_dim_index == LAST_DIM_INDEX)) begin
                    next_state = RUN_K;
                end
            end

            RUN_K: begin
                if ((k_head_index == LAST_K_HEAD_INDEX) && (k_dim_index == LAST_DIM_INDEX)) begin
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
            q_head_index <= 'd0;
            k_head_index <= 'd0;
            q_dim_index  <= 'd0;
            k_dim_index  <= 'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    q_head_index <= 'd0;
                    k_head_index <= 'd0;
                    q_dim_index  <= 'd0;
                    k_dim_index  <= 'd0;
                end

                RUN_Q: begin
                    if (q_dim_index == LAST_DIM_INDEX) begin
                        q_dim_index <= 'd0;
                        if (q_head_index != LAST_Q_HEAD_INDEX) begin
                            q_head_index <= q_head_index + 1'b1;
                        end
                    end
                    else begin
                        q_dim_index <= q_dim_index + 1'b1;
                    end
                end

                RUN_K: begin
                    if (k_dim_index == LAST_DIM_INDEX) begin
                        k_dim_index <= 'd0;
                        if (k_head_index != LAST_K_HEAD_INDEX) begin
                            k_head_index <= k_head_index + 1'b1;
                        end
                    end
                    else begin
                        k_dim_index <= k_dim_index + 1'b1;
                    end
                end

                DONE: begin
                    q_head_index <= 'd0;
                    k_head_index <= 'd0;
                    q_dim_index  <= 'd0;
                    k_dim_index  <= 'd0;
                end

                default: begin
                    q_head_index <= 'd0;
                    k_head_index <= 'd0;
                    q_dim_index  <= 'd0;
                    k_dim_index  <= 'd0;
                end
            endcase
        end
    end

    always @* begin
        if (current_state == RUN_K) begin
            current_dim_index = k_dim_index;
        end
        else begin
            current_dim_index = q_dim_index;
        end

        if (current_dim_index < HALF_DIM_INDEX) begin
            rotate_dim_index = current_dim_index + HALF_DIM_INDEX;
        end
        else begin
            rotate_dim_index = current_dim_index - HALF_DIM_INDEX;
        end

        q_element_index        = (q_head_index * HEAD_DIM) + current_dim_index;
        q_rotate_element_index = (q_head_index * HEAD_DIM) + rotate_dim_index;
        k_element_index        = (k_head_index * HEAD_DIM) + current_dim_index;
        k_rotate_element_index = (k_head_index * HEAD_DIM) + rotate_dim_index;

        if (current_state == RUN_K) begin
            current_input = i_k_flat[k_element_index*IN_WIDTH +: IN_WIDTH];
            rotate_source = i_k_flat[k_rotate_element_index*IN_WIDTH +: IN_WIDTH];
        end
        else begin
            current_input = i_q_flat[q_element_index*IN_WIDTH +: IN_WIDTH];
            rotate_source = i_q_flat[q_rotate_element_index*IN_WIDTH +: IN_WIDTH];
        end

        current_input_extended = {current_input[IN_WIDTH-1], current_input};
        rotate_source_extended = {rotate_source[IN_WIDTH-1], rotate_source};

        if (current_dim_index < HALF_DIM_INDEX) begin
            rotated_input = -rotate_source_extended;
        end
        else begin
            rotated_input = rotate_source_extended;
        end

        current_cos = i_cos_flat[current_dim_index*TRIG_WIDTH +: TRIG_WIDTH];
        current_sin = i_sin_flat[current_dim_index*TRIG_WIDTH +: TRIG_WIDTH];

        product_x_cos            = current_input_extended * current_cos;
        product_rot_sin          = rotated_input * current_sin;
        product_x_cos_extended   = {{(SUM_WIDTH-PRODUCT_WIDTH){product_x_cos[PRODUCT_WIDTH-1]}}, product_x_cos};
        product_rot_sin_extended = {{(SUM_WIDTH-PRODUCT_WIDTH){product_rot_sin[PRODUCT_WIDTH-1]}}, product_rot_sin};
        rope_sum                 = product_x_cos_extended + product_rot_sin_extended;
        shifted_sum              = rope_sum >>> OUTPUT_SHIFT;

        out_max_extended = {{(SUM_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
        out_min_extended = {{(SUM_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};

        if (shifted_sum > out_max_extended) begin
            saturated_output  = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (shifted_sum < out_min_extended) begin
            saturated_output  = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_output  = shifted_sum[OUT_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            o_q_rope_flat  <= 'd0;
            o_k_rope_flat  <= 'd0;
            saturation_reg <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        o_q_rope_flat  <= 'd0;
                        o_k_rope_flat  <= 'd0;
                        saturation_reg <= 1'b0;
                    end
                end

                RUN_Q: begin
                    o_q_rope_flat[q_element_index*OUT_WIDTH +: OUT_WIDTH] <= saturated_output;
                    saturation_reg <= saturation_reg | current_saturates;
                end

                RUN_K: begin
                    o_k_rope_flat[k_element_index*OUT_WIDTH +: OUT_WIDTH] <= saturated_output;
                    saturation_reg <= saturation_reg | current_saturates;
                end

                DONE: begin
                    o_q_rope_flat  <= o_q_rope_flat;
                    o_k_rope_flat  <= o_k_rope_flat;
                    saturation_reg <= saturation_reg;
                end

                default: begin
                    o_q_rope_flat  <= 'd0;
                    o_k_rope_flat  <= 'd0;
                    saturation_reg <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
