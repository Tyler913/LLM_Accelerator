`default_nettype none

// Sequential BRAM-backed RoPE engine for one attention head.
//
// The flat-vector rope_qk_layer_128 core is useful as a numerical reference,
// but a variable select across 128 packed lanes synthesizes into a large mux.
// This board-oriented engine mirrors the input vector into two inferred BRAMs
// so the direct and rotate-half elements can be read together. One output
// element is produced per cycle after a one-cycle read prime.
module rope_head_bram #(
    parameter int HEAD_DIM       = 128,
    parameter int IN_WIDTH       = 24,
    parameter int IN_FRAC        = 12,
    parameter int TRIG_WIDTH     = 16,
    parameter int TRIG_FRAC      = 15,
    parameter int OUT_WIDTH      = 24,
    parameter int OUT_FRAC       = 12,
    parameter int DIM_INDEX_W    =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int INPUT_SIGNED_WIDTH = IN_WIDTH + 1,
    parameter int PRODUCT_WIDTH  = INPUT_SIGNED_WIDTH + TRIG_WIDTH,
    parameter int SUM_WIDTH      = PRODUCT_WIDTH + 1,
    parameter int OUTPUT_SHIFT   =
        IN_FRAC + TRIG_FRAC - OUT_FRAC
) (
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_input_wr_en,
    input  wire logic [DIM_INDEX_W-1 : 0]          i_input_wr_addr,
    input  wire logic signed [IN_WIDTH-1 : 0]      i_input_wr_data,
    input  wire logic                              i_cos_wr_en,
    input  wire logic [DIM_INDEX_W-1 : 0]          i_cos_wr_addr,
    input  wire logic signed [TRIG_WIDTH-1 : 0]    i_cos_wr_data,
    input  wire logic                              i_sin_wr_en,
    input  wire logic [DIM_INDEX_W-1 : 0]          i_sin_wr_addr,
    input  wire logic signed [TRIG_WIDTH-1 : 0]    i_sin_wr_data,

    input  wire logic                              i_start,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_saturation,

    output logic                                   o_output_wr_valid,
    output logic [DIM_INDEX_W-1 : 0]               o_output_wr_addr,
    output logic signed [OUT_WIDTH-1 : 0]          o_output_wr_data
);

    localparam int HALF_DIM = HEAD_DIM / 2;
    localparam logic [DIM_INDEX_W-1 : 0] LAST_DIM =
        DIM_INDEX_W'(HEAD_DIM - 1);
    localparam logic [DIM_INDEX_W-1 : 0] HALF_DIM_INDEX =
        DIM_INDEX_W'(HALF_DIM);
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {OUT_WIDTH-1{1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {OUT_WIDTH-1{1'b0}}};

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_PRIME,
        S_RUN,
        S_DONE
    } state_t;

    state_t state;
    logic [DIM_INDEX_W-1 : 0] element_index;
    logic [DIM_INDEX_W-1 : 0] read_dim;
    logic [DIM_INDEX_W-1 : 0] rotate_read_dim;

    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] input_direct_mem [0 : HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] input_rotate_mem [0 : HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [TRIG_WIDTH-1 : 0] cos_mem [0 : HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [TRIG_WIDTH-1 : 0] sin_mem [0 : HEAD_DIM-1];

    logic signed [IN_WIDTH-1 : 0] direct_read_data;
    logic signed [IN_WIDTH-1 : 0] rotate_read_data;
    logic signed [TRIG_WIDTH-1 : 0] cos_read_data;
    logic signed [TRIG_WIDTH-1 : 0] sin_read_data;

    logic signed [INPUT_SIGNED_WIDTH-1 : 0] direct_extended;
    logic signed [INPUT_SIGNED_WIDTH-1 : 0] rotate_extended;
    logic signed [INPUT_SIGNED_WIDTH-1 : 0] rotated_input;
    logic signed [PRODUCT_WIDTH-1 : 0] product_x_cos;
    logic signed [PRODUCT_WIDTH-1 : 0] product_rot_sin;
    logic signed [SUM_WIDTH-1 : 0] product_x_cos_extended;
    logic signed [SUM_WIDTH-1 : 0] product_rot_sin_extended;
    logic signed [SUM_WIDTH-1 : 0] rope_sum;
    logic signed [SUM_WIDTH-1 : 0] shifted_sum;
    logic signed [SUM_WIDTH-1 : 0] out_max_extended;
    logic signed [SUM_WIDTH-1 : 0] out_min_extended;
    logic signed [OUT_WIDTH-1 : 0] saturated_output;
    logic current_saturates;

    assign o_busy = (state == S_PRIME) || (state == S_RUN);
    assign o_done = (state == S_DONE);
    assign o_output_wr_valid = (state == S_RUN);
    assign o_output_wr_addr = element_index;
    assign o_output_wr_data = saturated_output;

    always_comb begin
        if ((state == S_RUN) && (element_index != LAST_DIM)) begin
            read_dim = element_index + 1'b1;
        end
        else begin
            read_dim = element_index;
        end

        if (read_dim < HALF_DIM_INDEX) begin
            rotate_read_dim = read_dim + HALF_DIM_INDEX;
        end
        else begin
            rotate_read_dim = read_dim - HALF_DIM_INDEX;
        end
    end

    // None of the vector memories is reset. Each is loaded in full before
    // i_start, which keeps the templates eligible for block-RAM inference.
    always_ff @(posedge i_clk) begin
        if (i_input_wr_en && (state == S_IDLE)) begin
            input_direct_mem[i_input_wr_addr] <= i_input_wr_data;
            input_rotate_mem[i_input_wr_addr] <= i_input_wr_data;
        end
        direct_read_data <= input_direct_mem[read_dim];
        rotate_read_data <= input_rotate_mem[rotate_read_dim];

        if (i_cos_wr_en && (state == S_IDLE)) begin
            cos_mem[i_cos_wr_addr] <= i_cos_wr_data;
        end
        cos_read_data <= cos_mem[read_dim];

        if (i_sin_wr_en && (state == S_IDLE)) begin
            sin_mem[i_sin_wr_addr] <= i_sin_wr_data;
        end
        sin_read_data <= sin_mem[read_dim];
    end

    always_comb begin
        direct_extended = {direct_read_data[IN_WIDTH-1], direct_read_data};
        rotate_extended = {rotate_read_data[IN_WIDTH-1], rotate_read_data};
        if (element_index < HALF_DIM_INDEX) begin
            rotated_input = -rotate_extended;
        end
        else begin
            rotated_input = rotate_extended;
        end

        product_x_cos = direct_extended * cos_read_data;
        product_rot_sin = rotated_input * sin_read_data;
        product_x_cos_extended = {
            {(SUM_WIDTH-PRODUCT_WIDTH){product_x_cos[PRODUCT_WIDTH-1]}},
            product_x_cos
        };
        product_rot_sin_extended = {
            {(SUM_WIDTH-PRODUCT_WIDTH){product_rot_sin[PRODUCT_WIDTH-1]}},
            product_rot_sin
        };
        rope_sum = product_x_cos_extended + product_rot_sin_extended;
        shifted_sum = rope_sum >>> OUTPUT_SHIFT;
        out_max_extended = {
            {(SUM_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}},
            OUT_MAX
        };
        out_min_extended = {
            {(SUM_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}},
            OUT_MIN
        };

        if (shifted_sum > out_max_extended) begin
            saturated_output = OUT_MAX;
            current_saturates = 1'b1;
        end
        else if (shifted_sum < out_min_extended) begin
            saturated_output = OUT_MIN;
            current_saturates = 1'b1;
        end
        else begin
            saturated_output = shifted_sum[OUT_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
            element_index <= '0;
            o_saturation <= 1'b0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        element_index <= '0;
                        o_saturation <= 1'b0;
                        state <= S_PRIME;
                    end
                end

                S_PRIME: begin
                    state <= S_RUN;
                end

                S_RUN: begin
                    o_saturation <=
                        o_saturation | current_saturates;
                    if (element_index == LAST_DIM) begin
                        state <= S_DONE;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

    initial begin
        if ((HEAD_DIM != 128) || ((HEAD_DIM % 2) != 0)) begin
            $fatal(1, "rope_head_bram: unsupported HEAD_DIM");
        end
        if ((IN_WIDTH != 24) ||
            (TRIG_WIDTH != 16) ||
            (OUT_WIDTH != 24)) begin
            $fatal(1, "rope_head_bram: unsupported fixed-point format");
        end
    end

endmodule

`default_nettype wire
