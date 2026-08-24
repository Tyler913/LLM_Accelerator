`default_nettype none

// BRAM-backed sequential SiLU(gate) * up engine.
//
// Gate, up, sigmoid-LUT, and hidden-output storage all use synchronous inferred
// block RAMs.  One element completes every two compute cycles: the first cycle
// reads the sigmoid value selected by the gate input, and the second performs
// the exact fixed-point arithmetic used by mlp_silu_mul_stage.
module mlp_silu_mul_bram_stage #(
    parameter int FEATURES               = 3072,
    parameter int IN_WIDTH               = 24,
    parameter int IN_FRAC                = 12,
    parameter int SIGMOID_WIDTH          = 16,
    parameter int SIGMOID_FRAC           = 16,
    parameter int SIGMOID_LUT_INDEX_FRAC = 6,
    parameter int SIGMOID_LUT_MIN_INT    = -8,
    parameter int SIGMOID_LUT_MAX_INT    = 8,
    parameter int SIGMOID_LUT_SIZE       =
        ((SIGMOID_LUT_MAX_INT - SIGMOID_LUT_MIN_INT) <<
         SIGMOID_LUT_INDEX_FRAC) + 1,
    parameter int OUT_WIDTH              = 24,
    parameter int OUT_FRAC               = 12,
    parameter int ROW_INDEX_W            =
        (FEATURES <= 1) ? 1 : $clog2(FEATURES),
    parameter int LUT_INDEX_W            =
        (SIGMOID_LUT_SIZE <= 1) ? 1 : $clog2(SIGMOID_LUT_SIZE),
    parameter int SILU_PRODUCT_WIDTH     =
        IN_WIDTH + SIGMOID_WIDTH + 1,
    parameter int HIDDEN_PRODUCT_WIDTH   =
        OUT_WIDTH + IN_WIDTH
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_gate_wr_en,
    input  wire logic [ROW_INDEX_W-1 : 0]          i_gate_wr_addr,
    input  wire logic [IN_WIDTH-1 : 0]             i_gate_wr_data,
    input  wire logic                              i_up_wr_en,
    input  wire logic [ROW_INDEX_W-1 : 0]          i_up_wr_addr,
    input  wire logic [IN_WIDTH-1 : 0]             i_up_wr_data,
    input  wire logic                              i_lut_wr_en,
    input  wire logic [LUT_INDEX_W-1 : 0]          i_lut_wr_addr,
    input  wire logic [SIGMOID_WIDTH-1 : 0]        i_lut_wr_data,

    input  wire logic                              i_start,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic                                   o_saturation,
    output logic [31 : 0]                          o_input_count,
    output logic [31 : 0]                          o_output_count,

    input  wire logic [ROW_INDEX_W-1 : 0]          i_hidden_rd_addr,
    output logic [OUT_WIDTH-1 : 0]                 o_hidden_rd_data
);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_VECTOR_PRIME,
        S_LUT_PRIME,
        S_COMPUTE,
        S_DONE
    } state_t;

    localparam int LUT_SHIFT =
        IN_FRAC - SIGMOID_LUT_INDEX_FRAC;
    // The logical table has 1025 entries for the default [-8, +8] range.
    // Padding the physical array to the address width's power-of-two depth
    // keeps all valid addresses unchanged and lets Vivado implement it as a
    // single 2048x16 block RAM instead of 340 LUTRAMs.
    localparam int LUT_STORAGE_DEPTH = 1 << LUT_INDEX_W;
    localparam logic signed [IN_WIDTH-1 : 0] LUT_MIN_Q =
        SIGMOID_LUT_MIN_INT <<< IN_FRAC;
    localparam logic signed [IN_WIDTH-1 : 0] LUT_MAX_Q =
        SIGMOID_LUT_MAX_INT <<< IN_FRAC;
    localparam logic [ROW_INDEX_W-1 : 0] LAST_ROW_INDEX =
        FEATURES - 1;
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam logic signed [OUT_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam logic signed [SILU_PRODUCT_WIDTH-1 : 0]
        OUT_MAX_SILU_EXT =
            {{(SILU_PRODUCT_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}},
             OUT_MAX};
    localparam logic signed [SILU_PRODUCT_WIDTH-1 : 0]
        OUT_MIN_SILU_EXT =
            {{(SILU_PRODUCT_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}},
             OUT_MIN};
    localparam logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0]
        OUT_MAX_HIDDEN_EXT =
            {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}},
             OUT_MAX};
    localparam logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0]
        OUT_MIN_HIDDEN_EXT =
            {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}},
             OUT_MIN};

    state_t state;
    logic [ROW_INDEX_W-1 : 0] element_index;
    logic [ROW_INDEX_W-1 : 0] vector_rd_addr;
    logic [LUT_INDEX_W-1 : 0] lut_rd_addr;

    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] gate_mem [0 : FEATURES-1];
    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] up_mem [0 : FEATURES-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0] hidden_mem [0 : FEATURES-1];

    logic signed [IN_WIDTH-1 : 0] gate_rd_data;
    logic signed [IN_WIDTH-1 : 0] up_rd_data;
    logic [SIGMOID_WIDTH-1 : 0] sigmoid_rd_data;
    logic signed [IN_WIDTH-1 : 0] gate_hold;
    logic signed [IN_WIDTH-1 : 0] up_hold;

    logic signed [SILU_PRODUCT_WIDTH-1 : 0] gate_ext;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] sigmoid_ext;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] silu_product;
    logic signed [SILU_PRODUCT_WIDTH-1 : 0] silu_shifted;
    logic signed [OUT_WIDTH-1 : 0] silu_gate;
    logic silu_saturates;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] silu_gate_ext;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] up_ext;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] hidden_product;
    logic signed [HIDDEN_PRODUCT_WIDTH-1 : 0] hidden_shifted;
    logic signed [OUT_WIDTH-1 : 0] hidden_result;
    logic hidden_saturates;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);

    function automatic logic [LUT_INDEX_W-1 : 0] gate_to_lut_index(
        input logic signed [IN_WIDTH-1 : 0] gate_value
    );
        logic signed [IN_WIDTH : 0] clipped;
        logic signed [IN_WIDTH : 0] shifted_from_min;
        logic signed [IN_WIDTH : 0] rounded_steps;
        begin
            if (gate_value <= LUT_MIN_Q) begin
                gate_to_lut_index = '0;
            end
            else if (gate_value >= LUT_MAX_Q) begin
                gate_to_lut_index = SIGMOID_LUT_SIZE - 1;
            end
            else begin
                clipped = {gate_value[IN_WIDTH-1], gate_value};
                shifted_from_min =
                    clipped - {LUT_MIN_Q[IN_WIDTH-1], LUT_MIN_Q};
                rounded_steps =
                    (shifted_from_min +
                     ({{IN_WIDTH{1'b0}}, 1'b1} <<
                      (LUT_SHIFT - 1))) >>> LUT_SHIFT;
                gate_to_lut_index =
                    rounded_steps[LUT_INDEX_W-1 : 0];
            end
        end
    endfunction

    always_comb begin
        vector_rd_addr = element_index;
        if ((state == S_COMPUTE) &&
            (element_index != LAST_ROW_INDEX)) begin
            vector_rd_addr = element_index + 1'b1;
        end
        lut_rd_addr = gate_to_lut_index(gate_rd_data);
    end

    // All memories are intentionally reset-free for RAMB inference.
    always_ff @(posedge i_clk) begin
        if (i_gate_wr_en) begin
            gate_mem[i_gate_wr_addr] <= i_gate_wr_data;
        end
        gate_rd_data <= gate_mem[vector_rd_addr];

        if (i_up_wr_en) begin
            up_mem[i_up_wr_addr] <= i_up_wr_data;
        end
        up_rd_data <= up_mem[vector_rd_addr];

        if (state == S_COMPUTE) begin
            hidden_mem[element_index] <= hidden_result;
        end
        o_hidden_rd_data <= hidden_mem[i_hidden_rd_addr];
    end

    qmap_sdp_bram #(
        .DATA_WIDTH(SIGMOID_WIDTH),
        .DEPTH(LUT_STORAGE_DEPTH),
        .ADDR_WIDTH(LUT_INDEX_W)
    ) sigmoid_lut_store (
        .i_clk(i_clk),
        .i_wr_en(i_lut_wr_en),
        .i_wr_addr(i_lut_wr_addr),
        .i_wr_data(i_lut_wr_data),
        .i_rd_en(1'b1),
        .i_rd_addr(lut_rd_addr),
        .o_rd_data(sigmoid_rd_data)
    );

    always_comb begin
        gate_ext =
            {{(SILU_PRODUCT_WIDTH-IN_WIDTH){gate_hold[IN_WIDTH-1]}},
             gate_hold};
        sigmoid_ext =
            {{(SILU_PRODUCT_WIDTH-SIGMOID_WIDTH){1'b0}},
             sigmoid_rd_data};
        silu_product = gate_ext * sigmoid_ext;
        silu_shifted = silu_product >>> SIGMOID_FRAC;

        if (silu_shifted > OUT_MAX_SILU_EXT) begin
            silu_gate = OUT_MAX;
            silu_saturates = 1'b1;
        end
        else if (silu_shifted < OUT_MIN_SILU_EXT) begin
            silu_gate = OUT_MIN;
            silu_saturates = 1'b1;
        end
        else begin
            silu_gate = silu_shifted[OUT_WIDTH-1 : 0];
            silu_saturates = 1'b0;
        end

        silu_gate_ext =
            {{(HIDDEN_PRODUCT_WIDTH-OUT_WIDTH){
                silu_gate[OUT_WIDTH-1]}},
             silu_gate};
        up_ext =
            {{(HIDDEN_PRODUCT_WIDTH-IN_WIDTH){up_hold[IN_WIDTH-1]}},
             up_hold};
        hidden_product = silu_gate_ext * up_ext;
        hidden_shifted = hidden_product >>> IN_FRAC;

        if (hidden_shifted > OUT_MAX_HIDDEN_EXT) begin
            hidden_result = OUT_MAX;
            hidden_saturates = 1'b1;
        end
        else if (hidden_shifted < OUT_MIN_HIDDEN_EXT) begin
            hidden_result = OUT_MIN;
            hidden_saturates = 1'b1;
        end
        else begin
            hidden_result = hidden_shifted[OUT_WIDTH-1 : 0];
            hidden_saturates = 1'b0;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            element_index <= '0;
            gate_hold <= '0;
            up_hold <= '0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_input_count <= 32'd0;
            o_output_count <= 32'd0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_VECTOR_PRIME;
                        element_index <= '0;
                        gate_hold <= '0;
                        up_hold <= '0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_input_count <= 32'd0;
                        o_output_count <= 32'd0;
                    end
                end

                S_VECTOR_PRIME: begin
                    element_index <= '0;
                    state <= S_LUT_PRIME;
                end

                S_LUT_PRIME: begin
                    gate_hold <= gate_rd_data;
                    up_hold <= up_rd_data;
                    o_input_count <= o_input_count + 1'b1;
                    state <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    o_output_count <= o_output_count + 1'b1;
                    o_saturation <=
                        o_saturation |
                        silu_saturates |
                        hidden_saturates;
                    if (element_index == LAST_ROW_INDEX) begin
                        element_index <= '0;
                        state <= S_DONE;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                        state <= S_LUT_PRIME;
                    end
                end

                S_DONE: begin
                    if ((o_input_count != FEATURES) ||
                        (o_output_count != FEATURES)) begin
                        o_error <= 1'b1;
                    end
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                    o_error <= 1'b1;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
