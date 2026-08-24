`default_nettype none

// BRAM-backed post-attention residual plus RMSNorm stage.
//
// The residual and O-projection vectors are loaded through native write
// ports.  A single sequential adder writes the saturated residual result both
// to a block RAM and directly into the input RAM of rmsnorm_bram.  This avoids
// the five wide flattened vectors used by the legacy validation stage.
module post_attention_residual_norm_bram_stage #(
    parameter int INPUT_SIZE        = 1024,
    parameter int RESIDUAL_WIDTH    = 24,
    parameter int RESIDUAL_FRAC     = 10,
    parameter int O_PROJ_WIDTH      = 24,
    parameter int O_PROJ_FRAC       = 12,
    parameter int GAMMA_WIDTH       = 16,
    parameter int GAMMA_FRAC        = 7,
    parameter int INV_RMS_WIDTH     = 24,
    parameter int INV_RMS_FRAC      = 16,
    parameter int NORM_OUT_WIDTH    = 24,
    parameter int NORM_OUT_FRAC     = 12,
    parameter int SUM_WIDTH         = 64,
    parameter int SUM_FRAC          = 2 * RESIDUAL_FRAC,
    parameter int MEAN_SHIFT        = $clog2(INPUT_SIZE),
    parameter int RMS_WIDTH         = RESIDUAL_WIDTH,
    parameter int RMS_FRAC          = RESIDUAL_FRAC,
    parameter int DIV_NUM_WIDTH     = 48,
    parameter int DIV_NUM_SHIFT     = RMS_FRAC + INV_RMS_FRAC,
    parameter int EPS_Q20           = 1,
    parameter int ELEMENT_INDEX_W   =
        (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE),
    parameter int ACC_WIDTH         = RESIDUAL_WIDTH + 2
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_residual_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_residual_wr_addr,
    input  wire logic [RESIDUAL_WIDTH-1 : 0]       i_residual_wr_data,

    input  wire logic                              i_o_proj_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_o_proj_wr_addr,
    input  wire logic [O_PROJ_WIDTH-1 : 0]         i_o_proj_wr_data,

    input  wire logic                              i_gamma_wr_en,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_gamma_wr_addr,
    input  wire logic [GAMMA_WIDTH-1 : 0]          i_gamma_wr_data,

    input  wire logic                              i_start,
    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic                                   o_residual_saturation,
    output logic                                   o_norm_saturation,
    output logic [31 : 0]                          o_residual_count,
    output logic [31 : 0]                          o_cycle_count,

    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_post_hidden_rd_addr,
    output logic [RESIDUAL_WIDTH-1 : 0]            o_post_hidden_rd_data,
    input  wire logic [ELEMENT_INDEX_W-1 : 0]      i_post_norm_rd_addr,
    output logic [NORM_OUT_WIDTH-1 : 0]            o_post_norm_rd_data,

    output logic [SUM_WIDTH-1 : 0]                 o_sum_squares,
    output logic [SUM_WIDTH-1 : 0]                 o_mean_square,
    output logic [INV_RMS_WIDTH-1 : 0]             o_inv_rms
);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_RESIDUAL_PRIME,
        S_RESIDUAL_RUN,
        S_NORM_START,
        S_NORM_WAIT,
        S_DONE
    } state_t;

    localparam logic [ELEMENT_INDEX_W-1 : 0] LAST_ELEMENT_INDEX =
        INPUT_SIZE - 1;
    localparam int O_PROJ_TO_OUT_SHIFT =
        O_PROJ_FRAC - RESIDUAL_FRAC;
    localparam logic signed [RESIDUAL_WIDTH-1 : 0] OUT_MAX =
        {1'b0, {(RESIDUAL_WIDTH-1){1'b1}}};
    localparam logic signed [RESIDUAL_WIDTH-1 : 0] OUT_MIN =
        {1'b1, {(RESIDUAL_WIDTH-1){1'b0}}};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MAX_EXT =
        {{(ACC_WIDTH-RESIDUAL_WIDTH){OUT_MAX[RESIDUAL_WIDTH-1]}},
         OUT_MAX};
    localparam logic signed [ACC_WIDTH-1 : 0] OUT_MIN_EXT =
        {{(ACC_WIDTH-RESIDUAL_WIDTH){OUT_MIN[RESIDUAL_WIDTH-1]}},
         OUT_MIN};

    state_t state;
    logic [ELEMENT_INDEX_W-1 : 0] element_index;
    logic [ELEMENT_INDEX_W-1 : 0] vector_internal_rd_addr;

    (* ram_style = "block" *)
    logic signed [RESIDUAL_WIDTH-1 : 0]
        residual_mem [0 : INPUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [O_PROJ_WIDTH-1 : 0]
        o_proj_mem [0 : INPUT_SIZE-1];
    (* ram_style = "block" *)
    logic signed [RESIDUAL_WIDTH-1 : 0]
        post_hidden_mem [0 : INPUT_SIZE-1];

    logic signed [RESIDUAL_WIDTH-1 : 0] residual_rd_data;
    logic signed [O_PROJ_WIDTH-1 : 0] o_proj_rd_data;
    logic signed [O_PROJ_WIDTH-1 : 0] o_proj_scaled_wide;
    logic signed [RESIDUAL_WIDTH-1 : 0] o_proj_scaled;
    logic signed [ACC_WIDTH-1 : 0] residual_ext;
    logic signed [ACC_WIDTH-1 : 0] o_proj_ext;
    logic signed [ACC_WIDTH-1 : 0] sum_ext;
    logic signed [RESIDUAL_WIDTH-1 : 0] saturated_sum;
    logic current_saturates;

    logic norm_start;
    logic norm_done;
    logic norm_error;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);
    assign norm_start = (state == S_NORM_START);

    always_comb begin
        vector_internal_rd_addr = '0;
        case (state)
            S_RESIDUAL_PRIME: begin
                vector_internal_rd_addr = '0;
            end

            S_RESIDUAL_RUN: begin
                if (element_index == LAST_ELEMENT_INDEX) begin
                    vector_internal_rd_addr = element_index;
                end
                else begin
                    vector_internal_rd_addr = element_index + 1'b1;
                end
            end

            default: begin
                vector_internal_rd_addr = '0;
            end
        endcase
    end

    // Vector memories deliberately have no reset so Vivado can infer RAMB
    // primitives.  All locations are loaded before i_start.
    always_ff @(posedge i_clk) begin
        if (i_residual_wr_en) begin
            residual_mem[i_residual_wr_addr] <= i_residual_wr_data;
        end
        residual_rd_data <= residual_mem[vector_internal_rd_addr];

        if (i_o_proj_wr_en) begin
            o_proj_mem[i_o_proj_wr_addr] <= i_o_proj_wr_data;
        end
        o_proj_rd_data <= o_proj_mem[vector_internal_rd_addr];

        if (state == S_RESIDUAL_RUN) begin
            post_hidden_mem[element_index] <= saturated_sum;
        end
        o_post_hidden_rd_data <=
            post_hidden_mem[i_post_hidden_rd_addr];
    end

    always_comb begin
        o_proj_scaled_wide =
            o_proj_rd_data >>> O_PROJ_TO_OUT_SHIFT;
        o_proj_scaled = o_proj_scaled_wide[RESIDUAL_WIDTH-1 : 0];
        residual_ext =
            {{(ACC_WIDTH-RESIDUAL_WIDTH){
                residual_rd_data[RESIDUAL_WIDTH-1]}},
             residual_rd_data};
        o_proj_ext =
            {{(ACC_WIDTH-RESIDUAL_WIDTH){
                o_proj_scaled[RESIDUAL_WIDTH-1]}},
             o_proj_scaled};
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
            saturated_sum = sum_ext[RESIDUAL_WIDTH-1 : 0];
            current_saturates = 1'b0;
        end
    end

    rmsnorm_bram #(
        .INPUT_SIZE(INPUT_SIZE),
        .IN_WIDTH(RESIDUAL_WIDTH),
        .IN_FRAC(RESIDUAL_FRAC),
        .GAMMA_WIDTH(GAMMA_WIDTH),
        .GAMMA_FRAC(GAMMA_FRAC),
        .GAMMA_SIGNED(1),
        .INV_RMS_WIDTH(INV_RMS_WIDTH),
        .INV_RMS_FRAC(INV_RMS_FRAC),
        .OUT_WIDTH(NORM_OUT_WIDTH),
        .OUT_FRAC(NORM_OUT_FRAC),
        .SUM_WIDTH(SUM_WIDTH),
        .SUM_FRAC(SUM_FRAC),
        .MEAN_SHIFT(MEAN_SHIFT),
        .RMS_WIDTH(RMS_WIDTH),
        .RMS_FRAC(RMS_FRAC),
        .DIV_NUM_WIDTH(DIV_NUM_WIDTH),
        .DIV_NUM_SHIFT(DIV_NUM_SHIFT),
        .EPS_Q20(EPS_Q20),
        .ELEMENT_INDEX_W(ELEMENT_INDEX_W)
    ) inst_post_attention_rmsnorm_bram (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input_wr_en(state == S_RESIDUAL_RUN),
        .i_input_wr_addr(element_index),
        .i_input_wr_data(saturated_sum),
        .i_gamma_wr_en(i_gamma_wr_en),
        .i_gamma_wr_addr(i_gamma_wr_addr),
        .i_gamma_wr_data(i_gamma_wr_data),
        .i_start(norm_start),
        .o_busy(),
        .o_done(norm_done),
        .o_error(norm_error),
        .o_saturation(o_norm_saturation),
        .i_output_rd_addr(i_post_norm_rd_addr),
        .o_output_rd_data(o_post_norm_rd_data),
        .o_output_wr_valid(),
        .o_output_wr_addr(),
        .o_output_wr_data(),
        .o_sum_squares(o_sum_squares),
        .o_mean_square(o_mean_square),
        .o_inv_rms(o_inv_rms)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            element_index <= '0;
            o_error <= 1'b0;
            o_residual_saturation <= 1'b0;
            o_residual_count <= 32'd0;
            o_cycle_count <= 32'd0;
        end
        else begin
            if ((state != S_IDLE) && (state != S_DONE)) begin
                o_cycle_count <= o_cycle_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_RESIDUAL_PRIME;
                        element_index <= '0;
                        o_error <= 1'b0;
                        o_residual_saturation <= 1'b0;
                        o_residual_count <= 32'd0;
                        o_cycle_count <= 32'd0;
                    end
                end

                S_RESIDUAL_PRIME: begin
                    element_index <= '0;
                    state <= S_RESIDUAL_RUN;
                end

                S_RESIDUAL_RUN: begin
                    o_residual_saturation <=
                        o_residual_saturation | current_saturates;
                    o_residual_count <= o_residual_count + 1'b1;
                    if (element_index == LAST_ELEMENT_INDEX) begin
                        element_index <= '0;
                        state <= S_NORM_START;
                    end
                    else begin
                        element_index <= element_index + 1'b1;
                    end
                end

                S_NORM_START: begin
                    state <= S_NORM_WAIT;
                end

                S_NORM_WAIT: begin
                    if (norm_done) begin
                        if (norm_error ||
                            (o_residual_count != INPUT_SIZE)) begin
                            o_error <= 1'b1;
                        end
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
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
