`default_nettype none

// Layer 0 final MLP residual add:
//
//   post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]
//
// Fixed-point contract:
//   post_attn_hidden: signed Q14.10
//   down_out:         signed Q12.12
//   layer_out:        signed Q14.10
//
// This wrapper reuses the sequential residual_add_1024 block and gives the
// final MLP residual step a stable top-level control/status surface.
module mlp_residual_add_stage #(
    parameter int INPUT_SIZE              = 1024,
    parameter int POST_ATTENTION_WIDTH    = 24,
    parameter int POST_ATTENTION_FRAC     = 10,
    parameter int DOWN_WIDTH              = 24,
    parameter int DOWN_FRAC               = 12,
    parameter int OUT_WIDTH               = 24,
    parameter int OUT_FRAC                = 10,
    parameter int ELEMENT_INDEX_W         = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE),
    parameter int ACC_WIDTH               = OUT_WIDTH + 2
)
(
    input  wire logic                                             i_clk,
    input  wire logic                                             i_rst_n,

    input  wire logic                                             i_start,
    input  wire logic [INPUT_SIZE*POST_ATTENTION_WIDTH-1 : 0]     i_post_attn_hidden_flat,
    input  wire logic [INPUT_SIZE*DOWN_WIDTH-1 : 0]               i_down_out_flat,

    output logic                                             o_busy,
    output logic                                             o_done,
    output logic                                             o_error,
    output logic                                             o_saturation,
    output logic [31 : 0]                                    o_output_count,
    output logic [31 : 0]                                    o_cycle_count,
    output logic [INPUT_SIZE*OUT_WIDTH-1 : 0]                o_layer_out_flat
);

    localparam IDLE      = 2'd0;
    localparam START_ADD = 2'd1;
    localparam WAIT_ADD  = 2'd2;
    localparam DONE      = 2'd3;

    logic [1 : 0] current_state;
    logic [1 : 0] next_state;
    logic         add_start;
    logic         add_busy;
    logic         add_done;
    logic         add_saturation;
    logic [31:0]  add_output_count;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0] layer_out_flat;
    logic         error_reg;

    assign add_start = (current_state == START_ADD);
    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_saturation = add_saturation;
    assign o_output_count = add_output_count;
    assign o_layer_out_flat = layer_out_flat;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    next_state = START_ADD;
                end
            end

            START_ADD: begin
                next_state = WAIT_ADD;
            end

            WAIT_ADD: begin
                if (add_done == 1'b1) begin
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
            error_reg <= 1'b0;
            o_cycle_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        error_reg <= 1'b0;
                        o_cycle_count <= 32'd0;
                    end
                end

                START_ADD,
                WAIT_ADD: begin
                    o_cycle_count <= o_cycle_count + 1'b1;
                end

                DONE: begin
                    error_reg <= error_reg || (add_output_count != INPUT_SIZE);
                end

                default: begin
                    error_reg <= 1'b1;
                    o_cycle_count <= 32'd0;
                end
            endcase
        end
    end

    residual_add_1024 #(
        .INPUT_SIZE      (INPUT_SIZE),
        .RESIDUAL_WIDTH  (POST_ATTENTION_WIDTH),
        .RESIDUAL_FRAC   (POST_ATTENTION_FRAC),
        .O_PROJ_WIDTH    (DOWN_WIDTH),
        .O_PROJ_FRAC     (DOWN_FRAC),
        .OUT_WIDTH       (OUT_WIDTH),
        .OUT_FRAC        (OUT_FRAC),
        .ELEMENT_INDEX_W (ELEMENT_INDEX_W),
        .ACC_WIDTH       (ACC_WIDTH)
    ) inst_residual_add_1024 (
        .i_clk          (i_clk),
        .i_rst_n        (i_rst_n),
        .i_start        (add_start),
        .i_residual_flat(i_post_attn_hidden_flat),
        .i_o_proj_flat  (i_down_out_flat),
        .o_busy         (add_busy),
        .o_done         (add_done),
        .o_saturation   (add_saturation),
        .o_output_count (add_output_count),
        .o_output_flat  (layer_out_flat)
    );

endmodule

`default_nettype wire
