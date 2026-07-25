`default_nettype none

// Append one token's K/V vectors into the per-layer KV cache.
//
// Shape contract:
//
//   K input: [8, 128], post-RoPE, flattened as k[head][dim]
//   V input: [8, 128], v_proj output, flattened as v[head][dim]
//
// Write order:
//
//   1. all K heads/dims, kind = 0
//   2. all V heads/dims, kind = 1
//
// Each signed 24-bit Q12.12 element is sign-extended into a 32-bit write word.
// The write stream holds address/data/control stable while i_wr_ready is low.
module kv_cache_append #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DATA_WIDTH       = 32,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int ELEMENT_WIDTH    = 24,
    parameter int ELEMENT_BYTES    = 4,
    parameter int LAYER_INDEX_W    = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int HEAD_INDEX_W     = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int ELEMENT_INDEX_W  = ((NUM_KV_HEADS*HEAD_DIM) <= 1) ? 1 : $clog2(NUM_KV_HEADS*HEAD_DIM)
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]                        i_base_addr,
    input  wire logic [LAYER_INDEX_W-1 : 0]                     i_layer_id,
    input  wire logic [POSITION_INDEX_W-1 : 0]                  i_position,

    input  wire logic [NUM_KV_HEADS*HEAD_DIM*ELEMENT_WIDTH-1:0] i_k_flat,
    input  wire logic [NUM_KV_HEADS*HEAD_DIM*ELEMENT_WIDTH-1:0] i_v_flat,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_error,

    output logic                                           o_wr_valid,
    input  wire logic                                           i_wr_ready,
    output logic [ADDR_WIDTH-1 : 0]                       o_wr_addr,
    output logic [DATA_WIDTH-1 : 0]                       o_wr_data,
    output logic                                           o_wr_last,

    output logic                                           o_current_kind,
    output logic [HEAD_INDEX_W-1 : 0]                     o_current_head,
    output logic [DIM_INDEX_W-1 : 0]                      o_current_dim,
    output logic [31 : 0]                                 o_write_count
);

    localparam IDLE = 2'd0;
    localparam RUN  = 2'd1;
    localparam DONE = 2'd2;

    localparam logic [HEAD_INDEX_W-1 : 0] LAST_HEAD_INDEX = NUM_KV_HEADS - 1;
    localparam logic [DIM_INDEX_W-1 : 0]  LAST_DIM_INDEX  = HEAD_DIM - 1;
    localparam int TOTAL_WRITES = 2 * NUM_KV_HEADS * HEAD_DIM;

    logic [1 : 0]                         current_state;
    logic [1 : 0]                         next_state;
    logic                                 kind_index;
    logic [HEAD_INDEX_W-1 : 0]            head_index;
    logic [DIM_INDEX_W-1 : 0]             dim_index;
    logic                                 error_reg;
    logic                                 start_params_valid;
    logic                                 addr_valid;
    logic [ADDR_WIDTH-1 : 0]              addr_offset_bytes;
    logic [ADDR_WIDTH-1 : 0]              byte_addr;
    logic [ELEMENT_INDEX_W-1 : 0]         element_index;
    logic signed [ELEMENT_WIDTH-1 : 0]    current_element;
    logic                                 current_is_last;
    logic                                 write_fire;

    assign o_busy = (current_state == RUN);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;

    assign o_wr_valid = (current_state == RUN) && addr_valid;
    assign o_wr_addr = byte_addr;
    assign o_wr_data = {{(DATA_WIDTH-ELEMENT_WIDTH){current_element[ELEMENT_WIDTH-1]}}, current_element};
    assign o_wr_last = o_wr_valid && current_is_last;

    assign o_current_kind = kind_index;
    assign o_current_head = head_index;
    assign o_current_dim = dim_index;

    assign element_index = (head_index * HEAD_DIM) + dim_index;
    assign current_is_last =
        (kind_index == 1'b1) &&
        (head_index == LAST_HEAD_INDEX) &&
        (dim_index == LAST_DIM_INDEX);
    assign write_fire = o_wr_valid && i_wr_ready;
    assign start_params_valid =
        (i_layer_id < NUM_LAYERS) &&
        (i_position < MAX_CONTEXT);

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    if (start_params_valid == 1'b1) begin
                        next_state = RUN;
                    end
                    else begin
                        next_state = DONE;
                    end
                end
            end

            RUN: begin
                if ((addr_valid == 1'b0) || (write_fire && current_is_last)) begin
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

    always_comb begin
        if (kind_index == 1'b0) begin
            current_element = i_k_flat[element_index*ELEMENT_WIDTH +: ELEMENT_WIDTH];
        end
        else begin
            current_element = i_v_flat[element_index*ELEMENT_WIDTH +: ELEMENT_WIDTH];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            kind_index   <= 1'b0;
            head_index   <= 'd0;
            dim_index    <= 'd0;
            o_write_count <= 32'd0;
            error_reg    <= 1'b0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        kind_index   <= 1'b0;
                        head_index   <= 'd0;
                        dim_index    <= 'd0;
                        o_write_count <= 32'd0;
                        error_reg    <= !start_params_valid;
                    end
                end

                RUN: begin
                    if (addr_valid == 1'b0) begin
                        error_reg <= 1'b1;
                    end
                    else if (write_fire == 1'b1) begin
                        o_write_count <= o_write_count + 1'b1;

                        if (current_is_last == 1'b0) begin
                            if (dim_index == LAST_DIM_INDEX) begin
                                dim_index <= 'd0;
                                if (head_index == LAST_HEAD_INDEX) begin
                                    head_index <= 'd0;
                                    kind_index <= 1'b1;
                                end
                                else begin
                                    head_index <= head_index + 1'b1;
                                end
                            end
                            else begin
                                dim_index <= dim_index + 1'b1;
                            end
                        end
                    end
                end

                DONE: begin
                    kind_index   <= kind_index;
                    head_index   <= head_index;
                    dim_index    <= dim_index;
                    o_write_count <= o_write_count;
                    error_reg    <= error_reg || (o_write_count != TOTAL_WRITES);
                end

                default: begin
                    kind_index   <= 1'b0;
                    head_index   <= 'd0;
                    dim_index    <= 'd0;
                    o_write_count <= 32'd0;
                    error_reg    <= 1'b1;
                end
            endcase
        end
    end

    kv_cache_addr_gen #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .NUM_LAYERS      (NUM_LAYERS),
        .NUM_KV_HEADS    (NUM_KV_HEADS),
        .HEAD_DIM        (HEAD_DIM),
        .MAX_CONTEXT     (MAX_CONTEXT),
        .ELEMENT_BYTES   (ELEMENT_BYTES),
        .LAYER_INDEX_W   (LAYER_INDEX_W),
        .HEAD_INDEX_W    (HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .DIM_INDEX_W     (DIM_INDEX_W)
    ) addr_gen (
        .i_base_addr    (i_base_addr),
        .i_layer_id     (i_layer_id),
        .i_kv_kind      (kind_index),
        .i_head_id      (head_index),
        .i_position     (i_position),
        .i_dim          (dim_index),
        .o_valid        (addr_valid),
        .o_offset_bytes (addr_offset_bytes),
        .o_byte_addr    (byte_addr)
    );

endmodule

`default_nettype wire
