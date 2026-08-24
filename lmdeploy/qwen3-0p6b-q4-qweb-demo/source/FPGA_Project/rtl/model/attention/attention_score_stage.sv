`default_nettype none

// Generate current-token attention scores from Q RoPE and cached K values.
//
// This module intentionally uses a small K-cache request/response interface
// instead of a giant flattened K-cache port. That keeps the local simulation
// close to the later DDR/AXI-facing design:
//
//   for q_head = 0..15:
//     kv_head = q_head / 2
//     for position = 0..cache_length-1:
//       raw_score = sum_dim(q_rope[q_head][dim] * k_cache[kv_head][position][dim])
//       scaled_score = raw_score * attention_scale_q0_31 >> 31
//
// Inputs are signed Q12.12. The raw dot product and scaled score are both
// emitted as signed Q24.24 integers in a 64-bit container.
module attention_score_stage #(
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int IN_WIDTH         = 24,
    parameter int SCORE_WIDTH      = 64,
    parameter int SCALE_WIDTH      = 32,
    parameter int SCALE_FRAC       = 31,
    parameter int USE_Q_READ_PORT  = 0,
    parameter int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM,
    parameter int Q_ADDR_WIDTH     =
        (Q_COUNT <= 1) ? 1 : $clog2(Q_COUNT),
    parameter int Q_HEAD_INDEX_W   = (NUM_Q_HEADS <= 1) ? 1 : $clog2(NUM_Q_HEADS),
    parameter int KV_HEAD_INDEX_W  = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int CACHE_LENGTH_W   = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT + 1),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,
    input  wire logic [CACHE_LENGTH_W-1 : 0]                    i_cache_length,
    input  wire logic signed [SCALE_WIDTH-1 : 0]                i_score_scale_q0_31,
    input  wire logic [NUM_Q_HEADS*HEAD_DIM*IN_WIDTH-1 : 0]     i_q_rope_flat,
    output logic [Q_ADDR_WIDTH-1 : 0]                       o_q_read_addr,
    input  wire logic signed [IN_WIDTH-1 : 0]                    i_q_read_data,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_error,

    output logic                                           o_k_req_valid,
    input  wire logic                                           i_k_req_ready,
    output logic [KV_HEAD_INDEX_W-1 : 0]                   o_k_req_kv_head,
    output logic [POSITION_INDEX_W-1 : 0]                  o_k_req_position,
    output logic [DIM_INDEX_W-1 : 0]                       o_k_req_dim,

    input  wire logic                                           i_k_rsp_valid,
    output logic                                           o_k_rsp_ready,
    input  wire logic signed [IN_WIDTH-1 : 0]                    i_k_rsp_data,

    output logic                                           o_score_valid,
    input  wire logic                                           i_score_ready,
    output logic [Q_HEAD_INDEX_W-1 : 0]                    o_score_q_head,
    output logic [KV_HEAD_INDEX_W-1 : 0]                   o_score_kv_head,
    output logic [POSITION_INDEX_W-1 : 0]                  o_score_position,
    output logic signed [SCORE_WIDTH-1 : 0]                o_score_raw,
    output logic signed [SCORE_WIDTH-1 : 0]                o_score_scaled,
    output logic                                           o_score_last,

    output logic [31 : 0]                                 o_k_request_count,
    output logic [31 : 0]                                 o_k_response_count,
    output logic [31 : 0]                                 o_score_count
);

    localparam int KV_REPEAT = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam bit CONFIG_VALID =
        (NUM_Q_HEADS >= NUM_KV_HEADS) &&
        ((NUM_Q_HEADS % NUM_KV_HEADS) == 0) &&
        (HEAD_DIM > 0) &&
        (MAX_CONTEXT > 0);
    localparam logic [Q_HEAD_INDEX_W-1 : 0] LAST_Q_HEAD_INDEX = NUM_Q_HEADS - 1;
    localparam logic [DIM_INDEX_W-1 : 0]    LAST_DIM_INDEX    = HEAD_DIM - 1;

    localparam int PRODUCT_WIDTH = IN_WIDTH * 2;
    localparam int SCALED_PRODUCT_WIDTH = SCORE_WIDTH + SCALE_WIDTH;

    localparam IDLE      = 3'd0;
    localparam ISSUE_REQ = 3'd1;
    localparam WAIT_RSP  = 3'd2;
    localparam EMIT      = 3'd3;
    localparam DONE      = 3'd4;

    logic [2 : 0] current_state;
    logic [2 : 0] next_state;

    logic [Q_HEAD_INDEX_W-1 : 0]        q_head_index;
    logic [KV_HEAD_INDEX_W-1 : 0]       kv_head_index;
    logic [POSITION_INDEX_W-1 : 0]      position_index;
    logic [DIM_INDEX_W-1 : 0]           dim_index;
    logic [CACHE_LENGTH_W-1 : 0]        cache_length_reg;
    logic signed [SCORE_WIDTH-1 : 0]    dot_acc;
    logic signed [SCORE_WIDTH-1 : 0]    dot_acc_next;
    logic signed [SCORE_WIDTH-1 : 0]    score_raw_reg;
    logic signed [SCORE_WIDTH-1 : 0]    score_scaled_reg;
    logic [Q_HEAD_INDEX_W-1 : 0]        score_q_head_reg;
    logic [KV_HEAD_INDEX_W-1 : 0]       score_kv_head_reg;
    logic [POSITION_INDEX_W-1 : 0]      score_position_reg;
    logic                               score_last_reg;
    logic                               error_reg;

    logic signed [IN_WIDTH-1 : 0]       current_q;
    logic signed [PRODUCT_WIDTH-1 : 0]  product_qk;
    logic signed [SCALED_PRODUCT_WIDTH-1 : 0] scaled_product_next;
    logic [CACHE_LENGTH_W-1 : 0]        position_index_ext;
    logic [CACHE_LENGTH_W-1 : 0]        last_cache_position;
    logic [31 : 0]                      expected_score_count;
    logic                               start_params_valid;
    logic                               req_fire;
    logic                               rsp_fire;
    logic                               score_fire;
    logic                               current_score_is_last;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;

    assign o_k_req_valid = (current_state == ISSUE_REQ);
    assign o_k_req_kv_head = kv_head_index;
    assign o_k_req_position = position_index;
    assign o_k_req_dim = dim_index;
    assign o_k_rsp_ready = (current_state == WAIT_RSP);

    assign o_score_valid = (current_state == EMIT);
    assign o_score_q_head = score_q_head_reg;
    assign o_score_kv_head = score_kv_head_reg;
    assign o_score_position = score_position_reg;
    assign o_score_raw = score_raw_reg;
    assign o_score_scaled = score_scaled_reg;
    assign o_score_last = o_score_valid && score_last_reg;

    assign req_fire = o_k_req_valid && i_k_req_ready;
    assign rsp_fire = i_k_rsp_valid && o_k_rsp_ready;
    assign score_fire = o_score_valid && i_score_ready;

    assign start_params_valid =
        CONFIG_VALID &&
        (i_cache_length > 'd0) &&
        (i_cache_length <= MAX_CONTEXT);

    assign position_index_ext = {{(CACHE_LENGTH_W-POSITION_INDEX_W){1'b0}}, position_index};
    assign last_cache_position = cache_length_reg - 1'b1;
    assign expected_score_count = NUM_Q_HEADS * cache_length_reg;
    assign current_score_is_last =
        (q_head_index == LAST_Q_HEAD_INDEX) &&
        (position_index_ext == last_cache_position);

    assign o_q_read_addr =
        Q_ADDR_WIDTH'((q_head_index * HEAD_DIM) + dim_index);
    assign current_q =
        (USE_Q_READ_PORT != 0) ?
        i_q_read_data :
        i_q_rope_flat[
            (q_head_index*HEAD_DIM + dim_index)*IN_WIDTH +: IN_WIDTH
        ];
    assign product_qk = current_q * i_k_rsp_data;
    assign dot_acc_next =
        dot_acc + {{(SCORE_WIDTH-PRODUCT_WIDTH){product_qk[PRODUCT_WIDTH-1]}}, product_qk};
    assign scaled_product_next = dot_acc_next * i_score_scale_q0_31;

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    if (start_params_valid == 1'b1) begin
                        next_state = ISSUE_REQ;
                    end
                    else begin
                        next_state = DONE;
                    end
                end
            end

            ISSUE_REQ: begin
                if (req_fire == 1'b1) begin
                    next_state = WAIT_RSP;
                end
            end

            WAIT_RSP: begin
                if (rsp_fire == 1'b1) begin
                    if (dim_index == LAST_DIM_INDEX) begin
                        next_state = EMIT;
                    end
                    else begin
                        next_state = ISSUE_REQ;
                    end
                end
            end

            EMIT: begin
                if (score_fire == 1'b1) begin
                    if (score_last_reg == 1'b1) begin
                        next_state = DONE;
                    end
                    else begin
                        next_state = ISSUE_REQ;
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
            q_head_index      <= 'd0;
            kv_head_index     <= 'd0;
            position_index    <= 'd0;
            dim_index         <= 'd0;
            cache_length_reg  <= 'd0;
            dot_acc           <= 'd0;
            score_raw_reg     <= 'd0;
            score_scaled_reg  <= 'd0;
            score_q_head_reg  <= 'd0;
            score_kv_head_reg <= 'd0;
            score_position_reg <= 'd0;
            score_last_reg    <= 1'b0;
            error_reg         <= 1'b0;
            o_k_request_count <= 32'd0;
            o_k_response_count <= 32'd0;
            o_score_count     <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        q_head_index      <= 'd0;
                        kv_head_index     <= 'd0;
                        position_index    <= 'd0;
                        dim_index         <= 'd0;
                        cache_length_reg  <= i_cache_length;
                        dot_acc           <= 'd0;
                        score_raw_reg     <= 'd0;
                        score_scaled_reg  <= 'd0;
                        score_q_head_reg  <= 'd0;
                        score_kv_head_reg <= 'd0;
                        score_position_reg <= 'd0;
                        score_last_reg    <= 1'b0;
                        error_reg         <= !start_params_valid;
                        o_k_request_count <= 32'd0;
                        o_k_response_count <= 32'd0;
                        o_score_count     <= 32'd0;
                    end
                end

                ISSUE_REQ: begin
                    if (req_fire == 1'b1) begin
                        o_k_request_count <= o_k_request_count + 1'b1;
                    end
                end

                WAIT_RSP: begin
                    if (rsp_fire == 1'b1) begin
                        o_k_response_count <= o_k_response_count + 1'b1;
                        if (dim_index == LAST_DIM_INDEX) begin
                            score_raw_reg <= dot_acc_next;
                            score_scaled_reg <= scaled_product_next >>> SCALE_FRAC;
                            score_q_head_reg <= q_head_index;
                            score_kv_head_reg <= kv_head_index;
                            score_position_reg <= position_index;
                            score_last_reg <= current_score_is_last;
                            dim_index <= dim_index;
                            dot_acc <= dot_acc_next;
                        end
                        else begin
                            dim_index <= dim_index + 1'b1;
                            dot_acc <= dot_acc_next;
                        end
                    end
                end

                EMIT: begin
                    if (score_fire == 1'b1) begin
                        o_score_count <= o_score_count + 1'b1;
                        dim_index <= 'd0;
                        dot_acc <= 'd0;

                        if (score_last_reg == 1'b0) begin
                            if (position_index_ext == last_cache_position) begin
                                position_index <= 'd0;
                                q_head_index <= q_head_index + 1'b1;
                                kv_head_index <= (q_head_index + 1'b1) / KV_REPEAT;
                            end
                            else begin
                                position_index <= position_index + 1'b1;
                            end
                        end
                    end
                end

                DONE: begin
                    error_reg <= error_reg || (o_score_count != expected_score_count);
                end

                default: begin
                    q_head_index      <= 'd0;
                    kv_head_index     <= 'd0;
                    position_index    <= 'd0;
                    dim_index         <= 'd0;
                    cache_length_reg  <= 'd0;
                    dot_acc           <= 'd0;
                    score_raw_reg     <= 'd0;
                    score_scaled_reg  <= 'd0;
                    score_q_head_reg  <= 'd0;
                    score_kv_head_reg <= 'd0;
                    score_position_reg <= 'd0;
                    score_last_reg    <= 1'b0;
                    error_reg         <= 1'b1;
                    o_k_request_count <= 32'd0;
                    o_k_response_count <= 32'd0;
                    o_score_count     <= 32'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
