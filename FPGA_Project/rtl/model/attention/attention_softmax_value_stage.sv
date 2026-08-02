`default_nettype none

// Fixed-point attention softmax + value accumulation.
//
// Input score stream order is expected to match attention_score_stage:
// q_head-major, then cache position. Scores are signed Q24.24.
//
// For each Q head:
//   1. max-subtract scores.
//   2. approximate exp(score - max) with an external UQ0.20 LUT indexed in
//      1/16 score steps, clamped at -16.
//   3. normalize to UQ0.16 probabilities.
//   4. read V cache with GQA mapping kv_head = q_head / 2.
//   5. emit attn_out[q_head][dim] as signed Q12.12.
module attention_softmax_value_stage #(
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int SCORE_WIDTH      = 64,
    parameter int SCORE_FRAC       = 24,
    parameter int VALUE_WIDTH      = 24,
    parameter int OUT_WIDTH        = 24,
    parameter int EXP_WIDTH        = 24,
    parameter int EXP_FRAC         = 20,
    parameter int EXP_LUT_STEP_FRAC = 4,
    parameter int EXP_LUT_SIZE     = 257,
    parameter int USE_EXP_READ_PORT = 0,
    parameter int PROB_WIDTH       = 24,
    parameter int PROB_FRAC        = 16,
    parameter int Q_HEAD_INDEX_W   = (NUM_Q_HEADS <= 1) ? 1 : $clog2(NUM_Q_HEADS),
    parameter int KV_HEAD_INDEX_W  = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int CACHE_LENGTH_W   = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT + 1),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int EXP_LUT_INDEX_W  = (EXP_LUT_SIZE <= 1) ? 1 : $clog2(EXP_LUT_SIZE),
    parameter int EXP_SUM_WIDTH    = EXP_WIDTH + CACHE_LENGTH_W + 2,
    parameter int VALUE_ACC_WIDTH  = VALUE_WIDTH + PROB_WIDTH + CACHE_LENGTH_W + 4
)
(
    input  wire logic                                           i_clk,
    input  wire logic                                           i_rst_n,

    input  wire logic                                           i_start,
    input  wire logic [CACHE_LENGTH_W-1 : 0]                    i_cache_length,
    input  wire logic [EXP_LUT_SIZE*EXP_WIDTH-1 : 0]            i_exp_lut_flat,
    output logic [EXP_LUT_INDEX_W-1 : 0]                        o_exp_lut_rd_addr,
    input  wire logic [EXP_WIDTH-1 : 0]                          i_exp_lut_rd_data,

    input  wire logic                                           i_score_valid,
    output logic                                           o_score_ready,
    input  wire logic [Q_HEAD_INDEX_W-1 : 0]                    i_score_q_head,
    input  wire logic [KV_HEAD_INDEX_W-1 : 0]                   i_score_kv_head,
    input  wire logic [POSITION_INDEX_W-1 : 0]                  i_score_position,
    input  wire logic signed [SCORE_WIDTH-1 : 0]                i_score_scaled,
    input  wire logic                                           i_score_last,

    output logic                                           o_v_req_valid,
    input  wire logic                                           i_v_req_ready,
    output logic [KV_HEAD_INDEX_W-1 : 0]                   o_v_req_kv_head,
    output logic [POSITION_INDEX_W-1 : 0]                  o_v_req_position,
    output logic [DIM_INDEX_W-1 : 0]                       o_v_req_dim,
    output logic [PROB_WIDTH-1 : 0]                         o_v_req_prob,

    input  wire logic                                           i_v_rsp_valid,
    output logic                                           o_v_rsp_ready,
    input  wire logic signed [VALUE_WIDTH-1 : 0]                i_v_rsp_data,

    output logic                                           o_out_valid,
    input  wire logic                                           i_out_ready,
    output logic [Q_HEAD_INDEX_W-1 : 0]                    o_out_q_head,
    output logic [DIM_INDEX_W-1 : 0]                       o_out_dim,
    output logic signed [OUT_WIDTH-1 : 0]                  o_out_data,
    output logic                                           o_out_last,

    output logic                                           o_busy,
    output logic                                           o_done,
    output logic                                           o_error,
    output logic                                           o_saturation,
    output logic [31 : 0]                                 o_score_count,
    output logic [31 : 0]                                 o_v_request_count,
    output logic [31 : 0]                                 o_v_response_count,
    output logic [31 : 0]                                 o_output_count
);

    localparam int KV_REPEAT = NUM_Q_HEADS / NUM_KV_HEADS;
    localparam bit CONFIG_VALID =
        (NUM_Q_HEADS >= NUM_KV_HEADS) &&
        ((NUM_Q_HEADS % NUM_KV_HEADS) == 0) &&
        (HEAD_DIM > 0) &&
        (MAX_CONTEXT > 0) &&
        (SCORE_FRAC > EXP_LUT_STEP_FRAC);

    localparam logic [Q_HEAD_INDEX_W-1 : 0] LAST_Q_HEAD_INDEX = NUM_Q_HEADS - 1;
    localparam logic [DIM_INDEX_W-1 : 0] LAST_DIM_INDEX = HEAD_DIM - 1;

    localparam logic [4 : 0] IDLE          = 5'd0;
    localparam logic [4 : 0] LOAD_SCORE    = 5'd1;
    localparam logic [4 : 0] PREP_HEAD     = 5'd2;
    localparam logic [4 : 0] MAX_READ      = 5'd3;
    localparam logic [4 : 0] MAX_USE       = 5'd4;
    localparam logic [4 : 0] SUM_READ      = 5'd5;
    localparam logic [4 : 0] SUM_USE       = 5'd6;
    localparam logic [4 : 0] PROB_READ     = 5'd7;
    localparam logic [4 : 0] PROB_USE      = 5'd8;
    localparam logic [4 : 0] PROB_WAIT     = 5'd9;
    localparam logic [4 : 0] PREP_VALUE    = 5'd10;
    localparam logic [4 : 0] FETCH_PROB    = 5'd11;
    localparam logic [4 : 0] ISSUE_V_REQ   = 5'd12;
    localparam logic [4 : 0] WAIT_V_RSP    = 5'd13;
    localparam logic [4 : 0] FINALIZE_OUT  = 5'd14;
    localparam logic [4 : 0] EMIT_OUT      = 5'd15;
    localparam logic [4 : 0] DONE          = 5'd16;
    localparam logic [4 : 0] SUM_LUT_WAIT  = 5'd17;
    localparam logic [4 : 0] PROB_LUT_WAIT = 5'd18;

    localparam signed [OUT_WIDTH-1 : 0] OUT_MAX = {1'b0, {(OUT_WIDTH-1){1'b1}}};
    localparam signed [OUT_WIDTH-1 : 0] OUT_MIN = {1'b1, {(OUT_WIDTH-1){1'b0}}};
    localparam signed [VALUE_ACC_WIDTH-1 : 0] OUT_MAX_EXT =
        {{(VALUE_ACC_WIDTH-OUT_WIDTH){OUT_MAX[OUT_WIDTH-1]}}, OUT_MAX};
    localparam signed [VALUE_ACC_WIDTH-1 : 0] OUT_MIN_EXT =
        {{(VALUE_ACC_WIDTH-OUT_WIDTH){OUT_MIN[OUT_WIDTH-1]}}, OUT_MIN};
    localparam int PROB_NUM_WIDTH = EXP_WIDTH + PROB_FRAC + 1;

    typedef logic signed [SCORE_WIDTH-1 : 0] score_t;
    typedef logic [PROB_WIDTH-1 : 0] prob_t;

    localparam int SCORE_DEPTH =
        NUM_Q_HEADS * MAX_CONTEXT;
    localparam int SCORE_ADDR_W =
        (SCORE_DEPTH <= 1) ? 1 : $clog2(SCORE_DEPTH);

    logic [4 : 0] current_state;
    logic [4 : 0] next_state;
    logic [CACHE_LENGTH_W-1 : 0] cache_length_reg;
    logic [Q_HEAD_INDEX_W-1 : 0] q_head_index;
    logic [KV_HEAD_INDEX_W-1 : 0] kv_head_index;
    logic [POSITION_INDEX_W-1 : 0] position_index;
    logic [POSITION_INDEX_W-1 : 0] softmax_position;
    logic [DIM_INDEX_W-1 : 0] dim_index;
    logic signed [VALUE_ACC_WIDTH-1 : 0] value_acc;
    logic signed [VALUE_ACC_WIDTH-1 : 0] value_acc_next;
    logic value_saturation_from_acc;

    (* ram_style = "block" *) score_t score_mem [0:SCORE_DEPTH-1];
    logic [SCORE_ADDR_W-1 : 0] score_write_addr;
    logic [SCORE_ADDR_W-1 : 0] score_read_addr;
    score_t score_read_data;
    prob_t prob_read_data;
    score_t max_score_reg;
    logic [EXP_SUM_WIDTH-1 : 0] exp_sum_reg;
    logic [EXP_LUT_INDEX_W-1 : 0] lut_index_current;
    logic [EXP_WIDTH-1 : 0] exp_value_current;
    logic [EXP_SUM_WIDTH-1 : 0] exp_sum_next;
    logic [PROB_NUM_WIDTH-1 : 0] prob_numerator_current;
    prob_t prob_value_current;
    logic prob_div_start;
    logic prob_div_busy;
    logic prob_div_done;
    logic prob_divide_by_zero;
    logic prob_write_en;
    logic [PROB_WIDTH-1 : 0] prob_div_quotient;
    logic [EXP_SUM_WIDTH-1 : 0] prob_div_remainder;

    logic [CACHE_LENGTH_W-1 : 0] position_index_ext;
    logic [CACHE_LENGTH_W-1 : 0] softmax_position_ext;
    logic [CACHE_LENGTH_W-1 : 0] score_position_ext;
    logic [CACHE_LENGTH_W-1 : 0] last_cache_position;
    logic [31 : 0] expected_score_count;
    logic [31 : 0] expected_v_count;
    logic [31 : 0] expected_output_count;
    logic start_params_valid;
    logic score_fire;
    logic v_req_fire;
    logic v_rsp_fire;
    logic out_fire;
    logic current_output_is_last;
    logic error_reg;
    logic saturation_reg;

    logic [Q_HEAD_INDEX_W-1 : 0] out_q_head_reg;
    logic [DIM_INDEX_W-1 : 0] out_dim_reg;
    logic signed [OUT_WIDTH-1 : 0] out_data_reg;
    logic out_last_reg;

    assign o_busy = (current_state != IDLE) && (current_state != DONE);
    assign o_done = (current_state == DONE);
    assign o_error = error_reg;
    assign o_saturation = saturation_reg;

    assign o_score_ready = (current_state == LOAD_SCORE);
    assign score_fire = i_score_valid && o_score_ready;

    assign o_v_req_valid = (current_state == ISSUE_V_REQ);
    assign o_v_req_kv_head = kv_head_index;
    assign o_v_req_position = position_index;
    assign o_v_req_dim = dim_index;
    assign o_v_req_prob = prob_read_data;
    assign v_req_fire = o_v_req_valid && i_v_req_ready;

    assign o_v_rsp_ready = (current_state == WAIT_V_RSP);
    assign v_rsp_fire = i_v_rsp_valid && o_v_rsp_ready;

    assign o_out_valid = (current_state == EMIT_OUT);
    assign o_out_q_head = out_q_head_reg;
    assign o_out_dim = out_dim_reg;
    assign o_out_data = out_data_reg;
    assign o_out_last = o_out_valid && out_last_reg;
    assign out_fire = o_out_valid && i_out_ready;

    assign position_index_ext = {{(CACHE_LENGTH_W-POSITION_INDEX_W){1'b0}}, position_index};
    assign softmax_position_ext =
        {{(CACHE_LENGTH_W-POSITION_INDEX_W){1'b0}}, softmax_position};
    assign score_position_ext = {{(CACHE_LENGTH_W-POSITION_INDEX_W){1'b0}}, i_score_position};
    assign score_write_addr =
        (i_score_q_head * MAX_CONTEXT) + i_score_position;
    assign score_read_addr =
        (q_head_index * MAX_CONTEXT) + softmax_position;
    assign last_cache_position = cache_length_reg - 1'b1;
    assign expected_score_count = NUM_Q_HEADS * cache_length_reg;
    assign expected_v_count = NUM_Q_HEADS * HEAD_DIM * cache_length_reg;
    assign expected_output_count = NUM_Q_HEADS * HEAD_DIM;
    assign current_output_is_last =
        (q_head_index == LAST_Q_HEAD_INDEX) &&
        (dim_index == LAST_DIM_INDEX);

    assign start_params_valid =
        CONFIG_VALID &&
        (i_cache_length > 'd0) &&
        (i_cache_length <= MAX_CONTEXT);

    function automatic logic [EXP_LUT_INDEX_W-1 : 0] diff_to_lut_index(
        input score_t max_score,
        input score_t score_value
    );
        score_t neg_diff;
        logic [SCORE_WIDTH-1 : 0] rounded_steps;
        begin
            neg_diff = max_score - score_value;
            if (neg_diff <= 'd0) begin
                diff_to_lut_index = 'd0;
            end
            else begin
                rounded_steps =
                    (neg_diff + ({{(SCORE_WIDTH-1){1'b0}}, 1'b1} << (SCORE_FRAC - EXP_LUT_STEP_FRAC - 1)))
                    >>> (SCORE_FRAC - EXP_LUT_STEP_FRAC);
                if (rounded_steps >= EXP_LUT_SIZE) begin
                    diff_to_lut_index = EXP_LUT_SIZE - 1;
                end
                else begin
                    diff_to_lut_index = rounded_steps[EXP_LUT_INDEX_W-1 : 0];
                end
            end
        end
    endfunction

    function automatic logic signed [OUT_WIDTH-1 : 0] saturate_to_out(
        input logic signed [VALUE_ACC_WIDTH-1 : 0] value
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

    localparam int VALUE_PRODUCT_WIDTH = VALUE_WIDTH + PROB_WIDTH + 1;
    logic signed [VALUE_PRODUCT_WIDTH-1 : 0] prob_product_operand;
    logic signed [VALUE_PRODUCT_WIDTH-1 : 0] value_product_operand;
    logic signed [VALUE_PRODUCT_WIDTH-1 : 0] value_product;

    always @* begin
        lut_index_current = diff_to_lut_index(max_score_reg, score_read_data);
        if (USE_EXP_READ_PORT != 0) begin
            exp_value_current = i_exp_lut_rd_data;
        end
        else begin
            exp_value_current =
                i_exp_lut_flat[
                    lut_index_current*EXP_WIDTH +: EXP_WIDTH
                ];
        end
        exp_sum_next =
            exp_sum_reg +
            {{(EXP_SUM_WIDTH-EXP_WIDTH){1'b0}}, exp_value_current};
        prob_numerator_current =
            {{(PROB_NUM_WIDTH-EXP_WIDTH){1'b0}}, exp_value_current} << PROB_FRAC;
    end

    assign o_exp_lut_rd_addr = lut_index_current;
    assign prob_div_start = (current_state == PROB_LUT_WAIT);
    assign prob_value_current =
        prob_divide_by_zero ? 'd0 : prob_div_quotient;
    assign prob_write_en =
        i_rst_n && (current_state == PROB_WAIT) && prob_div_done;

    // Keep the iterative divider as a sequential hierarchy. Without this
    // boundary Vivado can fold the fixed-latency transaction into the
    // probability RAM write cone, recreating a long combinational divider.
    (* DONT_TOUCH = "yes" *)
    fixed_udiv #(
        .NUMERATOR_WIDTH   (PROB_NUM_WIDTH),
        .DENOMINATOR_WIDTH (EXP_SUM_WIDTH),
        .QUOTIENT_WIDTH    (PROB_WIDTH),
        .REMAINDER_WIDTH   (EXP_SUM_WIDTH)
    ) prob_divider (
        .i_clk            (i_clk),
        .i_rst_n          (i_rst_n),
        .i_start          (prob_div_start),
        .i_numerator      (prob_numerator_current),
        .i_denominator    (exp_sum_reg),
        .o_busy           (prob_div_busy),
        .o_done           (prob_div_done),
        .o_divide_by_zero (prob_divide_by_zero),
        .o_quotient       (prob_div_quotient),
        .o_remainder      (prob_div_remainder)
    );

    always @* begin
        prob_product_operand =
            {{(VALUE_PRODUCT_WIDTH-PROB_WIDTH-1){1'b0}}, 1'b0, prob_read_data};
        value_product_operand =
            {{(VALUE_PRODUCT_WIDTH-VALUE_WIDTH){i_v_rsp_data[VALUE_WIDTH-1]}}, i_v_rsp_data};
        value_product = prob_product_operand * value_product_operand;
        value_acc_next =
            value_acc +
            {{(VALUE_ACC_WIDTH-VALUE_PRODUCT_WIDTH){value_product[VALUE_PRODUCT_WIDTH-1]}},
             value_product};
        value_saturation_from_acc =
            ((value_acc >>> PROB_FRAC) > OUT_MAX_EXT) ||
            ((value_acc >>> PROB_FRAC) < OUT_MIN_EXT);
    end

    always_comb begin
        next_state = current_state;

        case (current_state)
            IDLE: begin
                if (i_start == 1'b1) begin
                    if (start_params_valid == 1'b1) begin
                        next_state = LOAD_SCORE;
                    end
                    else begin
                        next_state = DONE;
                    end
                end
            end

            LOAD_SCORE: begin
                if ((score_fire == 1'b1) && (i_score_last == 1'b1)) begin
                    next_state = PREP_HEAD;
                end
            end

            PREP_HEAD: begin
                next_state = MAX_READ;
            end

            MAX_READ: begin
                next_state = MAX_USE;
            end

            MAX_USE: begin
                if (softmax_position_ext == last_cache_position) begin
                    next_state = SUM_READ;
                end
                else begin
                    next_state = MAX_READ;
                end
            end

            SUM_READ: begin
                next_state = SUM_USE;
            end

            SUM_USE: begin
                next_state = SUM_LUT_WAIT;
            end

            SUM_LUT_WAIT: begin
                if (softmax_position_ext == last_cache_position) begin
                    next_state = PROB_READ;
                end
                else begin
                    next_state = SUM_READ;
                end
            end

            PROB_READ: begin
                next_state = PROB_USE;
            end

            PROB_USE: begin
                next_state = PROB_LUT_WAIT;
            end

            PROB_LUT_WAIT: begin
                next_state = PROB_WAIT;
            end

            PROB_WAIT: begin
                if (prob_div_done) begin
                    if (softmax_position_ext == last_cache_position) begin
                        next_state = PREP_VALUE;
                    end
                    else begin
                        next_state = PROB_READ;
                    end
                end
            end

            PREP_VALUE: begin
                next_state = FETCH_PROB;
            end

            FETCH_PROB: begin
                next_state = ISSUE_V_REQ;
            end

            ISSUE_V_REQ: begin
                if (v_req_fire == 1'b1) begin
                    next_state = WAIT_V_RSP;
                end
            end

            WAIT_V_RSP: begin
                if (v_rsp_fire == 1'b1) begin
                    if (position_index_ext == last_cache_position) begin
                        next_state = FINALIZE_OUT;
                    end
                    else begin
                        next_state = FETCH_PROB;
                    end
                end
            end

            FINALIZE_OUT: begin
                next_state = EMIT_OUT;
            end

            EMIT_OUT: begin
                if (out_fire == 1'b1) begin
                    if (out_last_reg == 1'b1) begin
                        next_state = DONE;
                    end
                    else if (dim_index == LAST_DIM_INDEX) begin
                        next_state = PREP_HEAD;
                    end
                    else begin
                        next_state = FETCH_PROB;
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

    // A flat, synchronous-read score store maps to a simple dual-port BRAM.
    // Reset only gates writes; stored data is overwritten by every invocation.
    always_ff @(posedge i_clk) begin
        if (i_rst_n && score_fire &&
            (i_score_q_head < NUM_Q_HEADS) &&
            (score_position_ext < cache_length_reg)) begin
            score_mem[score_write_addr] <= i_score_scaled;
        end
        score_read_data <= score_mem[score_read_addr];
    end

    // Probabilities are produced serially after max and exp-sum passes. A
    // synchronous BRAM read is primed in FETCH_PROB before each V-cache
    // request.
    qmap_sdp_bram #(
        .DATA_WIDTH(PROB_WIDTH),
        .DEPTH(MAX_CONTEXT),
        .ADDR_WIDTH(POSITION_INDEX_W)
    ) prob_store (
        .i_clk(i_clk),
        .i_wr_en(prob_write_en),
        .i_wr_addr(softmax_position),
        .i_wr_data(prob_value_current),
        .i_rd_en(1'b1),
        .i_rd_addr(position_index),
        .o_rd_data(prob_read_data)
    );

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
            cache_length_reg <= 'd0;
            q_head_index <= 'd0;
            kv_head_index <= 'd0;
            position_index <= 'd0;
            softmax_position <= 'd0;
            dim_index <= 'd0;
            value_acc <= 'd0;
            max_score_reg <= 'd0;
            exp_sum_reg <= 'd0;
            out_q_head_reg <= 'd0;
            out_dim_reg <= 'd0;
            out_data_reg <= 'd0;
            out_last_reg <= 1'b0;
            error_reg <= 1'b0;
            saturation_reg <= 1'b0;
            o_score_count <= 32'd0;
            o_v_request_count <= 32'd0;
            o_v_response_count <= 32'd0;
            o_output_count <= 32'd0;
        end
        else begin
            case (current_state)
                IDLE: begin
                    if (i_start == 1'b1) begin
                        cache_length_reg <= i_cache_length;
                        q_head_index <= 'd0;
                        kv_head_index <= 'd0;
                        position_index <= 'd0;
                        softmax_position <= 'd0;
                        dim_index <= 'd0;
                        value_acc <= 'd0;
                        max_score_reg <= 'd0;
                        exp_sum_reg <= 'd0;
                        out_q_head_reg <= 'd0;
                        out_dim_reg <= 'd0;
                        out_data_reg <= 'd0;
                        out_last_reg <= 1'b0;
                        error_reg <= !start_params_valid;
                        saturation_reg <= 1'b0;
                        o_score_count <= 32'd0;
                        o_v_request_count <= 32'd0;
                        o_v_response_count <= 32'd0;
                        o_output_count <= 32'd0;
                    end
                end

                LOAD_SCORE: begin
                    if (score_fire == 1'b1) begin
                        if (!((i_score_q_head < NUM_Q_HEADS) &&
                              (score_position_ext < cache_length_reg))) begin
                            error_reg <= 1'b1;
                        end
                        o_score_count <= o_score_count + 1'b1;
                        if (i_score_last == 1'b1) begin
                            error_reg <= error_reg || ((o_score_count + 1'b1) != expected_score_count);
                            q_head_index <= 'd0;
                            kv_head_index <= 'd0;
                            position_index <= 'd0;
                            softmax_position <= 'd0;
                            dim_index <= 'd0;
                            value_acc <= 'd0;
                            max_score_reg <= 'd0;
                            exp_sum_reg <= 'd0;
                        end
                    end
                end

                PREP_HEAD: begin
                    softmax_position <= 'd0;
                    max_score_reg <= 'd0;
                    exp_sum_reg <= 'd0;
                end

                MAX_USE: begin
                    if ((softmax_position == 'd0) ||
                        (score_read_data > max_score_reg)) begin
                        max_score_reg <= score_read_data;
                    end
                    if (softmax_position_ext == last_cache_position) begin
                        softmax_position <= 'd0;
                        exp_sum_reg <= 'd0;
                    end
                    else begin
                        softmax_position <= softmax_position + 1'b1;
                    end
                end

                SUM_LUT_WAIT: begin
                    exp_sum_reg <= exp_sum_next;
                    if (softmax_position_ext == last_cache_position) begin
                        softmax_position <= 'd0;
                    end
                    else begin
                        softmax_position <= softmax_position + 1'b1;
                    end
                end

                PROB_WAIT: begin
                    if (prob_div_done) begin
                        if (softmax_position_ext == last_cache_position) begin
                            softmax_position <= 'd0;
                        end
                        else begin
                            softmax_position <= softmax_position + 1'b1;
                        end
                    end
                end

                PREP_VALUE: begin
                    kv_head_index <= q_head_index / KV_REPEAT;
                    position_index <= 'd0;
                    dim_index <= 'd0;
                    value_acc <= 'd0;
                end

                MAX_READ,
                SUM_READ,
                SUM_USE,
                PROB_READ,
                PROB_USE,
                PROB_LUT_WAIT,
                FETCH_PROB: begin
                end

                ISSUE_V_REQ: begin
                    if (v_req_fire == 1'b1) begin
                        o_v_request_count <= o_v_request_count + 1'b1;
                    end
                end

                WAIT_V_RSP: begin
                    if (v_rsp_fire == 1'b1) begin
                        o_v_response_count <= o_v_response_count + 1'b1;

                        if (position_index_ext == last_cache_position) begin
                            value_acc <= value_acc_next;
                        end
                        else begin
                            position_index <= position_index + 1'b1;
                            value_acc <= value_acc_next;
                        end
                    end
                end

                FINALIZE_OUT: begin
                    out_q_head_reg <= q_head_index;
                    out_dim_reg <= dim_index;
                    out_data_reg <= saturate_to_out(value_acc >>> PROB_FRAC);
                    out_last_reg <= current_output_is_last;
                    saturation_reg <= saturation_reg || value_saturation_from_acc;
                end

                EMIT_OUT: begin
                    if (out_fire == 1'b1) begin
                        o_output_count <= o_output_count + 1'b1;
                        value_acc <= 'd0;
                        position_index <= 'd0;

                        if (out_last_reg == 1'b0) begin
                            if (dim_index == LAST_DIM_INDEX) begin
                                q_head_index <= q_head_index + 1'b1;
                                kv_head_index <= (q_head_index + 1'b1) / KV_REPEAT;
                                dim_index <= 'd0;
                            end
                            else begin
                                dim_index <= dim_index + 1'b1;
                            end
                        end
                    end
                end

                DONE: begin
                    error_reg <=
                        error_reg ||
                        (o_score_count != expected_score_count) ||
                        (o_v_request_count != expected_v_count) ||
                        (o_v_response_count != expected_v_count) ||
                        (o_output_count != expected_output_count);
                end

                default: begin
                    cache_length_reg <= 'd0;
                    q_head_index <= 'd0;
                    kv_head_index <= 'd0;
                    position_index <= 'd0;
                    softmax_position <= 'd0;
                    dim_index <= 'd0;
                    value_acc <= 'd0;
                    max_score_reg <= 'd0;
                    exp_sum_reg <= 'd0;
                    out_q_head_reg <= 'd0;
                    out_dim_reg <= 'd0;
                    out_data_reg <= 'd0;
                    out_last_reg <= 1'b0;
                    error_reg <= 1'b1;
                    saturation_reg <= 1'b0;
                    o_score_count <= 32'd0;
                    o_v_request_count <= 32'd0;
                    o_v_response_count <= 32'd0;
                    o_output_count <= 32'd0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
