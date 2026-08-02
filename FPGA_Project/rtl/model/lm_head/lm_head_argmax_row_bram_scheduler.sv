`default_nettype none

// Resource-first LM-head scanner using one BRAM-backed Q4 row engine.
//
// The 1024-element activation is loaded once through the native activation
// port.  Each vocabulary row then performs:
//   1. one packed-Q4 weight-row read,
//   2. one scale-row read,
//   3. one q4_gemv_row_bram run,
//   4. a strict-greater greedy argmax update.
//
// TILE_ROWS remains a software-visible accounting unit only.  Arithmetic is
// intentionally single-row/single-group for the first board image.
module lm_head_argmax_row_bram_scheduler #(
    parameter int ADDR_WIDTH       = 64,
    parameter int MAX_TILES        = 9496,
    parameter int TILE_COUNT_WIDTH =
        (MAX_TILES <= 1) ? 1 : $clog2(MAX_TILES + 1),
    parameter int TILE_ROWS        = 16,
    parameter int INPUT_SIZE       = 1024,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int PARTIAL_WIDTH    =
        ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    =
        SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TOKEN_ID_WIDTH   = 32,
    parameter int MEM_DATA_WIDTH   = 32
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_act_wr_valid,
    output logic                                   o_act_wr_ready,
    input  wire logic [$clog2(INPUT_SIZE)-1 : 0]   i_act_wr_addr,
    input  wire logic [ACT_WIDTH-1 : 0]            i_act_wr_data,

    input  wire logic                              i_start,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]       i_token_base,
    input  wire logic [TILE_COUNT_WIDTH-1 : 0]     i_tile_count,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_weight_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_scale_base_addr,

    output logic                                   o_mem_req_valid,
    input  wire logic                              i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                o_mem_req_addr,
    output logic [15 : 0]                          o_mem_req_len_bytes,
    input  wire logic                              i_mem_rsp_valid,
    output logic                                   o_mem_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]       i_mem_rsp_data,
    input  wire logic                              i_mem_rsp_last,

    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic [TOKEN_ID_WIDTH-1 : 0]            o_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0]      o_best_score_q26,
    output logic [TILE_COUNT_WIDTH-1 : 0]          o_current_tile_index,
    output logic [31 : 0]                          o_tiles_started,
    output logic [31 : 0]                          o_tiles_completed,
    output logic [31 : 0]                          o_compute_cycle_count,
    output logic [31 : 0]                          o_mem_read_burst_count,
    output logic [31 : 0]                          o_mem_read_word_count,

    output logic                                   o_row_result_valid,
    output logic [TOKEN_ID_WIDTH-1 : 0]            o_row_result_token,
    output logic signed [ROW_ACC_WIDTH-1 : 0]      o_row_result_score,
    output logic                                   o_tile_complete_pulse,
    output logic [3 : 0]                           o_state_debug,
    output logic [2 : 0]                           o_engine_state_debug
);

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int WEIGHT_ROW_BYTES =
        (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES =
        (GROUP_COUNT * SCALE_WIDTH) / 8;
    localparam int WEIGHT_WORDS =
        WEIGHT_ROW_BYTES / MEM_DATA_BYTES;
    localparam int SCALE_WORDS =
        SCALE_ROW_BYTES / MEM_DATA_BYTES;
    localparam int ACT_ADDR_WIDTH =
        (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);
    localparam int WEIGHT_ADDR_WIDTH =
        (WEIGHT_WORDS <= 1) ? 1 : $clog2(WEIGHT_WORDS);
    localparam int SCALE_ADDR_WIDTH =
        (SCALE_WORDS <= 1) ? 1 : $clog2(SCALE_WORDS);
    localparam logic [15 : 0] WEIGHT_ROW_BYTES_U16 =
        WEIGHT_ROW_BYTES;
    localparam logic [15 : 0] SCALE_ROW_BYTES_U16 =
        SCALE_ROW_BYTES;

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_WEIGHT_REQ,
        S_WEIGHT_READ,
        S_SCALE_REQ,
        S_SCALE_READ,
        S_ROW_START,
        S_ROW_WAIT,
        S_UPDATE,
        S_DONE
    } state_t;

    state_t state;
    logic [TOKEN_ID_WIDTH-1 : 0] token_base_reg;
    logic [TILE_COUNT_WIDTH-1 : 0] tile_count_reg;
    logic [31 : 0] total_row_count;
    logic [31 : 0] row_offset;
    logic [15 : 0] read_word_index;
    logic best_valid;

    logic engine_weight_wr_valid;
    logic engine_weight_wr_ready;
    logic [WEIGHT_ADDR_WIDTH-1 : 0] engine_weight_wr_addr;
    logic engine_scale_wr_valid;
    logic engine_scale_wr_ready;
    logic [SCALE_ADDR_WIDTH-1 : 0] engine_scale_wr_addr;
    logic engine_start;
    logic engine_busy;
    logic engine_done;
    logic engine_error;
    logic signed [ROW_ACC_WIDTH-1 : 0] engine_row_sum_q26;
    logic [TOKEN_ID_WIDTH-1 : 0] current_token;
    logic expected_response_last;
    logic response_fire;
    logic row_is_first_in_tile;
    logic row_is_last_in_tile;
    logic last_row;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);
    assign o_state_debug = state;
    assign o_engine_state_debug = engine.state;

    assign current_token =
        token_base_reg + row_offset[TOKEN_ID_WIDTH-1 : 0];
    assign o_current_tile_index =
        row_offset / TILE_ROWS;
    assign row_is_first_in_tile =
        ((row_offset % TILE_ROWS) == 0);
    assign row_is_last_in_tile =
        ((row_offset % TILE_ROWS) == (TILE_ROWS - 1));
    assign last_row =
        ((row_offset + 1'b1) >= total_row_count);

    assign o_mem_req_valid =
        (state == S_WEIGHT_REQ) || (state == S_SCALE_REQ);
    assign o_mem_req_addr =
        (state == S_SCALE_REQ) ?
        (i_scale_base_addr +
         ({{(ADDR_WIDTH-TOKEN_ID_WIDTH){1'b0}}, current_token} *
          SCALE_ROW_BYTES)) :
        (i_weight_base_addr +
         ({{(ADDR_WIDTH-TOKEN_ID_WIDTH){1'b0}}, current_token} *
          WEIGHT_ROW_BYTES));
    assign o_mem_req_len_bytes =
        (state == S_SCALE_REQ) ?
        SCALE_ROW_BYTES_U16 :
        WEIGHT_ROW_BYTES_U16;

    assign o_mem_rsp_ready =
        (state == S_WEIGHT_READ) ?
        engine_weight_wr_ready :
        (state == S_SCALE_READ) ?
        engine_scale_wr_ready :
        1'b0;
    assign response_fire =
        i_mem_rsp_valid && o_mem_rsp_ready;
    assign expected_response_last =
        (state == S_WEIGHT_READ) ?
        (read_word_index == (WEIGHT_WORDS - 1)) :
        (read_word_index == (SCALE_WORDS - 1));

    assign engine_weight_wr_valid =
        (state == S_WEIGHT_READ) && i_mem_rsp_valid;
    assign engine_weight_wr_addr =
        read_word_index[WEIGHT_ADDR_WIDTH-1 : 0];
    assign engine_scale_wr_valid =
        (state == S_SCALE_READ) && i_mem_rsp_valid;
    assign engine_scale_wr_addr =
        read_word_index[SCALE_ADDR_WIDTH-1 : 0];
    assign engine_start = (state == S_ROW_START);

    assign o_row_result_valid = (state == S_UPDATE);
    assign o_row_result_token = current_token;
    assign o_row_result_score = engine_row_sum_q26;
    assign o_tile_complete_pulse =
        (state == S_UPDATE) && row_is_last_in_tile;

    q4_gemv_row_bram #(
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH),
        .ACT_ADDR_WIDTH(ACT_ADDR_WIDTH),
        .WEIGHT_WORDS(WEIGHT_WORDS),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .SCALE_WORDS(SCALE_WORDS),
        .SCALE_ADDR_WIDTH(SCALE_ADDR_WIDTH)
    ) engine (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_act_wr_valid(i_act_wr_valid),
        .o_act_wr_ready(o_act_wr_ready),
        .i_act_wr_addr(i_act_wr_addr),
        .i_act_wr_data(i_act_wr_data),
        .i_weight_wr_valid(engine_weight_wr_valid),
        .o_weight_wr_ready(engine_weight_wr_ready),
        .i_weight_wr_addr(engine_weight_wr_addr),
        .i_weight_wr_data(i_mem_rsp_data),
        .i_scale_wr_valid(engine_scale_wr_valid),
        .o_scale_wr_ready(engine_scale_wr_ready),
        .i_scale_wr_addr(engine_scale_wr_addr),
        .i_scale_wr_data(i_mem_rsp_data),
        .i_start(engine_start),
        .o_busy(engine_busy),
        .o_done(engine_done),
        .o_error(engine_error),
        .o_row_sum_q26(engine_row_sum_q26)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
            token_base_reg <= '0;
            tile_count_reg <= '0;
            total_row_count <= 32'd0;
            row_offset <= 32'd0;
            read_word_index <= 16'd0;
            best_valid <= 1'b0;
            o_error <= 1'b0;
            o_best_token_id <= '0;
            o_best_score_q26 <= '0;
            o_tiles_started <= 32'd0;
            o_tiles_completed <= 32'd0;
            o_compute_cycle_count <= 32'd0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count <= 32'd0;
        end
        else begin
            if ((state == S_ROW_START) || (state == S_ROW_WAIT)) begin
                o_compute_cycle_count <=
                    o_compute_cycle_count + 1'b1;
            end
            if (response_fire) begin
                o_mem_read_word_count <=
                    o_mem_read_word_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        token_base_reg <= i_token_base;
                        tile_count_reg <= i_tile_count;
                        total_row_count <= i_tile_count * TILE_ROWS;
                        row_offset <= 32'd0;
                        read_word_index <= 16'd0;
                        best_valid <= 1'b0;
                        o_error <=
                            (i_tile_count == 0) ||
                            (i_tile_count > MAX_TILES) ||
                            engine_error;
                        o_best_token_id <= '0;
                        o_best_score_q26 <= '0;
                        o_tiles_started <= 32'd0;
                        o_tiles_completed <= 32'd0;
                        o_compute_cycle_count <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        if ((i_tile_count == 0) ||
                            (i_tile_count > MAX_TILES) ||
                            engine_error) begin
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_WEIGHT_REQ;
                        end
                    end
                end

                S_WEIGHT_REQ: begin
                    if (i_mem_req_ready) begin
                        read_word_index <= 16'd0;
                        o_mem_read_burst_count <=
                            o_mem_read_burst_count + 1'b1;
                        if (row_is_first_in_tile) begin
                            o_tiles_started <=
                                o_tiles_started + 1'b1;
                        end
                        state <= S_WEIGHT_READ;
                    end
                end

                S_WEIGHT_READ: begin
                    if (response_fire) begin
                        if (i_mem_rsp_last !=
                            expected_response_last) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else if (expected_response_last) begin
                            read_word_index <= 16'd0;
                            state <= S_SCALE_REQ;
                        end
                        else begin
                            read_word_index <=
                                read_word_index + 1'b1;
                        end
                    end
                end

                S_SCALE_REQ: begin
                    if (i_mem_req_ready) begin
                        read_word_index <= 16'd0;
                        o_mem_read_burst_count <=
                            o_mem_read_burst_count + 1'b1;
                        state <= S_SCALE_READ;
                    end
                end

                S_SCALE_READ: begin
                    if (response_fire) begin
                        if (i_mem_rsp_last !=
                            expected_response_last) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else if (expected_response_last) begin
                            read_word_index <= 16'd0;
                            state <= S_ROW_START;
                        end
                        else begin
                            read_word_index <=
                                read_word_index + 1'b1;
                        end
                    end
                end

                S_ROW_START: begin
                    state <= S_ROW_WAIT;
                end

                S_ROW_WAIT: begin
                    if (engine_done) begin
                        if (engine_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_UPDATE;
                        end
                    end
                end

                S_UPDATE: begin
                    if (!best_valid ||
                        (engine_row_sum_q26 > o_best_score_q26)) begin
                        best_valid <= 1'b1;
                        o_best_token_id <= current_token;
                        o_best_score_q26 <= engine_row_sum_q26;
                    end

                    if (row_is_last_in_tile) begin
                        o_tiles_completed <=
                            o_tiles_completed + 1'b1;
                    end

                    if (last_row) begin
                        state <= S_DONE;
                    end
                    else begin
                        row_offset <= row_offset + 1'b1;
                        read_word_index <= 16'd0;
                        state <= S_WEIGHT_REQ;
                    end
                end

                S_DONE: begin
                    if (!best_valid && !o_error) begin
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
