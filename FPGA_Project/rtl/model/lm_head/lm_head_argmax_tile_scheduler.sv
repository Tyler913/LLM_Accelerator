`default_nettype none

// Runtime tile scheduler for the memory-backed Q4 LM-head argmax path.
//
// The scheduler scans a descriptor-provided number of TILE_ROWS-row LM-head
// tiles. ROW_PARALLEL controls how many rows are fetched and computed at once:
// a board build can keep the external TILE_ROWS-row QMAP contract while
// issuing several smaller memory-backed sub-runs per outer tile. This avoids a
// very wide dynamic row selector when ROW_PARALLEL is smaller than TILE_ROWS.
// The block keeps the global best token/score across all sub-runs and reports
// outer-tile counters to preserve the software-visible contract.
module lm_head_argmax_tile_scheduler #(
    parameter int ADDR_WIDTH       = 64,
    parameter int MAX_TILES        = 9496,
    parameter int TILE_COUNT_WIDTH = (MAX_TILES <= 1) ? 1 : $clog2(MAX_TILES + 1),
    parameter int TILE_ROWS        = 16,
    parameter int ROW_PARALLEL     = 1,
    parameter int INPUT_SIZE       = 1024,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL   = 1,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TOKEN_ID_WIDTH   = 32,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_start,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]       i_token_base,
    input  wire logic [TILE_COUNT_WIDTH-1 : 0]     i_tile_count,
    input  wire logic [INPUT_SIZE*ACT_WIDTH-1 : 0] i_activation_flat,
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
    output logic [31 : 0]                          o_mem_read_word_count
);

    localparam int ROW_PARALLEL_LOCAL =
        (ROW_PARALLEL < 1) ? 1 :
        ((ROW_PARALLEL > TILE_ROWS) ? TILE_ROWS : ROW_PARALLEL);
    localparam int ROW_BATCH_COUNT =
        (TILE_ROWS + ROW_PARALLEL_LOCAL - 1) / ROW_PARALLEL_LOCAL;
    localparam int ROW_BATCH_INDEX_WIDTH =
        (ROW_BATCH_COUNT <= 1) ? 1 : $clog2(ROW_BATCH_COUNT);
    localparam int SINGLE_TILE_SCAN_ROWS = ROW_PARALLEL_LOCAL;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_START_TILE,
        S_WAIT_TILE,
        S_DONE
    } state_t;

    state_t state;
    state_t next_state;

    logic [TOKEN_ID_WIDTH-1 : 0] token_base_reg;
    logic [TILE_COUNT_WIDTH-1 : 0] tile_count_reg;
    logic [TILE_COUNT_WIDTH-1 : 0] current_tile_index;
    logic [ROW_BATCH_INDEX_WIDTH-1 : 0] row_batch_index;
    logic best_valid_reg;
    logic error_reg;

    logic tile_stage_start;
    logic tile_stage_busy;
    logic tile_stage_done;
    logic tile_stage_error;
    logic [TOKEN_ID_WIDTH-1 : 0] tile_stage_token_base;
    logic [TOKEN_ID_WIDTH-1 : 0] tile_stage_best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] tile_stage_best_score_q26;
    logic [31 : 0] tile_stage_tiles_requested;
    logic [31 : 0] tile_stage_tiles_completed;
    logic [31 : 0] tile_stage_compute_cycle_count;
    logic [31 : 0] tile_stage_mem_read_burst_count;
    logic [31 : 0] tile_stage_mem_read_word_count;
    logic last_row_batch;
    logic last_scheduler_tile;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);
    assign o_error = error_reg;
    assign o_current_tile_index = current_tile_index;

    assign tile_stage_start = (state == S_START_TILE);
    assign tile_stage_token_base =
        token_base_reg +
        ({{(TOKEN_ID_WIDTH-TILE_COUNT_WIDTH){1'b0}}, current_tile_index} * TILE_ROWS) +
        ({{(TOKEN_ID_WIDTH-ROW_BATCH_INDEX_WIDTH){1'b0}}, row_batch_index} *
         ROW_PARALLEL_LOCAL);
    assign last_row_batch = (row_batch_index == (ROW_BATCH_COUNT - 1));
    assign last_scheduler_tile =
        last_row_batch && ((current_tile_index + 1'b1) >= tile_count_reg);

    always_comb begin
        next_state = state;

        case (state)
            S_IDLE: begin
                if (i_start == 1'b1) begin
                    if ((i_tile_count == 'd0) || (i_tile_count > MAX_TILES)) begin
                        next_state = S_DONE;
                    end
                    else begin
                        next_state = S_START_TILE;
                    end
                end
            end

            S_START_TILE: begin
                next_state = S_WAIT_TILE;
            end

            S_WAIT_TILE: begin
                if (tile_stage_done == 1'b1) begin
                    if (last_scheduler_tile) begin
                        next_state = S_DONE;
                    end
                    else begin
                        next_state = S_START_TILE;
                    end
                end
            end

            S_DONE: begin
                next_state = S_IDLE;
            end

            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    lm_head_argmax_mem_stage #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .SCAN_ROWS      (SINGLE_TILE_SCAN_ROWS),
        .TILE_ROWS      (ROW_PARALLEL_LOCAL),
        .ROW_PARALLEL   (ROW_PARALLEL_LOCAL),
        .INPUT_SIZE     (INPUT_SIZE),
        .GROUP_SIZE     (GROUP_SIZE),
        .GROUP_COUNT    (GROUP_COUNT),
        .GROUP_PARALLEL (GROUP_PARALLEL),
        .ACT_WIDTH      (ACT_WIDTH),
        .ACT_FRAC       (ACT_FRAC),
        .WEIGHT_WIDTH   (WEIGHT_WIDTH),
        .SCALE_WIDTH    (SCALE_WIDTH),
        .SCALE_FRAC     (SCALE_FRAC),
        .PARTIAL_WIDTH  (PARTIAL_WIDTH),
        .SCALED_WIDTH   (SCALED_WIDTH),
        .ROW_ACC_WIDTH  (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH (TOKEN_ID_WIDTH),
        .TILE_COUNT     (1),
        .TILE_INDEX_W   (1),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH),
        .MAX_READ_BYTES (MAX_READ_BYTES)
    ) tile_stage (
        .i_clk                  (i_clk),
        .i_rst_n                (i_rst_n),
        .i_start                (tile_stage_start),
        .i_token_base           (tile_stage_token_base),
        .i_activation_flat      (i_activation_flat),
        .i_weight_base_addr     (i_weight_base_addr),
        .i_scale_base_addr      (i_scale_base_addr),
        .o_mem_req_valid        (o_mem_req_valid),
        .i_mem_req_ready        (i_mem_req_ready),
        .o_mem_req_addr         (o_mem_req_addr),
        .o_mem_req_len_bytes    (o_mem_req_len_bytes),
        .i_mem_rsp_valid        (i_mem_rsp_valid),
        .o_mem_rsp_ready        (o_mem_rsp_ready),
        .i_mem_rsp_data         (i_mem_rsp_data),
        .i_mem_rsp_last         (i_mem_rsp_last),
        .o_busy                 (tile_stage_busy),
        .o_done                 (tile_stage_done),
        .o_error                (tile_stage_error),
        .o_best_token_id        (tile_stage_best_token_id),
        .o_best_score_q26       (tile_stage_best_score_q26),
        .o_tiles_requested      (tile_stage_tiles_requested),
        .o_tiles_completed      (tile_stage_tiles_completed),
        .o_compute_cycle_count  (tile_stage_compute_cycle_count),
        .o_mem_read_burst_count (tile_stage_mem_read_burst_count),
        .o_mem_read_word_count  (tile_stage_mem_read_word_count)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
        end
        else begin
            state <= next_state;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            token_base_reg          <= 'd0;
            tile_count_reg          <= 'd0;
            current_tile_index      <= 'd0;
            row_batch_index         <= 'd0;
            best_valid_reg          <= 1'b0;
            error_reg               <= 1'b0;
            o_best_token_id         <= 'd0;
            o_best_score_q26        <= 'd0;
            o_tiles_started         <= 32'd0;
            o_tiles_completed       <= 32'd0;
            o_compute_cycle_count   <= 32'd0;
            o_mem_read_burst_count  <= 32'd0;
            o_mem_read_word_count   <= 32'd0;
        end
        else begin
            if (o_busy == 1'b1) begin
                o_compute_cycle_count <= o_compute_cycle_count + 1'b1;
            end

            if (o_mem_req_valid && i_mem_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end

            if (i_mem_rsp_valid && o_mem_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start == 1'b1) begin
                        token_base_reg         <= i_token_base;
                        tile_count_reg         <= i_tile_count;
                        current_tile_index     <= 'd0;
                        row_batch_index        <= 'd0;
                        best_valid_reg         <= 1'b0;
                        error_reg              <= (i_tile_count == 'd0) || (i_tile_count > MAX_TILES);
                        o_best_token_id        <= 'd0;
                        o_best_score_q26       <= 'd0;
                        o_tiles_started        <= 32'd0;
                        o_tiles_completed      <= 32'd0;
                        o_compute_cycle_count  <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count  <= 32'd0;
                    end
                end

                S_START_TILE: begin
                    if (row_batch_index == 'd0) begin
                        o_tiles_started <= o_tiles_started + 1'b1;
                    end
                end

                S_WAIT_TILE: begin
                    if (tile_stage_done == 1'b1) begin
                        if (last_row_batch) begin
                            o_tiles_completed <= o_tiles_completed + 1'b1;
                        end
                        error_reg <=
                            error_reg ||
                            tile_stage_error ||
                            (tile_stage_tiles_requested != 32'd1) ||
                            (tile_stage_tiles_completed != 32'd1) ||
                            (last_scheduler_tile &&
                             ((o_tiles_started != tile_count_reg) ||
                              ((o_tiles_completed + 1'b1) != tile_count_reg) ||
                              ((best_valid_reg == 1'b0) && (tile_stage_error == 1'b1))));

                        if ((tile_stage_error == 1'b0) &&
                            ((best_valid_reg == 1'b0) ||
                             (tile_stage_best_score_q26 > o_best_score_q26))) begin
                            best_valid_reg <= 1'b1;
                            o_best_score_q26 <= tile_stage_best_score_q26;
                            o_best_token_id <= tile_stage_best_token_id;
                        end

                        if (last_row_batch) begin
                            row_batch_index <= 'd0;
                            if ((current_tile_index + 1'b1) < tile_count_reg) begin
                                current_tile_index <= current_tile_index + 1'b1;
                            end
                        end
                        else begin
                            row_batch_index <= row_batch_index + 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    error_reg <=
                        error_reg ||
                        (best_valid_reg == 1'b0) ||
                        (o_tiles_started != tile_count_reg) ||
                        (o_tiles_completed != tile_count_reg);
                    current_tile_index <= 'd0;
                    row_batch_index <= 'd0;
                end

                default: begin
                    error_reg <= 1'b1;
                    current_tile_index <= 'd0;
                    row_batch_index <= 'd0;
                end
            endcase
        end
    end

    if ((TILE_ROWS % ROW_PARALLEL_LOCAL) != 0) begin : gen_invalid_row_parallel
        initial begin
            $error(
                "lm_head_argmax_tile_scheduler requires TILE_ROWS (%0d) to be divisible by ROW_PARALLEL (%0d)",
                TILE_ROWS,
                ROW_PARALLEL_LOCAL
            );
        end
    end

endmodule

`default_nettype wire
