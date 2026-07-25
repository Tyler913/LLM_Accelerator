`default_nettype none

// Reads one contiguous LM-head Q4 tile from a 32-bit project-local memory
// interface.
//
// The reader splits the large TILE_ROWS-row packed Q4 weight tile into
// MAX_READ_BYTES bursts so it can later sit behind axi4_read_master, whose
// first-version burst limit is 256 32-bit beats. The scale tile follows as one
// smaller burst with the default Qwen3-0.6B shape.
module lm_head_tile_mem_reader #(
    parameter int ADDR_WIDTH     = 64,
    parameter int TILE_ROWS      = 16,
    parameter int INPUT_SIZE     = 1024,
    parameter int GROUP_SIZE     = 64,
    parameter int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE,
    parameter int WEIGHT_WIDTH   = 4,
    parameter int SCALE_WIDTH    = 16,
    parameter int MEM_DATA_WIDTH = 32,
    parameter int MAX_READ_BYTES = 1024
)
(
    input  wire logic                                             i_clk,
    input  wire logic                                             i_rst_n,

    input  wire logic                                             i_start,
    input  wire logic [31 : 0]                                    i_tile_token_base,
    input  wire logic [ADDR_WIDTH-1 : 0]                          i_weight_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]                          i_scale_base_addr,

    output logic                                                  o_busy,
    output logic                                                  o_done,
    output logic                                                  o_error,

    output logic                                                  o_mem_req_valid,
    input  wire logic                                             i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                               o_mem_req_addr,
    output logic [15 : 0]                                         o_mem_req_len_bytes,

    input  wire logic                                             i_mem_rsp_valid,
    output logic                                                  o_mem_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]                      i_mem_rsp_data,
    input  wire logic                                             i_mem_rsp_last,

    output logic [TILE_ROWS*INPUT_SIZE*WEIGHT_WIDTH-1 : 0]        o_tile_weight_flat,
    output logic [TILE_ROWS*GROUP_COUNT*SCALE_WIDTH-1 : 0]        o_tile_scale_flat,
    output logic [31 : 0]                                         o_read_burst_count,
    output logic [31 : 0]                                         o_read_word_count
);

    localparam int MEM_DATA_BYTES    = MEM_DATA_WIDTH / 8;
    localparam int WEIGHT_ROW_BYTES  = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES   = (GROUP_COUNT * SCALE_WIDTH) / 8;
    localparam int TILE_WEIGHT_BYTES = TILE_ROWS * WEIGHT_ROW_BYTES;
    localparam int TILE_SCALE_BYTES  = TILE_ROWS * SCALE_ROW_BYTES;
    localparam int MAX_READ_WORDS    = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int WEIGHT_WORDS      = TILE_WEIGHT_BYTES / MEM_DATA_BYTES;
    localparam int SCALE_WORDS       = TILE_SCALE_BYTES / MEM_DATA_BYTES;
    localparam logic [15 : 0] TILE_WEIGHT_BYTES_U16 = TILE_WEIGHT_BYTES;
    localparam logic [15 : 0] TILE_SCALE_BYTES_U16  = TILE_SCALE_BYTES;
    localparam logic [15 : 0] MAX_READ_BYTES_U16    = MAX_READ_BYTES;
    localparam logic [15 : 0] WEIGHT_WORDS_U16      = WEIGHT_WORDS;
    localparam logic [15 : 0] SCALE_WORDS_U16       = SCALE_WORDS;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_REQ,
        S_READ,
        S_DONE
    } state_t;

    typedef enum logic {
        TARGET_WEIGHT,
        TARGET_SCALE
    } target_t;

    state_t state;
    target_t target;

    logic [31 : 0] tile_token_base_reg;
    logic [15 : 0] chunk_byte_offset;
    logic [8 : 0] word_index;
    logic [15 : 0] current_len_bytes;
    logic [15 : 0] current_total_bytes;
    logic [15 : 0] remaining_bytes;
    logic [8 : 0] current_words;
    logic [15 : 0] write_word_index;
    logic last_expected;
    logic chunk_finishes_target;
    logic target_is_weight;

    assign target_is_weight = (target == TARGET_WEIGHT);

    always @* begin
        current_total_bytes = TILE_WEIGHT_BYTES_U16;
        if (target == TARGET_SCALE) begin
            current_total_bytes = TILE_SCALE_BYTES_U16;
        end

        remaining_bytes = current_total_bytes - chunk_byte_offset;
        if (remaining_bytes > MAX_READ_BYTES_U16) begin
            current_len_bytes = MAX_READ_BYTES_U16;
        end
        else begin
            current_len_bytes = remaining_bytes;
        end
    end

    assign current_words = (current_len_bytes + MEM_DATA_BYTES - 1) >> $clog2(MEM_DATA_BYTES);
    assign last_expected = (word_index == (current_words - 1'b1));
    assign write_word_index = (chunk_byte_offset >> $clog2(MEM_DATA_BYTES)) + word_index;
    assign chunk_finishes_target = (chunk_byte_offset + current_len_bytes) >= current_total_bytes;

    assign o_busy              = (state != S_IDLE);
    assign o_mem_req_valid     = (state == S_REQ);
    assign o_mem_rsp_ready     = (state == S_READ);
    assign o_mem_req_len_bytes = current_len_bytes;

    always @* begin
        o_mem_req_addr =
            i_weight_base_addr +
            ({{(ADDR_WIDTH-32){1'b0}}, tile_token_base_reg} * WEIGHT_ROW_BYTES) +
            chunk_byte_offset;

        if (target == TARGET_SCALE) begin
            o_mem_req_addr =
                i_scale_base_addr +
                ({{(ADDR_WIDTH-32){1'b0}}, tile_token_base_reg} * SCALE_ROW_BYTES) +
                chunk_byte_offset;
        end
    end

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                <= S_IDLE;
            target               <= TARGET_WEIGHT;
            tile_token_base_reg  <= 32'd0;
            chunk_byte_offset    <= 16'd0;
            word_index           <= 9'd0;
            o_done               <= 1'b0;
            o_error              <= 1'b0;
            o_tile_weight_flat   <= 'd0;
            o_tile_scale_flat    <= 'd0;
            o_read_burst_count   <= 32'd0;
            o_read_word_count    <= 32'd0;
        end
        else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        target              <= TARGET_WEIGHT;
                        tile_token_base_reg <= i_tile_token_base;
                        chunk_byte_offset   <= 16'd0;
                        word_index          <= 9'd0;
                        o_error             <= 1'b0;
                        o_tile_weight_flat  <= 'd0;
                        o_tile_scale_flat   <= 'd0;
                        o_read_burst_count  <= 32'd0;
                        o_read_word_count   <= 32'd0;
                        state               <= S_REQ;
                    end
                end

                S_REQ: begin
                    if (i_mem_req_ready) begin
                        o_read_burst_count <= o_read_burst_count + 1'b1;
                        word_index         <= 9'd0;
                        state              <= S_READ;
                    end
                end

                S_READ: begin
                    if (i_mem_rsp_valid) begin
                        o_read_word_count <= o_read_word_count + 1'b1;
                        if (i_mem_rsp_last != last_expected) begin
                            o_error <= 1'b1;
                        end

                        if (target_is_weight) begin
                            if (write_word_index < WEIGHT_WORDS_U16) begin
                                o_tile_weight_flat[write_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <= i_mem_rsp_data;
                            end
                            else begin
                                o_error <= 1'b1;
                            end
                        end
                        else begin
                            if (write_word_index < SCALE_WORDS_U16) begin
                                o_tile_scale_flat[write_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <= i_mem_rsp_data;
                            end
                            else begin
                                o_error <= 1'b1;
                            end
                        end

                        if (i_mem_rsp_last || last_expected) begin
                            if (i_mem_rsp_last != last_expected) begin
                                state <= S_DONE;
                            end
                            else if (!chunk_finishes_target) begin
                                chunk_byte_offset <= chunk_byte_offset + current_len_bytes;
                                word_index        <= 9'd0;
                                state             <= S_REQ;
                            end
                            else if (target_is_weight) begin
                                target            <= TARGET_SCALE;
                                chunk_byte_offset <= 16'd0;
                                word_index        <= 9'd0;
                                state             <= S_REQ;
                            end
                            else begin
                                state <= S_DONE;
                            end
                        end
                        else begin
                            word_index <= word_index + 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
