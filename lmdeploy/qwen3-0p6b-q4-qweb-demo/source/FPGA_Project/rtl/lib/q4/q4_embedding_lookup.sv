`default_nettype none

// Reads one tied Q4 embedding row, dequantizes it to signed Q14.10, and writes
// the 1024-word hidden vector through the project-local memory interface.
module q4_embedding_lookup #(
    parameter int ADDR_WIDTH     = 64,
    parameter int MEM_DATA_WIDTH = 32,
    parameter int VOCAB_SIZE     = 151936,
    parameter int INPUT_SIZE     = 1024,
    parameter int GROUP_SIZE     = 64,
    parameter int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE,
    parameter int WEIGHT_WIDTH   = 4,
    parameter int SCALE_WIDTH    = 16,
    parameter int SCALE_FRAC     = 14,
    parameter int OUTPUT_FRAC    = 10
)(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,
    input  wire logic                         i_start,
    input  wire logic [31 : 0]                i_token_id,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_weight_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_scale_base_addr,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,

    output logic                              o_mem_rd_req_valid,
    input  wire logic                         i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_rd_req_addr,
    output logic [15 : 0]                     o_mem_rd_req_len_bytes,
    input  wire logic                         i_mem_rd_rsp_valid,
    output logic                              o_mem_rd_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]  i_mem_rd_rsp_data,
    input  wire logic                         i_mem_rd_rsp_last,

    output logic                              o_mem_wr_req_valid,
    input  wire logic                         i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_wr_req_addr,
    output logic [15 : 0]                     o_mem_wr_req_len_bytes,
    output logic [MEM_DATA_WIDTH-1 : 0]       o_mem_wr_data,
    output logic                              o_mem_wr_data_valid,
    input  wire logic                         i_mem_wr_data_ready,
    output logic                              o_mem_wr_data_last,
    input  wire logic                         i_mem_wr_done,
    input  wire logic                         i_mem_wr_error,

    output logic [31 : 0]                     o_read_burst_count,
    output logic [31 : 0]                     o_read_word_count,
    output logic [31 : 0]                     o_write_req_count,
    output logic [31 : 0]                     o_write_word_count
);

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int ADDR_LSB = $clog2(MEM_DATA_BYTES);
    localparam int WEIGHT_ROW_BYTES = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES = (GROUP_COUNT * SCALE_WIDTH) / 8;
    localparam int OUTPUT_BYTES = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int WEIGHT_WORDS = WEIGHT_ROW_BYTES / MEM_DATA_BYTES;
    localparam int SCALE_WORDS = SCALE_ROW_BYTES / MEM_DATA_BYTES;
    localparam int OUTPUT_INDEX_WIDTH = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);
    localparam int WEIGHT_WORD_INDEX_WIDTH =
        (WEIGHT_WORDS <= 1) ? 1 : $clog2(WEIGHT_WORDS);
    localparam int DEQUANT_SHIFT = SCALE_FRAC - OUTPUT_FRAC;
    localparam int PRODUCT_WIDTH = WEIGHT_WIDTH + SCALE_WIDTH + 1;

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_WEIGHT_REQ,
        S_WEIGHT_READ,
        S_SCALE_REQ,
        S_SCALE_READ,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_DONE
    } state_t;

    state_t state;
    logic [31 : 0] token_id_reg;
    logic [ADDR_WIDTH-1 : 0] weight_base_reg;
    logic [ADDR_WIDTH-1 : 0] scale_base_reg;
    logic [ADDR_WIDTH-1 : 0] output_base_reg;
    logic [15 : 0] read_word_index;
    logic [OUTPUT_INDEX_WIDTH-1 : 0] output_index;
    (* ram_style = "block" *)
    logic [MEM_DATA_WIDTH-1 : 0] weight_word_mem [0:WEIGHT_WORDS-1];
    logic [WEIGHT_WORD_INDEX_WIDTH-1 : 0] weight_word_rd_index;
    logic [MEM_DATA_WIDTH-1 : 0] weight_word_rd_data;
    logic [MEM_DATA_WIDTH-1 : 0] current_weight_word;
    logic [GROUP_COUNT*SCALE_WIDTH-1 : 0] scale_flat;
    logic read_last_expected;
    logic signed [WEIGHT_WIDTH-1 : 0] current_weight;
    logic [SCALE_WIDTH-1 : 0] current_scale;
    logic signed [SCALE_WIDTH : 0] current_scale_signed;
    logic signed [PRODUCT_WIDTH-1 : 0] current_product_q14;
    logic signed [PRODUCT_WIDTH-1 : 0] current_output_q10;
    logic request_aligned;

    assign request_aligned =
        (i_weight_base_addr[ADDR_LSB-1 : 0] == 'd0) &&
        (i_scale_base_addr[ADDR_LSB-1 : 0] == 'd0) &&
        (i_output_base_addr[ADDR_LSB-1 : 0] == 'd0);

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_mem_rd_req_valid = (state == S_WEIGHT_REQ) || (state == S_SCALE_REQ);
    assign o_mem_rd_rsp_ready = (state == S_WEIGHT_READ) || (state == S_SCALE_READ);
    assign o_mem_wr_req_valid = (state == S_WRITE_REQ);
    assign o_mem_wr_data_valid = (state == S_WRITE_DATA);
    assign o_mem_wr_data_last = (output_index == (INPUT_SIZE - 1));
    assign o_mem_wr_req_addr = output_base_reg;
    assign o_mem_wr_req_len_bytes = OUTPUT_BYTES;

    always_comb begin
        o_mem_rd_req_addr =
            weight_base_reg +
            ({{(ADDR_WIDTH-32){1'b0}}, token_id_reg} * WEIGHT_ROW_BYTES);
        o_mem_rd_req_len_bytes = WEIGHT_ROW_BYTES;
        read_last_expected = (read_word_index == (WEIGHT_WORDS - 1));
        if ((state == S_SCALE_REQ) || (state == S_SCALE_READ)) begin
            o_mem_rd_req_addr =
                scale_base_reg +
                ({{(ADDR_WIDTH-32){1'b0}}, token_id_reg} * SCALE_ROW_BYTES);
            o_mem_rd_req_len_bytes = SCALE_ROW_BYTES;
            read_last_expected = (read_word_index == (SCALE_WORDS - 1));
        end
    end

    always_comb begin
        current_weight =
            current_weight_word[output_index[2 : 0]*WEIGHT_WIDTH +: WEIGHT_WIDTH];
        current_scale = scale_flat[(output_index/GROUP_SIZE)*SCALE_WIDTH +: SCALE_WIDTH];
        current_scale_signed = $signed({1'b0, current_scale});
        current_product_q14 = current_weight * current_scale_signed;
        current_output_q10 = current_product_q14 >>> DEQUANT_SHIFT;
        o_mem_wr_data = {{(MEM_DATA_WIDTH-PRODUCT_WIDTH){current_output_q10[PRODUCT_WIDTH-1]}},
                         current_output_q10};
    end

    // The 512-byte tied-Q4 row is buffered as 128 words in block RAM.  The
    // next word is prefetched while the current word emits its eight nibbles,
    // avoiding the 4096-bit variable part-select mux of the legacy flat vector.
    always_comb begin
        weight_word_rd_index = '0;
        if ((state == S_WRITE_REQ) ||
            (state == S_WRITE_DATA) ||
            (state == S_WRITE_WAIT)) begin
            if ((output_index >> 3) < (WEIGHT_WORDS - 1)) begin
                weight_word_rd_index =
                    (output_index >> 3) + 1'b1;
            end
            else begin
                weight_word_rd_index = WEIGHT_WORDS - 1;
            end
        end
    end

    always_ff @(posedge i_clk) begin
        if ((state == S_WEIGHT_READ) && i_mem_rd_rsp_valid) begin
            weight_word_mem[read_word_index[WEIGHT_WORD_INDEX_WIDTH-1:0]] <=
                i_mem_rd_rsp_data;
        end
        else begin
            weight_word_rd_data <= weight_word_mem[weight_word_rd_index];
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
            token_id_reg <= 32'd0;
            weight_base_reg <= 'd0;
            scale_base_reg <= 'd0;
            output_base_reg <= 'd0;
            read_word_index <= 16'd0;
            output_index <= 'd0;
            current_weight_word <= '0;
            scale_flat <= 'd0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_read_burst_count <= 32'd0;
            o_read_word_count <= 32'd0;
            o_write_req_count <= 32'd0;
            o_write_word_count <= 32'd0;
        end else begin
            o_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        token_id_reg <= i_token_id;
                        weight_base_reg <= i_weight_base_addr;
                        scale_base_reg <= i_scale_base_addr;
                        output_base_reg <= i_output_base_addr;
                        read_word_index <= 16'd0;
                        output_index <= 'd0;
                        current_weight_word <= '0;
                        scale_flat <= 'd0;
                        o_error <= 1'b0;
                        o_read_burst_count <= 32'd0;
                        o_read_word_count <= 32'd0;
                        o_write_req_count <= 32'd0;
                        o_write_word_count <= 32'd0;
                        if ((i_token_id >= VOCAB_SIZE) || !request_aligned) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            state <= S_WEIGHT_REQ;
                        end
                    end
                end

                S_WEIGHT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        o_read_burst_count <= o_read_burst_count + 1'b1;
                        read_word_index <= 16'd0;
                        state <= S_WEIGHT_READ;
                    end
                end

                S_WEIGHT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        o_read_word_count <= o_read_word_count + 1'b1;
                        if (i_mem_rd_rsp_last != read_last_expected) begin
                            o_error <= 1'b1;
                        end
                        if (i_mem_rd_rsp_last || read_last_expected) begin
                            if (i_mem_rd_rsp_last != read_last_expected) begin
                                state <= S_DONE;
                            end else begin
                                read_word_index <= 16'd0;
                                state <= S_SCALE_REQ;
                            end
                        end else begin
                            read_word_index <= read_word_index + 1'b1;
                        end
                    end
                end

                S_SCALE_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        o_read_burst_count <= o_read_burst_count + 1'b1;
                        read_word_index <= 16'd0;
                        state <= S_SCALE_READ;
                    end
                end

                S_SCALE_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        o_read_word_count <= o_read_word_count + 1'b1;
                        scale_flat[read_word_index*MEM_DATA_WIDTH +: MEM_DATA_WIDTH] <= i_mem_rd_rsp_data;
                        if (i_mem_rd_rsp_last != read_last_expected) begin
                            o_error <= 1'b1;
                        end
                        if (i_mem_rd_rsp_last || read_last_expected) begin
                            if (i_mem_rd_rsp_last != read_last_expected) begin
                                state <= S_DONE;
                            end else begin
                                output_index <= 'd0;
                                current_weight_word <= weight_word_rd_data;
                                state <= S_WRITE_REQ;
                            end
                        end else begin
                            read_word_index <= read_word_index + 1'b1;
                        end
                    end
                end

                S_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        o_write_req_count <= o_write_req_count + 1'b1;
                        output_index <= 'd0;
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
                        o_write_word_count <= o_write_word_count + 1'b1;
                        if (o_mem_wr_data_last) begin
                            state <= S_WRITE_WAIT;
                        end else begin
                            if (output_index[2 : 0] == 3'd7) begin
                                current_weight_word <= weight_word_rd_data;
                            end
                            output_index <= output_index + 1'b1;
                        end
                    end
                end

                S_WRITE_WAIT: begin
                    if (i_mem_wr_error) begin
                        o_error <= 1'b1;
                    end
                    if (i_mem_wr_done) begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    o_error <= 1'b1;
                    state <= S_DONE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
