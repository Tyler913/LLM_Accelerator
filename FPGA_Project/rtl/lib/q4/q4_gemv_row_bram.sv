`timescale 1ns/1ps
`default_nettype none

// BRAM-backed, one-MAC-per-cycle Q4 GEMV engine for one output row.
//
// Unlike q4_gemv_row_1024, this module does not expose INPUT_SIZE-wide packed
// vectors. Callers fill three native memories while the engine is idle:
//
//   activation_mem[element]     : one signed Q12.12 activation
//   weight_mem[word]            : eight signed int4 values per 32-bit word
//   scale_mem[word]             : two unsigned UQ2.14 scales per 32-bit word
//
// Activation and weight storage use synchronous block-RAM read templates and
// are never reset or cleared. Only control, pipeline, and result registers are
// reset. A new activation and packed-weight word are prefetched while the
// current element is consumed, so group scaling overlaps the next group's MACs.
//
// Arithmetic is bit-identical to q4_gemv_row_1024:
//
//   partial[g] = sum(activation[64*g+j] * signed_int4_weight[64*g+j])
//   scaled[g]  = partial[g] * unsigned_uq2_14_scale[g]
//   row_q26    = sum(scaled[g])
//
// There is no rounding, shifting, saturation, or requantization in this block.
module q4_gemv_row_bram #(
    parameter int INPUT_SIZE      = 1024,
    parameter int GROUP_SIZE      = 64,
    parameter int GROUP_COUNT     = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH       = 24,
    parameter int ACT_FRAC        = 12,
    parameter int WEIGHT_WIDTH    = 4,
    parameter int SCALE_WIDTH     = 16,
    parameter int SCALE_FRAC      = 14,
    parameter int PRODUCT_WIDTH   = ACT_WIDTH + WEIGHT_WIDTH,
    parameter int PARTIAL_WIDTH   = PRODUCT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH    = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH   = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int ACT_ADDR_WIDTH  = (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE),
    parameter int WEIGHT_WORDS    = (INPUT_SIZE * WEIGHT_WIDTH) / 32,
    parameter int WEIGHT_ADDR_WIDTH =
        (WEIGHT_WORDS <= 1) ? 1 : $clog2(WEIGHT_WORDS),
    parameter int SCALE_WORDS     = (GROUP_COUNT * SCALE_WIDTH) / 32,
    parameter int SCALE_ADDR_WIDTH =
        (SCALE_WORDS <= 1) ? 1 : $clog2(SCALE_WORDS)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,

    input  wire logic                                  i_act_wr_valid,
    output logic                                       o_act_wr_ready,
    input  wire logic [ACT_ADDR_WIDTH-1 : 0]           i_act_wr_addr,
    input  wire logic signed [ACT_WIDTH-1 : 0]         i_act_wr_data,

    input  wire logic                                  i_weight_wr_valid,
    output logic                                       o_weight_wr_ready,
    input  wire logic [WEIGHT_ADDR_WIDTH-1 : 0]        i_weight_wr_addr,
    input  wire logic [31 : 0]                         i_weight_wr_data,

    input  wire logic                                  i_scale_wr_valid,
    output logic                                       o_scale_wr_ready,
    input  wire logic [SCALE_ADDR_WIDTH-1 : 0]         i_scale_wr_addr,
    input  wire logic [31 : 0]                         i_scale_wr_data,

    input  wire logic                                  i_start,
    output logic                                       o_busy,
    output logic                                       o_done,
    // Sticky protocol/address error. Once asserted, reset is required to clear.
    output logic                                       o_error,
    output logic signed [ROW_ACC_WIDTH-1 : 0]          o_row_sum_q26
);

    localparam int WEIGHTS_PER_WORD = 32 / WEIGHT_WIDTH;
    localparam int SCALES_PER_WORD  = 32 / SCALE_WIDTH;
    localparam int WEIGHT_SELECT_WIDTH =
        (WEIGHTS_PER_WORD <= 1) ? 1 : $clog2(WEIGHTS_PER_WORD);
    localparam int GROUP_INDEX_WIDTH =
        (GROUP_COUNT <= 1) ? 1 : $clog2(GROUP_COUNT);
    localparam int SCALE_PRODUCT_WIDTH =
        PARTIAL_WIDTH + SCALE_WIDTH + 1;

    localparam logic [ACT_ADDR_WIDTH-1 : 0] LAST_ELEMENT_INDEX =
        ACT_ADDR_WIDTH'(INPUT_SIZE - 1);
    localparam logic [GROUP_INDEX_WIDTH-1 : 0] LAST_GROUP_INDEX =
        GROUP_INDEX_WIDTH'(GROUP_COUNT - 1);
    localparam logic [WEIGHT_ADDR_WIDTH-1 : 0] LAST_WEIGHT_WORD_INDEX =
        WEIGHT_ADDR_WIDTH'(WEIGHT_WORDS - 1);
    localparam logic [ACT_ADDR_WIDTH : 0] ACT_COUNT_LIMIT =
        (ACT_ADDR_WIDTH + 1)'(INPUT_SIZE);
    localparam logic [WEIGHT_ADDR_WIDTH : 0] WEIGHT_COUNT_LIMIT =
        (WEIGHT_ADDR_WIDTH + 1)'(WEIGHT_WORDS);
    localparam logic [SCALE_ADDR_WIDTH : 0] SCALE_COUNT_LIMIT =
        (SCALE_ADDR_WIDTH + 1)'(SCALE_WORDS);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_PRIME_READ,
        S_PRIME_CAPTURE,
        S_RUN,
        S_DRAIN_LAST_SCALE,
        S_DONE
    } state_t;

    state_t state;
    state_t next_state;

    logic act_wr_in_range;
    logic weight_wr_in_range;
    logic scale_wr_in_range;

    assign act_wr_in_range =
        ({1'b0, i_act_wr_addr} < ACT_COUNT_LIMIT);
    assign weight_wr_in_range =
        ({1'b0, i_weight_wr_addr} < WEIGHT_COUNT_LIMIT);
    assign scale_wr_in_range =
        ({1'b0, i_scale_wr_addr} < SCALE_COUNT_LIMIT);

    assign o_act_wr_ready = (state == S_IDLE);
    assign o_weight_wr_ready = (state == S_IDLE);
    assign o_scale_wr_ready = (state == S_IDLE);
    assign o_busy =
        (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);

    // Memory arrays intentionally have no reset branch and no whole-array
    // assignment. The two large stores follow simple dual-port BRAM templates.
    (* ram_style = "block" *)
    logic signed [ACT_WIDTH-1 : 0] activation_mem [0 : INPUT_SIZE-1];
    logic signed [ACT_WIDTH-1 : 0] activation_read_data;
    logic [ACT_ADDR_WIDTH-1 : 0] activation_read_addr;

    (* ram_style = "block" *)
    logic [31 : 0] weight_mem [0 : WEIGHT_WORDS-1];
    logic [31 : 0] weight_read_data;
    logic [WEIGHT_ADDR_WIDTH-1 : 0] weight_read_addr;

    (* ram_style = "distributed" *)
    logic [31 : 0] scale_mem [0 : SCALE_WORDS-1];

    always_ff @(posedge i_clk) begin
        if (i_act_wr_valid && o_act_wr_ready && act_wr_in_range) begin
            activation_mem[i_act_wr_addr] <= i_act_wr_data;
        end
        activation_read_data <= activation_mem[activation_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (i_weight_wr_valid && o_weight_wr_ready && weight_wr_in_range) begin
            weight_mem[i_weight_wr_addr] <= i_weight_wr_data;
        end
        weight_read_data <= weight_mem[weight_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (i_scale_wr_valid && o_scale_wr_ready && scale_wr_in_range) begin
            scale_mem[i_scale_wr_addr] <= i_scale_wr_data;
        end
    end

    logic [ACT_ADDR_WIDTH-1 : 0] element_index;
    logic [GROUP_INDEX_WIDTH-1 : 0] group_index;
    logic [WEIGHT_SELECT_WIDTH-1 : 0] weight_nibble_index;
    logic [31 : 0] current_weight_word;

    logic signed [WEIGHT_WIDTH-1 : 0] current_weight_q4;
    logic signed [PRODUCT_WIDTH-1 : 0] current_product;
    logic signed [PARTIAL_WIDTH-1 : 0] partial_acc;
    logic signed [PARTIAL_WIDTH-1 : 0] partial_with_current;

    logic [31 : 0] current_scale_word;
    logic [SCALE_WIDTH-1 : 0] current_scale_q2_14;
    logic signed [SCALE_WIDTH : 0] current_signed_scale_q2_14;

    logic scale_s0_valid;
    logic scale_s0_last_group;
    logic signed [PARTIAL_WIDTH-1 : 0] scale_s0_partial;
    logic signed [SCALE_WIDTH : 0] scale_s0_scale;

    logic scale_s1_valid;
    logic scale_s1_last_group;
    logic signed [SCALE_PRODUCT_WIDTH-1 : 0] scale_s1_product_full;
    logic signed [SCALED_WIDTH-1 : 0] scale_commit_q26;
    logic signed [ROW_ACC_WIDTH-1 : 0] scale_commit_extended;
    logic scale_commit_last;

    logic current_group_last_element;
    logic current_row_last_element;
    logic prefetch_next_weight_word;
    logic [WEIGHT_ADDR_WIDTH-1 : 0] current_weight_word_index;

    always_comb begin
        weight_nibble_index =
            element_index[WEIGHT_SELECT_WIDTH-1 : 0];
        current_weight_q4 =
            current_weight_word[
                weight_nibble_index*WEIGHT_WIDTH +: WEIGHT_WIDTH
            ];
        current_product = activation_read_data * current_weight_q4;
        partial_with_current = partial_acc + current_product;

        current_scale_word =
            scale_mem[group_index / SCALES_PER_WORD];
        current_scale_q2_14 =
            current_scale_word[
                (group_index % SCALES_PER_WORD)*SCALE_WIDTH +: SCALE_WIDTH
            ];
        current_signed_scale_q2_14 =
            $signed({1'b0, current_scale_q2_14});

        scale_commit_q26 =
            scale_s1_product_full[SCALED_WIDTH-1 : 0];
        scale_commit_extended =
            {{(ROW_ACC_WIDTH-SCALED_WIDTH){scale_commit_q26[SCALED_WIDTH-1]}},
             scale_commit_q26};

        current_group_last_element =
            (weight_nibble_index == WEIGHT_SELECT_WIDTH'(WEIGHTS_PER_WORD - 1)) &&
            (element_index[5 : 0] == 6'd63);
        current_row_last_element =
            (element_index == LAST_ELEMENT_INDEX);
        current_weight_word_index =
            WEIGHT_ADDR_WIDTH'(element_index / WEIGHTS_PER_WORD);
        prefetch_next_weight_word =
            (weight_nibble_index == WEIGHT_SELECT_WIDTH'(WEIGHTS_PER_WORD - 2)) &&
            (current_weight_word_index < LAST_WEIGHT_WORD_INDEX);
        scale_commit_last =
            scale_s1_valid && scale_s1_last_group;
    end

    // PRIME_READ requests element/word zero. PRIME_CAPTURE gives the inferred
    // BRAM outputs one full cycle before RUN consumes element zero.
    always_comb begin
        activation_read_addr = '0;
        weight_read_addr = '0;

        case (state)
            S_PRIME_READ,
            S_PRIME_CAPTURE: begin
                activation_read_addr = '0;
                weight_read_addr = '0;
            end

            S_RUN: begin
                if (!current_row_last_element) begin
                    activation_read_addr = element_index + 1'b1;
                end
                else begin
                    activation_read_addr = element_index;
                end

                if (prefetch_next_weight_word) begin
                    weight_read_addr =
                        WEIGHT_ADDR_WIDTH'(
                            (element_index / WEIGHTS_PER_WORD) + 1'b1
                        );
                end
                else begin
                    weight_read_addr = current_weight_word_index;
                end
            end

            default: begin
                activation_read_addr = '0;
                weight_read_addr = '0;
            end
        endcase
    end

    always_comb begin
        next_state = state;

        case (state)
            S_IDLE: begin
                if (i_start) begin
                    next_state = S_PRIME_READ;
                end
            end

            S_PRIME_READ: begin
                next_state = S_PRIME_CAPTURE;
            end

            S_PRIME_CAPTURE: begin
                next_state = S_RUN;
            end

            S_RUN: begin
                if (current_row_last_element) begin
                    next_state = S_DRAIN_LAST_SCALE;
                end
            end

            S_DRAIN_LAST_SCALE: begin
                if (scale_commit_last) begin
                    next_state = S_DONE;
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

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
        end
        else begin
            state <= next_state;
        end
    end

    // Main MAC and independent scale pipeline. At a group boundary S0 captures
    // partial_acc + current_product, so element 63 is never omitted. The MAC
    // immediately continues with the next group while S1 multiplies and the
    // following cycle commits the prior group's signed 50-bit result.
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            element_index <= '0;
            group_index <= '0;
            current_weight_word <= '0;
            partial_acc <= '0;
            scale_s0_valid <= 1'b0;
            scale_s0_last_group <= 1'b0;
            scale_s0_partial <= '0;
            scale_s0_scale <= '0;
            scale_s1_valid <= 1'b0;
            scale_s1_last_group <= 1'b0;
            scale_s1_product_full <= '0;
            o_row_sum_q26 <= '0;
        end
        else begin
            scale_s0_valid <= 1'b0;
            scale_s1_valid <= scale_s0_valid;
            scale_s1_last_group <= scale_s0_last_group;

            if (scale_s0_valid) begin
                scale_s1_product_full <=
                    scale_s0_partial * scale_s0_scale;
            end

            if (scale_s1_valid) begin
                o_row_sum_q26 <=
                    o_row_sum_q26 + scale_commit_extended;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        element_index <= '0;
                        group_index <= '0;
                        current_weight_word <= '0;
                        partial_acc <= '0;
                        scale_s0_valid <= 1'b0;
                        scale_s0_last_group <= 1'b0;
                        scale_s1_valid <= 1'b0;
                        scale_s1_last_group <= 1'b0;
                        o_row_sum_q26 <= '0;
                    end
                end

                S_PRIME_CAPTURE: begin
                    current_weight_word <= weight_read_data;
                end

                S_RUN: begin
                    if (current_group_last_element) begin
                        scale_s0_valid <= 1'b1;
                        scale_s0_last_group <=
                            (group_index == LAST_GROUP_INDEX);
                        scale_s0_partial <= partial_with_current;
                        scale_s0_scale <= current_signed_scale_q2_14;
                        partial_acc <= '0;

                        if (group_index != LAST_GROUP_INDEX) begin
                            group_index <= group_index + 1'b1;
                        end
                    end
                    else begin
                        partial_acc <= partial_with_current;
                    end

                    if (!current_row_last_element) begin
                        element_index <= element_index + 1'b1;
                    end

                    if ((weight_nibble_index ==
                         WEIGHT_SELECT_WIDTH'(WEIGHTS_PER_WORD - 1)) &&
                        !current_row_last_element) begin
                        current_weight_word <= weight_read_data;
                    end
                end

                default: begin
                end
            endcase
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_error <= 1'b0;
        end
        else begin
            if (i_start && (state != S_IDLE)) begin
                o_error <= 1'b1;
            end
            // Fill ports use normal ready/valid backpressure: valid may remain
            // asserted while ready is low. Address checks apply only when a
            // write is accepted in IDLE.
            if (i_act_wr_valid && o_act_wr_ready &&
                !act_wr_in_range) begin
                o_error <= 1'b1;
            end
            if (i_weight_wr_valid && o_weight_wr_ready &&
                !weight_wr_in_range) begin
                o_error <= 1'b1;
            end
            if (i_scale_wr_valid && o_scale_wr_ready &&
                !scale_wr_in_range) begin
                o_error <= 1'b1;
            end
        end
    end

    initial begin
        if (!((INPUT_SIZE == 1024) ||
              (INPUT_SIZE == 2048) ||
              (INPUT_SIZE == 3072))) begin
            $fatal(1, "q4_gemv_row_bram: unsupported INPUT_SIZE=%0d", INPUT_SIZE);
        end
        if ((INPUT_SIZE % GROUP_SIZE) != 0) begin
            $fatal(1, "q4_gemv_row_bram: INPUT_SIZE must divide into groups");
        end
        if ((GROUP_SIZE != 64) ||
            (ACT_WIDTH != 24) ||
            (WEIGHT_WIDTH != 4) ||
            (SCALE_WIDTH != 16)) begin
            $fatal(1, "q4_gemv_row_bram: unsupported arithmetic format");
        end
        if (((INPUT_SIZE * WEIGHT_WIDTH) % 32) != 0 ||
            ((GROUP_COUNT * SCALE_WIDTH) % 32) != 0) begin
            $fatal(1, "q4_gemv_row_bram: fill ports require whole 32-bit words");
        end
    end

endmodule

`default_nettype wire
