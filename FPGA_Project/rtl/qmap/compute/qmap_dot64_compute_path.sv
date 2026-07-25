`default_nettype none

`include "qmap_defs.svh"

// Synthesizable dot64 smoke/bring-up controller.
//
// This is not a full-model operator. It is the first narrow hardware wrapper
// for the proven QMAP dot64 image:
//   1. read QMAP header/descriptors
//   2. fetch the activation, packed Q4 weight, scale, and expected/debug data
//   3. run q4_dot_product_64
//   4. compare the result with the expected/debug payload
//
// The external memory side intentionally stays on the project-local
// request/response interface. Simulation can back it with a fake memory, and a
// later Vivado-facing layer can back it with axi4_read_master.
module qmap_dot64_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 4,
    parameter int GROUP_SIZE       = 64,
    parameter int ACT_WIDTH        = 16,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int PRODUCT_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH,
    parameter int PARTIAL_WIDTH    = PRODUCT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH
)
(
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,

    input  wire logic                                  i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]               i_qmap_base_addr,

    output logic                                  o_busy,
    output logic                                  o_done,
    output logic                                  o_error,
    output logic                                  o_compare_match,

    output logic                                  o_mem_req_valid,
    input  wire logic                                  i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]               o_mem_req_addr,
    output logic [15 : 0]                         o_mem_req_len_bytes,

    input  wire logic                                  i_mem_rsp_valid,
    output logic                                  o_mem_rsp_ready,
    input  wire logic [31 : 0]                         i_mem_rsp_data,
    input  wire logic                                  i_mem_rsp_last,

    output logic signed [63 : 0]                  o_partial_sum,
    output logic signed [63 : 0]                  o_scaled_sum_q26,
    output logic signed [63 : 0]                  o_expected_partial_sum,
    output logic signed [63 : 0]                  o_expected_scaled_sum_q26
);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_FETCHER_START,
        S_FETCHER_WAIT,
        S_DOT_START,
        S_DOT_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic reader_start;
    logic reader_done;
    logic reader_error;
    logic reader_req_valid;
    logic reader_req_ready;
    logic [ADDR_WIDTH-1 : 0] reader_req_addr;
    logic [15 : 0] reader_req_len_bytes;
    logic reader_rsp_valid;
    logic reader_rsp_ready;
    logic [31 : 0] reader_rsp_data;
    logic reader_rsp_last;

    logic [31 : 0] header_magic;
    logic [31 : 0] header_version;
    logic [31 : 0] header_bytes;
    logic [31 : 0] descriptor_bytes;
    logic [31 : 0] descriptor_count;
    logic [31 : 0] descriptor_capacity;
    logic [63 : 0] descriptor_table_addr;
    logic [63 : 0] payload_base_addr;
    logic [63 : 0] image_base_addr;
    logic [63 : 0] image_bytes;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_role_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_dtype_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_rank_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_flags_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_element_bits_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_group_size_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_scale_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] desc_base_addr_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] desc_nbytes_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_dim0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_dim1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_dim2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_dim3_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] desc_aux3_flat;

    logic fetcher_start;
    logic fetcher_done;
    logic fetcher_error;
    logic fetcher_req_valid;
    logic fetcher_req_ready;
    logic [ADDR_WIDTH-1 : 0] fetcher_req_addr;
    logic [15 : 0] fetcher_req_len_bytes;
    logic fetcher_rsp_valid;
    logic fetcher_rsp_ready;
    logic [31 : 0] fetcher_rsp_data;
    logic fetcher_rsp_last;

    logic [GROUP_SIZE*ACT_WIDTH-1 : 0] activation_flat;
    logic [GROUP_SIZE*WEIGHT_WIDTH-1 : 0] weight_packed;
    logic [SCALE_WIDTH-1 : 0] scale_q2_14;
    logic signed [63 : 0] expected_partial_sum;
    logic signed [63 : 0] expected_scaled_sum_q26;

    logic dot_start;
    logic dot_done;
    logic signed [PARTIAL_WIDTH-1 : 0] dot_partial_sum;
    logic signed [SCALED_WIDTH-1 : 0] dot_scaled_sum_q26;
    logic signed [63 : 0] dot_partial_sum_ext;
    logic signed [63 : 0] dot_scaled_sum_q26_ext;

    logic [63 : 0] activation_base_addr;
    logic [63 : 0] activation_nbytes;
    logic [63 : 0] weight_base_addr;
    logic [63 : 0] weight_nbytes;
    logic [63 : 0] scale_base_addr;
    logic [63 : 0] scale_nbytes;
    logic [63 : 0] expected_base_addr;
    logic [63 : 0] expected_nbytes;

    assign o_busy        = (state != S_IDLE);
    assign reader_start  = (state == S_READER_START);
    assign fetcher_start = (state == S_FETCHER_START);
    assign dot_start     = (state == S_DOT_START);

    assign activation_base_addr = desc_base_addr_flat[0*64 +: 64];
    assign activation_nbytes    = desc_nbytes_flat[0*64 +: 64];
    assign weight_base_addr     = desc_base_addr_flat[1*64 +: 64];
    assign weight_nbytes        = desc_nbytes_flat[1*64 +: 64];
    assign scale_base_addr      = desc_base_addr_flat[2*64 +: 64];
    assign scale_nbytes         = desc_nbytes_flat[2*64 +: 64];
    assign expected_base_addr   = desc_base_addr_flat[3*64 +: 64];
    assign expected_nbytes      = desc_nbytes_flat[3*64 +: 64];

    assign dot_partial_sum_ext =
        {{(64-PARTIAL_WIDTH){dot_partial_sum[PARTIAL_WIDTH-1]}}, dot_partial_sum};
    assign dot_scaled_sum_q26_ext =
        {{(64-SCALED_WIDTH){dot_scaled_sum_q26[SCALED_WIDTH-1]}}, dot_scaled_sum_q26};

    assign o_mem_req_valid = (state == S_READER_WAIT) ? reader_req_valid :
                             (state == S_FETCHER_WAIT) ? fetcher_req_valid :
                             1'b0;
    assign o_mem_req_addr = (state == S_READER_WAIT) ? reader_req_addr :
                            fetcher_req_addr;
    assign o_mem_req_len_bytes = (state == S_READER_WAIT) ? reader_req_len_bytes :
                                 fetcher_req_len_bytes;
    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_req_ready : 1'b0;
    assign fetcher_req_ready = (state == S_FETCHER_WAIT) ? i_mem_req_ready : 1'b0;

    assign o_mem_rsp_ready = (state == S_READER_WAIT) ? reader_rsp_ready :
                             (state == S_FETCHER_WAIT) ? fetcher_rsp_ready :
                             1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rsp_valid : 1'b0;
    assign fetcher_rsp_valid = (state == S_FETCHER_WAIT) ? i_mem_rsp_valid : 1'b0;
    assign reader_rsp_data = i_mem_rsp_data;
    assign fetcher_rsp_data = i_mem_rsp_data;
    assign reader_rsp_last = i_mem_rsp_last;
    assign fetcher_rsp_last = i_mem_rsp_last;

    qmap_dot64_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS)
    ) reader (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(reader_start),
        .i_qmap_base_addr(i_qmap_base_addr),
        .o_busy(),
        .o_done(reader_done),
        .o_error(reader_error),
        .o_mem_req_valid(reader_req_valid),
        .i_mem_req_ready(reader_req_ready),
        .o_mem_req_addr(reader_req_addr),
        .o_mem_req_len_bytes(reader_req_len_bytes),
        .i_mem_rsp_valid(reader_rsp_valid),
        .o_mem_rsp_ready(reader_rsp_ready),
        .i_mem_rsp_data(reader_rsp_data),
        .i_mem_rsp_last(reader_rsp_last),
        .o_header_magic(header_magic),
        .o_header_version(header_version),
        .o_header_bytes(header_bytes),
        .o_descriptor_bytes(descriptor_bytes),
        .o_descriptor_count(descriptor_count),
        .o_descriptor_capacity(descriptor_capacity),
        .o_descriptor_table_addr(descriptor_table_addr),
        .o_payload_base_addr(payload_base_addr),
        .o_image_base_addr(image_base_addr),
        .o_image_bytes(image_bytes),
        .o_desc_tensor_id_flat(desc_tensor_id_flat),
        .o_desc_role_flat(desc_role_flat),
        .o_desc_dtype_flat(desc_dtype_flat),
        .o_desc_rank_flat(desc_rank_flat),
        .o_desc_flags_flat(desc_flags_flat),
        .o_desc_element_bits_flat(desc_element_bits_flat),
        .o_desc_group_size_flat(desc_group_size_flat),
        .o_desc_scale_tensor_id_flat(desc_scale_tensor_id_flat),
        .o_desc_base_addr_flat(desc_base_addr_flat),
        .o_desc_nbytes_flat(desc_nbytes_flat),
        .o_desc_dim0_flat(desc_dim0_flat),
        .o_desc_dim1_flat(desc_dim1_flat),
        .o_desc_dim2_flat(desc_dim2_flat),
        .o_desc_dim3_flat(desc_dim3_flat),
        .o_desc_aux0_flat(desc_aux0_flat),
        .o_desc_aux1_flat(desc_aux1_flat),
        .o_desc_aux2_flat(desc_aux2_flat),
        .o_desc_aux3_flat(desc_aux3_flat)
    );

    qmap_dot64_payload_fetcher #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .GROUP_SIZE(GROUP_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH)
    ) fetcher (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(fetcher_start),
        .i_activation_base_addr(activation_base_addr),
        .i_activation_nbytes(activation_nbytes),
        .i_weight_base_addr(weight_base_addr),
        .i_weight_nbytes(weight_nbytes),
        .i_scale_base_addr(scale_base_addr),
        .i_scale_nbytes(scale_nbytes),
        .i_expected_base_addr(expected_base_addr),
        .i_expected_nbytes(expected_nbytes),
        .o_busy(),
        .o_done(fetcher_done),
        .o_error(fetcher_error),
        .o_mem_req_valid(fetcher_req_valid),
        .i_mem_req_ready(fetcher_req_ready),
        .o_mem_req_addr(fetcher_req_addr),
        .o_mem_req_len_bytes(fetcher_req_len_bytes),
        .i_mem_rsp_valid(fetcher_rsp_valid),
        .o_mem_rsp_ready(fetcher_rsp_ready),
        .i_mem_rsp_data(fetcher_rsp_data),
        .i_mem_rsp_last(fetcher_rsp_last),
        .o_activation_flat(activation_flat),
        .o_weight_packed(weight_packed),
        .o_scale_q2_14(scale_q2_14),
        .o_expected_partial_sum(expected_partial_sum),
        .o_expected_scaled_sum_q26(expected_scaled_sum_q26)
    );

    q4_dot_product_64 #(
        .GROUP_SIZE(GROUP_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PRODUCT_WIDTH(PRODUCT_WIDTH),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH)
    ) dot64 (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(dot_start),
        .i_activation_flat(activation_flat),
        .i_weight_packed(weight_packed),
        .i_scale_q2_14(scale_q2_14),
        .o_busy(),
        .o_done(dot_done),
        .o_partial_sum(dot_partial_sum),
        .o_scaled_sum_q26(dot_scaled_sum_q26)
    );

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                     <= S_IDLE;
            o_done                    <= 1'b0;
            o_error                   <= 1'b0;
            o_compare_match           <= 1'b0;
            o_partial_sum             <= 64'sd0;
            o_scaled_sum_q26          <= 64'sd0;
            o_expected_partial_sum    <= 64'sd0;
            o_expected_scaled_sum_q26 <= 64'sd0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        o_error                   <= 1'b0;
                        o_compare_match           <= 1'b0;
                        o_partial_sum             <= 64'sd0;
                        o_scaled_sum_q26          <= 64'sd0;
                        o_expected_partial_sum    <= 64'sd0;
                        o_expected_scaled_sum_q26 <= 64'sd0;
                        state                     <= S_READER_START;
                    end
                end

                S_READER_START: begin
                    state <= S_READER_WAIT;
                end

                S_READER_WAIT: begin
                    if (reader_done) begin
                        if (reader_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            state <= S_FETCHER_START;
                        end
                    end
                end

                S_FETCHER_START: begin
                    state <= S_FETCHER_WAIT;
                end

                S_FETCHER_WAIT: begin
                    if (fetcher_done) begin
                        if (fetcher_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            state <= S_DOT_START;
                        end
                    end
                end

                S_DOT_START: begin
                    state <= S_DOT_WAIT;
                end

                S_DOT_WAIT: begin
                    if (dot_done) begin
                        o_partial_sum             <= dot_partial_sum_ext;
                        o_scaled_sum_q26          <= dot_scaled_sum_q26_ext;
                        o_expected_partial_sum    <= expected_partial_sum;
                        o_expected_scaled_sum_q26 <= expected_scaled_sum_q26;
                        o_compare_match           <= (dot_partial_sum_ext == expected_partial_sum) &&
                                                     (dot_scaled_sum_q26_ext == expected_scaled_sum_q26);
                        o_error                   <= (dot_partial_sum_ext != expected_partial_sum) ||
                                                     (dot_scaled_sum_q26_ext != expected_scaled_sum_q26);
                        state                     <= S_DONE;
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
