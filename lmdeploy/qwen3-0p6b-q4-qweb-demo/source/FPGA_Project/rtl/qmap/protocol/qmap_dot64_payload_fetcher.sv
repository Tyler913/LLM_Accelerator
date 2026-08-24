`default_nettype none

`include "qmap_defs.svh"

// Fetches the real dot64 payload tensors described by QMAP descriptors.
//
// This is the second layer after descriptor parsing:
//   descriptor reader: learns where tensors live
//   payload fetcher: reads those tensor bytes into RTL-local packed signals
module qmap_dot64_payload_fetcher #(
    parameter int ADDR_WIDTH    = 64,
    parameter int GROUP_SIZE    = 64,
    parameter int ACT_WIDTH     = 16,
    parameter int WEIGHT_WIDTH  = 4,
    parameter int SCALE_WIDTH   = 16
)
(
    input  wire logic                                             i_clk,
    input  wire logic                                             i_rst_n,

    input  wire logic                                             i_start,

    input  wire logic [ADDR_WIDTH-1 : 0]                          i_activation_base_addr,
    input  wire logic [63 : 0]                                    i_activation_nbytes,
    input  wire logic [ADDR_WIDTH-1 : 0]                          i_weight_base_addr,
    input  wire logic [63 : 0]                                    i_weight_nbytes,
    input  wire logic [ADDR_WIDTH-1 : 0]                          i_scale_base_addr,
    input  wire logic [63 : 0]                                    i_scale_nbytes,
    input  wire logic [ADDR_WIDTH-1 : 0]                          i_expected_base_addr,
    input  wire logic [63 : 0]                                    i_expected_nbytes,

    output logic                                             o_busy,
    output logic                                             o_done,
    output logic                                             o_error,

    output logic                                             o_mem_req_valid,
    input  wire logic                                             i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                          o_mem_req_addr,
    output logic [15 : 0]                                    o_mem_req_len_bytes,

    input  wire logic                                             i_mem_rsp_valid,
    output logic                                             o_mem_rsp_ready,
    input  wire logic [31 : 0]                                    i_mem_rsp_data,
    input  wire logic                                             i_mem_rsp_last,

    output logic [GROUP_SIZE*ACT_WIDTH-1 : 0]                o_activation_flat,
    output logic [GROUP_SIZE*WEIGHT_WIDTH-1 : 0]             o_weight_packed,
    output logic [SCALE_WIDTH-1 : 0]                         o_scale_q2_14,
    output logic signed [63 : 0]                             o_expected_partial_sum,
    output logic signed [63 : 0]                             o_expected_scaled_sum_q26
);

    localparam logic [1 : 0] TARGET_ACTIVATION = 2'd0;
    localparam logic [1 : 0] TARGET_WEIGHT     = 2'd1;
    localparam logic [1 : 0] TARGET_SCALE      = 2'd2;
    localparam logic [1 : 0] TARGET_EXPECTED   = 2'd3;

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_REQ,
        S_READ,
        S_DONE
    } state_t;

    state_t state;
    logic [1 : 0] target;
    logic [7 : 0] word_index;

    logic [ADDR_WIDTH-1 : 0] current_addr;
    logic [63 : 0] current_nbytes;
    logic [8 : 0] current_words;
    logic last_expected;
    logic descriptors_match_dot64;

    always @* begin
        current_addr   = i_activation_base_addr;
        current_nbytes = i_activation_nbytes;

        case (target)
            TARGET_ACTIVATION: begin
                current_addr   = i_activation_base_addr;
                current_nbytes = i_activation_nbytes;
            end
            TARGET_WEIGHT: begin
                current_addr   = i_weight_base_addr;
                current_nbytes = i_weight_nbytes;
            end
            TARGET_SCALE: begin
                current_addr   = i_scale_base_addr;
                current_nbytes = i_scale_nbytes;
            end
            TARGET_EXPECTED: begin
                current_addr   = i_expected_base_addr;
                current_nbytes = i_expected_nbytes;
            end
            default: begin
                current_addr   = i_activation_base_addr;
                current_nbytes = i_activation_nbytes;
            end
        endcase
    end

    assign current_words = (current_nbytes[15 : 0] + 16'd3) >> 2;
    assign last_expected = (word_index == (current_words - 1'b1));

    assign descriptors_match_dot64 =
        (i_activation_nbytes == `QMAP_DOT64_ACTIVATION_BYTES) &&
        (i_weight_nbytes     == `QMAP_DOT64_WEIGHT_BYTES) &&
        (i_scale_nbytes      == `QMAP_DOT64_SCALE_BYTES) &&
        (i_expected_nbytes   == `QMAP_DOT64_EXPECTED_BYTES);

    assign o_busy              = (state != S_IDLE);
    assign o_mem_req_valid     = (state == S_REQ);
    assign o_mem_req_addr      = current_addr;
    assign o_mem_req_len_bytes = current_nbytes[15 : 0];
    assign o_mem_rsp_ready     = (state == S_READ);

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                     <= S_IDLE;
            target                    <= TARGET_ACTIVATION;
            word_index                <= 8'd0;
            o_done                    <= 1'b0;
            o_error                   <= 1'b0;
            o_activation_flat         <= 'd0;
            o_weight_packed           <= 'd0;
            o_scale_q2_14             <= 'd0;
            o_expected_partial_sum    <= 64'sd0;
            o_expected_scaled_sum_q26 <= 64'sd0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        o_error                   <= !descriptors_match_dot64;
                        o_activation_flat         <= 'd0;
                        o_weight_packed           <= 'd0;
                        o_scale_q2_14             <= 'd0;
                        o_expected_partial_sum    <= 64'sd0;
                        o_expected_scaled_sum_q26 <= 64'sd0;
                        target                    <= TARGET_ACTIVATION;
                        word_index                <= 8'd0;
                        state                     <= descriptors_match_dot64 ? S_REQ : S_DONE;
                    end
                end

                S_REQ: begin
                    if (i_mem_req_ready) begin
                        word_index <= 8'd0;
                        state      <= S_READ;
                    end
                end

                S_READ: begin
                    if (i_mem_rsp_valid) begin
                        if ((i_mem_rsp_last && !last_expected) ||
                            (!i_mem_rsp_last && last_expected)) begin
                            o_error <= 1'b1;
                        end

                        case (target)
                            TARGET_ACTIVATION: begin
                                o_activation_flat[word_index*32 +: 32] <= i_mem_rsp_data;
                            end
                            TARGET_WEIGHT: begin
                                o_weight_packed[word_index*32 +: 32] <= i_mem_rsp_data;
                            end
                            TARGET_SCALE: begin
                                if (word_index == 8'd0) begin
                                    o_scale_q2_14 <= i_mem_rsp_data[15 : 0];
                                end
                            end
                            TARGET_EXPECTED: begin
                                case (word_index)
                                    8'd0: o_expected_partial_sum[31 : 0]       <= i_mem_rsp_data;
                                    8'd1: o_expected_partial_sum[63 : 32]      <= i_mem_rsp_data;
                                    8'd2: o_expected_scaled_sum_q26[31 : 0]    <= i_mem_rsp_data;
                                    8'd3: o_expected_scaled_sum_q26[63 : 32]   <= i_mem_rsp_data;
                                    default: begin
                                    end
                                endcase
                            end
                            default: begin
                            end
                        endcase

                        if (i_mem_rsp_last || last_expected) begin
                            if (target == TARGET_EXPECTED) begin
                                state <= S_DONE;
                            end else begin
                                target <= target + 1'b1;
                                state  <= S_REQ;
                            end
                        end else begin
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
