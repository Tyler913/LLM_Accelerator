`default_nettype none

`include "qmap_defs.svh"

// Reads and decodes the fixed QMAP v1 header prefix.
//
// Memory interface:
//   - one request per transaction
//   - 32-bit little-endian response words
//   - o_mem_req_len_bytes tells the lower layer how many bytes to return
module qmap_header_reader #(
    parameter int ADDR_WIDTH = 64
)
(
    input  logic                    i_clk,
    input  logic                    i_rst_n,

    input  logic                    i_start,
    input  logic [ADDR_WIDTH-1 : 0] i_qmap_base_addr,

    output logic                    o_busy,
    output logic                    o_done,
    output logic                    o_error,

    output logic                    o_mem_req_valid,
    input  logic                    i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0] o_mem_req_addr,
    output logic [15 : 0]           o_mem_req_len_bytes,

    input  logic                    i_mem_rsp_valid,
    output logic                    o_mem_rsp_ready,
    input  logic [31 : 0]           i_mem_rsp_data,
    input  logic                    i_mem_rsp_last,

    output logic [31 : 0]           o_magic,
    output logic [31 : 0]           o_version,
    output logic [31 : 0]           o_header_bytes,
    output logic [31 : 0]           o_descriptor_bytes,
    output logic [31 : 0]           o_descriptor_count,
    output logic [31 : 0]           o_descriptor_capacity,
    output logic [63 : 0]           o_descriptor_table_addr,
    output logic [63 : 0]           o_payload_base_addr,
    output logic [63 : 0]           o_image_base_addr,
    output logic [63 : 0]           o_image_bytes,
    output logic [31 : 0]           o_flags,
    output logic [31 : 0]           o_checksum32
);

    localparam int HEADER_WORDS = `QMAP_HEADER_FETCH_BYTES / 4;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_REQ,
        S_READ,
        S_CHECK
    } state_t;

    state_t state;
    logic [$clog2(HEADER_WORDS)-1 : 0] word_index;
    logic protocol_error;

    assign o_busy              = (state != S_IDLE);
    assign o_mem_req_valid     = (state == S_REQ);
    assign o_mem_req_addr      = i_qmap_base_addr;
    assign o_mem_req_len_bytes = `QMAP_HEADER_FETCH_BYTES;
    assign o_mem_rsp_ready     = (state == S_READ);

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                   <= S_IDLE;
            word_index              <= 'd0;
            protocol_error          <= 1'b0;
            o_done                  <= 1'b0;
            o_error                 <= 1'b0;
            o_magic                 <= 32'd0;
            o_version               <= 32'd0;
            o_header_bytes          <= 32'd0;
            o_descriptor_bytes      <= 32'd0;
            o_descriptor_count      <= 32'd0;
            o_descriptor_capacity   <= 32'd0;
            o_descriptor_table_addr <= 64'd0;
            o_payload_base_addr     <= 64'd0;
            o_image_base_addr       <= 64'd0;
            o_image_bytes           <= 64'd0;
            o_flags                 <= 32'd0;
            o_checksum32            <= 32'd0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        word_index     <= 'd0;
                        protocol_error <= 1'b0;
                        o_error        <= 1'b0;
                        state          <= S_REQ;
                    end
                end

                S_REQ: begin
                    if (i_mem_req_ready) begin
                        state <= S_READ;
                    end
                end

                S_READ: begin
                    if (i_mem_rsp_valid) begin
                        case (word_index)
                            0:  o_magic                       <= i_mem_rsp_data;
                            1:  o_version                     <= i_mem_rsp_data;
                            2:  o_header_bytes                <= i_mem_rsp_data;
                            3:  o_descriptor_bytes            <= i_mem_rsp_data;
                            4:  o_descriptor_count            <= i_mem_rsp_data;
                            5:  o_descriptor_capacity         <= i_mem_rsp_data;
                            6:  o_descriptor_table_addr[31:0] <= i_mem_rsp_data;
                            7:  o_descriptor_table_addr[63:32] <= i_mem_rsp_data;
                            8:  o_payload_base_addr[31:0]     <= i_mem_rsp_data;
                            9:  o_payload_base_addr[63:32]    <= i_mem_rsp_data;
                            10: o_image_base_addr[31:0]       <= i_mem_rsp_data;
                            11: o_image_base_addr[63:32]      <= i_mem_rsp_data;
                            12: o_image_bytes[31:0]           <= i_mem_rsp_data;
                            13: o_image_bytes[63:32]          <= i_mem_rsp_data;
                            14: o_flags                       <= i_mem_rsp_data;
                            15: o_checksum32                  <= i_mem_rsp_data;
                            default: begin
                            end
                        endcase

                        if ((i_mem_rsp_last && (word_index != HEADER_WORDS-1)) ||
                            (!i_mem_rsp_last && (word_index == HEADER_WORDS-1))) begin
                            protocol_error <= 1'b1;
                        end

                        if (word_index == HEADER_WORDS-1) begin
                            state <= S_CHECK;
                        end else begin
                            word_index <= word_index + 1'b1;
                        end
                    end
                end

                S_CHECK: begin
                    o_done  <= 1'b1;
                    o_error <= protocol_error ||
                               (o_magic != `QMAP_MAGIC) ||
                               (o_version != `QMAP_VERSION) ||
                               (o_header_bytes != `QMAP_HEADER_BYTES) ||
                               (o_descriptor_bytes != `QMAP_DESCRIPTOR_BYTES);
                    state   <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
