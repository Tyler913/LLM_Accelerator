`default_nettype none

`include "qmap_defs.svh"

// Reads and decodes one 128-byte QMAP tensor descriptor.
module qmap_descriptor_reader #(
    parameter int ADDR_WIDTH       = 64,
    parameter int SLOT_INDEX_WIDTH = 8
)
(
    input  wire logic                          i_clk,
    input  wire logic                          i_rst_n,

    input  wire logic                          i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]       i_descriptor_table_addr,
    input  wire logic [SLOT_INDEX_WIDTH-1 : 0] i_slot_index,

    output logic                          o_busy,
    output logic                          o_done,
    output logic                          o_error,

    output logic                          o_mem_req_valid,
    input  wire logic                          i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]       o_mem_req_addr,
    output logic [15 : 0]                 o_mem_req_len_bytes,

    input  wire logic                          i_mem_rsp_valid,
    output logic                          o_mem_rsp_ready,
    input  wire logic [31 : 0]                 i_mem_rsp_data,
    input  wire logic                          i_mem_rsp_last,

    output logic [31 : 0]                 o_tensor_id,
    output logic [31 : 0]                 o_role,
    output logic [31 : 0]                 o_dtype,
    output logic [31 : 0]                 o_rank,
    output logic [31 : 0]                 o_flags,
    output logic [31 : 0]                 o_element_bits,
    output logic [31 : 0]                 o_group_size,
    output logic [31 : 0]                 o_scale_tensor_id,
    output logic [63 : 0]                 o_base_addr,
    output logic [63 : 0]                 o_nbytes,
    output logic [31 : 0]                 o_dim0,
    output logic [31 : 0]                 o_dim1,
    output logic [31 : 0]                 o_dim2,
    output logic [31 : 0]                 o_dim3,
    output logic [63 : 0]                 o_stride0_bytes,
    output logic [63 : 0]                 o_stride1_bytes,
    output logic [63 : 0]                 o_stride2_bytes,
    output logic [63 : 0]                 o_stride3_bytes,
    output logic [31 : 0]                 o_aux0,
    output logic [31 : 0]                 o_aux1,
    output logic [31 : 0]                 o_aux2,
    output logic [31 : 0]                 o_aux3,
    output logic [31 : 0]                 o_checksum32,
    output logic [31 : 0]                 o_reserved0,
    output logic [31 : 0]                 o_reserved1,
    output logic [31 : 0]                 o_reserved2
);

    localparam int DESCRIPTOR_WORDS = `QMAP_DESCRIPTOR_BYTES / 4;

    typedef enum logic [1 : 0] {
        S_IDLE,
        S_REQ,
        S_READ,
        S_CHECK
    } state_t;

    state_t state;
    logic [$clog2(DESCRIPTOR_WORDS)-1 : 0] word_index;
    logic protocol_error;
    logic [ADDR_WIDTH-1 : 0] slot_offset_bytes;

    always @* begin
        slot_offset_bytes = 'd0;
        slot_offset_bytes[SLOT_INDEX_WIDTH+6 : 7] = i_slot_index;
    end

    assign o_busy              = (state != S_IDLE);
    assign o_mem_req_valid     = (state == S_REQ);
    assign o_mem_req_addr      = i_descriptor_table_addr + slot_offset_bytes;
    assign o_mem_req_len_bytes = 16'd128;
    assign o_mem_rsp_ready     = (state == S_READ);

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state             <= S_IDLE;
            word_index        <= 'd0;
            protocol_error    <= 1'b0;
            o_done            <= 1'b0;
            o_error           <= 1'b0;
            o_tensor_id       <= 32'd0;
            o_role            <= 32'd0;
            o_dtype           <= 32'd0;
            o_rank            <= 32'd0;
            o_flags           <= 32'd0;
            o_element_bits    <= 32'd0;
            o_group_size      <= 32'd0;
            o_scale_tensor_id <= 32'd0;
            o_base_addr       <= 64'd0;
            o_nbytes          <= 64'd0;
            o_dim0            <= 32'd0;
            o_dim1            <= 32'd0;
            o_dim2            <= 32'd0;
            o_dim3            <= 32'd0;
            o_stride0_bytes   <= 64'd0;
            o_stride1_bytes   <= 64'd0;
            o_stride2_bytes   <= 64'd0;
            o_stride3_bytes   <= 64'd0;
            o_aux0            <= 32'd0;
            o_aux1            <= 32'd0;
            o_aux2            <= 32'd0;
            o_aux3            <= 32'd0;
            o_checksum32      <= 32'd0;
            o_reserved0       <= 32'd0;
            o_reserved1       <= 32'd0;
            o_reserved2       <= 32'd0;
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
                            0:  o_tensor_id              <= i_mem_rsp_data;
                            1:  o_role                   <= i_mem_rsp_data;
                            2:  o_dtype                  <= i_mem_rsp_data;
                            3:  o_rank                   <= i_mem_rsp_data;
                            4:  o_flags                  <= i_mem_rsp_data;
                            5:  o_element_bits           <= i_mem_rsp_data;
                            6:  o_group_size             <= i_mem_rsp_data;
                            7:  o_scale_tensor_id        <= i_mem_rsp_data;
                            8:  o_base_addr[31:0]        <= i_mem_rsp_data;
                            9:  o_base_addr[63:32]       <= i_mem_rsp_data;
                            10: o_nbytes[31:0]           <= i_mem_rsp_data;
                            11: o_nbytes[63:32]          <= i_mem_rsp_data;
                            12: o_dim0                   <= i_mem_rsp_data;
                            13: o_dim1                   <= i_mem_rsp_data;
                            14: o_dim2                   <= i_mem_rsp_data;
                            15: o_dim3                   <= i_mem_rsp_data;
                            16: o_stride0_bytes[31:0]    <= i_mem_rsp_data;
                            17: o_stride0_bytes[63:32]   <= i_mem_rsp_data;
                            18: o_stride1_bytes[31:0]    <= i_mem_rsp_data;
                            19: o_stride1_bytes[63:32]   <= i_mem_rsp_data;
                            20: o_stride2_bytes[31:0]    <= i_mem_rsp_data;
                            21: o_stride2_bytes[63:32]   <= i_mem_rsp_data;
                            22: o_stride3_bytes[31:0]    <= i_mem_rsp_data;
                            23: o_stride3_bytes[63:32]   <= i_mem_rsp_data;
                            24: o_aux0                   <= i_mem_rsp_data;
                            25: o_aux1                   <= i_mem_rsp_data;
                            26: o_aux2                   <= i_mem_rsp_data;
                            27: o_aux3                   <= i_mem_rsp_data;
                            28: o_checksum32             <= i_mem_rsp_data;
                            29: o_reserved0              <= i_mem_rsp_data;
                            30: o_reserved1              <= i_mem_rsp_data;
                            31: o_reserved2              <= i_mem_rsp_data;
                            default: begin
                            end
                        endcase

                        if ((i_mem_rsp_last && (word_index != DESCRIPTOR_WORDS-1)) ||
                            (!i_mem_rsp_last && (word_index == DESCRIPTOR_WORDS-1))) begin
                            protocol_error <= 1'b1;
                        end

                        if (word_index == DESCRIPTOR_WORDS-1) begin
                            state <= S_CHECK;
                        end else begin
                            word_index <= word_index + 1'b1;
                        end
                    end
                end

                S_CHECK: begin
                    o_done  <= 1'b1;
                    o_error <= protocol_error ||
                               (o_reserved0 != 32'd0) ||
                               (o_reserved1 != 32'd0) ||
                               (o_reserved2 != 32'd0);
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
