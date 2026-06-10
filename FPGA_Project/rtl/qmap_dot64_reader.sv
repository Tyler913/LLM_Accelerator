`default_nettype none

`include "qmap_defs.svh"

// First narrow QMAP reader for the dot64 smoke image.
//
// This module intentionally uses the project-local memory request/response
// interface instead of AXI. A lower layer can later implement that interface
// with AXI4, BRAM, or a simulation memory model.
module qmap_dot64_reader #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 4
)
(
    input  logic                              i_clk,
    input  logic                              i_rst_n,

    input  logic                              i_start,
    input  logic [ADDR_WIDTH-1 : 0]           i_qmap_base_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,

    output logic                              o_mem_req_valid,
    input  logic                              i_mem_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_req_addr,
    output logic [15 : 0]                     o_mem_req_len_bytes,

    input  logic                              i_mem_rsp_valid,
    output logic                              o_mem_rsp_ready,
    input  logic [31 : 0]                     i_mem_rsp_data,
    input  logic                              i_mem_rsp_last,

    output logic [31 : 0]                     o_header_magic,
    output logic [31 : 0]                     o_header_version,
    output logic [31 : 0]                     o_header_bytes,
    output logic [31 : 0]                     o_descriptor_bytes,
    output logic [31 : 0]                     o_descriptor_count,
    output logic [31 : 0]                     o_descriptor_capacity,
    output logic [63 : 0]                     o_descriptor_table_addr,
    output logic [63 : 0]                     o_payload_base_addr,
    output logic [63 : 0]                     o_image_base_addr,
    output logic [63 : 0]                     o_image_bytes,

    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_tensor_id_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_role_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_dtype_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_rank_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_flags_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_element_bits_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_group_size_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_scale_tensor_id_flat,
    output logic [DESCRIPTOR_SLOTS*64-1 : 0]  o_desc_base_addr_flat,
    output logic [DESCRIPTOR_SLOTS*64-1 : 0]  o_desc_nbytes_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_dim0_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_dim1_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_dim2_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_dim3_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_aux0_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_aux1_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_aux2_flat,
    output logic [DESCRIPTOR_SLOTS*32-1 : 0]  o_desc_aux3_flat
);

    localparam int SLOT_INDEX_WIDTH = (DESCRIPTOR_SLOTS <= 1) ? 1 : $clog2(DESCRIPTOR_SLOTS);

    typedef enum logic [2 : 0] {
        S_IDLE,
        S_HEADER_START,
        S_HEADER_WAIT,
        S_DESCRIPTOR_START,
        S_DESCRIPTOR_WAIT,
        S_DONE
    } state_t;

    state_t state;
    logic [SLOT_INDEX_WIDTH-1 : 0] slot_index;

    logic header_start;
    logic header_busy;
    logic header_done;
    logic header_error;
    logic header_req_valid;
    logic header_req_ready;
    logic [ADDR_WIDTH-1 : 0] header_req_addr;
    logic [15 : 0] header_req_len_bytes;
    logic header_rsp_valid;
    logic header_rsp_ready;
    logic [31 : 0] header_rsp_data;
    logic header_rsp_last;
    logic [31 : 0] header_flags;
    logic [31 : 0] header_checksum32;

    logic descriptor_start;
    logic descriptor_busy;
    logic descriptor_done;
    logic descriptor_error;
    logic descriptor_req_valid;
    logic descriptor_req_ready;
    logic [ADDR_WIDTH-1 : 0] descriptor_req_addr;
    logic [15 : 0] descriptor_req_len_bytes;
    logic descriptor_rsp_valid;
    logic descriptor_rsp_ready;
    logic [31 : 0] descriptor_rsp_data;
    logic descriptor_rsp_last;

    logic [31 : 0] desc_tensor_id;
    logic [31 : 0] desc_role;
    logic [31 : 0] desc_dtype;
    logic [31 : 0] desc_rank;
    logic [31 : 0] desc_flags;
    logic [31 : 0] desc_element_bits;
    logic [31 : 0] desc_group_size;
    logic [31 : 0] desc_scale_tensor_id;
    logic [63 : 0] desc_base_addr;
    logic [63 : 0] desc_nbytes;
    logic [31 : 0] desc_dim0;
    logic [31 : 0] desc_dim1;
    logic [31 : 0] desc_dim2;
    logic [31 : 0] desc_dim3;
    logic [63 : 0] desc_stride0_bytes;
    logic [63 : 0] desc_stride1_bytes;
    logic [63 : 0] desc_stride2_bytes;
    logic [63 : 0] desc_stride3_bytes;
    logic [31 : 0] desc_aux0;
    logic [31 : 0] desc_aux1;
    logic [31 : 0] desc_aux2;
    logic [31 : 0] desc_aux3;
    logic [31 : 0] desc_checksum32;
    logic [31 : 0] desc_reserved0;
    logic [31 : 0] desc_reserved1;
    logic [31 : 0] desc_reserved2;

    assign o_busy           = (state != S_IDLE);
    assign header_start     = (state == S_HEADER_START);
    assign descriptor_start = (state == S_DESCRIPTOR_START);

    assign o_mem_req_valid     = (state == S_HEADER_WAIT) ? header_req_valid :
                                 (state == S_DESCRIPTOR_WAIT) ? descriptor_req_valid :
                                 1'b0;
    assign o_mem_req_addr      = (state == S_HEADER_WAIT) ? header_req_addr :
                                 descriptor_req_addr;
    assign o_mem_req_len_bytes = (state == S_HEADER_WAIT) ? header_req_len_bytes :
                                 descriptor_req_len_bytes;
    assign header_req_ready    = (state == S_HEADER_WAIT) ? i_mem_req_ready : 1'b0;
    assign descriptor_req_ready = (state == S_DESCRIPTOR_WAIT) ? i_mem_req_ready : 1'b0;

    assign o_mem_rsp_ready     = (state == S_HEADER_WAIT) ? header_rsp_ready :
                                 (state == S_DESCRIPTOR_WAIT) ? descriptor_rsp_ready :
                                 1'b0;
    assign header_rsp_valid    = (state == S_HEADER_WAIT) ? i_mem_rsp_valid : 1'b0;
    assign header_rsp_data     = i_mem_rsp_data;
    assign header_rsp_last     = i_mem_rsp_last;
    assign descriptor_rsp_valid = (state == S_DESCRIPTOR_WAIT) ? i_mem_rsp_valid : 1'b0;
    assign descriptor_rsp_data  = i_mem_rsp_data;
    assign descriptor_rsp_last  = i_mem_rsp_last;

    qmap_header_reader #(
        .ADDR_WIDTH(ADDR_WIDTH)
    ) header_reader (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(header_start),
        .i_qmap_base_addr(i_qmap_base_addr),
        .o_busy(header_busy),
        .o_done(header_done),
        .o_error(header_error),
        .o_mem_req_valid(header_req_valid),
        .i_mem_req_ready(header_req_ready),
        .o_mem_req_addr(header_req_addr),
        .o_mem_req_len_bytes(header_req_len_bytes),
        .i_mem_rsp_valid(header_rsp_valid),
        .o_mem_rsp_ready(header_rsp_ready),
        .i_mem_rsp_data(header_rsp_data),
        .i_mem_rsp_last(header_rsp_last),
        .o_magic(o_header_magic),
        .o_version(o_header_version),
        .o_header_bytes(o_header_bytes),
        .o_descriptor_bytes(o_descriptor_bytes),
        .o_descriptor_count(o_descriptor_count),
        .o_descriptor_capacity(o_descriptor_capacity),
        .o_descriptor_table_addr(o_descriptor_table_addr),
        .o_payload_base_addr(o_payload_base_addr),
        .o_image_base_addr(o_image_base_addr),
        .o_image_bytes(o_image_bytes),
        .o_flags(header_flags),
        .o_checksum32(header_checksum32)
    );

    qmap_descriptor_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .SLOT_INDEX_WIDTH(SLOT_INDEX_WIDTH)
    ) descriptor_reader (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(descriptor_start),
        .i_descriptor_table_addr(o_descriptor_table_addr),
        .i_slot_index(slot_index),
        .o_busy(descriptor_busy),
        .o_done(descriptor_done),
        .o_error(descriptor_error),
        .o_mem_req_valid(descriptor_req_valid),
        .i_mem_req_ready(descriptor_req_ready),
        .o_mem_req_addr(descriptor_req_addr),
        .o_mem_req_len_bytes(descriptor_req_len_bytes),
        .i_mem_rsp_valid(descriptor_rsp_valid),
        .o_mem_rsp_ready(descriptor_rsp_ready),
        .i_mem_rsp_data(descriptor_rsp_data),
        .i_mem_rsp_last(descriptor_rsp_last),
        .o_tensor_id(desc_tensor_id),
        .o_role(desc_role),
        .o_dtype(desc_dtype),
        .o_rank(desc_rank),
        .o_flags(desc_flags),
        .o_element_bits(desc_element_bits),
        .o_group_size(desc_group_size),
        .o_scale_tensor_id(desc_scale_tensor_id),
        .o_base_addr(desc_base_addr),
        .o_nbytes(desc_nbytes),
        .o_dim0(desc_dim0),
        .o_dim1(desc_dim1),
        .o_dim2(desc_dim2),
        .o_dim3(desc_dim3),
        .o_stride0_bytes(desc_stride0_bytes),
        .o_stride1_bytes(desc_stride1_bytes),
        .o_stride2_bytes(desc_stride2_bytes),
        .o_stride3_bytes(desc_stride3_bytes),
        .o_aux0(desc_aux0),
        .o_aux1(desc_aux1),
        .o_aux2(desc_aux2),
        .o_aux3(desc_aux3),
        .o_checksum32(desc_checksum32),
        .o_reserved0(desc_reserved0),
        .o_reserved1(desc_reserved1),
        .o_reserved2(desc_reserved2)
    );

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                       <= S_IDLE;
            slot_index                  <= 'd0;
            o_done                      <= 1'b0;
            o_error                     <= 1'b0;
            o_desc_tensor_id_flat       <= 'd0;
            o_desc_role_flat            <= 'd0;
            o_desc_dtype_flat           <= 'd0;
            o_desc_rank_flat            <= 'd0;
            o_desc_flags_flat           <= 'd0;
            o_desc_element_bits_flat    <= 'd0;
            o_desc_group_size_flat      <= 'd0;
            o_desc_scale_tensor_id_flat <= 'd0;
            o_desc_base_addr_flat       <= 'd0;
            o_desc_nbytes_flat          <= 'd0;
            o_desc_dim0_flat            <= 'd0;
            o_desc_dim1_flat            <= 'd0;
            o_desc_dim2_flat            <= 'd0;
            o_desc_dim3_flat            <= 'd0;
            o_desc_aux0_flat            <= 'd0;
            o_desc_aux1_flat            <= 'd0;
            o_desc_aux2_flat            <= 'd0;
            o_desc_aux3_flat            <= 'd0;
        end else begin
            o_done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        slot_index                  <= 'd0;
                        o_error                     <= 1'b0;
                        o_desc_tensor_id_flat       <= 'd0;
                        o_desc_role_flat            <= 'd0;
                        o_desc_dtype_flat           <= 'd0;
                        o_desc_rank_flat            <= 'd0;
                        o_desc_flags_flat           <= 'd0;
                        o_desc_element_bits_flat    <= 'd0;
                        o_desc_group_size_flat      <= 'd0;
                        o_desc_scale_tensor_id_flat <= 'd0;
                        o_desc_base_addr_flat       <= 'd0;
                        o_desc_nbytes_flat          <= 'd0;
                        o_desc_dim0_flat            <= 'd0;
                        o_desc_dim1_flat            <= 'd0;
                        o_desc_dim2_flat            <= 'd0;
                        o_desc_dim3_flat            <= 'd0;
                        o_desc_aux0_flat            <= 'd0;
                        o_desc_aux1_flat            <= 'd0;
                        o_desc_aux2_flat            <= 'd0;
                        o_desc_aux3_flat            <= 'd0;
                        state                       <= S_HEADER_START;
                    end
                end

                S_HEADER_START: begin
                    state <= S_HEADER_WAIT;
                end

                S_HEADER_WAIT: begin
                    if (header_done) begin
                        if (header_error ||
                            (o_descriptor_count < DESCRIPTOR_SLOTS) ||
                            (o_descriptor_capacity < DESCRIPTOR_SLOTS)) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            slot_index <= 'd0;
                            state      <= S_DESCRIPTOR_START;
                        end
                    end
                end

                S_DESCRIPTOR_START: begin
                    state <= S_DESCRIPTOR_WAIT;
                end

                S_DESCRIPTOR_WAIT: begin
                    if (descriptor_done) begin
                        if (descriptor_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end else begin
                            o_desc_tensor_id_flat[slot_index*32 +: 32]       <= desc_tensor_id;
                            o_desc_role_flat[slot_index*32 +: 32]            <= desc_role;
                            o_desc_dtype_flat[slot_index*32 +: 32]           <= desc_dtype;
                            o_desc_rank_flat[slot_index*32 +: 32]            <= desc_rank;
                            o_desc_flags_flat[slot_index*32 +: 32]           <= desc_flags;
                            o_desc_element_bits_flat[slot_index*32 +: 32]    <= desc_element_bits;
                            o_desc_group_size_flat[slot_index*32 +: 32]      <= desc_group_size;
                            o_desc_scale_tensor_id_flat[slot_index*32 +: 32] <= desc_scale_tensor_id;
                            o_desc_base_addr_flat[slot_index*64 +: 64]       <= desc_base_addr;
                            o_desc_nbytes_flat[slot_index*64 +: 64]          <= desc_nbytes;
                            o_desc_dim0_flat[slot_index*32 +: 32]            <= desc_dim0;
                            o_desc_dim1_flat[slot_index*32 +: 32]            <= desc_dim1;
                            o_desc_dim2_flat[slot_index*32 +: 32]            <= desc_dim2;
                            o_desc_dim3_flat[slot_index*32 +: 32]            <= desc_dim3;
                            o_desc_aux0_flat[slot_index*32 +: 32]            <= desc_aux0;
                            o_desc_aux1_flat[slot_index*32 +: 32]            <= desc_aux1;
                            o_desc_aux2_flat[slot_index*32 +: 32]            <= desc_aux2;
                            o_desc_aux3_flat[slot_index*32 +: 32]            <= desc_aux3;

                            if (slot_index == DESCRIPTOR_SLOTS-1) begin
                                state <= S_DONE;
                            end else begin
                                slot_index <= slot_index + 1'b1;
                                state      <= S_DESCRIPTOR_START;
                            end
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
