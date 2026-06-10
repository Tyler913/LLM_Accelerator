`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// RTL smoke test for the first QMAP dot64 reader.
//
// Regenerate the memory vector from the repository root:
//
//   conda run -n llm_fpga python FPGA_Project/sim/tools/qmap_bin_to_mem.py
//
// Run:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_dot64_reader.vvp \
//     FPGA_Project/sim/tb_qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv
//   vvp FPGA_Project/sim/tb_qmap_dot64_reader.vvp

module tb_qmap_dot64_reader;

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 4;
    localparam int IMAGE_BYTES      = 16'h0600;
    localparam int MEM_WORDS        = IMAGE_BYTES / 4;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;

    logic mem_req_valid;
    logic mem_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_req_addr;
    logic [15 : 0] mem_req_len_bytes;
    logic mem_rsp_valid;
    logic mem_rsp_ready;
    logic [31 : 0] mem_rsp_data;
    logic mem_rsp_last;

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

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic active_read;
    logic [31 : 0] read_index;
    logic [31 : 0] beats_remaining;
    integer cycle_count;
    integer mismatch_count;

    qmap_dot64_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(`QMAP_DOT64_BASE_ADDR),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_mem_req_valid(mem_req_valid),
        .i_mem_req_ready(mem_req_ready),
        .o_mem_req_addr(mem_req_addr),
        .o_mem_req_len_bytes(mem_req_len_bytes),
        .i_mem_rsp_valid(mem_rsp_valid),
        .o_mem_rsp_ready(mem_rsp_ready),
        .i_mem_rsp_data(mem_rsp_data),
        .i_mem_rsp_last(mem_rsp_last),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $readmemh("FPGA_Project/sim/vectors/qmap_dot64_image_words32.hex", mem);
    end

    assign mem_req_ready = !active_read;
    assign mem_rsp_data  = mem[read_index];
    assign mem_rsp_last  = active_read && (beats_remaining == 32'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            active_read     <= 1'b0;
            read_index      <= 32'd0;
            beats_remaining <= 32'd0;
            mem_rsp_valid   <= 1'b0;
        end else begin
            if (mem_req_valid && mem_req_ready) begin
                active_read     <= 1'b1;
                read_index      <= (mem_req_addr - `QMAP_DOT64_BASE_ADDR) >> 2;
                beats_remaining <= (mem_req_len_bytes + 16'd3) >> 2;
                mem_rsp_valid   <= 1'b1;
            end else if (mem_rsp_valid && mem_rsp_ready) begin
                if (beats_remaining == 32'd1) begin
                    active_read     <= 1'b0;
                    mem_rsp_valid   <= 1'b0;
                    beats_remaining <= 32'd0;
                end else begin
                    read_index      <= read_index + 1'b1;
                    beats_remaining <= beats_remaining - 1'b1;
                end
            end
        end
    end

    function automatic logic [31 : 0] desc32;
        input logic [DESCRIPTOR_SLOTS*32-1 : 0] flat;
        input int slot;
        begin
            desc32 = flat[slot*32 +: 32];
        end
    endfunction

    function automatic logic [63 : 0] desc64;
        input logic [DESCRIPTOR_SLOTS*64-1 : 0] flat;
        input int slot;
        begin
            desc64 = flat[slot*64 +: 64];
        end
    endfunction

    task check32;
        input string name;
        input logic [31 : 0] actual;
        input logic [31 : 0] expected;
        begin
            $display("%-32s actual=0x%08h expected=0x%08h", name, actual, expected);
            if (actual !== expected) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check64;
        input string name;
        input logic [63 : 0] actual;
        input logic [63 : 0] expected;
        begin
            $display("%-32s actual=0x%016h expected=0x%016h", name, actual, expected);
            if (actual !== expected) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 1000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for qmap_dot64_reader");
            $finish(1);
        end

        if (error) begin
            $display("FAIL: qmap_dot64_reader reported error");
            $finish(1);
        end

        $display("QMAP dot64 reader header checks");
        check32("magic", header_magic, `QMAP_MAGIC);
        check32("version", header_version, `QMAP_VERSION);
        check32("header_bytes", header_bytes, `QMAP_HEADER_BYTES);
        check32("descriptor_bytes", descriptor_bytes, `QMAP_DESCRIPTOR_BYTES);
        check32("descriptor_count", descriptor_count, `QMAP_DESCRIPTOR_COUNT_DOT64);
        check32("descriptor_capacity", descriptor_capacity, `QMAP_DESCRIPTOR_CAPACITY_DOT64);
        check64("descriptor_table_addr", descriptor_table_addr, `QMAP_DOT64_DESCRIPTOR_TABLE_ADDR);
        check64("payload_base_addr", payload_base_addr, `QMAP_DOT64_PAYLOAD_BASE_ADDR);
        check64("image_base_addr", image_base_addr, `QMAP_DOT64_BASE_ADDR);
        check64("image_bytes", image_bytes, `QMAP_DOT64_IMAGE_BYTES);

        $display("QMAP dot64 reader descriptor checks");
        check32("slot0 tensor_id", desc32(desc_tensor_id_flat, 0), `QMAP_TENSOR_ID_ACTIVATION);
        check32("slot0 role", desc32(desc_role_flat, 0), `QMAP_ROLE_ACTIVATION);
        check32("slot0 dtype", desc32(desc_dtype_flat, 0), `QMAP_DTYPE_I16_Q4_12);
        check64("slot0 base_addr", desc64(desc_base_addr_flat, 0), 64'h0000_0004_1B10_0500);
        check64("slot0 nbytes", desc64(desc_nbytes_flat, 0), 64'd128);

        check32("slot1 tensor_id", desc32(desc_tensor_id_flat, 1), `QMAP_TENSOR_ID_WEIGHT);
        check32("slot1 role", desc32(desc_role_flat, 1), `QMAP_ROLE_Q4_WEIGHT);
        check32("slot1 dtype", desc32(desc_dtype_flat, 1), `QMAP_DTYPE_PACKED_Q4_S4);
        check32("slot1 group_size", desc32(desc_group_size_flat, 1), `QMAP_Q4_GROUP_SIZE);
        check32("slot1 scale_tensor_id", desc32(desc_scale_tensor_id_flat, 1), `QMAP_TENSOR_ID_SCALE);
        check64("slot1 base_addr", desc64(desc_base_addr_flat, 1), 64'h0000_0004_1B10_0580);
        check64("slot1 nbytes", desc64(desc_nbytes_flat, 1), 64'd32);

        check32("slot2 tensor_id", desc32(desc_tensor_id_flat, 2), `QMAP_TENSOR_ID_SCALE);
        check32("slot2 role", desc32(desc_role_flat, 2), `QMAP_ROLE_Q4_SCALE);
        check32("slot2 dtype", desc32(desc_dtype_flat, 2), `QMAP_DTYPE_U16_Q2_14);
        check64("slot2 base_addr", desc64(desc_base_addr_flat, 2), 64'h0000_0004_1B10_05A0);
        check64("slot2 nbytes", desc64(desc_nbytes_flat, 2), 64'd2);

        check32("slot3 tensor_id", desc32(desc_tensor_id_flat, 3), `QMAP_TENSOR_ID_EXPECTED);
        check32("slot3 role", desc32(desc_role_flat, 3), `QMAP_ROLE_EXPECTED);
        check32("slot3 dtype", desc32(desc_dtype_flat, 3), `QMAP_DTYPE_I64);
        check64("slot3 base_addr", desc64(desc_base_addr_flat, 3), 64'h0000_0004_1B10_05C0);
        check64("slot3 nbytes", desc64(desc_nbytes_flat, 3), 64'd16);

        check32("slot0 aux0", desc32(desc_aux0_flat, 0), `QMAP_MATRIX_ID_Q_PROJ);
        check32("slot1 aux3", desc32(desc_aux3_flat, 1), 32'd0);
        check32("slot3 debug flag", desc32(desc_flags_flat, 3) & `QMAP_TENSOR_F_DEBUG_ONLY, `QMAP_TENSOR_F_DEBUG_ONLY);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d QMAP reader mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_dot64_reader decoded the QMAP header and four descriptors.");
        $finish;
    end

endmodule

`default_nettype wire
