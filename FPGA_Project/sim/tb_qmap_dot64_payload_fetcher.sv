`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// End-to-end metadata-to-payload smoke test.
//
// This test first runs qmap_dot64_reader to decode the QMAP header and
// descriptors, then runs qmap_dot64_payload_fetcher using the descriptor
// base_addr/nbytes fields.
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_dot64_payload_fetcher.vvp \
//     FPGA_Project/sim/tb_qmap_dot64_payload_fetcher.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_payload_fetcher.sv
//   vvp FPGA_Project/sim/tb_qmap_dot64_payload_fetcher.vvp

module tb_qmap_dot64_payload_fetcher;

    localparam int ADDR_WIDTH       = 64;
    localparam int DESCRIPTOR_SLOTS = 4;
    localparam int IMAGE_BYTES      = 16'h0600;
    localparam int MEM_WORDS        = IMAGE_BYTES / 4;
    localparam int GROUP_SIZE       = 64;
    localparam int ACT_WIDTH        = 16;
    localparam int WEIGHT_WIDTH     = 4;
    localparam int SCALE_WIDTH      = 16;

    localparam logic [GROUP_SIZE*ACT_WIDTH-1 : 0] EXPECTED_ACTIVATION_FLAT =
        1024'h01f40103ff9df636ff5c009402e9ff06f191fde100a3fd120162ffe70074ff07fe9e04b3066dfdc9fd1cfeebfb3a022dffa3fe230450fca2e91dfcda014806e3043606ee01bd02d4df3e026300de032704c5ffdfffff017afc5afef6fa10011e0669edf1ecdd06e50391f9eb05dffc81ea89fb7709f6fcfcebcbe96b0b5c01b8;

    localparam logic [GROUP_SIZE*WEIGHT_WIDTH-1 : 0] EXPECTED_WEIGHT_PACKED =
        256'h3913f0e3d30d1131efde3002fd1eff1e3503fc4f1012f1ed0fe150ee30023ef1;

    localparam logic [SCALE_WIDTH-1 : 0] EXPECTED_SCALE_Q2_14 = 16'd122;
    localparam logic signed [63 : 0] EXPECTED_PARTIAL_SUM = 64'sd24751;
    localparam logic signed [63 : 0] EXPECTED_SCALED_SUM_Q26 = 64'sd3019622;

    logic clk;
    logic rst_n;

    logic reader_start;
    logic reader_busy;
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
    logic fetcher_busy;
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

    logic use_fetcher;
    logic mem_req_valid;
    logic mem_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_req_addr;
    logic [15 : 0] mem_req_len_bytes;
    logic mem_rsp_valid;
    logic mem_rsp_ready;
    logic [31 : 0] mem_rsp_data;
    logic mem_rsp_last;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic active_read;
    logic [31 : 0] read_index;
    logic [31 : 0] beats_remaining;
    integer cycle_count;
    integer mismatch_count;

    logic [63 : 0] activation_base_addr;
    logic [63 : 0] activation_nbytes;
    logic [63 : 0] weight_base_addr;
    logic [63 : 0] weight_nbytes;
    logic [63 : 0] scale_base_addr;
    logic [63 : 0] scale_nbytes;
    logic [63 : 0] expected_base_addr;
    logic [63 : 0] expected_nbytes;

    assign activation_base_addr = desc_base_addr_flat[0*64 +: 64];
    assign activation_nbytes    = desc_nbytes_flat[0*64 +: 64];
    assign weight_base_addr     = desc_base_addr_flat[1*64 +: 64];
    assign weight_nbytes        = desc_nbytes_flat[1*64 +: 64];
    assign scale_base_addr      = desc_base_addr_flat[2*64 +: 64];
    assign scale_nbytes         = desc_nbytes_flat[2*64 +: 64];
    assign expected_base_addr   = desc_base_addr_flat[3*64 +: 64];
    assign expected_nbytes      = desc_nbytes_flat[3*64 +: 64];

    qmap_dot64_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS)
    ) reader (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(reader_start),
        .i_qmap_base_addr(`QMAP_DOT64_BASE_ADDR),
        .o_busy(reader_busy),
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
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(fetcher_start),
        .i_activation_base_addr(activation_base_addr),
        .i_activation_nbytes(activation_nbytes),
        .i_weight_base_addr(weight_base_addr),
        .i_weight_nbytes(weight_nbytes),
        .i_scale_base_addr(scale_base_addr),
        .i_scale_nbytes(scale_nbytes),
        .i_expected_base_addr(expected_base_addr),
        .i_expected_nbytes(expected_nbytes),
        .o_busy(fetcher_busy),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $readmemh("FPGA_Project/sim/vectors/qmap_dot64_image_words32.hex", mem);
    end

    assign mem_req_valid     = use_fetcher ? fetcher_req_valid : reader_req_valid;
    assign mem_req_addr      = use_fetcher ? fetcher_req_addr : reader_req_addr;
    assign mem_req_len_bytes = use_fetcher ? fetcher_req_len_bytes : reader_req_len_bytes;
    assign mem_rsp_ready     = use_fetcher ? fetcher_rsp_ready : reader_rsp_ready;

    assign reader_req_ready  = use_fetcher ? 1'b0 : mem_req_ready;
    assign fetcher_req_ready = use_fetcher ? mem_req_ready : 1'b0;
    assign reader_rsp_valid  = use_fetcher ? 1'b0 : mem_rsp_valid;
    assign fetcher_rsp_valid = use_fetcher ? mem_rsp_valid : 1'b0;
    assign reader_rsp_data   = mem_rsp_data;
    assign fetcher_rsp_data  = mem_rsp_data;
    assign reader_rsp_last   = mem_rsp_last;
    assign fetcher_rsp_last  = mem_rsp_last;

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

    task check_bit_vector;
        input string name;
        input logic actual_match;
        begin
            $display("%-30s match=%0d", name, actual_match);
            if (!actual_match) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check64;
        input string name;
        input logic signed [63 : 0] actual;
        input logic signed [63 : 0] expected;
        begin
            $display("%-30s actual=%0d expected=%0d", name, actual, expected);
            if (actual !== expected) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        reader_start = 1'b0;
        fetcher_start = 1'b0;
        use_fetcher = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        reader_start = 1'b1;
        @(negedge clk);
        reader_start = 1'b0;

        while ((reader_done != 1'b1) && (cycle_count < 1000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (reader_done != 1'b1) begin
            $display("FAIL: timed out waiting for QMAP descriptor reader");
            $finish(1);
        end

        if (reader_error) begin
            $display("FAIL: QMAP descriptor reader reported error");
            $finish(1);
        end

        @(negedge clk);
        use_fetcher = 1'b1;
        fetcher_start = 1'b1;
        @(negedge clk);
        fetcher_start = 1'b0;

        cycle_count = 0;
        while ((fetcher_done != 1'b1) && (cycle_count < 1000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (fetcher_done != 1'b1) begin
            $display("FAIL: timed out waiting for QMAP payload fetcher");
            $finish(1);
        end

        if (fetcher_error) begin
            $display("FAIL: QMAP payload fetcher reported error");
            $finish(1);
        end

        $display("QMAP dot64 payload fetch checks");
        check_bit_vector("activation_flat", activation_flat === EXPECTED_ACTIVATION_FLAT);
        check_bit_vector("weight_packed", weight_packed === EXPECTED_WEIGHT_PACKED);
        check_bit_vector("scale_q2_14", scale_q2_14 === EXPECTED_SCALE_Q2_14);
        check64("expected_partial_sum", expected_partial_sum, EXPECTED_PARTIAL_SUM);
        check64("expected_scaled_sum_q26", expected_scaled_sum_q26, EXPECTED_SCALED_SUM_Q26);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d QMAP payload mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_dot64_payload_fetcher loaded activation, weight, scale, and expected payloads.");
        $finish;
    end

endmodule

`default_nettype wire
