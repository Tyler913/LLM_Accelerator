`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// End-to-end QMAP-backed row1024 compute-path wrapper smoke test.
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_row1024_compute_path.vvp \
//     FPGA_Project/sim/tb_qmap_row1024_compute_path.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_row1024_payload_fetcher.sv \
//     FPGA_Project/rtl/qmap_row1024_compute_path.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv \
//     FPGA_Project/rtl/q4_gemv_row_1024.sv
//   vvp FPGA_Project/sim/tb_qmap_row1024_compute_path.vvp

module tb_qmap_row1024_compute_path;

    localparam int ADDR_WIDTH     = 64;
    localparam int IMAGE_BYTES    = 16'h1000;
    localparam int MEM_WORDS      = IMAGE_BYTES / 4;
    localparam int INPUT_SIZE     = 1024;
    localparam int GROUP_SIZE     = 64;
    localparam int GROUP_COUNT    = INPUT_SIZE / GROUP_SIZE;
    localparam int ACT_WIDTH      = 16;
    localparam int ACT_FRAC       = 12;
    localparam int WEIGHT_WIDTH   = 4;
    localparam int SCALE_WIDTH    = 16;
    localparam int SCALE_FRAC     = 14;
    localparam int PARTIAL_WIDTH  = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH   = PARTIAL_WIDTH + SCALE_WIDTH;
    localparam int ROW_ACC_WIDTH  = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2;

    localparam logic signed [63 : 0] EXPECTED_ROW_SUM_Q26 = -64'sd3482169;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic compare_match;

    logic mem_req_valid;
    logic mem_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_req_addr;
    logic [15 : 0] mem_req_len_bytes;
    logic mem_rsp_valid;
    logic mem_rsp_ready;
    logic [31 : 0] mem_rsp_data;
    logic mem_rsp_last;

    logic signed [63 : 0] row_sum_q26;
    logic signed [63 : 0] expected_row_sum_q26;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic active_read;
    logic [31 : 0] read_index;
    logic [31 : 0] beats_remaining;
    integer cycle_count;
    integer request_count;

    qmap_row1024_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(`QMAP_ROW1024_BASE_ADDR),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_compare_match(compare_match),
        .o_mem_req_valid(mem_req_valid),
        .i_mem_req_ready(mem_req_ready),
        .o_mem_req_addr(mem_req_addr),
        .o_mem_req_len_bytes(mem_req_len_bytes),
        .i_mem_rsp_valid(mem_rsp_valid),
        .o_mem_rsp_ready(mem_rsp_ready),
        .i_mem_rsp_data(mem_rsp_data),
        .i_mem_rsp_last(mem_rsp_last),
        .o_row_sum_q26(row_sum_q26),
        .o_expected_row_sum_q26(expected_row_sum_q26)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        $readmemh("FPGA_Project/sim/vectors/qmap_row1024_image_words32.hex", mem);
    end

    assign mem_req_ready = !active_read;
    assign mem_rsp_data  = active_read ? mem[read_index] : 32'd0;
    assign mem_rsp_last  = active_read && (beats_remaining == 32'd1);

    always @(posedge clk) begin
        if (!rst_n) begin
            active_read     <= 1'b0;
            read_index      <= 32'd0;
            beats_remaining <= 32'd0;
            mem_rsp_valid   <= 1'b0;
            request_count   <= 0;
        end else begin
            if (mem_req_valid && mem_req_ready) begin
                active_read     <= 1'b1;
                read_index      <= (mem_req_addr - `QMAP_ROW1024_BASE_ADDR) >> 2;
                beats_remaining <= (mem_req_len_bytes + 16'd3) >> 2;
                mem_rsp_valid   <= 1'b1;
                request_count   <= request_count + 1;
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

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        cycle_count = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while ((done != 1'b1) && (cycle_count < 10000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for qmap_row1024_compute_path");
            $finish(1);
        end

        #1;
        $display("QMAP-backed row1024 compute wrapper checks");
        $display("  memory requests       = %0d", request_count);
        $display("  row_sum_q26           actual=%0d expected_payload=%0d expected_const=%0d",
                 row_sum_q26, expected_row_sum_q26, EXPECTED_ROW_SUM_Q26);
        $display("  compare_match=%0d error=%0d", compare_match, error);

        if (error) begin
            $display("FAIL: qmap_row1024_compute_path reported error");
            $finish(1);
        end

        if (!compare_match) begin
            $display("FAIL: qmap_row1024_compute_path compare did not match");
            $finish(1);
        end

        if ((row_sum_q26 !== EXPECTED_ROW_SUM_Q26) ||
            (expected_row_sum_q26 !== EXPECTED_ROW_SUM_Q26)) begin
            $display("FAIL: row_sum_q26 mismatch");
            $finish(1);
        end

        $display("PASS: qmap_row1024_compute_path completed the full QMAP-backed row1024 chain.");
        $finish;
    end

endmodule

`default_nettype wire
