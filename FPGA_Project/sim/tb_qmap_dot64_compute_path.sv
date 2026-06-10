`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// End-to-end QMAP-backed dot64 compute-path wrapper smoke test.
//
// Unlike the earlier manual sequencing test, this testbench only starts the
// synthesizable qmap_dot64_compute_path wrapper. The wrapper owns the sequence:
//
//   QMAP header/descriptor read
//     -> descriptor-driven payload fetch
//     -> q4_dot_product_64 compute
//     -> compare against the QMAP expected/debug payload
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_dot64_compute_path.vvp \
//     FPGA_Project/sim/tb_qmap_dot64_compute_path.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_payload_fetcher.sv \
//     FPGA_Project/rtl/qmap_dot64_compute_path.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv
//   vvp FPGA_Project/sim/tb_qmap_dot64_compute_path.vvp

module tb_qmap_dot64_compute_path;

    localparam int ADDR_WIDTH    = 64;
    localparam int IMAGE_BYTES   = 16'h0600;
    localparam int MEM_WORDS     = IMAGE_BYTES / 4;
    localparam int GROUP_SIZE    = 64;
    localparam int ACT_WIDTH     = 16;
    localparam int ACT_FRAC      = 12;
    localparam int WEIGHT_WIDTH  = 4;
    localparam int SCALE_WIDTH   = 16;
    localparam int SCALE_FRAC    = 14;
    localparam int PRODUCT_WIDTH = ACT_WIDTH + WEIGHT_WIDTH;
    localparam int PARTIAL_WIDTH = PRODUCT_WIDTH + $clog2(GROUP_SIZE);
    localparam int SCALED_WIDTH  = PARTIAL_WIDTH + SCALE_WIDTH;

    localparam logic signed [63 : 0] EXPECTED_PARTIAL_SUM = 64'sd24751;
    localparam logic signed [63 : 0] EXPECTED_SCALED_SUM_Q26 = 64'sd3019622;

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

    logic signed [63 : 0] partial_sum;
    logic signed [63 : 0] scaled_sum_q26;
    logic signed [63 : 0] expected_partial_sum;
    logic signed [63 : 0] expected_scaled_sum_q26;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic active_read;
    logic [31 : 0] read_index;
    logic [31 : 0] beats_remaining;
    integer cycle_count;

    qmap_dot64_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .GROUP_SIZE(GROUP_SIZE),
        .ACT_WIDTH(ACT_WIDTH),
        .ACT_FRAC(ACT_FRAC),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .SCALE_FRAC(SCALE_FRAC),
        .PRODUCT_WIDTH(PRODUCT_WIDTH),
        .PARTIAL_WIDTH(PARTIAL_WIDTH),
        .SCALED_WIDTH(SCALED_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_qmap_base_addr(`QMAP_DOT64_BASE_ADDR),
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
        .o_partial_sum(partial_sum),
        .o_scaled_sum_q26(scaled_sum_q26),
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

    assign mem_req_ready = !active_read;
    assign mem_rsp_data  = active_read ? mem[read_index] : 32'd0;
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

        while ((done != 1'b1) && (cycle_count < 4000)) begin
            @(posedge clk);
            cycle_count++;
        end

        if (done != 1'b1) begin
            $display("FAIL: timed out waiting for qmap_dot64_compute_path");
            $finish(1);
        end

        #1;
        $display("QMAP-backed dot64 compute wrapper checks");
        $display("  partial_sum        actual=%0d expected_payload=%0d expected_const=%0d",
                 partial_sum, expected_partial_sum, EXPECTED_PARTIAL_SUM);
        $display("  scaled_sum_q26     actual=%0d expected_payload=%0d expected_const=%0d",
                 scaled_sum_q26, expected_scaled_sum_q26, EXPECTED_SCALED_SUM_Q26);
        $display("  compare_match=%0d error=%0d", compare_match, error);

        if (error) begin
            $display("FAIL: qmap_dot64_compute_path reported error");
            $finish(1);
        end

        if (!compare_match) begin
            $display("FAIL: qmap_dot64_compute_path compare did not match");
            $finish(1);
        end

        if ((partial_sum !== EXPECTED_PARTIAL_SUM) ||
            (expected_partial_sum !== EXPECTED_PARTIAL_SUM)) begin
            $display("FAIL: partial_sum mismatch");
            $finish(1);
        end

        if ((scaled_sum_q26 !== EXPECTED_SCALED_SUM_Q26) ||
            (expected_scaled_sum_q26 !== EXPECTED_SCALED_SUM_Q26)) begin
            $display("FAIL: scaled_sum_q26 mismatch");
            $finish(1);
        end

        $display("PASS: qmap_dot64_compute_path wrapper completed the full QMAP-backed dot64 chain.");
        $finish;
    end

endmodule

`default_nettype wire
