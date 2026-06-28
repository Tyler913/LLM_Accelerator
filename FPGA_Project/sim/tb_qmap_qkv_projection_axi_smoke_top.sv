`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

// Integration test for the Vivado-facing QMAP Q/K/V projection AXI top.
//
// This test exercises the same top-level contract that Vivado will see:
//   QMAP packet in AXI memory -> AXI reads -> QKV compute -> AXI writes.
//
// The default plusargs run the compact q=4/k=2/v=2 packet. Larger generated
// packets can be selected without recompiling:
//
//   vvp FPGA_Project/sim/tb_qmap_qkv_projection_axi_smoke_top.vvp \
//     +IMAGE_HEX=artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_medium_image_words32.hex \
//     +EXPECTED_HEX=artifacts/test_vectors/qwen3_0p6b_qmap_v1/layer0_qkv_projection_medium_expected_words32.hex \
//     +IMAGE_WORDS=7168 +EXPECTED_WORDS=32 +TIMEOUT_CYCLES=250000
//
// Run from the repository root:
//
//   iverilog -g2012 -I FPGA_Project/rtl -o FPGA_Project/sim/tb_qmap_qkv_projection_axi_smoke_top.vvp \
//     FPGA_Project/sim/tb_qmap_qkv_projection_axi_smoke_top.sv \
//     FPGA_Project/rtl/qmap_header_reader.sv \
//     FPGA_Project/rtl/qmap_descriptor_reader.sv \
//     FPGA_Project/rtl/qmap_dot64_reader.sv \
//     FPGA_Project/rtl/qmap_qkv_projection_compute_path.sv \
//     FPGA_Project/rtl/q4_dot_product_64.sv \
//     FPGA_Project/rtl/q4_gemv_row_1024.sv \
//     FPGA_Project/rtl/axi4_read_master.sv \
//     FPGA_Project/rtl/axi4_write_master.sv \
//     FPGA_Project/rtl/qmap_qkv_projection_axi_smoke_top.sv
//   vvp FPGA_Project/sim/tb_qmap_qkv_projection_axi_smoke_top.vvp

module tb_qmap_qkv_projection_axi_smoke_top;

    localparam int ADDR_WIDTH       = 64;
    localparam int DATA_WIDTH       = 32;
    localparam int MAX_IMAGE_BYTES    = 32'h0024_0000;
    localparam int MEM_WORDS          = MAX_IMAGE_BYTES / 4;
    localparam int MAX_EXPECTED_WORDS = 4096;
    localparam int DEFAULT_IMAGE_WORDS = 32'h0000_4000 / 4;
    localparam int DEFAULT_EXPECTED_WORDS = 8;
    localparam int DEFAULT_TIMEOUT_CYCLES = 120000;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;
    localparam int DESC_DIM0_WORD    = 12;
    localparam int SLOT_Q_OUT        = 8;
    localparam int SLOT_K_OUT        = 9;
    localparam int SLOT_V_OUT        = 10;

    logic aclk;
    logic aresetn;
    logic start;
    logic clear;

    logic busy;
    logic done_sticky;
    logic error_sticky;
    logic success_sticky;
    logic [3 : 0] status;
    logic [31 : 0] rows_done;
    logic [31 : 0] last_output_q12_12;
    logic [31 : 0] last_row_sum_q26_low32;
    logic [63 : 0] last_row_sum_q26;
    logic axi_read_error_sticky;
    logic axi_write_error_sticky;

    logic [ADDR_WIDTH-1 : 0] M_AXI_AWADDR;
    logic [7 : 0] M_AXI_AWLEN;
    logic [2 : 0] M_AXI_AWSIZE;
    logic [1 : 0] M_AXI_AWBURST;
    logic [2 : 0] M_AXI_AWPROT;
    logic [3 : 0] M_AXI_AWCACHE;
    logic M_AXI_AWVALID;
    logic M_AXI_AWREADY;
    logic [DATA_WIDTH-1 : 0] M_AXI_WDATA;
    logic [(DATA_WIDTH/8)-1 : 0] M_AXI_WSTRB;
    logic M_AXI_WLAST;
    logic M_AXI_WVALID;
    logic M_AXI_WREADY;
    logic [1 : 0] M_AXI_BRESP;
    logic M_AXI_BVALID;
    logic M_AXI_BREADY;
    logic [ADDR_WIDTH-1 : 0] M_AXI_ARADDR;
    logic [7 : 0] M_AXI_ARLEN;
    logic [2 : 0] M_AXI_ARSIZE;
    logic [1 : 0] M_AXI_ARBURST;
    logic [2 : 0] M_AXI_ARPROT;
    logic [3 : 0] M_AXI_ARCACHE;
    logic M_AXI_ARVALID;
    logic M_AXI_ARREADY;
    logic [DATA_WIDTH-1 : 0] M_AXI_RDATA;
    logic [1 : 0] M_AXI_RRESP;
    logic M_AXI_RLAST;
    logic M_AXI_RVALID;
    logic M_AXI_RREADY;

    logic [31 : 0] mem [0 : MEM_WORDS-1];
    logic [31 : 0] expected [0 : MAX_EXPECTED_WORDS-1];

    logic axi_read_active;
    logic [31 : 0] axi_read_index;
    logic [8 : 0] axi_read_beats_left;

    logic axi_write_active;
    logic [31 : 0] axi_write_index;
    logic [8 : 0] axi_write_beats_left;

    integer cycle_count;
    integer read_burst_count;
    integer write_burst_count;
    integer write_word_count;
    integer mismatch_count;
    integer expected_index;
    integer image_word_count;
    integer expected_word_count;
    integer expected_read_burst_count;
    integer timeout_cycles;
    integer row;
    integer row_count;
    integer output_word_index;
    logic [63 : 0] output_base_addr;
    reg [8*256-1 : 0] image_hex_path;
    reg [8*256-1 : 0] expected_hex_path;

    qmap_qkv_projection_axi_smoke_top dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .i_start(start),
        .i_clear(clear),
        .o_busy(busy),
        .o_done_sticky(done_sticky),
        .o_error_sticky(error_sticky),
        .o_success_sticky(success_sticky),
        .o_status(status),
        .o_rows_done(rows_done),
        .o_last_output_q12_12(last_output_q12_12),
        .o_last_row_sum_q26_low32(last_row_sum_q26_low32),
        .o_last_row_sum_q26(last_row_sum_q26),
        .o_axi_read_error_sticky(axi_read_error_sticky),
        .o_axi_write_error_sticky(axi_write_error_sticky),
        .M_AXI_AWADDR(M_AXI_AWADDR),
        .M_AXI_AWLEN(M_AXI_AWLEN),
        .M_AXI_AWSIZE(M_AXI_AWSIZE),
        .M_AXI_AWBURST(M_AXI_AWBURST),
        .M_AXI_AWPROT(M_AXI_AWPROT),
        .M_AXI_AWCACHE(M_AXI_AWCACHE),
        .M_AXI_AWVALID(M_AXI_AWVALID),
        .M_AXI_AWREADY(M_AXI_AWREADY),
        .M_AXI_WDATA(M_AXI_WDATA),
        .M_AXI_WSTRB(M_AXI_WSTRB),
        .M_AXI_WLAST(M_AXI_WLAST),
        .M_AXI_WVALID(M_AXI_WVALID),
        .M_AXI_WREADY(M_AXI_WREADY),
        .M_AXI_BRESP(M_AXI_BRESP),
        .M_AXI_BVALID(M_AXI_BVALID),
        .M_AXI_BREADY(M_AXI_BREADY),
        .M_AXI_ARADDR(M_AXI_ARADDR),
        .M_AXI_ARLEN(M_AXI_ARLEN),
        .M_AXI_ARSIZE(M_AXI_ARSIZE),
        .M_AXI_ARBURST(M_AXI_ARBURST),
        .M_AXI_ARPROT(M_AXI_ARPROT),
        .M_AXI_ARCACHE(M_AXI_ARCACHE),
        .M_AXI_ARVALID(M_AXI_ARVALID),
        .M_AXI_ARREADY(M_AXI_ARREADY),
        .M_AXI_RDATA(M_AXI_RDATA),
        .M_AXI_RRESP(M_AXI_RRESP),
        .M_AXI_RLAST(M_AXI_RLAST),
        .M_AXI_RVALID(M_AXI_RVALID),
        .M_AXI_RREADY(M_AXI_RREADY)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    initial begin : load_vectors
        image_hex_path = "FPGA_Project/sim/vectors/qmap_qkv_projection_image_words32.hex";
        expected_hex_path = "FPGA_Project/sim/vectors/qmap_qkv_projection_expected_words32.hex";
        image_word_count = DEFAULT_IMAGE_WORDS;
        expected_word_count = DEFAULT_EXPECTED_WORDS;
        expected_read_burst_count = 0;
        timeout_cycles = DEFAULT_TIMEOUT_CYCLES;

        if (!$value$plusargs("IMAGE_HEX=%s", image_hex_path)) begin
        end
        if (!$value$plusargs("EXPECTED_HEX=%s", expected_hex_path)) begin
        end
        if (!$value$plusargs("IMAGE_WORDS=%d", image_word_count)) begin
        end
        if (!$value$plusargs("EXPECTED_WORDS=%d", expected_word_count)) begin
        end
        if (!$value$plusargs("EXPECTED_READ_BURSTS=%d", expected_read_burst_count)) begin
        end
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
        end

        if ((image_word_count < 1) || (image_word_count > MEM_WORDS)) begin
            $display("FAIL: IMAGE_WORDS=%0d is outside supported range 1..%0d",
                     image_word_count,
                     MEM_WORDS);
            $finish(1);
        end

        if ((expected_word_count < 1) || (expected_word_count > MAX_EXPECTED_WORDS)) begin
            $display("FAIL: EXPECTED_WORDS=%0d is outside supported range 1..%0d",
                     expected_word_count,
                     MAX_EXPECTED_WORDS);
            $finish(1);
        end

        if (expected_read_burst_count == 0) begin
            expected_read_burst_count = 17 + (expected_word_count * 2);
        end

        for (int mem_init_index = 0 ; mem_init_index < MEM_WORDS ; mem_init_index = mem_init_index + 1) begin
            mem[mem_init_index] = 32'd0;
        end
        for (int exp_init_index = 0 ; exp_init_index < MAX_EXPECTED_WORDS ; exp_init_index = exp_init_index + 1) begin
            expected[exp_init_index] = 32'd0;
        end

        $readmemh(image_hex_path, mem, 0, image_word_count - 1);
        $readmemh(expected_hex_path, expected, 0, expected_word_count - 1);
    end

    assign M_AXI_ARREADY = !axi_read_active;
    assign M_AXI_RRESP   = 2'b00;

    always @(posedge aclk) begin
        if (!aresetn) begin
            axi_read_active     <= 1'b0;
            axi_read_index      <= 32'd0;
            axi_read_beats_left <= 9'd0;
            M_AXI_RVALID        <= 1'b0;
            M_AXI_RDATA         <= 32'd0;
            M_AXI_RLAST         <= 1'b0;
            read_burst_count    <= 0;
        end else begin
            if (M_AXI_ARVALID && M_AXI_ARREADY) begin
                if ((M_AXI_ARSIZE != 3'd2) || (M_AXI_ARBURST != 2'b01)) begin
                    $display("FAIL: unsupported AXI read attributes");
                    $finish(1);
                end
                if ((M_AXI_ARADDR < `QMAP_QKV_BASE_ADDR) ||
                    (((M_AXI_ARADDR - `QMAP_QKV_BASE_ADDR) >> 2) >= image_word_count)) begin
                    $display("FAIL: AXI read address out of packet range: 0x%016h", M_AXI_ARADDR);
                    $finish(1);
                end

                axi_read_active     <= 1'b1;
                axi_read_index      <= (M_AXI_ARADDR - `QMAP_QKV_BASE_ADDR) >> 2;
                axi_read_beats_left <= {1'b0, M_AXI_ARLEN} + 1'b1;
                M_AXI_RVALID        <= 1'b1;
                M_AXI_RDATA         <= mem[(M_AXI_ARADDR - `QMAP_QKV_BASE_ADDR) >> 2];
                M_AXI_RLAST         <= (M_AXI_ARLEN == 8'd0);
                read_burst_count    <= read_burst_count + 1;
            end else if (M_AXI_RVALID && M_AXI_RREADY) begin
                if (axi_read_beats_left == 9'd1) begin
                    axi_read_active     <= 1'b0;
                    axi_read_beats_left <= 9'd0;
                    M_AXI_RVALID        <= 1'b0;
                    M_AXI_RLAST         <= 1'b0;
                end else begin
                    axi_read_index      <= axi_read_index + 1'b1;
                    axi_read_beats_left <= axi_read_beats_left - 1'b1;
                    M_AXI_RDATA         <= mem[axi_read_index + 1'b1];
                    M_AXI_RLAST         <= (axi_read_beats_left == 9'd2);
                end
            end
        end
    end

    assign M_AXI_AWREADY = !axi_write_active && !M_AXI_BVALID;
    assign M_AXI_WREADY  = axi_write_active;
    assign M_AXI_BRESP   = 2'b00;

    always @(posedge aclk) begin
        if (!aresetn) begin
            axi_write_active     <= 1'b0;
            axi_write_index      <= 32'd0;
            axi_write_beats_left <= 9'd0;
            M_AXI_BVALID         <= 1'b0;
            write_burst_count    <= 0;
            write_word_count     <= 0;
        end else begin
            if (M_AXI_AWVALID && M_AXI_AWREADY) begin
                if ((M_AXI_AWSIZE != 3'd2) || (M_AXI_AWBURST != 2'b01)) begin
                    $display("FAIL: unsupported AXI write attributes");
                    $finish(1);
                end
                if ((M_AXI_AWADDR < `QMAP_QKV_BASE_ADDR) ||
                    (((M_AXI_AWADDR - `QMAP_QKV_BASE_ADDR) >> 2) >= image_word_count)) begin
                    $display("FAIL: AXI write address out of packet range: 0x%016h", M_AXI_AWADDR);
                    $finish(1);
                end

                axi_write_active     <= 1'b1;
                axi_write_index      <= (M_AXI_AWADDR - `QMAP_QKV_BASE_ADDR) >> 2;
                axi_write_beats_left <= {1'b0, M_AXI_AWLEN} + 1'b1;
                write_burst_count    <= write_burst_count + 1;
            end

            if (M_AXI_WVALID && M_AXI_WREADY) begin
                if (M_AXI_WSTRB != 4'hF) begin
                    $display("FAIL: expected full-word AXI write strobe, got 0x%0h", M_AXI_WSTRB);
                    $finish(1);
                end
                if (M_AXI_WLAST != (axi_write_beats_left == 9'd1)) begin
                    $display("FAIL: AXI WLAST mismatch");
                    $finish(1);
                end

                mem[axi_write_index] <= M_AXI_WDATA;
                write_word_count <= write_word_count + 1;

                if (axi_write_beats_left == 9'd1) begin
                    axi_write_active     <= 1'b0;
                    axi_write_beats_left <= 9'd0;
                    M_AXI_BVALID         <= 1'b1;
                end else begin
                    axi_write_index      <= axi_write_index + 1'b1;
                    axi_write_beats_left <= axi_write_beats_left - 1'b1;
                end
            end

            if (M_AXI_BVALID && M_AXI_BREADY) begin
                M_AXI_BVALID <= 1'b0;
            end
        end
    end

    function automatic integer descriptor_word_index(input integer slot, input integer word_offset);
        begin
            descriptor_word_index =
                DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    task automatic check_output_slot(input integer slot);
        begin
            output_base_addr = {
                mem[descriptor_word_index(slot, DESC_BASE_HI_WORD)],
                mem[descriptor_word_index(slot, DESC_BASE_LO_WORD)]
            };
            output_word_index = (output_base_addr - `QMAP_QKV_BASE_ADDR) >> 2;
            row_count = mem[descriptor_word_index(slot, DESC_DIM0_WORD)];

            for (row = 0 ; row < row_count ; row = row + 1) begin
                if (mem[output_word_index + row] !== expected[expected_index]) begin
                    $display(
                        "FAIL: output mismatch slot=%0d row=%0d actual=0x%08h expected=0x%08h",
                        slot,
                        row,
                        mem[output_word_index + row],
                        expected[expected_index]
                    );
                    mismatch_count = mismatch_count + 1;
                end
                expected_index = expected_index + 1;
            end
        end
    endtask

    initial begin
        aresetn = 1'b0;
        start = 1'b0;
        clear = 1'b0;
        cycle_count = 0;
        mismatch_count = 0;
        expected_index = 0;

        repeat (4) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;

        @(negedge aclk);
        start = 1'b1;
        @(negedge aclk);
        start = 1'b0;

        while ((done_sticky != 1'b1) && (cycle_count < timeout_cycles)) begin
            @(posedge aclk);
            cycle_count++;
        end

        if (done_sticky != 1'b1) begin
            $display("FAIL: timed out waiting for qmap_qkv_projection_axi_smoke_top after %0d cycles",
                     timeout_cycles);
            $finish(1);
        end

        #1;
        $display("QMAP QKV AXI smoke top checks");
        $display("  image_hex                = %0s", image_hex_path);
        $display("  expected_hex             = %0s", expected_hex_path);
        $display("  image_words              = %0d", image_word_count);
        $display("  expected_words           = %0d", expected_word_count);
        $display("  AXI read bursts          = %0d", read_burst_count);
        $display("  AXI write bursts         = %0d", write_burst_count);
        $display("  AXI write words          = %0d", write_word_count);
        $display("  status                   = 0x%0h", status);
        $display("  rows_done                = %0d", rows_done);
        $display("  last_row_sum_q26_low32   = 0x%08h", last_row_sum_q26_low32);
        $display("  last_output_q12_12       = 0x%08h", last_output_q12_12);

        if (busy || error_sticky || !success_sticky ||
            axi_read_error_sticky || axi_write_error_sticky) begin
            $display("FAIL: unexpected QKV top status bits");
            $finish(1);
        end

        if (status !== 4'b1010) begin
            $display("FAIL: status expected 0b1010 {success,error,done,busy}");
            $finish(1);
        end

        if (read_burst_count != expected_read_burst_count) begin
            $display("FAIL: read_burst_count mismatch actual=%0d expected=%0d",
                     read_burst_count,
                     expected_read_burst_count);
            $finish(1);
        end

        if (write_burst_count != expected_word_count) begin
            $display("FAIL: write_burst_count mismatch actual=%0d expected=%0d",
                     write_burst_count,
                     expected_word_count);
            $finish(1);
        end

        if (write_word_count != expected_word_count) begin
            $display("FAIL: write_word_count mismatch actual=%0d expected=%0d",
                     write_word_count,
                     expected_word_count);
            $finish(1);
        end

        if (rows_done != expected_word_count) begin
            $display("FAIL: rows_done mismatch actual=%0d expected=%0d", rows_done, expected_word_count);
            $finish(1);
        end

        check_output_slot(SLOT_Q_OUT);
        check_output_slot(SLOT_K_OUT);
        check_output_slot(SLOT_V_OUT);

        if (expected_index != expected_word_count) begin
            $display("FAIL: expected index mismatch actual=%0d expected=%0d",
                     expected_index,
                     expected_word_count);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: %0d QKV AXI output mismatch(es)", mismatch_count);
            $finish(1);
        end

        @(negedge aclk);
        clear = 1'b1;
        @(negedge aclk);
        clear = 1'b0;
        #1;

        if (status !== 4'b0000) begin
            $display("FAIL: clear did not reset sticky status");
            $finish(1);
        end

        $display("PASS: qmap_qkv_projection_axi_smoke_top completed AXI-backed Q/K/V write-back.");
        $finish;
    end

endmodule

`default_nettype wire
