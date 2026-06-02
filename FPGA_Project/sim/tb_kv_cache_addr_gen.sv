`timescale 1ns/1ps
`default_nettype none

// Smoke test for kv_cache_addr_gen.
//
// Run from the repository root:
//
//   iverilog -g2012 -o FPGA_Project/sim/tb_kv_cache_addr_gen.vvp FPGA_Project/sim/tb_kv_cache_addr_gen.sv FPGA_Project/rtl/kv_cache_addr_gen.sv
//   vvp FPGA_Project/sim/tb_kv_cache_addr_gen.vvp

module tb_kv_cache_addr_gen;

    localparam int ADDR_WIDTH = 64;
    localparam logic [ADDR_WIDTH-1 : 0] BASE_ADDR = 64'h0000_0000_1410_0000;

    logic [ADDR_WIDTH-1 : 0] base_addr;

    logic [4 : 0] layer256;
    logic kind256;
    logic [2 : 0] head256;
    logic [7 : 0] position256;
    logic [6 : 0] dim256;
    logic valid256;
    logic [ADDR_WIDTH-1 : 0] offset256;
    logic [ADDR_WIDTH-1 : 0] addr256;

    logic [4 : 0] layer512;
    logic kind512;
    logic [2 : 0] head512;
    logic [8 : 0] position512;
    logic [6 : 0] dim512;
    logic valid512;
    logic [ADDR_WIDTH-1 : 0] offset512;
    logic [ADDR_WIDTH-1 : 0] addr512;

    integer mismatch_count;

    kv_cache_addr_gen #(.ADDR_WIDTH(ADDR_WIDTH), .MAX_CONTEXT(256), .ELEMENT_BYTES(4)) dut256 (.i_base_addr(base_addr), .i_layer_id(layer256), .i_kv_kind(kind256), .i_head_id(head256), .i_position(position256), .i_dim(dim256), .o_valid(valid256), .o_offset_bytes(offset256), .o_byte_addr(addr256));
    kv_cache_addr_gen #(.ADDR_WIDTH(ADDR_WIDTH), .MAX_CONTEXT(512), .ELEMENT_BYTES(4)) dut512 (.i_base_addr(base_addr), .i_layer_id(layer512), .i_kv_kind(kind512), .i_head_id(head512), .i_position(position512), .i_dim(dim512), .o_valid(valid512), .o_offset_bytes(offset512), .o_byte_addr(addr512));

    task check256;
        input string name;
        input logic [4 : 0] layer;
        input logic kind;
        input logic [2 : 0] head;
        input logic [7 : 0] position;
        input logic [6 : 0] dim;
        input logic [ADDR_WIDTH-1 : 0] expected_addr;
        input logic expected_valid;
        begin
            layer256 = layer;
            kind256 = kind;
            head256 = head;
            position256 = position;
            dim256 = dim;
            #1;
            $display("ctx256 %-28s addr = 0x%016h, expected = 0x%016h, valid = %0d", name, addr256, expected_addr, valid256);
            if ((addr256 !== expected_addr) || (valid256 !== expected_valid)) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check512;
        input string name;
        input logic [4 : 0] layer;
        input logic kind;
        input logic [2 : 0] head;
        input logic [8 : 0] position;
        input logic [6 : 0] dim;
        input logic [ADDR_WIDTH-1 : 0] expected_addr;
        input logic expected_valid;
        begin
            layer512 = layer;
            kind512 = kind;
            head512 = head;
            position512 = position;
            dim512 = dim;
            #1;
            $display("ctx512 %-28s addr = 0x%016h, expected = 0x%016h, valid = %0d", name, addr512, expected_addr, valid512);
            if ((addr512 !== expected_addr) || (valid512 !== expected_valid)) begin
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    initial begin
        base_addr = BASE_ADDR;
        layer256 = 'd0;
        kind256 = 1'b0;
        head256 = 'd0;
        position256 = 'd0;
        dim256 = 'd0;
        layer512 = 'd0;
        kind512 = 1'b0;
        head512 = 'd0;
        position512 = 'd0;
        dim512 = 'd0;
        mismatch_count = 0;

        check256("L0 K h0 p0 d0", 5'd0, 1'b0, 3'd0, 8'd0, 7'd0, 64'h0000_0000_1410_0000, 1'b1);
        check256("L0 K h0 p0 d1", 5'd0, 1'b0, 3'd0, 8'd0, 7'd1, 64'h0000_0000_1410_0004, 1'b1);
        check256("L0 K h0 p1 d0", 5'd0, 1'b0, 3'd0, 8'd1, 7'd0, 64'h0000_0000_1410_0200, 1'b1);
        check256("L0 K h1 p0 d0", 5'd0, 1'b0, 3'd1, 8'd0, 7'd0, 64'h0000_0000_1412_0000, 1'b1);
        check256("L0 V h0 p0 d0", 5'd0, 1'b1, 3'd0, 8'd0, 7'd0, 64'h0000_0000_1420_0000, 1'b1);
        check256("L1 K h0 p0 d0", 5'd1, 1'b0, 3'd0, 8'd0, 7'd0, 64'h0000_0000_1430_0000, 1'b1);
        check256("L2 V h3 p5 d7", 5'd2, 1'b1, 3'd3, 8'd5, 7'd7, 64'h0000_0000_1466_0a1c, 1'b1);
        check256("invalid layer31", 5'd31, 1'b0, 3'd0, 8'd0, 7'd0, 64'h0000_0000_17f0_0000, 1'b0);

        check512("L0 K h0 p0 d0", 5'd0, 1'b0, 3'd0, 9'd0, 7'd0, 64'h0000_0000_1410_0000, 1'b1);
        check512("L1 V h7 p511 d127", 5'd1, 1'b1, 3'd7, 9'd511, 7'd127, 64'h0000_0000_148f_fffc, 1'b1);
        check512("L27 V h7 p511 d127", 5'd27, 1'b1, 3'd7, 9'd511, 7'd127, 64'h0000_0000_1b0f_fffc, 1'b1);

        if (mismatch_count != 0) begin
            $display("FAIL: %0d kv_cache_addr_gen mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: kv_cache_addr_gen smoke vectors matched.");
        $finish;
    end

endmodule

`default_nettype wire
