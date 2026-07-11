`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_axil_top;
    localparam int ADDR_WIDTH = 64;
    localparam int MAX_LAYERS = 28;
    localparam int MAX_CONTEXT = 256;
    localparam int LAYER_INDEX_WIDTH = $clog2(MAX_LAYERS);
    localparam int LAYER_COUNT_WIDTH = $clog2(MAX_LAYERS + 1);
    localparam int POSITION_WIDTH = $clog2(MAX_CONTEXT);
    localparam logic [1 : 0] AXI_RESP_OKAY = 2'b00;

    localparam logic [11 : 0] REG_CTRL               = 12'h000;
    localparam logic [11 : 0] REG_STATUS             = 12'h004;
    localparam logic [11 : 0] REG_LAYER_START        = 12'h008;
    localparam logic [11 : 0] REG_LAYER_COUNT        = 12'h00C;
    localparam logic [11 : 0] REG_POSITION           = 12'h010;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_LO    = 12'h020;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_HI    = 12'h024;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_LO   = 12'h028;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_HI   = 12'h02C;
    localparam logic [11 : 0] REG_KV_CACHE_LO        = 12'h030;
    localparam logic [11 : 0] REG_KV_CACHE_HI        = 12'h034;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_LO = 12'h038;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_HI = 12'h03C;
    localparam logic [11 : 0] REG_LAYERS             = 12'h06C;
    localparam logic [11 : 0] REG_LAYER_ERROR_MASK   = 12'h074;
    localparam logic [11 : 0] REG_MEM_RD_REQS        = 12'h090;
    localparam logic [11 : 0] REG_MEM_RD_WORDS       = 12'h094;
    localparam logic [11 : 0] REG_MEM_WR_REQS        = 12'h098;
    localparam logic [11 : 0] REG_MEM_WR_WORDS       = 12'h09C;

    logic clk;
    logic rst_n;

    logic [11 : 0] s_axi_awaddr;
    logic [2 : 0]  s_axi_awprot;
    logic          s_axi_awvalid;
    logic          s_axi_awready;
    logic [31 : 0] s_axi_wdata;
    logic [3 : 0]  s_axi_wstrb;
    logic          s_axi_wvalid;
    logic          s_axi_wready;
    logic [1 : 0]  s_axi_bresp;
    logic          s_axi_bvalid;
    logic          s_axi_bready;
    logic [11 : 0] s_axi_araddr;
    logic [2 : 0]  s_axi_arprot;
    logic          s_axi_arvalid;
    logic          s_axi_arready;
    logic [31 : 0] s_axi_rdata;
    logic [1 : 0]  s_axi_rresp;
    logic          s_axi_rvalid;
    logic          s_axi_rready;

    logic axil_busy;
    logic busy;
    logic done;
    logic error;
    logic [7 : 0] state_debug;
    logic [7 : 0] phase_debug;
    logic scheduler_done_pulse;
    logic tail_start_pulse;
    logic tail_done_pulse;
    logic tail_active;
    logic [LAYER_INDEX_WIDTH-1:0] active_layer_index;
    logic [LAYER_COUNT_WIDTH-1:0] layers_started;
    logic [LAYER_COUNT_WIDTH-1:0] layers_completed;
    logic [MAX_LAYERS-1:0] layer_done_mask;
    logic [MAX_LAYERS-1:0] layer_error_mask;
    logic [ADDR_WIDTH-1 : 0] last_layer_output_base_addr;
    logic [ADDR_WIDTH-1 : 0] tail_effective_final_hidden_base_addr;
    logic [31 : 0] tail_best_token_id;
    logic signed [55 : 0] tail_best_score_q26;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [31 : 0] mem_rd_rsp_data;
    logic mem_rd_rsp_last;
    logic mem_wr_req_valid;
    logic mem_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [31 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_ready;
    logic mem_wr_data_last;
    logic mem_wr_done;
    logic mem_wr_error;

    integer fail_count;
    integer timeout_count;
    logic [31 : 0] read_data;

    always #5 clk = ~clk;

    qmap_one_token_axil_top dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_s_axi_awaddr(s_axi_awaddr),
        .i_s_axi_awprot(s_axi_awprot),
        .i_s_axi_awvalid(s_axi_awvalid),
        .o_s_axi_awready(s_axi_awready),
        .i_s_axi_wdata(s_axi_wdata),
        .i_s_axi_wstrb(s_axi_wstrb),
        .i_s_axi_wvalid(s_axi_wvalid),
        .o_s_axi_wready(s_axi_wready),
        .o_s_axi_bresp(s_axi_bresp),
        .o_s_axi_bvalid(s_axi_bvalid),
        .i_s_axi_bready(s_axi_bready),
        .i_s_axi_araddr(s_axi_araddr),
        .i_s_axi_arprot(s_axi_arprot),
        .i_s_axi_arvalid(s_axi_arvalid),
        .o_s_axi_arready(s_axi_arready),
        .o_s_axi_rdata(s_axi_rdata),
        .o_s_axi_rresp(s_axi_rresp),
        .o_s_axi_rvalid(s_axi_rvalid),
        .i_s_axi_rready(s_axi_rready),
        .o_axil_busy(axil_busy),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_phase_debug(phase_debug),
        .o_scheduler_done_pulse(scheduler_done_pulse),
        .o_tail_start_pulse(tail_start_pulse),
        .o_tail_done_pulse(tail_done_pulse),
        .o_tail_active(tail_active),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_last_layer_output_base_addr(last_layer_output_base_addr),
        .o_tail_effective_final_hidden_base_addr(tail_effective_final_hidden_base_addr),
        .o_tail_best_token_id(tail_best_token_id),
        .o_tail_best_score_q26(tail_best_score_q26),
        .o_mem_read_burst_count(mem_read_burst_count),
        .o_mem_read_word_count(mem_read_word_count),
        .o_mem_write_req_count(mem_write_req_count),
        .o_mem_write_word_count(mem_write_word_count),
        .o_mem_rd_req_valid(mem_rd_req_valid),
        .i_mem_rd_req_ready(mem_rd_req_ready),
        .o_mem_rd_req_addr(mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(mem_rd_rsp_valid),
        .o_mem_rd_rsp_ready(mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(mem_rd_rsp_data),
        .i_mem_rd_rsp_last(mem_rd_rsp_last),
        .o_mem_wr_req_valid(mem_wr_req_valid),
        .i_mem_wr_req_ready(mem_wr_req_ready),
        .o_mem_wr_req_addr(mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(mem_wr_req_len_bytes),
        .o_mem_wr_data(mem_wr_data),
        .o_mem_wr_data_valid(mem_wr_data_valid),
        .i_mem_wr_data_ready(mem_wr_data_ready),
        .o_mem_wr_data_last(mem_wr_data_last),
        .i_mem_wr_done(mem_wr_done),
        .i_mem_wr_error(mem_wr_error)
    );

    task check;
        input logic condition;
        input string message;
        begin
            if (!condition) begin
                $display("FAIL: %s", message);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task wait_clk;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task axil_write;
        input logic [11 : 0] addr;
        input logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_awaddr  = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata   = data;
            s_axi_wstrb   = 4'hF;
            s_axi_wvalid  = 1'b1;
            #1;
            while (!(s_axi_awready && s_axi_wready)) begin
                @(negedge clk);
                #1;
            end
            wait_clk();
            @(negedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid  = 1'b0;
            s_axi_awaddr  = 12'd0;
            s_axi_wdata   = 32'd0;

            while (!s_axi_bvalid) begin
                wait_clk();
            end
            check(s_axi_bresp == AXI_RESP_OKAY, "AXI write BRESP should be OKAY");
            @(negedge clk);
            s_axi_bready = 1'b1;
            wait_clk();
            check(!s_axi_bvalid, "AXI write BVALID should clear after BREADY");
            @(negedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task axil_read;
        input logic [11 : 0] addr;
        output logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_araddr  = addr;
            s_axi_arvalid = 1'b1;
            #1;
            while (!s_axi_arready) begin
                @(negedge clk);
                #1;
            end
            wait_clk();
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            s_axi_araddr  = 12'd0;

            while (!s_axi_rvalid) begin
                wait_clk();
            end
            data = s_axi_rdata;
            check(s_axi_rresp == AXI_RESP_OKAY, "AXI read RRESP should be OKAY");
            @(negedge clk);
            s_axi_rready = 1'b1;
            wait_clk();
            check(!s_axi_rvalid, "AXI read RVALID should clear after RREADY");
            @(negedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    task write_addr64;
        input logic [11 : 0] lo_addr;
        input logic [11 : 0] hi_addr;
        input logic [63 : 0] value;
        begin
            axil_write(lo_addr, value[31 : 0]);
            axil_write(hi_addr, value[63 : 32]);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        s_axi_awaddr = 12'd0;
        s_axi_awprot = 3'b000;
        s_axi_awvalid = 1'b0;
        s_axi_wdata = 32'd0;
        s_axi_wstrb = 4'hF;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b0;
        s_axi_araddr = 12'd0;
        s_axi_arprot = 3'b000;
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b0;
        mem_rd_req_ready = 1'b1;
        mem_rd_rsp_valid = 1'b0;
        mem_rd_rsp_data = 32'd0;
        mem_rd_rsp_last = 1'b0;
        mem_wr_req_ready = 1'b1;
        mem_wr_data_ready = 1'b1;
        mem_wr_done = 1'b0;
        mem_wr_error = 1'b0;
        fail_count = 0;
        timeout_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // Program a deliberately invalid layer request through the AXI-Lite
        // wrapper. The path should enter qmap_one_token_top and then exit via
        // the scheduler validation path without issuing memory traffic.
        axil_write(REG_LAYER_START, 32'd0);
        axil_write(REG_LAYER_COUNT, 32'd0);
        axil_write(REG_POSITION, 32'd4);
        write_addr64(REG_INPUT_HIDDEN_LO, REG_INPUT_HIDDEN_HI, 64'h0000_0004_0509_2540);
        write_addr64(REG_OUTPUT_HIDDEN_LO, REG_OUTPUT_HIDDEN_HI, 64'h0000_0004_1509_2540);
        write_addr64(REG_KV_CACHE_LO, REG_KV_CACHE_HI, 64'h0000_0004_1410_0000);
        write_addr64(REG_FINAL_TAIL_QMAP_LO, REG_FINAL_TAIL_QMAP_HI, 64'h0000_0004_0501_0000);

        axil_write(REG_CTRL, 32'd1);

        while (!done && timeout_count < 200) begin
            @(posedge clk);
            timeout_count = timeout_count + 1;
            check(!mem_rd_req_valid, "invalid AXI-Lite wrapper run must not issue read requests");
            check(!mem_wr_req_valid, "invalid AXI-Lite wrapper run must not issue write requests");
            check(!mem_wr_data_valid, "invalid AXI-Lite wrapper run must not issue write data");
        end

        check(timeout_count < 200, "AXI-Lite wrapper invalid run should finish quickly");
        check(error, "AXI-Lite wrapper invalid run should assert top error");
        check(layer_error_mask[0], "AXI-Lite wrapper should report layer0 validation error");
        check(mem_read_burst_count == 32'd0, "read request counter should stay zero");
        check(mem_read_word_count == 32'd0, "read word counter should stay zero");
        check(mem_write_req_count == 32'd0, "write request counter should stay zero");
        check(mem_write_word_count == 32'd0, "write word counter should stay zero");

        repeat (2) @(posedge clk);
        axil_read(REG_STATUS, read_data);
        check(read_data[1], "status.done sticky should be set through AXI-Lite");
        check(read_data[2], "status.error sticky should be set through AXI-Lite");
        axil_read(REG_LAYERS, read_data);
        check(read_data == 32'd0, "layers started/completed should be zero for validation exit");
        axil_read(REG_LAYER_ERROR_MASK, read_data);
        check(read_data[0], "layer error mask register should expose layer0 error through AXI-Lite");
        axil_read(REG_MEM_RD_REQS, read_data);
        check(read_data == 32'd0, "register read request counter should be zero");
        axil_read(REG_MEM_RD_WORDS, read_data);
        check(read_data == 32'd0, "register read word counter should be zero");
        axil_read(REG_MEM_WR_REQS, read_data);
        check(read_data == 32'd0, "register write request counter should be zero");
        axil_read(REG_MEM_WR_WORDS, read_data);
        check(read_data == 32'd0, "register write word counter should be zero");

        axil_write(REG_CTRL, 32'd2);
        axil_read(REG_STATUS, read_data);
        check(!read_data[1], "CTRL.clear should clear done sticky through AXI-Lite");
        check(!read_data[3], "CTRL.clear should clear command error sticky through AXI-Lite");
        check(read_data[2], "error sticky may relatch while top_error remains asserted");

        if (fail_count != 0) begin
            $display("FAIL: qmap_one_token_axil_top wrapper smoke failures=%0d", fail_count);
            $finish(1);
        end

        $display("PASS: qmap_one_token_axil_top connected AXI-Lite adapter to one-token validation path with no memory traffic.");
        $finish;
    end
endmodule

`default_nettype wire
