`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_axil_embedding_top;

    localparam int ADDR_WIDTH = 64;
    localparam int INPUT_SIZE = 1024;
    localparam int WEIGHT_WORDS = 128;
    localparam int SCALE_WORDS = 8;
    localparam logic [1 : 0] AXI_RESP_OKAY = 2'b00;
    localparam logic [ADDR_WIDTH-1 : 0] WEIGHT_BASE = 64'h0000_0004_0010_0000;
    localparam logic [ADDR_WIDTH-1 : 0] SCALE_BASE = 64'h0000_0004_04B3_0000;
    localparam logic [ADDR_WIDTH-1 : 0] INPUT_HIDDEN_BASE = 64'h0000_0004_0509_2540;

    localparam logic [11 : 0] REG_CTRL                = 12'h000;
    localparam logic [11 : 0] REG_STATUS              = 12'h004;
    localparam logic [11 : 0] REG_LAYER_START         = 12'h008;
    localparam logic [11 : 0] REG_LAYER_COUNT         = 12'h00C;
    localparam logic [11 : 0] REG_POSITION            = 12'h010;
    localparam logic [11 : 0] REG_INPUT_TOKEN         = 12'h014;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_LO     = 12'h020;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_HI     = 12'h024;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_LO    = 12'h028;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_HI    = 12'h02C;
    localparam logic [11 : 0] REG_KV_CACHE_LO         = 12'h030;
    localparam logic [11 : 0] REG_KV_CACHE_HI         = 12'h034;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_LO  = 12'h038;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_HI  = 12'h03C;
    localparam logic [11 : 0] REG_EMBEDDING_CTRL      = 12'h04C;
    localparam logic [11 : 0] REG_LAYER_ERROR_MASK    = 12'h074;
    localparam logic [11 : 0] REG_MEM_RD_REQS         = 12'h090;
    localparam logic [11 : 0] REG_MEM_RD_WORDS        = 12'h094;
    localparam logic [11 : 0] REG_MEM_WR_REQS         = 12'h098;
    localparam logic [11 : 0] REG_MEM_WR_WORDS        = 12'h09C;
    localparam logic [11 : 0] REG_EMBED_WEIGHT_LO     = 12'h0A0;
    localparam logic [11 : 0] REG_EMBED_WEIGHT_HI     = 12'h0A4;
    localparam logic [11 : 0] REG_EMBED_SCALE_LO      = 12'h0A8;
    localparam logic [11 : 0] REG_EMBED_SCALE_HI      = 12'h0AC;

    logic clk;
    logic rst_n;
    logic [11 : 0] s_axi_awaddr;
    logic [2 : 0] s_axi_awprot;
    logic s_axi_awvalid;
    logic s_axi_awready;
    logic [31 : 0] s_axi_wdata;
    logic [3 : 0] s_axi_wstrb;
    logic s_axi_wvalid;
    logic s_axi_wready;
    logic [1 : 0] s_axi_bresp;
    logic s_axi_bvalid;
    logic s_axi_bready;
    logic [11 : 0] s_axi_araddr;
    logic [2 : 0] s_axi_arprot;
    logic s_axi_arvalid;
    logic s_axi_arready;
    logic [31 : 0] s_axi_rdata;
    logic [1 : 0] s_axi_rresp;
    logic s_axi_rvalid;
    logic s_axi_rready;

    logic busy;
    logic done;
    logic error;
    logic [27 : 0] layer_error_mask;
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

    logic [31 : 0] weight_words [0 : WEIGHT_WORDS-1];
    logic [31 : 0] scale_words [0 : SCALE_WORDS-1];
    logic [31 : 0] expected_words [0 : INPUT_SIZE-1];
    logic [31 : 0] token_file [0 : 0];
    integer cycle_count;
    integer fail_count;
    integer mismatch_count;
    integer read_target;
    integer read_index;
    integer read_requests;
    integer read_words;
    integer write_requests;
    integer write_words;
    integer write_index;
    integer write_done_delay;
    integer last_write_cycle;
    integer layer_error_cycle;
    integer top_done_cycle;
    logic read_active;
    logic write_active;
    logic [31 : 0] read_data;
    string vector_dir;
    string vector_path;

    always #5 clk = ~clk;

    assign mem_rd_req_ready = !read_active && !mem_rd_rsp_valid && ((cycle_count % 5) != 1);
    assign mem_wr_req_ready = !write_active && ((cycle_count % 7) != 2);
    assign mem_wr_data_ready = write_active && ((cycle_count % 6) != 3);

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
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_layer_error_mask(layer_error_mask),
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

    task automatic check;
        input logic condition;
        input string message;
        begin
            if (!condition) begin
                $display("FAIL: %s", message);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task automatic wait_clk;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task automatic axil_write;
        input logic [11 : 0] address;
        input logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_awaddr = address;
            s_axi_awvalid = 1'b1;
            s_axi_wdata = data;
            s_axi_wvalid = 1'b1;
            while (!(s_axi_awready && s_axi_wready)) begin
                wait_clk();
            end
            wait_clk();
            @(negedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;
            while (!s_axi_bvalid) begin
                wait_clk();
            end
            check(s_axi_bresp == AXI_RESP_OKAY, "AXI-Lite write returned an error");
            @(negedge clk);
            s_axi_bready = 1'b1;
            wait_clk();
            @(negedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axil_read;
        input logic [11 : 0] address;
        output logic [31 : 0] data;
        begin
            @(negedge clk);
            s_axi_araddr = address;
            s_axi_arvalid = 1'b1;
            while (!s_axi_arready) begin
                wait_clk();
            end
            wait_clk();
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) begin
                wait_clk();
            end
            data = s_axi_rdata;
            check(s_axi_rresp == AXI_RESP_OKAY, "AXI-Lite read returned an error");
            @(negedge clk);
            s_axi_rready = 1'b1;
            wait_clk();
            @(negedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    task automatic write_addr64;
        input logic [11 : 0] lo_address;
        input logic [11 : 0] hi_address;
        input logic [63 : 0] value;
        begin
            axil_write(lo_address, value[31 : 0]);
            axil_write(hi_address, value[63 : 32]);
        end
    endtask

    initial begin
        vector_dir = "vectors";
        if (!$value$plusargs("VECTOR_DIR=%s", vector_dir)) begin
        end
        vector_path = {vector_dir, "/embedding_weight_words32.hex"};
        $readmemh(vector_path, weight_words);
        vector_path = {vector_dir, "/embedding_scale_words32.hex"};
        $readmemh(vector_path, scale_words);
        vector_path = {vector_dir, "/embedding_expected_q14_10.hex"};
        $readmemh(vector_path, expected_words);
        vector_path = {vector_dir, "/embedding_token_id.hex"};
        $readmemh(vector_path, token_file);
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
            mismatch_count <= 0;
            read_target <= 0;
            read_index <= 0;
            read_requests <= 0;
            read_words <= 0;
            write_requests <= 0;
            write_words <= 0;
            write_index <= 0;
            write_done_delay <= 0;
            last_write_cycle <= -1;
            layer_error_cycle <= -1;
            top_done_cycle <= -1;
            read_active <= 1'b0;
            write_active <= 1'b0;
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            if ((layer_error_cycle < 0) && layer_error_mask[0]) layer_error_cycle <= cycle_count;
            if ((top_done_cycle < 0) && done) top_done_cycle <= cycle_count;

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                read_requests <= read_requests + 1;
                read_active <= 1'b1;
                read_index <= 0;
                if (mem_rd_req_addr == (WEIGHT_BASE + (token_file[0] * 512))) begin
                    read_target <= 1;
                    check(mem_rd_req_len_bytes == 16'd512, "AXI-Lite top weight read length mismatch");
                end else if (mem_rd_req_addr == (SCALE_BASE + (token_file[0] * 32))) begin
                    read_target <= 2;
                    check(mem_rd_req_len_bytes == 16'd32, "AXI-Lite top scale read length mismatch");
                end else begin
                    read_target <= 0;
                    check(1'b0, "AXI-Lite top issued an unexpected read");
                end
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                read_words <= read_words + 1;
                mem_rd_rsp_valid <= 1'b0;
                if (mem_rd_rsp_last) read_active <= 1'b0;
                else read_index <= read_index + 1;
            end
            if (read_active && !mem_rd_rsp_valid && ((cycle_count % 4) != 2)) begin
                mem_rd_rsp_valid <= 1'b1;
                mem_rd_rsp_data <= (read_target == 1) ? weight_words[read_index] : scale_words[read_index];
                mem_rd_rsp_last <= (read_target == 1) ?
                    (read_index == (WEIGHT_WORDS - 1)) : (read_index == (SCALE_WORDS - 1));
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                write_requests <= write_requests + 1;
                write_active <= 1'b1;
                write_index <= 0;
                check(mem_wr_req_addr == INPUT_HIDDEN_BASE, "AXI-Lite top embedding write address mismatch");
                check(mem_wr_req_len_bytes == 16'd4096, "AXI-Lite top embedding write length mismatch");
            end
            if (mem_wr_data_valid && mem_wr_data_ready) begin
                write_words <= write_words + 1;
                if (mem_wr_data !== expected_words[write_index]) begin
                    mismatch_count <= mismatch_count + 1;
                end
                check(mem_wr_data_last == (write_index == (INPUT_SIZE - 1)),
                      "AXI-Lite top embedding write last mismatch");
                if (mem_wr_data_last) begin
                    write_active <= 1'b0;
                    write_done_delay <= 3;
                    last_write_cycle <= cycle_count;
                end else begin
                    write_index <= write_index + 1;
                end
            end
            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) mem_wr_done <= 1'b1;
            end
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        s_axi_awaddr = 12'd0;
        s_axi_awprot = 3'd0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata = 32'd0;
        s_axi_wstrb = 4'hF;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b0;
        s_axi_araddr = 12'd0;
        s_axi_arprot = 3'd0;
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b0;
        fail_count = 0;

        repeat (5) wait_clk();
        rst_n = 1'b1;
        repeat (2) wait_clk();

        axil_write(REG_LAYER_START, 32'd0);
        axil_write(REG_LAYER_COUNT, 32'd0);
        axil_write(REG_POSITION, 32'd4);
        axil_write(REG_INPUT_TOKEN, token_file[0]);
        write_addr64(REG_INPUT_HIDDEN_LO, REG_INPUT_HIDDEN_HI, INPUT_HIDDEN_BASE);
        write_addr64(REG_OUTPUT_HIDDEN_LO, REG_OUTPUT_HIDDEN_HI, 64'h0000_0004_1509_2540);
        write_addr64(REG_KV_CACHE_LO, REG_KV_CACHE_HI, 64'h0000_0004_1410_0000);
        write_addr64(REG_FINAL_TAIL_QMAP_LO, REG_FINAL_TAIL_QMAP_HI, 64'h0000_0004_0501_0000);
        write_addr64(REG_EMBED_WEIGHT_LO, REG_EMBED_WEIGHT_HI, WEIGHT_BASE);
        write_addr64(REG_EMBED_SCALE_LO, REG_EMBED_SCALE_HI, SCALE_BASE);
        axil_write(REG_EMBEDDING_CTRL, 32'd1);

        axil_read(REG_EMBEDDING_CTRL, read_data);
        check(read_data == 32'd1, "embedding enable register readback mismatch");
        axil_read(REG_EMBED_WEIGHT_LO, read_data);
        check(read_data == WEIGHT_BASE[31 : 0], "embedding weight low readback mismatch");
        axil_read(REG_EMBED_SCALE_HI, read_data);
        check(read_data == SCALE_BASE[63 : 32], "embedding scale high readback mismatch");

        axil_write(REG_CTRL, 32'd1);
        while (!done && (cycle_count < 10000)) begin
            wait_clk();
        end
        check(done, "AXI-Lite embedding top timed out");
        wait_clk();
        check(error, "scheduler validation error did not propagate after embedding");
        check(layer_error_mask[0], "scheduler layer0 validation error missing after embedding");
        check(mismatch_count == 0, "AXI-Lite embedding output mismatch");
        check(read_requests == 2 && read_words == 136, "AXI-Lite embedding read counts mismatch");
        check(write_requests == 1 && write_words == 1024, "AXI-Lite embedding write counts mismatch");
        check(mem_read_burst_count == 32'd2 && mem_read_word_count == 32'd136,
              "top embedding read counters mismatch");
        check(mem_write_req_count == 32'd1 && mem_write_word_count == 32'd1024,
              "top embedding write counters mismatch");
        check((last_write_cycle >= 0) && (last_write_cycle < layer_error_cycle) &&
              (layer_error_cycle < top_done_cycle),
              "embedding write, scheduler validation, and top done ordering mismatch");

        axil_read(REG_STATUS, read_data);
        check(read_data[1] && read_data[2], "status sticky done/error mismatch");
        axil_read(REG_LAYER_ERROR_MASK, read_data);
        check(read_data[0], "layer error register missing layer0 bit");
        axil_read(REG_MEM_RD_REQS, read_data);
        check(read_data == 32'd2, "register embedding read request count mismatch");
        axil_read(REG_MEM_RD_WORDS, read_data);
        check(read_data == 32'd136, "register embedding read word count mismatch");
        axil_read(REG_MEM_WR_REQS, read_data);
        check(read_data == 32'd1, "register embedding write request count mismatch");
        axil_read(REG_MEM_WR_WORDS, read_data);
        check(read_data == 32'd1024, "register embedding write word count mismatch");

        axil_write(REG_CTRL, 32'd2);
        if (fail_count != 0) begin
            $display("FAIL: tb_qmap_one_token_axil_embedding_top saw %0d failure(s).", fail_count);
            $finish(1);
        end
        $display("PASS: AXI-Lite one-token top ran exact tied-Q4 embedding before scheduler validation.");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT: tb_qmap_one_token_axil_embedding_top did not finish");
        $finish(1);
    end

endmodule

`default_nettype wire
