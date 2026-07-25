`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_axi_embedding_top #(
    parameter bit INJECT_MEM_ERROR = 1'b0
);

    localparam int INPUT_SIZE = 1024;
    localparam int WEIGHT_WORDS = 128;
    localparam int SCALE_WORDS = 8;
    localparam logic [1 : 0] AXI_RESP_OKAY = 2'b00;
    localparam logic [1 : 0] AXI_RESP_SLVERR = 2'b10;
    localparam logic [63 : 0] WEIGHT_BASE = 64'h0000_0004_0010_0000;
    localparam logic [63 : 0] SCALE_BASE = 64'h0000_0004_04B3_0000;
    localparam logic [63 : 0] INPUT_HIDDEN_BASE = 64'h0000_0004_0509_2540;

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
    logic mem_error;
    logic [31 : 0] status;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;
    logic [31 : 0] sticky_status;

    logic [63 : 0] m_axi_awaddr;
    logic [7 : 0] m_axi_awlen;
    logic [2 : 0] m_axi_awsize;
    logic [1 : 0] m_axi_awburst;
    logic [2 : 0] m_axi_awprot;
    logic [3 : 0] m_axi_awcache;
    logic m_axi_awvalid;
    logic m_axi_awready;
    logic [31 : 0] m_axi_wdata;
    logic [3 : 0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready;
    logic [1 : 0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;
    logic [63 : 0] m_axi_araddr;
    logic [7 : 0] m_axi_arlen;
    logic [2 : 0] m_axi_arsize;
    logic [1 : 0] m_axi_arburst;
    logic [2 : 0] m_axi_arprot;
    logic [3 : 0] m_axi_arcache;
    logic m_axi_arvalid;
    logic m_axi_arready;
    logic [31 : 0] m_axi_rdata;
    logic [1 : 0] m_axi_rresp;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    logic m_axi_rready;

    logic [31 : 0] weight_words [0 : WEIGHT_WORDS-1];
    logic [31 : 0] scale_words [0 : SCALE_WORDS-1];
    logic [31 : 0] expected_words [0 : INPUT_SIZE-1];
    logic [31 : 0] token_file [0 : 0];
    logic [63 : 0] expected_awaddr [0 : 4];
    integer expected_awbeats [0 : 4];
    integer cycle_count;
    integer fail_count;
    integer mismatch_count;
    integer ar_count;
    integer read_target;
    integer read_index;
    integer aw_count;
    integer active_aw_index;
    integer burst_write_index;
    integer total_write_words;
    integer final_b_cycle;
    integer top_done_cycle;
    logic read_active;
    logic write_active;
    string vector_dir;
    string vector_path;

    always #5 clk = ~clk;

    assign m_axi_arready = !read_active && !m_axi_rvalid && ((cycle_count % 5) != 1);
    assign m_axi_awready = !write_active && !m_axi_bvalid && ((cycle_count % 7) != 2);
    assign m_axi_wready = write_active && ((cycle_count % 6) != 3);
    assign m_axi_rresp = (INJECT_MEM_ERROR && (read_target == 1) && (read_index == 0)) ?
                         AXI_RESP_SLVERR : AXI_RESP_OKAY;

    qmap_one_token_axi_top dut (
        .aclk(clk),
        .aresetn(rst_n),
        .S_AXI_AWADDR(s_axi_awaddr),
        .S_AXI_AWPROT(s_axi_awprot),
        .S_AXI_AWVALID(s_axi_awvalid),
        .S_AXI_AWREADY(s_axi_awready),
        .S_AXI_WDATA(s_axi_wdata),
        .S_AXI_WSTRB(s_axi_wstrb),
        .S_AXI_WVALID(s_axi_wvalid),
        .S_AXI_WREADY(s_axi_wready),
        .S_AXI_BRESP(s_axi_bresp),
        .S_AXI_BVALID(s_axi_bvalid),
        .S_AXI_BREADY(s_axi_bready),
        .S_AXI_ARADDR(s_axi_araddr),
        .S_AXI_ARPROT(s_axi_arprot),
        .S_AXI_ARVALID(s_axi_arvalid),
        .S_AXI_ARREADY(s_axi_arready),
        .S_AXI_RDATA(s_axi_rdata),
        .S_AXI_RRESP(s_axi_rresp),
        .S_AXI_RVALID(s_axi_rvalid),
        .S_AXI_RREADY(s_axi_rready),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_mem_error(mem_error),
        .o_status(status),
        .o_mem_read_burst_count(mem_read_burst_count),
        .o_mem_read_word_count(mem_read_word_count),
        .o_mem_write_req_count(mem_write_req_count),
        .o_mem_write_word_count(mem_write_word_count),
        .M_AXI_AWADDR(m_axi_awaddr),
        .M_AXI_AWLEN(m_axi_awlen),
        .M_AXI_AWSIZE(m_axi_awsize),
        .M_AXI_AWBURST(m_axi_awburst),
        .M_AXI_AWPROT(m_axi_awprot),
        .M_AXI_AWCACHE(m_axi_awcache),
        .M_AXI_AWVALID(m_axi_awvalid),
        .M_AXI_AWREADY(m_axi_awready),
        .M_AXI_WDATA(m_axi_wdata),
        .M_AXI_WSTRB(m_axi_wstrb),
        .M_AXI_WLAST(m_axi_wlast),
        .M_AXI_WVALID(m_axi_wvalid),
        .M_AXI_WREADY(m_axi_wready),
        .M_AXI_BRESP(m_axi_bresp),
        .M_AXI_BVALID(m_axi_bvalid),
        .M_AXI_BREADY(m_axi_bready),
        .M_AXI_ARADDR(m_axi_araddr),
        .M_AXI_ARLEN(m_axi_arlen),
        .M_AXI_ARSIZE(m_axi_arsize),
        .M_AXI_ARBURST(m_axi_arburst),
        .M_AXI_ARPROT(m_axi_arprot),
        .M_AXI_ARCACHE(m_axi_arcache),
        .M_AXI_ARVALID(m_axi_arvalid),
        .M_AXI_ARREADY(m_axi_arready),
        .M_AXI_RDATA(m_axi_rdata),
        .M_AXI_RRESP(m_axi_rresp),
        .M_AXI_RLAST(m_axi_rlast),
        .M_AXI_RVALID(m_axi_rvalid),
        .M_AXI_RREADY(m_axi_rready)
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
            while (!(s_axi_awready && s_axi_wready)) wait_clk();
            wait_clk();
            @(negedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;
            while (!s_axi_bvalid) wait_clk();
            check(s_axi_bresp == AXI_RESP_OKAY, "BD-facing AXI-Lite write returned an error");
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
            while (!s_axi_arready) wait_clk();
            wait_clk();
            @(negedge clk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) wait_clk();
            data = s_axi_rdata;
            check(s_axi_rresp == AXI_RESP_OKAY, "BD-facing AXI-Lite read returned an error");
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
            ar_count <= 0;
            read_target <= 0;
            read_index <= 0;
            aw_count <= 0;
            active_aw_index <= -1;
            burst_write_index <= 0;
            total_write_words <= 0;
            final_b_cycle <= -1;
            top_done_cycle <= -1;
            read_active <= 1'b0;
            write_active <= 1'b0;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 32'd0;
            m_axi_rlast <= 1'b0;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= AXI_RESP_OKAY;
        end else begin
            cycle_count <= cycle_count + 1;
            if ((top_done_cycle < 0) && done) top_done_cycle <= cycle_count;

            if (m_axi_arvalid && m_axi_arready) begin
                ar_count <= ar_count + 1;
                read_active <= 1'b1;
                read_index <= 0;
                check(m_axi_arsize == 3'd2 && m_axi_arburst == 2'b01,
                      "BD-facing embedding read AXI attributes mismatch");
                if (m_axi_araddr == (WEIGHT_BASE + (token_file[0] * 512))) begin
                    read_target <= 1;
                    check(m_axi_arlen == 8'd127, "BD-facing weight ARLEN mismatch");
                end else if (m_axi_araddr == (SCALE_BASE + (token_file[0] * 32))) begin
                    read_target <= 2;
                    check(m_axi_arlen == 8'd7, "BD-facing scale ARLEN mismatch");
                end else begin
                    read_target <= 0;
                    check(1'b0, "BD-facing embedding issued unexpected ARADDR");
                end
            end
            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                if (m_axi_rlast) read_active <= 1'b0;
                else read_index <= read_index + 1;
            end
            if (read_active && !m_axi_rvalid && ((cycle_count % 4) != 2)) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= (read_target == 1) ? weight_words[read_index] : scale_words[read_index];
                m_axi_rlast <= (read_target == 1) ?
                    (read_index == (WEIGHT_WORDS - 1)) : (read_index == (SCALE_WORDS - 1));
            end

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
                if (aw_count == 5) final_b_cycle <= cycle_count;
            end
            if (m_axi_awvalid && m_axi_awready) begin
                check(aw_count < 5, "BD-facing embedding issued too many AW bursts");
                if (aw_count < 5) begin
                    check(m_axi_awaddr == expected_awaddr[aw_count], "BD-facing split AWADDR mismatch");
                    check(({1'b0, m_axi_awlen} + 1'b1) == expected_awbeats[aw_count],
                          "BD-facing split AWLEN mismatch");
                end
                check(m_axi_awsize == 3'd2 && m_axi_awburst == 2'b01,
                      "BD-facing embedding write AXI attributes mismatch");
                check((m_axi_awaddr[11 : 0] + (({1'b0, m_axi_awlen} + 1'b1) * 4)) <= 4096,
                      "BD-facing write burst crossed 4 KiB boundary");
                active_aw_index <= aw_count;
                aw_count <= aw_count + 1;
                burst_write_index <= 0;
                write_active <= 1'b1;
            end
            if (m_axi_wvalid && m_axi_wready) begin
                check(m_axi_wdata == expected_words[total_write_words],
                      "BD-facing embedding WDATA mismatch");
                check(m_axi_wstrb == 4'hF, "BD-facing embedding WSTRB mismatch");
                check(m_axi_wlast ==
                      (burst_write_index == (expected_awbeats[active_aw_index] - 1)),
                      "BD-facing split WLAST mismatch");
                if (m_axi_wdata !== expected_words[total_write_words]) mismatch_count <= mismatch_count + 1;
                total_write_words <= total_write_words + 1;
                if (m_axi_wlast) begin
                    write_active <= 1'b0;
                    active_aw_index <= -1;
                    m_axi_bvalid <= 1'b1;
                    burst_write_index <= 0;
                end else begin
                    burst_write_index <= burst_write_index + 1;
                end
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

        expected_awaddr[0] = 64'h0000_0004_0509_2540;
        expected_awaddr[1] = 64'h0000_0004_0509_2940;
        expected_awaddr[2] = 64'h0000_0004_0509_2D40;
        expected_awaddr[3] = 64'h0000_0004_0509_3000;
        expected_awaddr[4] = 64'h0000_0004_0509_3400;
        expected_awbeats[0] = 256;
        expected_awbeats[1] = 256;
        expected_awbeats[2] = 176;
        expected_awbeats[3] = 256;
        expected_awbeats[4] = 80;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

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
        axil_write(REG_CTRL, 32'd1);

        while (!done && (cycle_count < 12000)) wait_clk();
        check(done, "BD-facing embedding top timed out");
        check(status[1] && status[2], "BD-facing live done/error status mismatch");
        wait_clk();
        check(error, "BD-facing scheduler validation error missing after embedding");
        if (INJECT_MEM_ERROR) begin
            check(mem_error, "BD-facing wrapper did not preserve an early AXI read error");
        end else begin
            check(!mem_error, "BD-facing embedding reported an AXI memory error");
        end
        axil_read(REG_STATUS, sticky_status);
        check(sticky_status[1] && sticky_status[2], "BD-facing sticky done/error status mismatch");
        check(ar_count == 2, "BD-facing embedding AR burst count mismatch");
        check(aw_count == 5, "BD-facing 4 KiB AW split count mismatch");
        check(total_write_words == 1024, "BD-facing embedding W word count mismatch");
        check(mismatch_count == 0, "BD-facing embedding output mismatch");
        check(mem_read_burst_count == 32'd2 && mem_read_word_count == 32'd136,
              "BD-facing local read counters mismatch");
        check(mem_write_req_count == 32'd1 && mem_write_word_count == 32'd1024,
              "BD-facing local write counters mismatch");
        check((final_b_cycle >= 0) && (final_b_cycle < top_done_cycle),
              "BD-facing top done did not follow final AXI B response");

        if (fail_count != 0) begin
            $display("FAIL: tb_qmap_one_token_axi_embedding_top saw %0d failure(s).", fail_count);
            $finish(1);
        end
        $display("PASS: BD-facing one-token AXI top ran exact embedding and split its 4 KiB write into five legal bursts.");
        $finish;
    end

    initial begin
        #5000000;
        $display("TIMEOUT: tb_qmap_one_token_axi_embedding_top did not finish");
        $finish(1);
    end

endmodule

module tb_qmap_one_token_axi_embedding_top_mem_error;
    tb_qmap_one_token_axi_embedding_top #(
        .INJECT_MEM_ERROR(1'b1)
    ) testbench ();
endmodule

`default_nettype wire
