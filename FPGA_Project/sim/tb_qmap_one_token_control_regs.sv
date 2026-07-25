`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_control_regs;
    localparam int ADDR_WIDTH = 64;
    localparam int MAX_LAYERS = 28;
    localparam int MAX_CONTEXT = 256;
    localparam int TOKEN_ID_WIDTH = 32;
    localparam int SCORE_WIDTH = 56;
    localparam int LAYER_INDEX_WIDTH = $clog2(MAX_LAYERS);
    localparam int LAYER_COUNT_WIDTH = $clog2(MAX_LAYERS + 1);
    localparam int POSITION_WIDTH = $clog2(MAX_CONTEXT);

    localparam logic [11 : 0] REG_CTRL                 = 12'h000;
    localparam logic [11 : 0] REG_STATUS               = 12'h004;
    localparam logic [11 : 0] REG_LAYER_START          = 12'h008;
    localparam logic [11 : 0] REG_LAYER_COUNT          = 12'h00C;
    localparam logic [11 : 0] REG_POSITION             = 12'h010;
    localparam logic [11 : 0] REG_INPUT_TOKEN          = 12'h014;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_LO      = 12'h020;
    localparam logic [11 : 0] REG_INPUT_HIDDEN_HI      = 12'h024;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_LO     = 12'h028;
    localparam logic [11 : 0] REG_OUTPUT_HIDDEN_HI     = 12'h02C;
    localparam logic [11 : 0] REG_KV_CACHE_LO          = 12'h030;
    localparam logic [11 : 0] REG_KV_CACHE_HI          = 12'h034;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_LO   = 12'h038;
    localparam logic [11 : 0] REG_FINAL_TAIL_QMAP_HI   = 12'h03C;
    localparam logic [11 : 0] REG_FINAL_OVERRIDE_LO    = 12'h040;
    localparam logic [11 : 0] REG_FINAL_OVERRIDE_HI    = 12'h044;
    localparam logic [11 : 0] REG_FINAL_OVERRIDE_CTRL  = 12'h048;
    localparam logic [11 : 0] REG_EMBEDDING_CTRL       = 12'h04C;
    localparam logic [11 : 0] REG_TABLE_SELECT         = 12'h050;
    localparam logic [11 : 0] REG_TABLE_DATA_LO        = 12'h054;
    localparam logic [11 : 0] REG_TABLE_DATA_HI        = 12'h058;
    localparam logic [11 : 0] REG_TABLE_COMMIT         = 12'h05C;
    localparam logic [11 : 0] REG_OUT_TOKEN            = 12'h060;
    localparam logic [11 : 0] REG_OUT_SCORE_LO         = 12'h064;
    localparam logic [11 : 0] REG_OUT_SCORE_HI         = 12'h068;
    localparam logic [11 : 0] REG_LAYERS               = 12'h06C;
    localparam logic [11 : 0] REG_LAYER_DONE_MASK      = 12'h070;
    localparam logic [11 : 0] REG_LAYER_ERROR_MASK     = 12'h074;
    localparam logic [11 : 0] REG_LAST_OUTPUT_LO       = 12'h078;
    localparam logic [11 : 0] REG_LAST_OUTPUT_HI       = 12'h07C;
    localparam logic [11 : 0] REG_TAIL_HIDDEN_LO       = 12'h080;
    localparam logic [11 : 0] REG_TAIL_HIDDEN_HI       = 12'h084;
    localparam logic [11 : 0] REG_TAIL_TILES_STARTED   = 12'h088;
    localparam logic [11 : 0] REG_TAIL_TILES_COMPLETED = 12'h08C;
    localparam logic [11 : 0] REG_MEM_RD_REQS          = 12'h090;
    localparam logic [11 : 0] REG_MEM_RD_WORDS         = 12'h094;
    localparam logic [11 : 0] REG_MEM_WR_REQS          = 12'h098;
    localparam logic [11 : 0] REG_MEM_WR_WORDS         = 12'h09C;
    localparam logic [11 : 0] REG_EMBED_WEIGHT_LO      = 12'h0A0;
    localparam logic [11 : 0] REG_EMBED_WEIGHT_HI      = 12'h0A4;
    localparam logic [11 : 0] REG_EMBED_SCALE_LO       = 12'h0A8;
    localparam logic [11 : 0] REG_EMBED_SCALE_HI       = 12'h0AC;
    localparam logic [11 : 0] REG_RUNTIME_CTRL         = 12'h0B0;

    localparam logic [7 : 0] TABLE_QKV = 8'd0;
    localparam logic [7 : 0] TABLE_INPUT_NORM = 8'd1;
    localparam logic [7 : 0] TABLE_MLP_RESIDUAL = 8'd9;

    logic clk;
    logic rst_n;

    logic reg_wr_valid;
    logic reg_wr_ready;
    logic reg_rd_valid;
    logic reg_rd_ready;
    logic [11 : 0] reg_addr;
    logic [31 : 0] reg_wdata;
    logic [31 : 0] reg_rdata;
    logic reg_error;

    logic start_pulse;
    logic [31 : 0] input_token_id;
    logic embedding_enable;
    logic [ADDR_WIDTH-1 : 0] embedding_weight_base_addr;
    logic [ADDR_WIDTH-1 : 0] embedding_scale_base_addr;
    logic [LAYER_INDEX_WIDTH-1 : 0] layer_start_index;
    logic [LAYER_COUNT_WIDTH-1 : 0] layer_count;
    logic [POSITION_WIDTH-1 : 0] position;
    logic runtime_context_enable;
    logic [ADDR_WIDTH-1 : 0] input_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] kv_cache_base_addr;
    logic [ADDR_WIDTH-1 : 0] final_tail_qmap_base_addr;
    logic final_hidden_base_override_valid;
    logic [ADDR_WIDTH-1 : 0] final_hidden_base_override_addr;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] qkv_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] input_norm_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] attn_frontend_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] attn_score_value_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_proj_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] post_attn_norm_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_gate_up_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_silu_mul_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_down_qmap_base_addr_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] mlp_residual_add_qmap_base_addr_table;

    logic top_busy;
    logic top_done;
    logic top_error;
    logic [7 : 0] top_state_debug;
    logic [7 : 0] top_phase_debug;
    logic [LAYER_COUNT_WIDTH-1 : 0] layers_started;
    logic [LAYER_COUNT_WIDTH-1 : 0] layers_completed;
    logic [MAX_LAYERS-1 : 0] layer_done_mask;
    logic [MAX_LAYERS-1 : 0] layer_error_mask;
    logic [ADDR_WIDTH-1 : 0] last_layer_output_base_addr;
    logic tail_error;
    logic tail_norm_saturation;
    logic [ADDR_WIDTH-1 : 0] tail_effective_final_hidden_base_addr;
    logic [TOKEN_ID_WIDTH-1 : 0] tail_best_token_id;
    logic signed [SCORE_WIDTH-1 : 0] tail_best_score_q26;
    logic [31 : 0] tail_tiles_started;
    logic [31 : 0] tail_tiles_completed;
    logic [31 : 0] mem_read_burst_count;
    logic [31 : 0] mem_read_word_count;
    logic [31 : 0] mem_write_req_count;
    logic [31 : 0] mem_write_word_count;

    integer fail_count;
    logic [31 : 0] read_data;

    qmap_one_token_control_regs #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_LAYERS(MAX_LAYERS),
        .MAX_CONTEXT(MAX_CONTEXT),
        .TOKEN_ID_WIDTH(TOKEN_ID_WIDTH),
        .SCORE_WIDTH(SCORE_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_reg_wr_valid(reg_wr_valid),
        .o_reg_wr_ready(reg_wr_ready),
        .i_reg_rd_valid(reg_rd_valid),
        .o_reg_rd_ready(reg_rd_ready),
        .i_reg_addr(reg_addr),
        .i_reg_wdata(reg_wdata),
        .o_reg_rdata(reg_rdata),
        .o_reg_error(reg_error),
        .o_start_pulse(start_pulse),
        .o_input_token_id(input_token_id),
        .o_embedding_enable(embedding_enable),
        .o_embedding_weight_base_addr(embedding_weight_base_addr),
        .o_embedding_scale_base_addr(embedding_scale_base_addr),
        .o_layer_start_index(layer_start_index),
        .o_layer_count(layer_count),
        .o_position(position),
        .o_runtime_context_enable(runtime_context_enable),
        .o_input_hidden_base_addr(input_hidden_base_addr),
        .o_output_hidden_base_addr(output_hidden_base_addr),
        .o_kv_cache_base_addr(kv_cache_base_addr),
        .o_final_tail_qmap_base_addr(final_tail_qmap_base_addr),
        .o_final_hidden_base_override_valid(final_hidden_base_override_valid),
        .o_final_hidden_base_override_addr(final_hidden_base_override_addr),
        .o_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .o_input_norm_qmap_base_addr_table(input_norm_qmap_base_addr_table),
        .o_attn_frontend_qmap_base_addr_table(attn_frontend_qmap_base_addr_table),
        .o_attn_score_value_qmap_base_addr_table(attn_score_value_qmap_base_addr_table),
        .o_o_proj_qmap_base_addr_table(o_proj_qmap_base_addr_table),
        .o_post_attn_norm_qmap_base_addr_table(post_attn_norm_qmap_base_addr_table),
        .o_mlp_gate_up_qmap_base_addr_table(mlp_gate_up_qmap_base_addr_table),
        .o_mlp_silu_mul_qmap_base_addr_table(mlp_silu_mul_qmap_base_addr_table),
        .o_mlp_down_qmap_base_addr_table(mlp_down_qmap_base_addr_table),
        .o_mlp_residual_add_qmap_base_addr_table(mlp_residual_add_qmap_base_addr_table),
        .i_top_busy(top_busy),
        .i_top_done(top_done),
        .i_top_error(top_error),
        .i_top_state_debug(top_state_debug),
        .i_top_phase_debug(top_phase_debug),
        .i_layers_started(layers_started),
        .i_layers_completed(layers_completed),
        .i_layer_done_mask(layer_done_mask),
        .i_layer_error_mask(layer_error_mask),
        .i_last_layer_output_base_addr(last_layer_output_base_addr),
        .i_tail_error(tail_error),
        .i_tail_norm_saturation(tail_norm_saturation),
        .i_tail_effective_final_hidden_base_addr(tail_effective_final_hidden_base_addr),
        .i_tail_best_token_id(tail_best_token_id),
        .i_tail_best_score_q26(tail_best_score_q26),
        .i_tail_tiles_started(tail_tiles_started),
        .i_tail_tiles_completed(tail_tiles_completed),
        .i_mem_read_burst_count(mem_read_burst_count),
        .i_mem_read_word_count(mem_read_word_count),
        .i_mem_write_req_count(mem_write_req_count),
        .i_mem_write_word_count(mem_write_word_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

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

    task automatic mmio_write;
        input logic [11 : 0] addr;
        input logic [31 : 0] data;
        begin
            @(negedge clk);
            reg_addr = addr;
            reg_wdata = data;
            reg_wr_valid = 1'b1;
            @(negedge clk);
            reg_wr_valid = 1'b0;
        end
    endtask

    task automatic mmio_read;
        input logic [11 : 0] addr;
        output logic [31 : 0] data;
        begin
            @(negedge clk);
            reg_addr = addr;
            reg_rd_valid = 1'b1;
            #1;
            data = reg_rdata;
            @(negedge clk);
            reg_rd_valid = 1'b0;
        end
    endtask

    task automatic write_addr64;
        input logic [11 : 0] lo_addr;
        input logic [11 : 0] hi_addr;
        input logic [63 : 0] value;
        begin
            mmio_write(lo_addr, value[31 : 0]);
            mmio_write(hi_addr, value[63 : 32]);
        end
    endtask

    task automatic commit_table;
        input logic [7 : 0] table_id;
        input logic [7 : 0] layer_id;
        input logic [63 : 0] value;
        begin
            mmio_write(REG_TABLE_SELECT, {16'd0, layer_id, table_id});
            write_addr64(REG_TABLE_DATA_LO, REG_TABLE_DATA_HI, value);
            mmio_write(REG_TABLE_COMMIT, 32'd1);
        end
    endtask

    initial begin
        fail_count = 0;
        rst_n = 1'b0;
        reg_wr_valid = 1'b0;
        reg_rd_valid = 1'b0;
        reg_addr = 12'd0;
        reg_wdata = 32'd0;
        top_busy = 1'b0;
        top_done = 1'b0;
        top_error = 1'b0;
        top_state_debug = 8'h00;
        top_phase_debug = 8'h00;
        layers_started = '0;
        layers_completed = '0;
        layer_done_mask = '0;
        layer_error_mask = '0;
        last_layer_output_base_addr = 64'h0000_0004_2509_2540;
        tail_error = 1'b0;
        tail_norm_saturation = 1'b0;
        tail_effective_final_hidden_base_addr = 64'h0000_0004_2509_2540;
        tail_best_token_id = 32'd537;
        tail_best_score_q26 = 56'sd850086863;
        tail_tiles_started = 32'd9496;
        tail_tiles_completed = 32'd9496;
        mem_read_burst_count = 32'd178050;
        mem_read_word_count = 32'd24947620;
        mem_write_req_count = 32'd12312;
        mem_write_word_count = 32'd52227;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        mmio_read(REG_STATUS, read_data);
        check(read_data[2 : 0] == 3'b000, "reset status should be idle/clean");
        mmio_read(REG_RUNTIME_CTRL, read_data);
        check(read_data == 32'd0, "runtime context should default disabled");

        mmio_write(REG_LAYER_START, 32'd1);
        mmio_write(REG_LAYER_COUNT, 32'd2);
        mmio_write(REG_POSITION, 32'd17);
        mmio_write(REG_RUNTIME_CTRL, 32'd1);
        mmio_write(REG_INPUT_TOKEN, 32'd264);
        write_addr64(REG_INPUT_HIDDEN_LO, REG_INPUT_HIDDEN_HI, 64'h0000_0004_0509_2540);
        write_addr64(REG_OUTPUT_HIDDEN_LO, REG_OUTPUT_HIDDEN_HI, 64'h0000_0004_2509_2540);
        write_addr64(REG_KV_CACHE_LO, REG_KV_CACHE_HI, 64'h0000_0004_1800_0000);
        write_addr64(REG_FINAL_TAIL_QMAP_LO, REG_FINAL_TAIL_QMAP_HI, 64'h0000_0004_0501_0000);
        write_addr64(REG_FINAL_OVERRIDE_LO, REG_FINAL_OVERRIDE_HI, 64'h0000_0004_2509_2540);
        mmio_write(REG_FINAL_OVERRIDE_CTRL, 32'd1);
        mmio_write(REG_EMBEDDING_CTRL, 32'd1);
        write_addr64(REG_EMBED_WEIGHT_LO, REG_EMBED_WEIGHT_HI, 64'h0000_0004_0010_0000);
        write_addr64(REG_EMBED_SCALE_LO, REG_EMBED_SCALE_HI, 64'h0000_0004_04B3_0000);

        check(layer_start_index == 1, "layer_start_index writeback");
        check(layer_count == 2, "layer_count writeback");
        check(position == 17, "position writeback");
        check(runtime_context_enable, "runtime context enable writeback");
        check(input_token_id == 264, "input token writeback");
        check(input_hidden_base_addr == 64'h0000_0004_0509_2540, "input hidden addr writeback");
        check(output_hidden_base_addr == 64'h0000_0004_2509_2540, "output hidden addr writeback");
        check(kv_cache_base_addr == 64'h0000_0004_1800_0000, "kv cache addr writeback");
        check(final_tail_qmap_base_addr == 64'h0000_0004_0501_0000, "tail qmap addr writeback");
        check(final_hidden_base_override_valid, "override valid writeback");
        check(final_hidden_base_override_addr == 64'h0000_0004_2509_2540, "override addr writeback");
        check(embedding_enable, "embedding enable writeback");
        check(embedding_weight_base_addr == 64'h0000_0004_0010_0000,
              "embedding weight addr writeback");
        check(embedding_scale_base_addr == 64'h0000_0004_04B3_0000,
              "embedding scale addr writeback");
        mmio_read(REG_EMBEDDING_CTRL, read_data);
        check(read_data == 32'd1, "embedding enable register readback");
        mmio_read(REG_RUNTIME_CTRL, read_data);
        check(read_data == 32'd1, "runtime context enable register readback");
        mmio_read(REG_EMBED_WEIGHT_LO, read_data);
        check(read_data == 32'h0010_0000, "embedding weight lo register readback");
        mmio_read(REG_EMBED_WEIGHT_HI, read_data);
        check(read_data == 32'h0000_0004, "embedding weight hi register readback");
        mmio_read(REG_EMBED_SCALE_LO, read_data);
        check(read_data == 32'h04B3_0000, "embedding scale lo register readback");
        mmio_read(REG_EMBED_SCALE_HI, read_data);
        check(read_data == 32'h0000_0004, "embedding scale hi register readback");

        commit_table(TABLE_QKV, 8'd1, 64'h0000_0004_1008_0000);
        commit_table(TABLE_INPUT_NORM, 8'd1, 64'h0000_0004_150A_0000);
        commit_table(TABLE_MLP_RESIDUAL, 8'd2, 64'h0000_0004_2509_0000);
        check(qkv_qmap_base_addr_table[1*ADDR_WIDTH +: ADDR_WIDTH] == 64'h0000_0004_1008_0000, "qkv table layer1 commit");
        check(input_norm_qmap_base_addr_table[1*ADDR_WIDTH +: ADDR_WIDTH] == 64'h0000_0004_150A_0000, "input norm table layer1 commit");
        check(mlp_residual_add_qmap_base_addr_table[2*ADDR_WIDTH +: ADDR_WIDTH] == 64'h0000_0004_2509_0000, "residual table layer2 commit");

        mmio_write(REG_CTRL, 32'd1);
        @(posedge clk);
        check(start_pulse, "start pulse should assert after CTRL.start write");
        @(posedge clk);
        check(!start_pulse, "start pulse should be one cycle");

        top_done = 1'b1;
        layers_started = 5'd2;
        layers_completed = 5'd2;
        layer_done_mask = 28'h0000006;
        @(posedge clk);
        top_done = 1'b0;
        mmio_read(REG_STATUS, read_data);
        check(read_data[1], "done sticky should latch top done");
        mmio_read(REG_OUT_TOKEN, read_data);
        check(read_data == 32'd537, "output token readback");
        mmio_read(REG_OUT_SCORE_LO, read_data);
        check(read_data == 32'd850086863, "score low readback");
        mmio_read(REG_OUT_SCORE_HI, read_data);
        check(read_data == 32'd0, "positive score high readback");
        mmio_read(REG_LAYERS, read_data);
        check(read_data[4 : 0] == 5'd2 && read_data[20 : 16] == 5'd2, "layers started/completed readback");
        mmio_read(REG_LAYER_DONE_MASK, read_data);
        check(read_data[27 : 0] == 28'h0000006, "layer done mask readback");
        mmio_read(REG_LAST_OUTPUT_LO, read_data);
        check(read_data == 32'h2509_2540, "last output lo readback");
        mmio_read(REG_TAIL_TILES_COMPLETED, read_data);
        check(read_data == 32'd9496, "tail tiles completed readback");
        mmio_read(REG_MEM_RD_REQS, read_data);
        check(read_data == 32'd178050, "mem read request counter readback");
        mmio_read(REG_MEM_WR_WORDS, read_data);
        check(read_data == 32'd52227, "mem write word counter readback");

        top_busy = 1'b1;
        mmio_write(REG_LAYER_COUNT, 32'd7);
        check(layer_count == 2, "busy config write should not update layer_count");
        mmio_write(REG_EMBEDDING_CTRL, 32'd0);
        mmio_write(REG_RUNTIME_CTRL, 32'd0);
        mmio_write(REG_EMBED_WEIGHT_LO, 32'hDEAD_BEEF);
        check(embedding_enable, "busy write should not disable embedding");
        check(runtime_context_enable, "busy write should not disable runtime context");
        check(embedding_weight_base_addr == 64'h0000_0004_0010_0000,
              "busy write should not update embedding weight addr");
        mmio_write(REG_CTRL, 32'd1);
        @(posedge clk);
        check(!start_pulse, "busy start should not pulse");
        mmio_read(REG_STATUS, read_data);
        check(read_data[3] && read_data[2], "busy write/start should latch command error");

        top_busy = 1'b0;
        mmio_write(REG_CTRL, 32'd2);
        mmio_read(REG_STATUS, read_data);
        check(read_data[3 : 1] == 3'b000, "clear should drop sticky done/error/command error");

        mmio_write(REG_TABLE_SELECT, {16'd0, 8'd1, 8'd99});
        write_addr64(REG_TABLE_DATA_LO, REG_TABLE_DATA_HI, 64'h0000_0004_DEAD_BEEF);
        mmio_write(REG_TABLE_COMMIT, 32'd1);
        mmio_read(REG_STATUS, read_data);
        check(read_data[3] && read_data[2], "invalid table commit should latch command error");

        if (fail_count != 0) begin
            $display("FAIL: qmap_one_token_control_regs contract test failures=%0d", fail_count);
            $finish(1);
        end

        $display("PASS: qmap_one_token_control_regs MMIO contract test passed.");
        $finish;
    end
endmodule

`default_nettype wire
