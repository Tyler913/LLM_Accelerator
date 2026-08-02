`timescale 1ns/1ps
`default_nettype none

module tb_qmap_one_token_control_regs_bram;

    localparam int ADDR_WIDTH = 64;
    localparam int MAX_LAYERS = 28;
    localparam int LAYER_INDEX_WIDTH = 5;
    localparam int LAYER_COUNT_WIDTH = 5;

    localparam logic [11:0] REG_STATUS       = 12'h004;
    localparam logic [11:0] REG_TABLE_SELECT = 12'h050;
    localparam logic [11:0] REG_TABLE_LO     = 12'h054;
    localparam logic [11:0] REG_TABLE_HI     = 12'h058;
    localparam logic [11:0] REG_TABLE_COMMIT = 12'h05C;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic reg_wr_valid;
    logic reg_rd_valid;
    logic [11:0] reg_addr;
    logic [31:0] reg_wdata;
    logic [31:0] reg_rdata;
    logic reg_wr_ready;
    logic reg_rd_ready;
    logic reg_error;

    logic [LAYER_INDEX_WIDTH-1:0] runtime_layer;
    logic [ADDR_WIDTH-1:0] qkv_addr;
    logic [ADDR_WIDTH-1:0] input_norm_addr;
    logic [ADDR_WIDTH-1:0] frontend_addr;
    logic [ADDR_WIDTH-1:0] score_addr;
    logic [ADDR_WIDTH-1:0] o_proj_addr;
    logic [ADDR_WIDTH-1:0] post_norm_addr;
    logic [ADDR_WIDTH-1:0] gate_up_addr;
    logic [ADDR_WIDTH-1:0] silu_addr;
    logic [ADDR_WIDTH-1:0] down_addr;
    logic [ADDR_WIDTH-1:0] residual_addr;
    logic runtime_table_valid;
    logic [LAYER_INDEX_WIDTH-1:0] runtime_table_layer;

    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_qkv_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_input_norm_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_frontend_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_score_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_o_proj_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_post_norm_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_gate_up_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_silu_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_down_table;
    logic [MAX_LAYERS*ADDR_WIDTH-1:0] legacy_residual_table;

    qmap_one_token_control_regs #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_LAYERS(MAX_LAYERS),
        .LAYER_INDEX_WIDTH(LAYER_INDEX_WIDTH),
        .LAYER_COUNT_WIDTH(LAYER_COUNT_WIDTH),
        .USE_BRAM_BASE_TABLES(1'b1)
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
        .o_start_pulse(),
        .o_input_token_id(),
        .o_embedding_enable(),
        .o_embedding_weight_base_addr(),
        .o_embedding_scale_base_addr(),
        .o_layer_start_index(),
        .o_layer_count(),
        .o_position(),
        .o_runtime_context_enable(),
        .o_input_hidden_base_addr(),
        .o_output_hidden_base_addr(),
        .o_kv_cache_base_addr(),
        .o_final_tail_qmap_base_addr(),
        .o_final_hidden_base_override_valid(),
        .o_final_hidden_base_override_addr(),
        .o_qkv_qmap_base_addr_table(legacy_qkv_table),
        .o_input_norm_qmap_base_addr_table(legacy_input_norm_table),
        .o_attn_frontend_qmap_base_addr_table(legacy_frontend_table),
        .o_attn_score_value_qmap_base_addr_table(legacy_score_table),
        .o_o_proj_qmap_base_addr_table(legacy_o_proj_table),
        .o_post_attn_norm_qmap_base_addr_table(legacy_post_norm_table),
        .o_mlp_gate_up_qmap_base_addr_table(legacy_gate_up_table),
        .o_mlp_silu_mul_qmap_base_addr_table(legacy_silu_table),
        .o_mlp_down_qmap_base_addr_table(legacy_down_table),
        .o_mlp_residual_add_qmap_base_addr_table(legacy_residual_table),
        .i_runtime_table_layer(runtime_layer),
        .o_qkv_qmap_base_addr(qkv_addr),
        .o_input_norm_qmap_base_addr(input_norm_addr),
        .o_attn_frontend_qmap_base_addr(frontend_addr),
        .o_attn_score_value_qmap_base_addr(score_addr),
        .o_o_proj_qmap_base_addr(o_proj_addr),
        .o_post_attn_norm_qmap_base_addr(post_norm_addr),
        .o_mlp_gate_up_qmap_base_addr(gate_up_addr),
        .o_mlp_silu_mul_qmap_base_addr(silu_addr),
        .o_mlp_down_qmap_base_addr(down_addr),
        .o_mlp_residual_add_qmap_base_addr(residual_addr),
        .o_runtime_table_valid(runtime_table_valid),
        .o_runtime_table_layer(runtime_table_layer),
        .i_top_busy(1'b0),
        .i_top_done(1'b0),
        .i_top_error(1'b0),
        .i_top_state_debug(8'd0),
        .i_top_phase_debug(8'd0),
        .i_layers_started({LAYER_COUNT_WIDTH{1'b0}}),
        .i_layers_completed({LAYER_COUNT_WIDTH{1'b0}}),
        .i_layer_done_mask({MAX_LAYERS{1'b0}}),
        .i_layer_error_mask({MAX_LAYERS{1'b0}}),
        .i_last_layer_output_base_addr(64'd0),
        .i_tail_error(1'b0),
        .i_tail_norm_saturation(1'b0),
        .i_tail_effective_final_hidden_base_addr(64'd0),
        .i_tail_best_token_id(32'd0),
        .i_tail_best_score_q26(56'sd0),
        .i_tail_tiles_started(32'd0),
        .i_tail_tiles_completed(32'd0),
        .i_mem_read_burst_count(32'd0),
        .i_mem_read_word_count(32'd0),
        .i_mem_write_req_count(32'd0),
        .i_mem_write_word_count(32'd0)
    );

    task automatic check(input logic condition, input string message);
        if (!condition) begin
            $display("FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic mmio_write(input logic [11:0] addr, input logic [31:0] data);
        begin
            @(negedge clk);
            reg_addr = addr;
            reg_wdata = data;
            reg_wr_valid = 1'b1;
            @(posedge clk);
            #1;
            check(reg_wr_ready && !reg_error, "MMIO write handshake/error");
            @(negedge clk);
            reg_wr_valid = 1'b0;
        end
    endtask

    task automatic mmio_read(input logic [11:0] addr, output logic [31:0] data);
        begin
            @(negedge clk);
            reg_addr = addr;
            reg_rd_valid = 1'b1;
            @(posedge clk);
            #1;
            check(reg_rd_ready && !reg_error, "MMIO read handshake/error");
            data = reg_rdata;
            @(negedge clk);
            reg_rd_valid = 1'b0;
        end
    endtask

    task automatic program_table(
        input logic [7:0] table_id,
        input logic [7:0] layer_id,
        input logic [63:0] base_addr
    );
        begin
            mmio_write(REG_TABLE_SELECT, {16'd0, layer_id, table_id});
            mmio_write(REG_TABLE_LO, base_addr[31:0]);
            mmio_write(REG_TABLE_HI, base_addr[63:32]);
            mmio_write(REG_TABLE_COMMIT, 32'd1);
        end
    endtask

    logic [31:0] read_lo;
    logic [31:0] read_hi;

    initial begin
        reg_wr_valid = 1'b0;
        reg_rd_valid = 1'b0;
        reg_addr = '0;
        reg_wdata = '0;
        runtime_layer = '0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        program_table(8'd0, 8'd1, 64'h0000_0004_1008_0000);
        program_table(8'd1, 8'd1, 64'h0000_0004_150A_0000);
        program_table(8'd9, 8'd2, 64'h0000_0004_2509_0000);

        mmio_write(REG_TABLE_SELECT, {16'd0, 8'd1, 8'd0});
        repeat (2) @(posedge clk);
        mmio_read(REG_TABLE_LO, read_lo);
        mmio_read(REG_TABLE_HI, read_hi);
        check({read_hi, read_lo} == 64'h0000_0004_1008_0000,
              "BRAM software readback did not preserve layer1 QKV");

        runtime_layer = 5'd1;
        repeat (16) @(posedge clk);
        #1;
        check(runtime_table_valid && (runtime_table_layer == 5'd1),
              "BRAM runtime table scan did not complete for layer1");
        check(qkv_addr == 64'h0000_0004_1008_0000,
              "BRAM runtime QKV lookup mismatch for layer1");
        check(input_norm_addr == 64'h0000_0004_150A_0000,
              "BRAM runtime input-norm lookup mismatch for layer1");
        check(residual_addr == 64'd0,
              "BRAM runtime residual lookup should be zero for layer1");

        runtime_layer = 5'd2;
        repeat (16) @(posedge clk);
        #1;
        check(runtime_table_valid && (runtime_table_layer == 5'd2),
              "BRAM runtime table scan did not complete for layer2");
        check(residual_addr == 64'h0000_0004_2509_0000,
              "BRAM runtime residual lookup mismatch for layer2");
        check(qkv_addr == 64'd0,
              "BRAM runtime QKV lookup should be zero for layer2");

        check((legacy_qkv_table == '0) &&
              (legacy_input_norm_table == '0) &&
              (legacy_frontend_table == '0) &&
              (legacy_score_table == '0) &&
              (legacy_o_proj_table == '0) &&
              (legacy_post_norm_table == '0) &&
              (legacy_gate_up_table == '0) &&
              (legacy_silu_table == '0) &&
              (legacy_down_table == '0) &&
              (legacy_residual_table == '0),
              "legacy flattened outputs should remain prunable constants in BRAM mode");

        mmio_read(REG_STATUS, read_lo);
        check(read_lo[3:0] == 4'd0, "unexpected sticky status after valid BRAM commits");

        $display("PASS: qmap_one_token_control_regs BRAM tables preserve MMIO readback and synchronous runtime lookup.");
        $finish;
    end

endmodule

`default_nettype wire
