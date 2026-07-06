`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_one_token_layer_scheduler_validation;

    localparam int ADDR_WIDTH = 64;
    localparam int MEM_DATA_WIDTH = 32;
    localparam int MAX_LAYERS = 28;
    localparam int BASE_TABLE_BITS = MAX_LAYERS * ADDR_WIDTH;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic [7 : 0] state_debug;
    logic [4 : 0] layer_start_index;
    logic [4 : 0] layer_count;
    logic [7 : 0] token_position;
    logic [ADDR_WIDTH-1 : 0] input_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] output_hidden_base_addr;
    logic [ADDR_WIDTH-1 : 0] kv_cache_base_addr;
    logic [BASE_TABLE_BITS-1 : 0] qkv_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] attn_frontend_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] attn_score_value_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] o_proj_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] post_attn_norm_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_gate_up_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_silu_mul_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_down_qmap_base_addr_table;
    logic [BASE_TABLE_BITS-1 : 0] mlp_residual_add_qmap_base_addr_table;
    logic [4 : 0] active_layer_index;
    logic [4 : 0] layers_started;
    logic [4 : 0] layers_completed;
    logic [27 : 0] layer_done_mask;
    logic [27 : 0] layer_error_mask;
    logic mem_rd_req_valid;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_ready;
    logic mem_wr_req_valid;
    logic [ADDR_WIDTH-1 : 0] mem_wr_req_addr;
    logic [15 : 0] mem_wr_req_len_bytes;
    logic [31 : 0] mem_wr_data;
    logic mem_wr_data_valid;
    logic mem_wr_data_last;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;

    integer cycle_count;
    integer rd_req_seen;
    integer wr_req_seen;
    integer wr_data_seen;

    qmap_one_token_layer_scheduler dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_layer_start_index(layer_start_index),
        .i_layer_count(layer_count),
        .i_position(token_position),
        .i_input_hidden_base_addr(input_hidden_base_addr),
        .i_output_hidden_base_addr(output_hidden_base_addr),
        .i_kv_cache_base_addr(kv_cache_base_addr),
        .i_qkv_qmap_base_addr_table(qkv_qmap_base_addr_table),
        .i_attn_frontend_qmap_base_addr_table(attn_frontend_qmap_base_addr_table),
        .i_attn_score_value_qmap_base_addr_table(attn_score_value_qmap_base_addr_table),
        .i_o_proj_qmap_base_addr_table(o_proj_qmap_base_addr_table),
        .i_post_attn_norm_qmap_base_addr_table(post_attn_norm_qmap_base_addr_table),
        .i_mlp_gate_up_qmap_base_addr_table(mlp_gate_up_qmap_base_addr_table),
        .i_mlp_silu_mul_qmap_base_addr_table(mlp_silu_mul_qmap_base_addr_table),
        .i_mlp_down_qmap_base_addr_table(mlp_down_qmap_base_addr_table),
        .i_mlp_residual_add_qmap_base_addr_table(mlp_residual_add_qmap_base_addr_table),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_state_debug(state_debug),
        .o_active_layer_index(active_layer_index),
        .o_layers_started(layers_started),
        .o_layers_completed(layers_completed),
        .o_layer_done_mask(layer_done_mask),
        .o_layer_error_mask(layer_error_mask),
        .o_layer0_active_stage_debug(),
        .o_layer0_state_debug(),
        .o_layer0_stage_done_mask(),
        .o_layer0_stage_error_mask(),
        .o_layer0_full_stage_done_mask(),
        .o_layer0_full_stage_error_mask(),
        .o_body_stage_done_mask(),
        .o_body_stage_error_mask(),
        .o_qkv_rows_done(),
        .o_qkv_last_row_sum_q26(),
        .o_qkv_last_output_q12_12(),
        .o_mem_read_burst_count(dut_read_burst_count),
        .o_mem_read_word_count(dut_read_word_count),
        .o_mem_write_req_count(dut_write_req_count),
        .o_mem_write_word_count(dut_write_word_count),
        .o_mem_rd_req_valid(mem_rd_req_valid),
        .i_mem_rd_req_ready(1'b1),
        .o_mem_rd_req_addr(mem_rd_req_addr),
        .o_mem_rd_req_len_bytes(mem_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(1'b0),
        .o_mem_rd_rsp_ready(mem_rd_rsp_ready),
        .i_mem_rd_rsp_data(32'd0),
        .i_mem_rd_rsp_last(1'b0),
        .o_mem_wr_req_valid(mem_wr_req_valid),
        .i_mem_wr_req_ready(1'b1),
        .o_mem_wr_req_addr(mem_wr_req_addr),
        .o_mem_wr_req_len_bytes(mem_wr_req_len_bytes),
        .o_mem_wr_data(mem_wr_data),
        .o_mem_wr_data_valid(mem_wr_data_valid),
        .i_mem_wr_data_ready(1'b1),
        .o_mem_wr_data_last(mem_wr_data_last),
        .i_mem_wr_done(1'b0),
        .i_mem_wr_error(1'b0)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task set_layer_packet_bases;
        input integer layer_index;
        begin
            qkv_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_QKV_BASE_ADDR;
            attn_frontend_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_FRONTEND_BASE_ADDR;
            attn_score_value_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_ATTN_SCORE_VALUE_BASE_ADDR;
            o_proj_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_O_PROJ_BASE_ADDR;
            post_attn_norm_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_POST_ATTN_NORM_BASE_ADDR;
            mlp_gate_up_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_GATE_UP_BASE_ADDR;
            mlp_silu_mul_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_SILU_MUL_BASE_ADDR;
            mlp_down_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_DOWN_BASE_ADDR;
            mlp_residual_add_qmap_base_addr_table[layer_index*ADDR_WIDTH +: ADDR_WIDTH] = `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR;
        end
    endtask

    task set_valid_contract;
        begin
            layer_start_index = 5'd0;
            layer_count = 5'd1;
            token_position = 8'd4;
            input_hidden_base_addr = 64'h0000_0004_0008_2000;
            output_hidden_base_addr = 64'h0000_0004_0508_1400;
            kv_cache_base_addr = 64'h0000_0004_1410_0000;
            qkv_qmap_base_addr_table = '0;
            attn_frontend_qmap_base_addr_table = '0;
            attn_score_value_qmap_base_addr_table = '0;
            o_proj_qmap_base_addr_table = '0;
            post_attn_norm_qmap_base_addr_table = '0;
            mlp_gate_up_qmap_base_addr_table = '0;
            mlp_silu_mul_qmap_base_addr_table = '0;
            mlp_down_qmap_base_addr_table = '0;
            mlp_residual_add_qmap_base_addr_table = '0;
            set_layer_packet_bases(0);
        end
    endtask

    task reset_case;
        begin
            rst_n = 1'b0;
            start = 1'b0;
            cycle_count = 0;
            rd_req_seen = 0;
            wr_req_seen = 0;
            wr_data_seen = 0;
            set_valid_contract();
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_until_done;
        input integer timeout_cycles;
        begin
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            while ((done != 1'b1) && (cycle_count < timeout_cycles)) begin
                @(posedge clk);
            end
            if (!done) begin
                $display("FAIL: validation case timed out state=0x%0h active_layer=%0d", state_debug, active_layer_index);
                $finish(1);
            end
            @(posedge clk);
        end
    endtask

    task expect_no_memory_error;
        input string case_name;
        input logic [27 : 0] expected_error_mask;
        begin
            if (!error) begin
                $display("FAIL: %s did not assert error", case_name);
                $finish(1);
            end
            if ((layers_started != 5'd0) ||
                (layers_completed != 5'd0) ||
                (layer_done_mask != 28'h0000000) ||
                (layer_error_mask != expected_error_mask)) begin
                $display(
                    "FAIL: %s mask mismatch started=%0d completed=%0d done=0x%0h error=0x%0h expected_error=0x%0h",
                    case_name,
                    layers_started,
                    layers_completed,
                    layer_done_mask,
                    layer_error_mask,
                    expected_error_mask
                );
                $finish(1);
            end
            if ((dut_read_burst_count != 0) ||
                (dut_read_word_count != 0) ||
                (dut_write_req_count != 0) ||
                (dut_write_word_count != 0) ||
                (rd_req_seen != 0) ||
                (wr_req_seen != 0) ||
                (wr_data_seen != 0)) begin
                $display(
                    "FAIL: %s issued memory traffic rd=%0d/%0d wr=%0d/%0d seen=%0d/%0d/%0d",
                    case_name,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    rd_req_seen,
                    wr_req_seen,
                    wr_data_seen
                );
                $finish(1);
            end
            $display("PASS_CASE: %s no-memory error mask 0x%0h", case_name, layer_error_mask);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end
        else begin
            cycle_count <= cycle_count + 1;
            if (mem_rd_req_valid) begin
                rd_req_seen <= rd_req_seen + 1;
            end
            if (mem_wr_req_valid) begin
                wr_req_seen <= wr_req_seen + 1;
            end
            if (mem_wr_data_valid) begin
                wr_data_seen <= wr_data_seen + 1;
            end
        end
    end

    initial begin
        reset_case();
        qkv_qmap_base_addr_table[0*ADDR_WIDTH +: ADDR_WIDTH] = '0;
        run_until_done(20);
        expect_no_memory_error("missing_layer0_qkv_base", 28'h0000001);

        reset_case();
        layer_count = 5'd2;
        run_until_done(20);
        expect_no_memory_error("missing_layer1_packet_bases", 28'h0000002);

        reset_case();
        layer_start_index = 5'd28;
        run_until_done(20);
        expect_no_memory_error("out_of_range_layer_start", 28'h8000000);

        reset_case();
        layer_count = 5'd0;
        run_until_done(20);
        expect_no_memory_error("zero_layer_count", 28'h0000001);

        reset_case();
        input_hidden_base_addr = '0;
        run_until_done(20);
        expect_no_memory_error("zero_input_hidden_base", 28'h0000001);

        $display("PASS: qmap_one_token_layer_scheduler validation-only no-memory contract checks passed.");
        $finish;
    end

endmodule

`default_nettype wire
