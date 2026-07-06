`timescale 1ns/1ps
`default_nettype none

`include "qmap_defs.svh"

module tb_qmap_layer0_body_scheduler;

    localparam int ADDR_WIDTH = 64;
    localparam int MEM_DATA_WIDTH = 32;
    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;

    localparam int POST_IMAGE_BYTES = 32'h0000_8000;
    localparam int GATE_IMAGE_BYTES = 32'h0000_E000;
    localparam int SILU_IMAGE_BYTES = 32'h0000_E000;
    localparam int DOWN_IMAGE_BYTES = 32'h0000_6000;
    localparam int RESIDUAL_IMAGE_BYTES = 32'h0000_5000;

    localparam int POST_WORDS = POST_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int GATE_WORDS = GATE_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int SILU_WORDS = SILU_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int DOWN_WORDS = DOWN_IMAGE_BYTES / MEM_DATA_BYTES;
    localparam int RESIDUAL_WORDS = RESIDUAL_IMAGE_BYTES / MEM_DATA_BYTES;

    localparam int VEC1024 = 1024;
    localparam int VEC3072 = 3072;
    localparam int DESCRIPTOR_WORDS = 32;
    localparam int DESCRIPTOR_TABLE_WORD_OFFSET = 32'h0100 / 4;
    localparam int DESC_DTYPE_WORD = 2;
    localparam int DESC_BASE_LO_WORD = 8;
    localparam int DESC_BASE_HI_WORD = 9;

    localparam int POST_SLOT_GAMMA = 3;
    localparam int POST_SLOT_HIDDEN = 4;
    localparam int POST_SLOT_NORM = 5;
    localparam int GATE_SLOT_ACTIVATION = 1;
    localparam int GATE_SLOT_GATE_OUTPUT = 6;
    localparam int GATE_SLOT_UP_OUTPUT = 7;
    localparam int SILU_SLOT_GATE = 1;
    localparam int SILU_SLOT_UP = 2;
    localparam int SILU_SLOT_HIDDEN = 4;
    localparam int DOWN_SLOT_ACTIVATION = 1;
    localparam int DOWN_SLOT_OUTPUT = 4;
    localparam int RESIDUAL_SLOT_POST_ATTN = 1;
    localparam int RESIDUAL_SLOT_DOWN = 2;
    localparam int RESIDUAL_SLOT_OUTPUT = 3;

    localparam int GATE_WEIGHT_WORDS = (3072 * 512) / 4;
    localparam int GATE_SCALE_WORDS = (3072 * 32) / 4;
    localparam int DOWN_WEIGHT_WORDS = (1024 * 1536) / 4;
    localparam int DOWN_SCALE_WORDS = (1024 * 96) / 4;

    localparam int WRITE_POST_HIDDEN = 1;
    localparam int WRITE_POST_NORM = 2;
    localparam int WRITE_GATE = 3;
    localparam int WRITE_UP = 4;
    localparam int WRITE_SILU_HIDDEN = 5;
    localparam int WRITE_DOWN = 6;
    localparam int WRITE_LAYER = 7;

    localparam int EXPECTED_NORMAL_RD_REQS = 15465;
    localparam int EXPECTED_NORMAL_RD_WORDS = 1270961;
    localparam int EXPECTED_NORMAL_WR_REQS = 7;
    localparam int EXPECTED_NORMAL_WR_WORDS = 13312;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic error;
    logic [2 : 0] active_stage_debug;
    logic [7 : 0] state_debug;
    logic [4 : 0] stage_done_mask;
    logic [4 : 0] stage_error_mask;
    logic [31 : 0] dut_read_burst_count;
    logic [31 : 0] dut_read_word_count;
    logic [31 : 0] dut_write_req_count;
    logic [31 : 0] dut_write_word_count;

    logic mem_rd_req_valid;
    logic mem_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] mem_rd_req_addr;
    logic [15 : 0] mem_rd_req_len_bytes;
    logic mem_rd_rsp_valid;
    logic mem_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] mem_rd_rsp_data;
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

    logic [31 : 0] post_qmap [0 : POST_WORDS-1];
    logic [31 : 0] gate_qmap [0 : GATE_WORDS-1];
    logic [31 : 0] silu_qmap [0 : SILU_WORDS-1];
    logic [31 : 0] down_qmap [0 : DOWN_WORDS-1];
    logic [31 : 0] residual_qmap [0 : RESIDUAL_WORDS-1];

    logic [31 : 0] gate_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] gate_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] up_weight_mem [0 : GATE_WEIGHT_WORDS-1];
    logic [31 : 0] up_scale_mem [0 : GATE_SCALE_WORDS-1];
    logic [31 : 0] down_weight_mem [0 : DOWN_WEIGHT_WORDS-1];
    logic [31 : 0] down_scale_mem [0 : DOWN_SCALE_WORDS-1];

    logic [31 : 0] expected_post_hidden [0 : VEC1024-1];
    logic [31 : 0] expected_post_norm [0 : VEC1024-1];
    logic [31 : 0] expected_gate [0 : VEC3072-1];
    logic [31 : 0] expected_up [0 : VEC3072-1];
    logic [31 : 0] expected_silu_hidden [0 : VEC3072-1];
    logic [31 : 0] expected_down [0 : VEC1024-1];
    logic [31 : 0] expected_layer [0 : VEC1024-1];

    logic [ADDR_WIDTH-1 : 0] post_hidden_base;
    logic [ADDR_WIDTH-1 : 0] post_norm_base;
    logic [ADDR_WIDTH-1 : 0] gate_output_base;
    logic [ADDR_WIDTH-1 : 0] up_output_base;
    logic [ADDR_WIDTH-1 : 0] silu_hidden_base;
    logic [ADDR_WIDTH-1 : 0] down_output_base;
    logic [ADDR_WIDTH-1 : 0] layer_output_base;

    string tracefile;
    string wavefile;
    integer trace_fd;
    integer cycle_count;
    integer mismatch_count;
    integer print_count;
    integer done_seen_count;
    integer last_done_cycle;
    integer normal_done_cycle;
    integer invalid_done_cycle;
    integer spurious_start_seen_busy;

    integer mem_req_fire_count;
    integer mem_rsp_fire_count;
    integer mem_wr_req_count;
    integer mem_wr_word_count_total;
    integer write_mismatch_count;
    longint signed max_abs_diff;

    integer normal_read_bursts;
    integer normal_read_words;
    integer normal_write_reqs;
    integer normal_write_words;
    logic [4 : 0] normal_stage_done_mask;
    logic [4 : 0] normal_stage_error_mask;
    logic normal_error;
    integer invalid_read_bursts;
    integer invalid_read_words;
    integer invalid_write_reqs;
    integer invalid_write_words;
    logic [4 : 0] invalid_stage_done_mask;
    logic [4 : 0] invalid_stage_error_mask;
    logic invalid_error;

    logic read_active;
    integer active_read_index;
    integer active_words_left;
    integer active_total_words;
    integer read_gap_count;
    logic [ADDR_WIDTH-1 : 0] active_read_addr;
    logic [31 : 0] current_read_word;

    logic write_active;
    integer active_write_kind;
    integer active_write_index;
    integer active_write_words_left;
    integer active_write_total_words;
    integer write_done_delay;
    logic [ADDR_WIDTH-1 : 0] active_write_addr;

    logic post_hidden_written;
    logic post_norm_written;
    logic gate_written;
    logic up_written;
    logic silu_hidden_written;
    logic down_written;
    logic layer_written;

    logic rd_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_rd_req_addr;
    logic [15 : 0] stalled_rd_req_len;
    logic rd_rsp_stall_active;
    logic [31 : 0] stalled_rd_rsp_data;
    logic stalled_rd_rsp_last;
    logic wr_req_stall_active;
    logic [ADDR_WIDTH-1 : 0] stalled_wr_req_addr;
    logic [15 : 0] stalled_wr_req_len;
    logic wr_data_stall_active;
    logic [31 : 0] stalled_wr_data;
    logic stalled_wr_last;
    integer last_trace_stage;
    integer last_trace_state;
    integer last_trace_wr_words;

    qmap_layer0_body_scheduler dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_start(start),
        .i_post_attn_norm_qmap_base_addr(`QMAP_POST_ATTN_NORM_BASE_ADDR),
        .i_mlp_gate_up_qmap_base_addr(`QMAP_MLP_GATE_UP_BASE_ADDR),
        .i_mlp_silu_mul_qmap_base_addr(`QMAP_MLP_SILU_MUL_BASE_ADDR),
        .i_mlp_down_qmap_base_addr(`QMAP_MLP_DOWN_BASE_ADDR),
        .i_mlp_residual_add_qmap_base_addr(`QMAP_MLP_RESIDUAL_ADD_BASE_ADDR),
        .o_busy(busy),
        .o_done(done),
        .o_error(error),
        .o_active_stage_debug(active_stage_debug),
        .o_state_debug(state_debug),
        .o_stage_done_mask(stage_done_mask),
        .o_stage_error_mask(stage_error_mask),
        .o_mem_read_burst_count(dut_read_burst_count),
        .o_mem_read_word_count(dut_read_word_count),
        .o_mem_write_req_count(dut_write_req_count),
        .o_mem_write_word_count(dut_write_word_count),
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

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wavefile = "FPGA_Project/wave/qmap_layer0_body_scheduler.vcd";
        if ($value$plusargs("wavefile=%s", wavefile)) begin
        end
        if ($test$plusargs("dumpwaves")) begin
            $dumpfile(wavefile);
            $dumpvars(0, clk);
            $dumpvars(0, rst_n);
            $dumpvars(0, start);
            $dumpvars(0, busy);
            $dumpvars(0, done);
            $dumpvars(0, error);
            $dumpvars(0, active_stage_debug);
            $dumpvars(0, state_debug);
            $dumpvars(0, stage_done_mask);
            $dumpvars(0, stage_error_mask);
        end
    end

    assign mem_rd_req_ready =
        (!read_active) &&
        (!mem_rd_rsp_valid) &&
        ((cycle_count % 11) != 3) &&
        (((cycle_count + mem_req_fire_count) % 29) != 7);
    assign mem_wr_req_ready =
        (!write_active) &&
        ((cycle_count % 13) != 5) &&
        (((cycle_count + mem_wr_req_count) % 31) != 9);
    assign mem_wr_data_ready =
        write_active &&
        ((cycle_count % 7) != 2) &&
        (((cycle_count + mem_wr_word_count_total) % 23) != 10);

    function automatic integer desc_idx(input integer slot, input integer word_offset);
        begin
            desc_idx = DESCRIPTOR_TABLE_WORD_OFFSET + (slot * DESCRIPTOR_WORDS) + word_offset;
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] post_desc_base(input integer slot);
        begin
            post_desc_base = {post_qmap[desc_idx(slot, DESC_BASE_HI_WORD)], post_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]};
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] gate_desc_base(input integer slot);
        begin
            gate_desc_base = {gate_qmap[desc_idx(slot, DESC_BASE_HI_WORD)], gate_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]};
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] silu_desc_base(input integer slot);
        begin
            silu_desc_base = {silu_qmap[desc_idx(slot, DESC_BASE_HI_WORD)], silu_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]};
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] down_desc_base(input integer slot);
        begin
            down_desc_base = {down_qmap[desc_idx(slot, DESC_BASE_HI_WORD)], down_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]};
        end
    endfunction

    function automatic logic [ADDR_WIDTH-1 : 0] residual_desc_base(input integer slot);
        begin
            residual_desc_base = {residual_qmap[desc_idx(slot, DESC_BASE_HI_WORD)], residual_qmap[desc_idx(slot, DESC_BASE_LO_WORD)]};
        end
    endfunction

    task patch_gate_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            gate_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            gate_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_silu_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            silu_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            silu_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_down_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            down_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            down_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    task patch_residual_base;
        input integer slot;
        input logic [ADDR_WIDTH-1 : 0] base_addr;
        begin
            residual_qmap[desc_idx(slot, DESC_BASE_LO_WORD)] = base_addr[31 : 0];
            residual_qmap[desc_idx(slot, DESC_BASE_HI_WORD)] = base_addr[63 : 32];
        end
    endtask

    function automatic logic in_range(
        input logic [ADDR_WIDTH-1 : 0] addr,
        input logic [ADDR_WIDTH-1 : 0] base_addr,
        input integer bytes
    );
        begin
            in_range = (addr >= base_addr) && (addr < (base_addr + bytes));
        end
    endfunction

    task load_vectors;
        begin
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_image_words32.hex", post_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_image_words32.hex", gate_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_silu_mul_image_words32.hex", silu_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_down_image_words32.hex", down_qmap);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_residual_add_image_words32.hex", residual_qmap);

            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_gate_weight_words32.hex", gate_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_gate_scale_words32.hex", gate_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_up_weight_words32.hex", up_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_gate_up_proj_stage_real_up_scale_words32.hex", up_scale_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_down_proj_stage_real_weight_words32.hex", down_weight_mem);
            $readmemh("FPGA_Project/sim/vectors/mlp_down_proj_stage_real_scale_words32.hex", down_scale_mem);

            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_hidden_words32.hex", expected_post_hidden);
            $readmemh("FPGA_Project/sim/vectors/qmap_post_attention_residual_norm_expected_norm_words32.hex", expected_post_norm);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_expected_gate_words32.hex", expected_gate);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_gate_up_expected_up_words32.hex", expected_up);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_silu_mul_expected_hidden_words32.hex", expected_silu_hidden);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_down_expected_words32.hex", expected_down);
            $readmemh("FPGA_Project/sim/vectors/qmap_mlp_residual_add_expected_words32.hex", expected_layer);

            post_hidden_base = post_desc_base(POST_SLOT_HIDDEN);
            post_norm_base = post_desc_base(POST_SLOT_NORM);
            gate_output_base = gate_desc_base(GATE_SLOT_GATE_OUTPUT);
            up_output_base = gate_desc_base(GATE_SLOT_UP_OUTPUT);
            silu_hidden_base = silu_desc_base(SILU_SLOT_HIDDEN);
            down_output_base = down_desc_base(DOWN_SLOT_OUTPUT);
            layer_output_base = residual_desc_base(RESIDUAL_SLOT_OUTPUT);

            patch_gate_base(GATE_SLOT_ACTIVATION, post_norm_base);
            patch_silu_base(SILU_SLOT_GATE, gate_output_base);
            patch_silu_base(SILU_SLOT_UP, up_output_base);
            patch_down_base(DOWN_SLOT_ACTIVATION, silu_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_POST_ATTN, post_hidden_base);
            patch_residual_base(RESIDUAL_SLOT_DOWN, down_output_base);
        end
    endtask

    function automatic logic [31 : 0] memory_word(input logic [ADDR_WIDTH-1 : 0] addr);
        integer index;
        begin
            memory_word = 32'hBAD0_BAD0;
            if (in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2;
                memory_word = post_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2;
                memory_word = gate_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2;
                memory_word = silu_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_BASE_ADDR) >> 2;
                memory_word = down_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                memory_word = residual_qmap[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_WEIGHT_BASE_ADDR) >> 2;
                memory_word = gate_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_GATE_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_SCALE_BASE_ADDR) >> 2;
                memory_word = gate_scale_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_UP_WEIGHT_BASE_ADDR, GATE_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_UP_WEIGHT_BASE_ADDR) >> 2;
                memory_word = up_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_UP_SCALE_BASE_ADDR, GATE_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_UP_SCALE_BASE_ADDR) >> 2;
                memory_word = up_scale_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR, DOWN_WEIGHT_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_WEIGHT_BASE_ADDR) >> 2;
                memory_word = down_weight_mem[index];
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_SCALE_BASE_ADDR, DOWN_SCALE_WORDS * MEM_DATA_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_SCALE_BASE_ADDR) >> 2;
                memory_word = down_scale_mem[index];
            end
            else begin
                if (print_count < 32) begin
                    $display("FAIL: read from unknown address 0x%016h at cycle %0d", addr, cycle_count);
                    print_count = print_count + 1;
                end
                mismatch_count = mismatch_count + 1;
            end
        end
    endfunction

    task write_memory_word;
        input logic [ADDR_WIDTH-1 : 0] addr;
        input logic [31 : 0] data;
        integer index;
        begin
            if (in_range(addr, `QMAP_POST_ATTN_NORM_BASE_ADDR, POST_IMAGE_BYTES)) begin
                index = (addr - `QMAP_POST_ATTN_NORM_BASE_ADDR) >> 2;
                post_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_GATE_UP_BASE_ADDR, GATE_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_GATE_UP_BASE_ADDR) >> 2;
                gate_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_SILU_MUL_BASE_ADDR, SILU_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_SILU_MUL_BASE_ADDR) >> 2;
                silu_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_DOWN_BASE_ADDR, DOWN_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_DOWN_BASE_ADDR) >> 2;
                down_qmap[index] = data;
            end
            else if (in_range(addr, `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR, RESIDUAL_IMAGE_BYTES)) begin
                index = (addr - `QMAP_MLP_RESIDUAL_ADD_BASE_ADDR) >> 2;
                residual_qmap[index] = data;
            end
            else begin
                $display("FAIL: write to unknown address 0x%016h", addr);
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task check_chained_read_ready;
        input logic [ADDR_WIDTH-1 : 0] addr;
        begin
            if (in_range(addr, post_hidden_base, VEC1024 * MEM_DATA_BYTES) && !post_hidden_written) begin
                $display("FAIL: post_hidden read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
            if (in_range(addr, post_norm_base, VEC1024 * MEM_DATA_BYTES) && !post_norm_written) begin
                $display("FAIL: post_norm read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
            if (in_range(addr, gate_output_base, VEC3072 * MEM_DATA_BYTES) && !gate_written) begin
                $display("FAIL: gate output read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
            if (in_range(addr, up_output_base, VEC3072 * MEM_DATA_BYTES) && !up_written) begin
                $display("FAIL: up output read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
            if (in_range(addr, silu_hidden_base, VEC3072 * MEM_DATA_BYTES) && !silu_hidden_written) begin
                $display("FAIL: silu hidden read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
            if (in_range(addr, down_output_base, VEC1024 * MEM_DATA_BYTES) && !down_written) begin
                $display("FAIL: down output read before producer write completed");
                mismatch_count = mismatch_count + 1;
            end
        end
    endtask

    task classify_read_request;
        begin
            if ((mem_rd_req_addr[1:0] != 2'b00) || ((mem_rd_req_len_bytes % 4) != 0)) begin
                $display("FAIL: unaligned read request addr=0x%016h len=%0d", mem_rd_req_addr, mem_rd_req_len_bytes);
                mismatch_count = mismatch_count + 1;
            end
            check_chained_read_ready(mem_rd_req_addr);
            active_read_addr = mem_rd_req_addr;
            active_read_index = 0;
            active_words_left = mem_rd_req_len_bytes / MEM_DATA_BYTES;
            active_total_words = mem_rd_req_len_bytes / MEM_DATA_BYTES;
        end
    endtask

    task check_write_request;
        begin
            if (write_active) begin
                $display("FAIL: write request while another write is active");
                mismatch_count = mismatch_count + 1;
            end
            active_write_addr = mem_wr_req_addr;
            active_write_index = 0;
            active_write_words_left = mem_wr_req_len_bytes / MEM_DATA_BYTES;
            active_write_total_words = mem_wr_req_len_bytes / MEM_DATA_BYTES;
            write_active = 1'b1;
            write_done_delay = 0;

            if (mem_wr_req_addr == post_hidden_base) begin
                active_write_kind = WRITE_POST_HIDDEN;
                if (mem_wr_req_len_bytes != VEC1024 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == post_norm_base) begin
                active_write_kind = WRITE_POST_NORM;
                if (mem_wr_req_len_bytes != VEC1024 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == gate_output_base) begin
                active_write_kind = WRITE_GATE;
                if (mem_wr_req_len_bytes != VEC3072 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == up_output_base) begin
                active_write_kind = WRITE_UP;
                if (mem_wr_req_len_bytes != VEC3072 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == silu_hidden_base) begin
                active_write_kind = WRITE_SILU_HIDDEN;
                if (mem_wr_req_len_bytes != VEC3072 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == down_output_base) begin
                active_write_kind = WRITE_DOWN;
                if (mem_wr_req_len_bytes != VEC1024 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else if (mem_wr_req_addr == layer_output_base) begin
                active_write_kind = WRITE_LAYER;
                if (mem_wr_req_len_bytes != VEC1024 * MEM_DATA_BYTES) mismatch_count = mismatch_count + 1;
            end
            else begin
                $display("FAIL: unknown write request addr=0x%016h len=%0d", mem_wr_req_addr, mem_wr_req_len_bytes);
                active_write_kind = 0;
                mismatch_count = mismatch_count + 1;
            end

            mem_wr_req_count = mem_wr_req_count + 1;
        end
    endtask

    task check_write_word;
        logic [31 : 0] expected_word;
        longint signed expected_signed;
        longint signed actual_signed;
        longint signed diff;
        begin
            expected_word = 32'hDEAD_DEAD;
            case (active_write_kind)
                WRITE_POST_HIDDEN: expected_word = expected_post_hidden[active_write_index];
                WRITE_POST_NORM: expected_word = expected_post_norm[active_write_index];
                WRITE_GATE: expected_word = expected_gate[active_write_index];
                WRITE_UP: expected_word = expected_up[active_write_index];
                WRITE_SILU_HIDDEN: expected_word = expected_silu_hidden[active_write_index];
                WRITE_DOWN: expected_word = expected_down[active_write_index];
                WRITE_LAYER: expected_word = expected_layer[active_write_index];
                default: expected_word = 32'hDEAD_DEAD;
            endcase

            if (mem_wr_data !== expected_word) begin
                if (print_count < 32) begin
                    $display("FAIL: write mismatch kind=%0d idx=%0d actual=0x%08h expected=0x%08h",
                             active_write_kind, active_write_index, mem_wr_data, expected_word);
                    print_count = print_count + 1;
                end
                write_mismatch_count = write_mismatch_count + 1;
                mismatch_count = mismatch_count + 1;
            end

            expected_signed = $signed(expected_word);
            actual_signed = $signed(mem_wr_data);
            diff = actual_signed - expected_signed;
            if (diff < 0) diff = -diff;
            if (diff > max_abs_diff) max_abs_diff = diff;

            if (mem_wr_data_last != (active_write_words_left == 1)) begin
                $display("FAIL: write last mismatch kind=%0d idx=%0d last=%0d words_left=%0d",
                         active_write_kind, active_write_index, mem_wr_data_last, active_write_words_left);
                mismatch_count = mismatch_count + 1;
            end

            write_memory_word(active_write_addr + (active_write_index * MEM_DATA_BYTES), mem_wr_data);
            active_write_index = active_write_index + 1;
            active_write_words_left = active_write_words_left - 1;
            mem_wr_word_count_total = mem_wr_word_count_total + 1;

            if (active_write_words_left == 0) begin
                write_active = 1'b0;
                write_done_delay = 3;
                case (active_write_kind)
                    WRITE_POST_HIDDEN: post_hidden_written = 1'b1;
                    WRITE_POST_NORM: post_norm_written = 1'b1;
                    WRITE_GATE: gate_written = 1'b1;
                    WRITE_UP: up_written = 1'b1;
                    WRITE_SILU_HIDDEN: silu_hidden_written = 1'b1;
                    WRITE_DOWN: down_written = 1'b1;
                    WRITE_LAYER: layer_written = 1'b1;
                    default: begin
                    end
                endcase
            end
        end
    endtask

    task pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task wait_for_next_done;
        input integer prior_done_count;
        input integer timeout_delta;
        integer deadline;
        begin
            deadline = cycle_count + timeout_delta;
            while ((done_seen_count == prior_done_count) && (cycle_count < deadline)) begin
                @(negedge clk);
            end
            if (done_seen_count == prior_done_count) begin
                $display("FAIL: timed out waiting for scheduler done");
                mismatch_count = mismatch_count + 1;
                $finish(1);
            end
            @(negedge clk);
        end
    endtask

    task run_success;
        integer prior_done_count;
        begin
            prior_done_count = done_seen_count;
            pulse_start();

            repeat (256) @(negedge clk);
            if (busy) begin
                start = 1'b1;
                @(negedge clk);
                start = 1'b0;
            end

            wait_for_next_done(prior_done_count, 20000000);
            normal_done_cycle = last_done_cycle;
            normal_read_bursts = dut_read_burst_count;
            normal_read_words = dut_read_word_count;
            normal_write_reqs = dut_write_req_count;
            normal_write_words = dut_write_word_count;
            normal_stage_done_mask = stage_done_mask;
            normal_stage_error_mask = stage_error_mask;
            normal_error = error;
        end
    endtask

    task run_invalid_first_stage_descriptor;
        integer prior_done_count;
        logic [31 : 0] saved_dtype;
        integer saved_total_write_reqs;
        integer saved_total_write_words;
        begin
            prior_done_count = done_seen_count;
            saved_total_write_reqs = mem_wr_req_count;
            saved_total_write_words = mem_wr_word_count_total;
            saved_dtype = post_qmap[desc_idx(POST_SLOT_GAMMA, DESC_DTYPE_WORD)];
            post_qmap[desc_idx(POST_SLOT_GAMMA, DESC_DTYPE_WORD)] = `QMAP_DTYPE_U16_Q8_8;

            pulse_start();
            wait_for_next_done(prior_done_count, 200000);
            invalid_done_cycle = last_done_cycle;
            invalid_read_bursts = dut_read_burst_count;
            invalid_read_words = dut_read_word_count;
            invalid_write_reqs = dut_write_req_count;
            invalid_write_words = dut_write_word_count;
            invalid_stage_done_mask = stage_done_mask;
            invalid_stage_error_mask = stage_error_mask;
            invalid_error = error;

            if ((mem_wr_req_count != saved_total_write_reqs) ||
                (mem_wr_word_count_total != saved_total_write_words)) begin
                $display("FAIL: invalid scheduler run wrote data req=%0d/%0d words=%0d/%0d",
                         mem_wr_req_count, saved_total_write_reqs,
                         mem_wr_word_count_total, saved_total_write_words);
                mismatch_count = mismatch_count + 1;
            end

            post_qmap[desc_idx(POST_SLOT_GAMMA, DESC_DTYPE_WORD)] = saved_dtype;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            cycle_count <= 0;
            done_seen_count <= 0;
            last_done_cycle <= -1;
            last_trace_stage <= -1;
            last_trace_state <= -1;
            last_trace_wr_words <= -1;
        end
        else begin
            cycle_count <= cycle_count + 1;

            if ((start == 1'b1) && (busy == 1'b1)) begin
                spurious_start_seen_busy <= 1;
            end

            if (done == 1'b1) begin
                done_seen_count <= done_seen_count + 1;
                if ((last_done_cycle != -1) && ((cycle_count - last_done_cycle) == 1)) begin
                    $display("FAIL: adjacent done pulses at cycles %0d and %0d", last_done_cycle, cycle_count);
                    mismatch_count <= mismatch_count + 1;
                end
                last_done_cycle <= cycle_count;
            end

            if ((trace_fd != 0) &&
                (((mem_rd_req_valid == 1'b1) && (mem_rd_req_ready == 1'b1)) ||
                 ((mem_rd_rsp_valid == 1'b1) && (mem_rd_rsp_ready == 1'b1) && (mem_rd_rsp_last == 1'b1)) ||
                 ((mem_wr_req_valid == 1'b1) && (mem_wr_req_ready == 1'b1)) ||
                 ((mem_wr_data_valid == 1'b1) && (mem_wr_data_ready == 1'b1) && (mem_wr_data_last == 1'b1)) ||
                 (mem_wr_done == 1'b1) ||
                 (done == 1'b1) ||
                 (error == 1'b1) ||
                 (active_stage_debug != last_trace_stage) ||
                 (state_debug != last_trace_state) ||
                 (dut_write_word_count != last_trace_wr_words))) begin
                $fwrite(
                    trace_fd,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,0x%02h,0x%02h,0x%016h,%0d,%0d,%0d,0x%016h,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    cycle_count,
                    start,
                    busy,
                    done,
                    error,
                    active_stage_debug,
                    state_debug,
                    stage_done_mask,
                    stage_error_mask,
                    mem_rd_req_addr,
                    mem_rd_req_len_bytes,
                    mem_rd_req_valid && mem_rd_req_ready,
                    mem_rd_rsp_valid && mem_rd_rsp_ready && mem_rd_rsp_last,
                    mem_wr_req_addr,
                    mem_wr_req_len_bytes,
                    mem_wr_req_valid && mem_wr_req_ready,
                    mem_wr_data_valid && mem_wr_data_ready,
                    mem_wr_data_last,
                    dut_read_burst_count,
                    dut_read_word_count,
                    dut_write_req_count,
                    dut_write_word_count,
                    mem_wr_done,
                    post_norm_written,
                    silu_hidden_written,
                    down_written
                );
                last_trace_stage <= active_stage_debug;
                last_trace_state <= state_debug;
                last_trace_wr_words <= dut_write_word_count;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            mem_rd_rsp_valid <= 1'b0;
            mem_rd_rsp_data <= 32'd0;
            mem_rd_rsp_last <= 1'b0;
            read_active <= 1'b0;
            active_read_index <= 0;
            active_words_left <= 0;
            active_total_words <= 0;
            active_read_addr <= '0;
            read_gap_count <= 0;
            rd_req_stall_active <= 1'b0;
            rd_rsp_stall_active <= 1'b0;
        end
        else begin
            if (mem_rd_req_valid && !mem_rd_req_ready) begin
                if (!rd_req_stall_active) begin
                    rd_req_stall_active <= 1'b1;
                    stalled_rd_req_addr <= mem_rd_req_addr;
                    stalled_rd_req_len <= mem_rd_req_len_bytes;
                end
                else if ((mem_rd_req_addr !== stalled_rd_req_addr) ||
                         (mem_rd_req_len_bytes !== stalled_rd_req_len)) begin
                    $display("FAIL: read request changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                rd_req_stall_active <= 1'b0;
            end

            if (mem_rd_rsp_valid && !mem_rd_rsp_ready) begin
                if (!rd_rsp_stall_active) begin
                    rd_rsp_stall_active <= 1'b1;
                    stalled_rd_rsp_data <= mem_rd_rsp_data;
                    stalled_rd_rsp_last <= mem_rd_rsp_last;
                end
                else if ((mem_rd_rsp_data !== stalled_rd_rsp_data) ||
                         (mem_rd_rsp_last !== stalled_rd_rsp_last)) begin
                    $display("FAIL: read response changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                rd_rsp_stall_active <= 1'b0;
            end

            if (mem_rd_req_valid && mem_rd_req_ready) begin
                classify_read_request();
                read_active <= 1'b1;
                read_gap_count <= mem_req_fire_count % 3;
                mem_req_fire_count = mem_req_fire_count + 1;
            end

            if (mem_rd_rsp_valid && mem_rd_rsp_ready) begin
                mem_rsp_fire_count = mem_rsp_fire_count + 1;
                if (active_words_left <= 1) begin
                    mem_rd_rsp_valid <= 1'b0;
                    read_active <= 1'b0;
                    active_words_left <= 0;
                    mem_rd_rsp_last <= 1'b0;
                end
                else begin
                    mem_rd_rsp_valid <= 1'b0;
                    active_read_index <= active_read_index + 1;
                    active_words_left <= active_words_left - 1;
                    read_gap_count <= ((active_total_words - active_words_left) % 17 == 0) ? 1 : 0;
                end
            end
            else if (read_active && !mem_rd_rsp_valid) begin
                if (read_gap_count > 0) begin
                    read_gap_count <= read_gap_count - 1;
                end
                else begin
                    current_read_word = memory_word(active_read_addr + (active_read_index * MEM_DATA_BYTES));
                    mem_rd_rsp_data <= current_read_word;
                    mem_rd_rsp_last <= (active_words_left == 1);
                    mem_rd_rsp_valid <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 1'b0) begin
            write_active <= 1'b0;
            active_write_kind <= 0;
            active_write_index <= 0;
            active_write_words_left <= 0;
            active_write_total_words <= 0;
            active_write_addr <= '0;
            write_done_delay <= 0;
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;
            wr_req_stall_active <= 1'b0;
            wr_data_stall_active <= 1'b0;
        end
        else begin
            mem_wr_done <= 1'b0;
            mem_wr_error <= 1'b0;

            if (mem_wr_req_valid && !mem_wr_req_ready) begin
                if (!wr_req_stall_active) begin
                    wr_req_stall_active <= 1'b1;
                    stalled_wr_req_addr <= mem_wr_req_addr;
                    stalled_wr_req_len <= mem_wr_req_len_bytes;
                end
                else if ((mem_wr_req_addr !== stalled_wr_req_addr) ||
                         (mem_wr_req_len_bytes !== stalled_wr_req_len)) begin
                    $display("FAIL: write request changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                wr_req_stall_active <= 1'b0;
            end

            if (mem_wr_data_valid && !mem_wr_data_ready) begin
                if (!wr_data_stall_active) begin
                    wr_data_stall_active <= 1'b1;
                    stalled_wr_data <= mem_wr_data;
                    stalled_wr_last <= mem_wr_data_last;
                end
                else if ((mem_wr_data !== stalled_wr_data) ||
                         (mem_wr_data_last !== stalled_wr_last)) begin
                    $display("FAIL: write data changed while stalled");
                    mismatch_count = mismatch_count + 1;
                end
            end
            else begin
                wr_data_stall_active <= 1'b0;
            end

            if (mem_wr_req_valid && mem_wr_req_ready) begin
                check_write_request();
            end

            if (mem_wr_data_valid && mem_wr_data_ready) begin
                if (!write_active) begin
                    $display("FAIL: write data without active write request");
                    mismatch_count = mismatch_count + 1;
                end
                else begin
                    check_write_word();
                end
            end

            if (write_done_delay > 0) begin
                write_done_delay <= write_done_delay - 1;
                if (write_done_delay == 1) begin
                    mem_wr_done <= 1'b1;
                end
            end
        end
    end

    initial begin : main_test
        tracefile = "FPGA_Project/sim/qmap_layer0_body_scheduler_trace.csv";
        if ($value$plusargs("tracefile=%s", tracefile)) begin
        end

        trace_fd = $fopen(tracefile, "w");
        if (trace_fd == 0) begin
            $display("FAIL: could not open trace file %s", tracefile);
            $finish(1);
        end
        $fwrite(
            trace_fd,
            "cycle,start,busy,done,error,stage,state,stage_done_mask,stage_error_mask,rd_addr,rd_len,rd_req_fire,rd_rsp_last_fire,wr_addr,wr_len,wr_req_fire,wr_data_fire,wr_last,rd_bursts,rd_words,wr_reqs,wr_words,wr_done,post_norm_written,silu_hidden_written,down_written\n"
        );

        load_vectors();

        start = 1'b0;
        rst_n = 1'b0;
        mismatch_count = 0;
        print_count = 0;
        spurious_start_seen_busy = 0;
        mem_req_fire_count = 0;
        mem_rsp_fire_count = 0;
        mem_wr_req_count = 0;
        mem_wr_word_count_total = 0;
        write_mismatch_count = 0;
        max_abs_diff = 0;
        normal_done_cycle = 0;
        invalid_done_cycle = 0;
        normal_read_bursts = 0;
        normal_read_words = 0;
        normal_write_reqs = 0;
        normal_write_words = 0;
        normal_stage_done_mask = 5'd0;
        normal_stage_error_mask = 5'd0;
        normal_error = 1'b0;
        invalid_read_bursts = 0;
        invalid_read_words = 0;
        invalid_write_reqs = 0;
        invalid_write_words = 0;
        invalid_stage_done_mask = 5'd0;
        invalid_stage_error_mask = 5'd0;
        invalid_error = 1'b0;
        post_hidden_written = 1'b0;
        post_norm_written = 1'b0;
        gate_written = 1'b0;
        up_written = 1'b0;
        silu_hidden_written = 1'b0;
        down_written = 1'b0;
        layer_written = 1'b0;

        repeat (8) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        run_success();
        run_invalid_first_stage_descriptor();

        $fclose(trace_fd);
        trace_fd = 0;

        $display("qmap_layer0_body_scheduler chained Layer 0 body test");
        $display("  normal done cycle      = %0d", normal_done_cycle);
        $display("  invalid done cycle     = %0d", invalid_done_cycle);
        $display("  normal rd/wr           = %0d/%0d reads, %0d/%0d writes",
                 normal_read_bursts, normal_read_words, normal_write_reqs, normal_write_words);
        $display("  invalid rd/wr          = %0d/%0d reads, %0d/%0d writes",
                 invalid_read_bursts, invalid_read_words, invalid_write_reqs, invalid_write_words);
        $display("  external req/rsp       = %0d / %0d", mem_req_fire_count, mem_rsp_fire_count);
        $display("  external write req/word= %0d / %0d", mem_wr_req_count, mem_wr_word_count_total);
        $display("  stage masks normal     = done 0x%02h error 0x%02h",
                 normal_stage_done_mask, normal_stage_error_mask);
        $display("  stage masks invalid    = done 0x%02h error 0x%02h",
                 invalid_stage_done_mask, invalid_stage_error_mask);
        $display("  producer writes        = post_hidden %0d post_norm %0d gate %0d up %0d silu %0d down %0d layer %0d",
                 post_hidden_written, post_norm_written, gate_written, up_written,
                 silu_hidden_written, down_written, layer_written);
        $display("  write mismatches       = %0d", write_mismatch_count);
        $display("  max_abs write diff     = %0d", max_abs_diff);
        $display("  spurious start covered = %0d", spurious_start_seen_busy);
        $display("  trace                  = %s", tracefile);

        if (done_seen_count != 2) begin
            $display("FAIL: done pulse count mismatch actual=%0d expected=2", done_seen_count);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_error) begin
            $display("FAIL: valid scheduler run asserted error");
            mismatch_count = mismatch_count + 1;
        end
        if (normal_stage_done_mask != 5'b1_1111) begin
            $display("FAIL: normal stage done mask mismatch actual=0x%02h", normal_stage_done_mask);
            mismatch_count = mismatch_count + 1;
        end
        if (normal_stage_error_mask != 5'd0) begin
            $display("FAIL: normal stage error mask mismatch actual=0x%02h", normal_stage_error_mask);
            mismatch_count = mismatch_count + 1;
        end
        if ((normal_read_bursts != EXPECTED_NORMAL_RD_REQS) ||
            (normal_read_words != EXPECTED_NORMAL_RD_WORDS) ||
            (normal_write_reqs != EXPECTED_NORMAL_WR_REQS) ||
            (normal_write_words != EXPECTED_NORMAL_WR_WORDS)) begin
            $display("FAIL: normal aggregate counters mismatch rd=%0d/%0d wr=%0d/%0d",
                     normal_read_bursts, normal_read_words,
                     normal_write_reqs, normal_write_words);
            mismatch_count = mismatch_count + 1;
        end
        if (!post_hidden_written || !post_norm_written || !gate_written ||
            !up_written || !silu_hidden_written || !down_written || !layer_written) begin
            $display("FAIL: not all chained producer writes completed");
            mismatch_count = mismatch_count + 1;
        end
        if ((write_mismatch_count != 0) || (max_abs_diff != 0)) begin
            $display("FAIL: exact write-back expected mismatch=%0d max_abs=%0d",
                     write_mismatch_count, max_abs_diff);
            mismatch_count = mismatch_count + 1;
        end
        if (spurious_start_seen_busy != 1) begin
            $display("FAIL: busy-period spurious start was not covered");
            mismatch_count = mismatch_count + 1;
        end
        if (invalid_error != 1'b1) begin
            $display("FAIL: invalid scheduler run did not assert error");
            mismatch_count = mismatch_count + 1;
        end
        if ((invalid_stage_done_mask != 5'd0) || (invalid_stage_error_mask != 5'b0_0001)) begin
            $display("FAIL: invalid stage masks mismatch done=0x%02h error=0x%02h",
                     invalid_stage_done_mask, invalid_stage_error_mask);
            mismatch_count = mismatch_count + 1;
        end
        if ((invalid_write_reqs != 0) || (invalid_write_words != 0)) begin
            $display("FAIL: invalid scheduler run wrote data req=%0d words=%0d",
                     invalid_write_reqs, invalid_write_words);
            mismatch_count = mismatch_count + 1;
        end

        if (mismatch_count != 0) begin
            $display("FAIL: qmap_layer0_body_scheduler found %0d mismatch(es)", mismatch_count);
            $finish(1);
        end

        $display("PASS: qmap_layer0_body_scheduler chained all QMAP body wrappers with exact write-back and error propagation.");
        $display("Waveform: %s", wavefile);
        $finish;
    end

endmodule

`default_nettype wire
