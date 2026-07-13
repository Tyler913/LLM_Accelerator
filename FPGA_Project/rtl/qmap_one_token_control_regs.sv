`default_nettype none

// PS-visible control register block for the one-token top.
//
// This block intentionally uses a tiny valid/ready MMIO-style interface rather
// than AXI-Lite. It defines the software-visible contract now; an AXI-Lite
// adapter can wrap the same register map later.
module qmap_one_token_control_regs #(
    parameter int ADDR_WIDTH        = 64,
    parameter int MAX_LAYERS        = 28,
    parameter int MAX_CONTEXT       = 256,
    parameter int TOKEN_ID_WIDTH    = 32,
    parameter int SCORE_WIDTH       = 56,
    parameter int LAYER_INDEX_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS),
    parameter int LAYER_COUNT_WIDTH = (MAX_LAYERS <= 1) ? 1 : $clog2(MAX_LAYERS + 1),
    parameter int POSITION_WIDTH    = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT)
) (
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_reg_wr_valid,
    output logic                              o_reg_wr_ready,
    input  wire logic                         i_reg_rd_valid,
    output logic                              o_reg_rd_ready,
    input  wire logic [11 : 0]                i_reg_addr,
    input  wire logic [31 : 0]                i_reg_wdata,
    output logic [31 : 0]                     o_reg_rdata,
    output logic                              o_reg_error,

    output logic                              o_start_pulse,
    output logic [31 : 0]                     o_input_token_id,
    output logic                              o_embedding_enable,
    output logic [ADDR_WIDTH-1 : 0]           o_embedding_weight_base_addr,
    output logic [ADDR_WIDTH-1 : 0]           o_embedding_scale_base_addr,
    output logic [LAYER_INDEX_WIDTH-1 : 0]    o_layer_start_index,
    output logic [LAYER_COUNT_WIDTH-1 : 0]    o_layer_count,
    output logic [POSITION_WIDTH-1 : 0]       o_position,
    output logic [ADDR_WIDTH-1 : 0]           o_input_hidden_base_addr,
    output logic [ADDR_WIDTH-1 : 0]           o_output_hidden_base_addr,
    output logic [ADDR_WIDTH-1 : 0]           o_kv_cache_base_addr,
    output logic [ADDR_WIDTH-1 : 0]           o_final_tail_qmap_base_addr,
    output logic                              o_final_hidden_base_override_valid,
    output logic [ADDR_WIDTH-1 : 0]           o_final_hidden_base_override_addr,

    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_qkv_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_input_norm_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_attn_frontend_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_attn_score_value_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_o_proj_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_post_attn_norm_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_mlp_gate_up_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_mlp_silu_mul_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_mlp_down_qmap_base_addr_table,
    output logic [MAX_LAYERS*ADDR_WIDTH-1 : 0] o_mlp_residual_add_qmap_base_addr_table,

    input  wire logic                         i_top_busy,
    input  wire logic                         i_top_done,
    input  wire logic                         i_top_error,
    input  wire logic [7 : 0]                 i_top_state_debug,
    input  wire logic [7 : 0]                 i_top_phase_debug,
    input  wire logic [LAYER_COUNT_WIDTH-1:0] i_layers_started,
    input  wire logic [LAYER_COUNT_WIDTH-1:0] i_layers_completed,
    input  wire logic [MAX_LAYERS-1:0]        i_layer_done_mask,
    input  wire logic [MAX_LAYERS-1:0]        i_layer_error_mask,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_last_layer_output_base_addr,
    input  wire logic                         i_tail_error,
    input  wire logic                         i_tail_norm_saturation,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_tail_effective_final_hidden_base_addr,
    input  wire logic [TOKEN_ID_WIDTH-1 : 0]  i_tail_best_token_id,
    input  wire logic signed [SCORE_WIDTH-1 : 0] i_tail_best_score_q26,
    input  wire logic [31 : 0]                i_tail_tiles_started,
    input  wire logic [31 : 0]                i_tail_tiles_completed,
    input  wire logic [31 : 0]                i_mem_read_burst_count,
    input  wire logic [31 : 0]                i_mem_read_word_count,
    input  wire logic [31 : 0]                i_mem_write_req_count,
    input  wire logic [31 : 0]                i_mem_write_word_count
);

    localparam int TABLE_COUNT = 10;

    localparam logic [7 : 0] TABLE_QKV             = 8'd0;
    localparam logic [7 : 0] TABLE_INPUT_NORM      = 8'd1;
    localparam logic [7 : 0] TABLE_ATTN_FRONTEND   = 8'd2;
    localparam logic [7 : 0] TABLE_ATTN_SCORE      = 8'd3;
    localparam logic [7 : 0] TABLE_O_PROJ          = 8'd4;
    localparam logic [7 : 0] TABLE_POST_ATTN_NORM  = 8'd5;
    localparam logic [7 : 0] TABLE_MLP_GATE_UP     = 8'd6;
    localparam logic [7 : 0] TABLE_MLP_SILU_MUL    = 8'd7;
    localparam logic [7 : 0] TABLE_MLP_DOWN        = 8'd8;
    localparam logic [7 : 0] TABLE_MLP_RESIDUAL    = 8'd9;

    localparam logic [9 : 0] REG_CTRL                 = 10'h000;
    localparam logic [9 : 0] REG_STATUS               = 10'h001;
    localparam logic [9 : 0] REG_LAYER_START          = 10'h002;
    localparam logic [9 : 0] REG_LAYER_COUNT          = 10'h003;
    localparam logic [9 : 0] REG_POSITION             = 10'h004;
    localparam logic [9 : 0] REG_INPUT_TOKEN          = 10'h005;
    localparam logic [9 : 0] REG_INPUT_HIDDEN_LO      = 10'h008;
    localparam logic [9 : 0] REG_INPUT_HIDDEN_HI      = 10'h009;
    localparam logic [9 : 0] REG_OUTPUT_HIDDEN_LO     = 10'h00A;
    localparam logic [9 : 0] REG_OUTPUT_HIDDEN_HI     = 10'h00B;
    localparam logic [9 : 0] REG_KV_CACHE_LO          = 10'h00C;
    localparam logic [9 : 0] REG_KV_CACHE_HI          = 10'h00D;
    localparam logic [9 : 0] REG_FINAL_TAIL_QMAP_LO   = 10'h00E;
    localparam logic [9 : 0] REG_FINAL_TAIL_QMAP_HI   = 10'h00F;
    localparam logic [9 : 0] REG_FINAL_OVERRIDE_LO    = 10'h010;
    localparam logic [9 : 0] REG_FINAL_OVERRIDE_HI    = 10'h011;
    localparam logic [9 : 0] REG_FINAL_OVERRIDE_CTRL  = 10'h012;
    localparam logic [9 : 0] REG_EMBEDDING_CTRL       = 10'h013;
    localparam logic [9 : 0] REG_TABLE_SELECT         = 10'h014;
    localparam logic [9 : 0] REG_TABLE_DATA_LO        = 10'h015;
    localparam logic [9 : 0] REG_TABLE_DATA_HI        = 10'h016;
    localparam logic [9 : 0] REG_TABLE_COMMIT         = 10'h017;
    localparam logic [9 : 0] REG_OUT_TOKEN            = 10'h018;
    localparam logic [9 : 0] REG_OUT_SCORE_LO         = 10'h019;
    localparam logic [9 : 0] REG_OUT_SCORE_HI         = 10'h01A;
    localparam logic [9 : 0] REG_LAYERS               = 10'h01B;
    localparam logic [9 : 0] REG_LAYER_DONE_MASK      = 10'h01C;
    localparam logic [9 : 0] REG_LAYER_ERROR_MASK     = 10'h01D;
    localparam logic [9 : 0] REG_LAST_OUTPUT_LO       = 10'h01E;
    localparam logic [9 : 0] REG_LAST_OUTPUT_HI       = 10'h01F;
    localparam logic [9 : 0] REG_TAIL_HIDDEN_LO       = 10'h020;
    localparam logic [9 : 0] REG_TAIL_HIDDEN_HI       = 10'h021;
    localparam logic [9 : 0] REG_TAIL_TILES_STARTED   = 10'h022;
    localparam logic [9 : 0] REG_TAIL_TILES_COMPLETED = 10'h023;
    localparam logic [9 : 0] REG_MEM_RD_REQS          = 10'h024;
    localparam logic [9 : 0] REG_MEM_RD_WORDS         = 10'h025;
    localparam logic [9 : 0] REG_MEM_WR_REQS          = 10'h026;
    localparam logic [9 : 0] REG_MEM_WR_WORDS         = 10'h027;
    localparam logic [9 : 0] REG_EMBED_WEIGHT_LO      = 10'h028;
    localparam logic [9 : 0] REG_EMBED_WEIGHT_HI      = 10'h029;
    localparam logic [9 : 0] REG_EMBED_SCALE_LO       = 10'h02A;
    localparam logic [9 : 0] REG_EMBED_SCALE_HI       = 10'h02B;

    wire logic [9 : 0] word_addr = i_reg_addr[11 : 2];

    logic done_sticky;
    logic error_sticky;
    logic command_error_sticky;
    logic [7 : 0] table_select_id;
    logic [7 : 0] table_select_layer;
    logic [ADDR_WIDTH-1 : 0] table_shadow_addr;

    logic [ADDR_WIDTH-1 : 0] selected_table_addr;
    logic signed [63 : 0] tail_score_q26_ext;

    assign o_reg_wr_ready = 1'b1;
    assign o_reg_rd_ready = 1'b1;
    assign tail_score_q26_ext = {{(64-SCORE_WIDTH){i_tail_best_score_q26[SCORE_WIDTH-1]}}, i_tail_best_score_q26};

    function automatic logic table_select_valid;
        input logic [7 : 0] table_id;
        begin
            table_select_valid = (table_id < TABLE_COUNT[7 : 0]);
        end
    endfunction

    function automatic logic layer_select_valid;
        input logic [7 : 0] layer_id;
        begin
            layer_select_valid = (layer_id < MAX_LAYERS[7 : 0]);
        end
    endfunction

    always @* begin
        selected_table_addr = '0;
        if (layer_select_valid(table_select_layer)) begin
            case (table_select_id)
                TABLE_QKV:
                    selected_table_addr = o_qkv_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_INPUT_NORM:
                    selected_table_addr = o_input_norm_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_ATTN_FRONTEND:
                    selected_table_addr = o_attn_frontend_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_ATTN_SCORE:
                    selected_table_addr = o_attn_score_value_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_O_PROJ:
                    selected_table_addr = o_o_proj_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_POST_ATTN_NORM:
                    selected_table_addr = o_post_attn_norm_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_MLP_GATE_UP:
                    selected_table_addr = o_mlp_gate_up_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_MLP_SILU_MUL:
                    selected_table_addr = o_mlp_silu_mul_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_MLP_DOWN:
                    selected_table_addr = o_mlp_down_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                TABLE_MLP_RESIDUAL:
                    selected_table_addr = o_mlp_residual_add_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH];
                default:
                    selected_table_addr = '0;
            endcase
        end
    end

    always @* begin
        o_reg_error = 1'b0;
        unique case (word_addr)
            REG_CTRL,
            REG_STATUS,
            REG_LAYER_START,
            REG_LAYER_COUNT,
            REG_POSITION,
            REG_INPUT_TOKEN,
            REG_INPUT_HIDDEN_LO,
            REG_INPUT_HIDDEN_HI,
            REG_OUTPUT_HIDDEN_LO,
            REG_OUTPUT_HIDDEN_HI,
            REG_KV_CACHE_LO,
            REG_KV_CACHE_HI,
            REG_FINAL_TAIL_QMAP_LO,
            REG_FINAL_TAIL_QMAP_HI,
            REG_FINAL_OVERRIDE_LO,
            REG_FINAL_OVERRIDE_HI,
            REG_FINAL_OVERRIDE_CTRL,
            REG_EMBEDDING_CTRL,
            REG_TABLE_SELECT,
            REG_TABLE_DATA_LO,
            REG_TABLE_DATA_HI,
            REG_TABLE_COMMIT,
            REG_OUT_TOKEN,
            REG_OUT_SCORE_LO,
            REG_OUT_SCORE_HI,
            REG_LAYERS,
            REG_LAYER_DONE_MASK,
            REG_LAYER_ERROR_MASK,
            REG_LAST_OUTPUT_LO,
            REG_LAST_OUTPUT_HI,
            REG_TAIL_HIDDEN_LO,
            REG_TAIL_HIDDEN_HI,
            REG_TAIL_TILES_STARTED,
            REG_TAIL_TILES_COMPLETED,
            REG_MEM_RD_REQS,
            REG_MEM_RD_WORDS,
            REG_MEM_WR_REQS,
            REG_MEM_WR_WORDS,
            REG_EMBED_WEIGHT_LO,
            REG_EMBED_WEIGHT_HI,
            REG_EMBED_SCALE_LO,
            REG_EMBED_SCALE_HI: o_reg_error = 1'b0;
            default: o_reg_error = i_reg_wr_valid || i_reg_rd_valid;
        endcase
    end

    always @* begin
        o_reg_rdata = 32'd0;
        unique case (word_addr)
            REG_STATUS: begin
                o_reg_rdata[0] = i_top_busy;
                o_reg_rdata[1] = done_sticky;
                o_reg_rdata[2] = error_sticky;
                o_reg_rdata[3] = command_error_sticky;
                o_reg_rdata[4] = i_top_error;
                o_reg_rdata[5] = i_tail_error;
                o_reg_rdata[6] = i_tail_norm_saturation;
                o_reg_rdata[15 : 8] = i_top_state_debug;
                o_reg_rdata[23 : 16] = i_top_phase_debug;
            end
            REG_LAYER_START:          o_reg_rdata = {{(32-LAYER_INDEX_WIDTH){1'b0}}, o_layer_start_index};
            REG_LAYER_COUNT:          o_reg_rdata = {{(32-LAYER_COUNT_WIDTH){1'b0}}, o_layer_count};
            REG_POSITION:             o_reg_rdata = {{(32-POSITION_WIDTH){1'b0}}, o_position};
            REG_INPUT_TOKEN:          o_reg_rdata = o_input_token_id;
            REG_INPUT_HIDDEN_LO:      o_reg_rdata = o_input_hidden_base_addr[31 : 0];
            REG_INPUT_HIDDEN_HI:      o_reg_rdata = o_input_hidden_base_addr[63 : 32];
            REG_OUTPUT_HIDDEN_LO:     o_reg_rdata = o_output_hidden_base_addr[31 : 0];
            REG_OUTPUT_HIDDEN_HI:     o_reg_rdata = o_output_hidden_base_addr[63 : 32];
            REG_KV_CACHE_LO:          o_reg_rdata = o_kv_cache_base_addr[31 : 0];
            REG_KV_CACHE_HI:          o_reg_rdata = o_kv_cache_base_addr[63 : 32];
            REG_FINAL_TAIL_QMAP_LO:   o_reg_rdata = o_final_tail_qmap_base_addr[31 : 0];
            REG_FINAL_TAIL_QMAP_HI:   o_reg_rdata = o_final_tail_qmap_base_addr[63 : 32];
            REG_FINAL_OVERRIDE_LO:    o_reg_rdata = o_final_hidden_base_override_addr[31 : 0];
            REG_FINAL_OVERRIDE_HI:    o_reg_rdata = o_final_hidden_base_override_addr[63 : 32];
            REG_FINAL_OVERRIDE_CTRL:  o_reg_rdata = {31'd0, o_final_hidden_base_override_valid};
            REG_EMBEDDING_CTRL:       o_reg_rdata = {31'd0, o_embedding_enable};
            REG_TABLE_SELECT:         o_reg_rdata = {16'd0, table_select_layer, table_select_id};
            REG_TABLE_DATA_LO:        o_reg_rdata = selected_table_addr[31 : 0];
            REG_TABLE_DATA_HI:        o_reg_rdata = selected_table_addr[63 : 32];
            REG_OUT_TOKEN:            o_reg_rdata = {{(32-TOKEN_ID_WIDTH){1'b0}}, i_tail_best_token_id};
            REG_OUT_SCORE_LO:         o_reg_rdata = i_tail_best_score_q26[31 : 0];
            REG_OUT_SCORE_HI:         o_reg_rdata = tail_score_q26_ext[63 : 32];
            REG_LAYERS:               o_reg_rdata = {{(16-LAYER_COUNT_WIDTH){1'b0}}, i_layers_completed, {(16-LAYER_COUNT_WIDTH){1'b0}}, i_layers_started};
            REG_LAYER_DONE_MASK:      o_reg_rdata = {{(32-MAX_LAYERS){1'b0}}, i_layer_done_mask};
            REG_LAYER_ERROR_MASK:     o_reg_rdata = {{(32-MAX_LAYERS){1'b0}}, i_layer_error_mask};
            REG_LAST_OUTPUT_LO:       o_reg_rdata = i_last_layer_output_base_addr[31 : 0];
            REG_LAST_OUTPUT_HI:       o_reg_rdata = i_last_layer_output_base_addr[63 : 32];
            REG_TAIL_HIDDEN_LO:       o_reg_rdata = i_tail_effective_final_hidden_base_addr[31 : 0];
            REG_TAIL_HIDDEN_HI:       o_reg_rdata = i_tail_effective_final_hidden_base_addr[63 : 32];
            REG_TAIL_TILES_STARTED:   o_reg_rdata = i_tail_tiles_started;
            REG_TAIL_TILES_COMPLETED: o_reg_rdata = i_tail_tiles_completed;
            REG_MEM_RD_REQS:          o_reg_rdata = i_mem_read_burst_count;
            REG_MEM_RD_WORDS:         o_reg_rdata = i_mem_read_word_count;
            REG_MEM_WR_REQS:          o_reg_rdata = i_mem_write_req_count;
            REG_MEM_WR_WORDS:         o_reg_rdata = i_mem_write_word_count;
            REG_EMBED_WEIGHT_LO:      o_reg_rdata = o_embedding_weight_base_addr[31 : 0];
            REG_EMBED_WEIGHT_HI:      o_reg_rdata = o_embedding_weight_base_addr[63 : 32];
            REG_EMBED_SCALE_LO:       o_reg_rdata = o_embedding_scale_base_addr[31 : 0];
            REG_EMBED_SCALE_HI:       o_reg_rdata = o_embedding_scale_base_addr[63 : 32];
            default:                  o_reg_rdata = 32'd0;
        endcase
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_start_pulse <= 1'b0;
            o_input_token_id <= 32'd0;
            o_embedding_enable <= 1'b0;
            o_embedding_weight_base_addr <= '0;
            o_embedding_scale_base_addr <= '0;
            o_layer_start_index <= '0;
            o_layer_count <= '0;
            o_position <= '0;
            o_input_hidden_base_addr <= '0;
            o_output_hidden_base_addr <= '0;
            o_kv_cache_base_addr <= '0;
            o_final_tail_qmap_base_addr <= '0;
            o_final_hidden_base_override_valid <= 1'b0;
            o_final_hidden_base_override_addr <= '0;
            o_qkv_qmap_base_addr_table <= '0;
            o_input_norm_qmap_base_addr_table <= '0;
            o_attn_frontend_qmap_base_addr_table <= '0;
            o_attn_score_value_qmap_base_addr_table <= '0;
            o_o_proj_qmap_base_addr_table <= '0;
            o_post_attn_norm_qmap_base_addr_table <= '0;
            o_mlp_gate_up_qmap_base_addr_table <= '0;
            o_mlp_silu_mul_qmap_base_addr_table <= '0;
            o_mlp_down_qmap_base_addr_table <= '0;
            o_mlp_residual_add_qmap_base_addr_table <= '0;
            done_sticky <= 1'b0;
            error_sticky <= 1'b0;
            command_error_sticky <= 1'b0;
            table_select_id <= '0;
            table_select_layer <= '0;
            table_shadow_addr <= '0;
        end
        else begin
            o_start_pulse <= 1'b0;

            if (i_top_done) begin
                done_sticky <= 1'b1;
            end
            if (i_top_error || i_tail_error || i_tail_norm_saturation) begin
                error_sticky <= 1'b1;
            end

            if (i_reg_wr_valid) begin
                unique case (word_addr)
                    REG_CTRL: begin
                        if (i_reg_wdata[1]) begin
                            done_sticky <= 1'b0;
                            error_sticky <= 1'b0;
                            command_error_sticky <= 1'b0;
                        end
                        if (i_reg_wdata[0]) begin
                            if (i_top_busy) begin
                                command_error_sticky <= 1'b1;
                                error_sticky <= 1'b1;
                            end
                            else begin
                                o_start_pulse <= 1'b1;
                                done_sticky <= 1'b0;
                                error_sticky <= 1'b0;
                                command_error_sticky <= 1'b0;
                            end
                        end
                    end

                    REG_LAYER_START: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_layer_start_index <= i_reg_wdata[LAYER_INDEX_WIDTH-1 : 0];
                    end
                    REG_LAYER_COUNT: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_layer_count <= i_reg_wdata[LAYER_COUNT_WIDTH-1 : 0];
                    end
                    REG_POSITION: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_position <= i_reg_wdata[POSITION_WIDTH-1 : 0];
                    end
                    REG_INPUT_TOKEN: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_input_token_id <= i_reg_wdata;
                    end
                    REG_EMBEDDING_CTRL: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_embedding_enable <= i_reg_wdata[0];
                    end
                    REG_EMBED_WEIGHT_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_embedding_weight_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_EMBED_WEIGHT_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_embedding_weight_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_EMBED_SCALE_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_embedding_scale_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_EMBED_SCALE_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_embedding_scale_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_INPUT_HIDDEN_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_input_hidden_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_INPUT_HIDDEN_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_input_hidden_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_OUTPUT_HIDDEN_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_output_hidden_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_OUTPUT_HIDDEN_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_output_hidden_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_KV_CACHE_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_kv_cache_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_KV_CACHE_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_kv_cache_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_FINAL_TAIL_QMAP_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_final_tail_qmap_base_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_FINAL_TAIL_QMAP_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_final_tail_qmap_base_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_FINAL_OVERRIDE_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_final_hidden_base_override_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_FINAL_OVERRIDE_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_final_hidden_base_override_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_FINAL_OVERRIDE_CTRL: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else o_final_hidden_base_override_valid <= i_reg_wdata[0];
                    end
                    REG_TABLE_SELECT: begin
                        if (i_top_busy) begin
                            command_error_sticky <= 1'b1;
                        end
                        else begin
                            table_select_id <= i_reg_wdata[7 : 0];
                            table_select_layer <= i_reg_wdata[15 : 8];
                        end
                    end
                    REG_TABLE_DATA_LO: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else table_shadow_addr[31 : 0] <= i_reg_wdata;
                    end
                    REG_TABLE_DATA_HI: begin
                        if (i_top_busy) command_error_sticky <= 1'b1;
                        else table_shadow_addr[63 : 32] <= i_reg_wdata;
                    end
                    REG_TABLE_COMMIT: begin
                        if (i_top_busy ||
                            !table_select_valid(table_select_id) ||
                            !layer_select_valid(table_select_layer) ||
                            !i_reg_wdata[0]) begin
                            command_error_sticky <= 1'b1;
                            if (i_reg_wdata[0]) begin
                                error_sticky <= 1'b1;
                            end
                        end
                        else begin
                            unique case (table_select_id)
                                TABLE_QKV:
                                    o_qkv_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_INPUT_NORM:
                                    o_input_norm_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_ATTN_FRONTEND:
                                    o_attn_frontend_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_ATTN_SCORE:
                                    o_attn_score_value_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_O_PROJ:
                                    o_o_proj_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_POST_ATTN_NORM:
                                    o_post_attn_norm_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_MLP_GATE_UP:
                                    o_mlp_gate_up_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_MLP_SILU_MUL:
                                    o_mlp_silu_mul_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_MLP_DOWN:
                                    o_mlp_down_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                TABLE_MLP_RESIDUAL:
                                    o_mlp_residual_add_qmap_base_addr_table[table_select_layer*ADDR_WIDTH +: ADDR_WIDTH] <= table_shadow_addr;
                                default: begin
                                    command_error_sticky <= 1'b1;
                                    error_sticky <= 1'b1;
                                end
                            endcase
                        end
                    end
                    default: begin
                        if (o_reg_error) begin
                            command_error_sticky <= 1'b1;
                            error_sticky <= 1'b1;
                        end
                    end
                endcase
            end
        end
    end

endmodule

`default_nettype wire
