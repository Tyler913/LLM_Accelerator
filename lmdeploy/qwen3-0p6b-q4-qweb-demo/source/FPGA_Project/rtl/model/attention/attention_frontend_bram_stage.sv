`default_nettype none

// Resource-oriented Qwen3 attention front-end.
//
// The original bring-up path exposes whole-token packed Q/K/V buses. That is
// useful for simulation, but it builds very large variable-select muxes and
// resettable vector registers. This board path instead stores Q/K/V and the
// RoPE results in inferred block RAM, processes one 128-element head at a
// time, and emits the K/V cache stream after all heads are complete.
//
// The bit-proven RMSNorm and RoPE arithmetic contracts are retained, but the
// board path uses their BRAM-backed sequential forms. This avoids rebuilding
// 128-lane variable-select muxes for every head while still reusing one
// arithmetic engine across all 16 Q and 8 K heads.
module attention_frontend_bram_stage #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DATA_WIDTH       = 32,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_Q_HEADS      = 16,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int IN_WIDTH         = 24,
    parameter int IN_FRAC          = 12,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
    parameter int TRIG_WIDTH       = 16,
    parameter int TRIG_FRAC        = 15,
    parameter int OUT_WIDTH        = 24,
    parameter int OUT_FRAC         = 12,
    parameter int INV_RMS_WIDTH    = 24,
    parameter int INV_RMS_FRAC     = 16,
    parameter int SUM_WIDTH        = 64,
    parameter int EPS_Q24          = 17,
    parameter int Q_COUNT          = NUM_Q_HEADS * HEAD_DIM,
    parameter int KV_COUNT         = NUM_KV_HEADS * HEAD_DIM,
    parameter int Q_ADDR_WIDTH     =
        (Q_COUNT <= 1) ? 1 : $clog2(Q_COUNT),
    parameter int KV_ADDR_WIDTH    =
        (KV_COUNT <= 1) ? 1 : $clog2(KV_COUNT),
    parameter int PARAM_ADDR_WIDTH =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int LAYER_INDEX_W    =
        (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int POSITION_INDEX_W =
        (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int HEAD_INDEX_W     =
        (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int DIM_INDEX_W      =
        (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM),
    parameter int TOTAL_HEADS      = NUM_Q_HEADS + NUM_KV_HEADS,
    parameter int TOTAL_HEAD_INDEX_W =
        (TOTAL_HEADS <= 1) ? 1 : $clog2(TOTAL_HEADS)
) (
    input  wire logic                                  i_clk,
    input  wire logic                                  i_rst_n,

    input  wire logic                                  i_q_wr_valid,
    input  wire logic [Q_ADDR_WIDTH-1 : 0]             i_q_wr_addr,
    input  wire logic signed [IN_WIDTH-1 : 0]          i_q_wr_data,
    input  wire logic                                  i_k_wr_valid,
    input  wire logic [KV_ADDR_WIDTH-1 : 0]            i_k_wr_addr,
    input  wire logic signed [IN_WIDTH-1 : 0]          i_k_wr_data,
    input  wire logic                                  i_v_wr_valid,
    input  wire logic [KV_ADDR_WIDTH-1 : 0]            i_v_wr_addr,
    input  wire logic signed [IN_WIDTH-1 : 0]          i_v_wr_data,
    input  wire logic                                  i_q_gamma_wr_valid,
    input  wire logic [PARAM_ADDR_WIDTH-1 : 0]         i_q_gamma_wr_addr,
    input  wire logic signed [GAMMA_WIDTH-1 : 0]       i_q_gamma_wr_data,
    input  wire logic                                  i_k_gamma_wr_valid,
    input  wire logic [PARAM_ADDR_WIDTH-1 : 0]         i_k_gamma_wr_addr,
    input  wire logic signed [GAMMA_WIDTH-1 : 0]       i_k_gamma_wr_data,
    input  wire logic                                  i_cos_wr_valid,
    input  wire logic [PARAM_ADDR_WIDTH-1 : 0]         i_cos_wr_addr,
    input  wire logic signed [TRIG_WIDTH-1 : 0]        i_cos_wr_data,
    input  wire logic                                  i_sin_wr_valid,
    input  wire logic [PARAM_ADDR_WIDTH-1 : 0]         i_sin_wr_addr,
    input  wire logic signed [TRIG_WIDTH-1 : 0]        i_sin_wr_data,

    input  wire logic                                  i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]               i_cache_base_addr,
    input  wire logic [LAYER_INDEX_W-1 : 0]            i_layer_id,
    input  wire logic [POSITION_INDEX_W-1 : 0]         i_position,

    output logic                                       o_busy,
    output logic                                       o_done,
    output logic                                       o_error,
    output logic                                       o_saturation,
    output logic                                       o_norm_saturation,
    output logic                                       o_rope_saturation,

    output logic                                       o_cache_wr_valid,
    input  wire logic                                  i_cache_wr_ready,
    output logic [ADDR_WIDTH-1 : 0]                    o_cache_wr_addr,
    output logic [DATA_WIDTH-1 : 0]                    o_cache_wr_data,
    output logic                                       o_cache_wr_last,
    output logic                                       o_cache_wr_kind,
    output logic [HEAD_INDEX_W-1 : 0]                  o_cache_wr_head,
    output logic [DIM_INDEX_W-1 : 0]                   o_cache_wr_dim,
    output logic [31 : 0]                              o_cache_write_count,

    input  wire logic [Q_ADDR_WIDTH-1 : 0]             i_qrope_rd_addr,
    output logic signed [OUT_WIDTH-1 : 0]              o_qrope_rd_data
);

    localparam logic [TOTAL_HEAD_INDEX_W-1 : 0] LAST_TOTAL_HEAD =
        TOTAL_HEAD_INDEX_W'(TOTAL_HEADS - 1);
    localparam logic [DIM_INDEX_W-1 : 0] LAST_DIM =
        DIM_INDEX_W'(HEAD_DIM - 1);
    localparam logic [HEAD_INDEX_W-1 : 0] LAST_KV_HEAD =
        HEAD_INDEX_W'(NUM_KV_HEADS - 1);

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_HEAD_PRIME,
        S_HEAD_LOAD,
        S_NORM_START,
        S_NORM_WAIT,
        S_ROPE_START,
        S_ROPE_WAIT,
        S_CACHE_PRIME,
        S_CACHE_CAPTURE,
        S_CACHE_SEND,
        S_DONE
    } state_t;

    state_t state;

    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] q_mem [0 : Q_COUNT-1];
    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] k_mem [0 : KV_COUNT-1];
    (* ram_style = "block" *)
    logic signed [IN_WIDTH-1 : 0] v_mem [0 : KV_COUNT-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0] q_rope_mem [0 : Q_COUNT-1];
    (* ram_style = "block" *)
    logic signed [OUT_WIDTH-1 : 0] k_rope_mem [0 : KV_COUNT-1];

    logic [Q_ADDR_WIDTH-1 : 0] q_read_addr;
    logic [KV_ADDR_WIDTH-1 : 0] k_read_addr;
    logic [KV_ADDR_WIDTH-1 : 0] v_read_addr;
    logic [KV_ADDR_WIDTH-1 : 0] k_rope_read_addr;
    logic signed [IN_WIDTH-1 : 0] q_read_data;
    logic signed [IN_WIDTH-1 : 0] k_read_data;
    logic signed [IN_WIDTH-1 : 0] v_read_data;
    logic signed [OUT_WIDTH-1 : 0] k_rope_read_data;

    (* ram_style = "block" *)
    logic signed [GAMMA_WIDTH-1 : 0] q_gamma_mem [0 : HEAD_DIM-1];
    (* ram_style = "block" *)
    logic signed [GAMMA_WIDTH-1 : 0] k_gamma_mem [0 : HEAD_DIM-1];
    logic [PARAM_ADDR_WIDTH-1 : 0] gamma_read_addr;
    logic signed [GAMMA_WIDTH-1 : 0] q_gamma_read_data;
    logic signed [GAMMA_WIDTH-1 : 0] k_gamma_read_data;
    logic signed [GAMMA_WIDTH-1 : 0] head_gamma_read_data;

    logic [TOTAL_HEAD_INDEX_W-1 : 0] head_index;
    logic [DIM_INDEX_W-1 : 0] load_dim;
    logic signed [IN_WIDTH-1 : 0] head_read_data;

    logic norm_start;
    logic norm_done;
    logic norm_error;
    logic norm_saturation;
    logic norm_output_wr_valid;
    logic [DIM_INDEX_W-1 : 0] norm_output_wr_addr;
    logic signed [OUT_WIDTH-1 : 0] norm_output_wr_data;
    logic [SUM_WIDTH-1 : 0] norm_sum_squares;
    logic [SUM_WIDTH-1 : 0] norm_mean_square;
    logic [INV_RMS_WIDTH-1 : 0] norm_inv_rms;

    logic rope_start;
    logic rope_done;
    logic rope_saturation;
    logic rope_output_wr_valid;
    logic [DIM_INDEX_W-1 : 0] rope_output_wr_addr;
    logic signed [OUT_WIDTH-1 : 0] rope_output_wr_data;

    logic cache_kind;
    logic [HEAD_INDEX_W-1 : 0] cache_head;
    logic [DIM_INDEX_W-1 : 0] cache_dim;
    logic [KV_ADDR_WIDTH-1 : 0] cache_element_index;
    logic signed [OUT_WIDTH-1 : 0] cache_data_reg;
    logic cache_addr_valid;
    logic [ADDR_WIDTH-1 : 0] cache_addr_offset;
    logic [ADDR_WIDTH-1 : 0] cache_byte_addr;
    logic cache_is_last;
    logic cache_write_fire;

    logic [ADDR_WIDTH-1 : 0] active_cache_base_addr;
    logic [LAYER_INDEX_W-1 : 0] active_layer_id;
    logic [POSITION_INDEX_W-1 : 0] active_position;

    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_done = (state == S_DONE);
    assign o_saturation = o_norm_saturation | o_rope_saturation;

    assign norm_start = (state == S_NORM_START);
    assign rope_start = (state == S_ROPE_START);
    assign head_read_data =
        (head_index < NUM_Q_HEADS) ? q_read_data : k_read_data;
    assign head_gamma_read_data =
        (head_index < NUM_Q_HEADS) ?
            q_gamma_read_data :
            k_gamma_read_data;

    assign cache_element_index =
        KV_ADDR_WIDTH'((cache_head * HEAD_DIM) + cache_dim);
    assign v_read_addr = cache_element_index;
    assign k_rope_read_addr = cache_element_index;
    assign cache_is_last =
        cache_kind &&
        (cache_head == LAST_KV_HEAD) &&
        (cache_dim == LAST_DIM);
    assign o_cache_wr_valid = (state == S_CACHE_SEND) && cache_addr_valid;
    assign o_cache_wr_addr = cache_byte_addr;
    assign o_cache_wr_data = {
        {(DATA_WIDTH-OUT_WIDTH){cache_data_reg[OUT_WIDTH-1]}},
        cache_data_reg
    };
    assign o_cache_wr_last = o_cache_wr_valid && cache_is_last;
    assign o_cache_wr_kind = cache_kind;
    assign o_cache_wr_head = cache_head;
    assign o_cache_wr_dim = cache_dim;
    assign cache_write_fire = o_cache_wr_valid && i_cache_wr_ready;

    always_comb begin
        q_read_addr = '0;
        k_read_addr = '0;
        gamma_read_addr = load_dim;

        if ((state == S_HEAD_LOAD) && (load_dim != LAST_DIM)) begin
            gamma_read_addr = load_dim + 1'b1;
        end

        if (head_index < NUM_Q_HEADS) begin
            if ((state == S_HEAD_LOAD) && (load_dim != LAST_DIM)) begin
                q_read_addr =
                    Q_ADDR_WIDTH'((head_index * HEAD_DIM) + load_dim + 1'b1);
            end
            else begin
                q_read_addr =
                    Q_ADDR_WIDTH'((head_index * HEAD_DIM) + load_dim);
            end
        end
        else begin
            if ((state == S_HEAD_LOAD) && (load_dim != LAST_DIM)) begin
                k_read_addr =
                    KV_ADDR_WIDTH'(
                        ((head_index - NUM_Q_HEADS) * HEAD_DIM) +
                        load_dim +
                        1'b1
                    );
            end
            else begin
                k_read_addr =
                    KV_ADDR_WIDTH'(
                        ((head_index - NUM_Q_HEADS) * HEAD_DIM) +
                        load_dim
                    );
            end
        end
    end

    // Native-load plus synchronous-read block-RAM templates.
    always_ff @(posedge i_clk) begin
        if (i_q_wr_valid && (state == S_IDLE)) begin
            q_mem[i_q_wr_addr] <= i_q_wr_data;
        end
        q_read_data <= q_mem[q_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (i_k_wr_valid && (state == S_IDLE)) begin
            k_mem[i_k_wr_addr] <= i_k_wr_data;
        end
        k_read_data <= k_mem[k_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (i_v_wr_valid && (state == S_IDLE)) begin
            v_mem[i_v_wr_addr] <= i_v_wr_data;
        end
        v_read_data <= v_mem[v_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (rope_output_wr_valid && (head_index < NUM_Q_HEADS)) begin
            q_rope_mem[
                Q_ADDR_WIDTH'(
                    (head_index * HEAD_DIM) + rope_output_wr_addr
                )
            ] <= rope_output_wr_data;
        end
        o_qrope_rd_data <= q_rope_mem[i_qrope_rd_addr];
    end

    always_ff @(posedge i_clk) begin
        if (rope_output_wr_valid && (head_index >= NUM_Q_HEADS)) begin
            k_rope_mem[
                KV_ADDR_WIDTH'(
                    ((head_index - NUM_Q_HEADS) * HEAD_DIM) +
                    rope_output_wr_addr
                )
            ] <= rope_output_wr_data;
        end
        k_rope_read_data <= k_rope_mem[k_rope_read_addr];
    end

    always_ff @(posedge i_clk) begin
        if (i_q_gamma_wr_valid && (state == S_IDLE)) begin
            q_gamma_mem[i_q_gamma_wr_addr] <= i_q_gamma_wr_data;
        end
        if (i_k_gamma_wr_valid && (state == S_IDLE)) begin
            k_gamma_mem[i_k_gamma_wr_addr] <= i_k_gamma_wr_data;
        end
        q_gamma_read_data <= q_gamma_mem[gamma_read_addr];
        k_gamma_read_data <= k_gamma_mem[gamma_read_addr];
    end

    rmsnorm_bram #(
        .INPUT_SIZE    (HEAD_DIM),
        .IN_WIDTH      (IN_WIDTH),
        .IN_FRAC       (IN_FRAC),
        .GAMMA_WIDTH   (GAMMA_WIDTH),
        .GAMMA_FRAC    (GAMMA_FRAC),
        .GAMMA_SIGNED  (1),
        .INV_RMS_WIDTH (INV_RMS_WIDTH),
        .INV_RMS_FRAC  (INV_RMS_FRAC),
        .OUT_WIDTH     (OUT_WIDTH),
        .OUT_FRAC      (OUT_FRAC),
        .SUM_WIDTH     (SUM_WIDTH),
        .SUM_FRAC      (2 * IN_FRAC),
        .MEAN_SHIFT    ($clog2(HEAD_DIM)),
        .RMS_WIDTH     (IN_WIDTH),
        .RMS_FRAC      (IN_FRAC),
        .DIV_NUM_WIDTH (48),
        .DIV_NUM_SHIFT (IN_FRAC + INV_RMS_FRAC),
        .EPS_Q20       (EPS_Q24)
    ) head_norm (
        .i_clk             (i_clk),
        .i_rst_n           (i_rst_n),
        .i_input_wr_en     (state == S_HEAD_LOAD),
        .i_input_wr_addr   (load_dim),
        .i_input_wr_data   (head_read_data),
        .i_gamma_wr_en     (state == S_HEAD_LOAD),
        .i_gamma_wr_addr   (load_dim),
        .i_gamma_wr_data   (head_gamma_read_data),
        .i_start           (norm_start),
        .o_busy            (),
        .o_done            (norm_done),
        .o_error           (norm_error),
        .o_saturation      (norm_saturation),
        .i_output_rd_addr  ('0),
        .o_output_rd_data  (),
        .o_output_wr_valid (norm_output_wr_valid),
        .o_output_wr_addr  (norm_output_wr_addr),
        .o_output_wr_data  (norm_output_wr_data),
        .o_sum_squares     (norm_sum_squares),
        .o_mean_square     (norm_mean_square),
        .o_inv_rms         (norm_inv_rms)
    );

    rope_head_bram #(
        .HEAD_DIM   (HEAD_DIM),
        .IN_WIDTH   (OUT_WIDTH),
        .IN_FRAC    (OUT_FRAC),
        .TRIG_WIDTH (TRIG_WIDTH),
        .TRIG_FRAC  (TRIG_FRAC),
        .OUT_WIDTH  (OUT_WIDTH),
        .OUT_FRAC   (OUT_FRAC)
    ) head_rope (
        .i_clk             (i_clk),
        .i_rst_n           (i_rst_n),
        .i_input_wr_en     (norm_output_wr_valid),
        .i_input_wr_addr   (norm_output_wr_addr),
        .i_input_wr_data   (norm_output_wr_data),
        .i_cos_wr_en       (i_cos_wr_valid && (state == S_IDLE)),
        .i_cos_wr_addr     (i_cos_wr_addr),
        .i_cos_wr_data     (i_cos_wr_data),
        .i_sin_wr_en       (i_sin_wr_valid && (state == S_IDLE)),
        .i_sin_wr_addr     (i_sin_wr_addr),
        .i_sin_wr_data     (i_sin_wr_data),
        .i_start           (rope_start),
        .o_busy            (),
        .o_done            (rope_done),
        .o_saturation      (rope_saturation),
        .o_output_wr_valid (rope_output_wr_valid),
        .o_output_wr_addr  (rope_output_wr_addr),
        .o_output_wr_data  (rope_output_wr_data)
    );

    kv_cache_addr_gen #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .NUM_LAYERS      (NUM_LAYERS),
        .NUM_KV_HEADS    (NUM_KV_HEADS),
        .HEAD_DIM        (HEAD_DIM),
        .MAX_CONTEXT     (MAX_CONTEXT),
        .ELEMENT_BYTES   (DATA_WIDTH / 8),
        .LAYER_INDEX_W   (LAYER_INDEX_W),
        .HEAD_INDEX_W    (HEAD_INDEX_W),
        .POSITION_INDEX_W(POSITION_INDEX_W),
        .DIM_INDEX_W     (DIM_INDEX_W)
    ) cache_addr_gen (
        .i_base_addr    (active_cache_base_addr),
        .i_layer_id     (active_layer_id),
        .i_kv_kind      (cache_kind),
        .i_head_id      (cache_head),
        .i_position     (active_position),
        .i_dim          (cache_dim),
        .o_valid        (cache_addr_valid),
        .o_offset_bytes (cache_addr_offset),
        .o_byte_addr    (cache_byte_addr)
    );

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state <= S_IDLE;
            head_index <= '0;
            load_dim <= '0;
            cache_kind <= 1'b0;
            cache_head <= '0;
            cache_dim <= '0;
            cache_data_reg <= '0;
            active_cache_base_addr <= '0;
            active_layer_id <= '0;
            active_position <= '0;
            o_error <= 1'b0;
            o_norm_saturation <= 1'b0;
            o_rope_saturation <= 1'b0;
            o_cache_write_count <= 32'd0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        head_index <= '0;
                        load_dim <= '0;
                        cache_kind <= 1'b0;
                        cache_head <= '0;
                        cache_dim <= '0;
                        active_cache_base_addr <= i_cache_base_addr;
                        active_layer_id <= i_layer_id;
                        active_position <= i_position;
                        o_error <= 1'b0;
                        o_norm_saturation <= 1'b0;
                        o_rope_saturation <= 1'b0;
                        o_cache_write_count <= 32'd0;

                        if ((i_layer_id >= NUM_LAYERS) ||
                            (i_position >= MAX_CONTEXT) ||
                            (i_cache_base_addr[1 : 0] != 2'b00)) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_HEAD_PRIME;
                        end
                    end
                end

                S_HEAD_PRIME: begin
                    load_dim <= '0;
                    state <= S_HEAD_LOAD;
                end

                S_HEAD_LOAD: begin
                    if (load_dim == LAST_DIM) begin
                        load_dim <= '0;
                        state <= S_NORM_START;
                    end
                    else begin
                        load_dim <= load_dim + 1'b1;
                    end
                end

                S_NORM_START: begin
                    state <= S_NORM_WAIT;
                end

                S_NORM_WAIT: begin
                    if (norm_done) begin
                        o_norm_saturation <=
                            o_norm_saturation | norm_saturation;
                        o_error <= o_error | norm_error;
                        state <= S_ROPE_START;
                    end
                end

                S_ROPE_START: begin
                    state <= S_ROPE_WAIT;
                end

                S_ROPE_WAIT: begin
                    if (rope_done) begin
                        o_rope_saturation <=
                            o_rope_saturation | rope_saturation;
                        if (head_index == LAST_TOTAL_HEAD) begin
                            cache_kind <= 1'b0;
                            cache_head <= '0;
                            cache_dim <= '0;
                            state <= S_CACHE_PRIME;
                        end
                        else begin
                            head_index <= head_index + 1'b1;
                            load_dim <= '0;
                            state <= S_HEAD_PRIME;
                        end
                    end
                end

                S_CACHE_PRIME: begin
                    state <= S_CACHE_CAPTURE;
                end

                S_CACHE_CAPTURE: begin
                    cache_data_reg <=
                        cache_kind ? v_read_data : k_rope_read_data;
                    state <= S_CACHE_SEND;
                end

                S_CACHE_SEND: begin
                    if (!cache_addr_valid) begin
                        o_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else if (cache_write_fire) begin
                        o_cache_write_count <= o_cache_write_count + 1'b1;
                        if (cache_is_last) begin
                            state <= S_DONE;
                        end
                        else begin
                            if (cache_dim == LAST_DIM) begin
                                cache_dim <= '0;
                                if (cache_head == LAST_KV_HEAD) begin
                                    cache_head <= '0;
                                    cache_kind <= 1'b1;
                                end
                                else begin
                                    cache_head <= cache_head + 1'b1;
                                end
                            end
                            else begin
                                cache_dim <= cache_dim + 1'b1;
                            end
                            state <= S_CACHE_PRIME;
                        end
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    o_error <= 1'b1;
                    state <= S_DONE;
                end
            endcase
        end
    end

    initial begin
        if ((HEAD_DIM != 128) || ((HEAD_DIM % 2) != 0)) begin
            $fatal(1, "attention_frontend_bram_stage: unsupported HEAD_DIM");
        end
        if ((IN_WIDTH != 24) ||
            (GAMMA_WIDTH != 16) ||
            (TRIG_WIDTH != 16) ||
            (OUT_WIDTH != 24) ||
            (DATA_WIDTH != 32)) begin
            $fatal(1, "attention_frontend_bram_stage: unsupported format");
        end
    end

endmodule

`default_nettype wire
