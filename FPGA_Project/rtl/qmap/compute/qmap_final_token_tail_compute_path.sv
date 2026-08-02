`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed final-token tail:
//
//   final_hidden[1024] + final_norm.gamma[1024]
//       -> final_rmsnorm_stage
//       -> write final_norm[1024] to the QMAP activation descriptor
//       -> qmap_lm_head_argmax_compute_path
//       -> write {token, score_low32, score_high32}
//
// This is the first memory-mapped one-token wrapper boundary. It intentionally
// composes already-validated blocks while making every consumed/produced tensor
// visible through one QMAP runtime packet.
module qmap_final_token_tail_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 8,
    parameter int MAX_TILES        = 9496,
    parameter int TILE_COUNT_WIDTH = (MAX_TILES <= 1) ? 1 : $clog2(MAX_TILES + 1),
    parameter int TILE_ROWS        = 16,
    parameter int ROW_PARALLEL     = 1,
    parameter int INPUT_SIZE       = 1024,
    parameter int HIDDEN_WIDTH     = 24,
    parameter int HIDDEN_FRAC      = 10,
    parameter int GAMMA_WIDTH      = 16,
    parameter int GAMMA_FRAC       = 7,
    parameter int NORM_WIDTH       = 24,
    parameter int NORM_FRAC        = 12,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int GROUP_PARALLEL   = 1,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int PARTIAL_WIDTH    = NORM_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TOKEN_ID_WIDTH   = 32,
    parameter int USE_BRAM_STREAMING = 1,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,
    input  wire logic                         i_final_hidden_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_final_hidden_base_override_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic                              o_norm_saturation,
    output logic [ADDR_WIDTH-1 : 0]           o_effective_final_hidden_base_addr,
    output logic [TOKEN_ID_WIDTH-1 : 0]       o_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0] o_best_score_q26,
    output logic [31 : 0]                     o_tiles_started,
    output logic [31 : 0]                     o_tiles_completed,
    output logic [31 : 0]                     o_norm_cycle_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_word_count,

    output logic                              o_mem_rd_req_valid,
    input  wire logic                         i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_rd_req_addr,
    output logic [15 : 0]                     o_mem_rd_req_len_bytes,

    input  wire logic                         i_mem_rd_rsp_valid,
    output logic                              o_mem_rd_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]  i_mem_rd_rsp_data,
    input  wire logic                         i_mem_rd_rsp_last,

    output logic                              o_mem_wr_req_valid,
    input  wire logic                         i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]           o_mem_wr_req_addr,
    output logic [15 : 0]                     o_mem_wr_req_len_bytes,

    output logic [31 : 0]                     o_mem_wr_data,
    output logic                              o_mem_wr_data_valid,
    input  wire logic                         i_mem_wr_data_ready,
    output logic                              o_mem_wr_data_last,
    input  wire logic                         i_mem_wr_done,
    input  wire logic                         i_mem_wr_error
);

    localparam int SLOT_METADATA     = 0;
    localparam int SLOT_NORM_OUTPUT  = 1;
    localparam int SLOT_WEIGHT       = 2;
    localparam int SLOT_SCALE        = 3;
    localparam int SLOT_OUTPUT       = 4;
    localparam int SLOT_EXPECTED     = 5;
    localparam int SLOT_FINAL_HIDDEN = 6;
    localparam int SLOT_FINAL_GAMMA  = 7;

    localparam int MEM_DATA_BYTES       = MEM_DATA_WIDTH / 8;
    localparam int CHUNK_WORDS          = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int HIDDEN_BYTES         = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int GAMMA_BYTES          = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int NORM_BYTES           = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int HIDDEN_CHUNK_COUNT   = HIDDEN_BYTES / MAX_READ_BYTES;
    localparam int GAMMA_CHUNK_COUNT    = GAMMA_BYTES / MAX_READ_BYTES;
    localparam int ELEMENT_INDEX_W      =
        (INPUT_SIZE <= 1) ? 1 : $clog2(INPUT_SIZE);
    localparam logic [15 : 0] CHUNK_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] NORM_BYTES_U16 = NORM_BYTES;

    typedef enum logic [4 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_HIDDEN_REQ,
        S_HIDDEN_READ,
        S_GAMMA_REQ,
        S_GAMMA_READ,
        S_RMS_START,
        S_RMS_WAIT,
        S_NORM_WRITE_REQ,
        S_NORM_WRITE_PRIME,
        S_NORM_WRITE_DATA,
        S_NORM_WRITE_WAIT,
        S_LM_START,
        S_LM_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic reader_start;
    logic reader_done;
    logic reader_error;
    logic reader_req_valid;
    logic reader_req_ready;
    logic [ADDR_WIDTH-1 : 0] reader_req_addr;
    logic [15 : 0] reader_req_len_bytes;
    logic reader_rsp_valid;
    logic reader_rsp_ready;
    logic [31 : 0] reader_rsp_data;
    logic reader_rsp_last;
    logic [31 : 0] reader_descriptor_count;
    logic [31 : 0] reader_descriptor_capacity;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_role_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dtype_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_rank_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_flags_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_element_bits_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_group_size_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_scale_tensor_id_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_base_addr_flat;
    logic [DESCRIPTOR_SLOTS*64-1 : 0] reader_desc_nbytes_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_dim3_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux3_flat;

    logic [31 : 0] desc_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_role [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dtype [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_rank [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];

    logic [ADDR_WIDTH-1 : 0] effective_final_hidden_base_addr;

    logic [INPUT_SIZE*HIDDEN_WIDTH-1 : 0] hidden_flat;
    logic [INPUT_SIZE*GAMMA_WIDTH-1 : 0] gamma_flat;
    logic [INPUT_SIZE*NORM_WIDTH-1 : 0] norm_output_flat;
    logic [31 : 0] hidden_chunk_index;
    logic [31 : 0] gamma_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] norm_write_word_index;
    logic [31 : 0] element_index;
    logic hidden_protocol_error;
    logic gamma_protocol_error;
    logic validate_error;
    logic norm_start;
    logic norm_busy;
    logic norm_done;
    logic norm_error;
    logic [63 : 0] norm_sum_squares;
    logic [63 : 0] norm_mean_square;
    logic [23 : 0] norm_inv_rms;
    logic [31 : 0] norm_write_data_word;
    logic norm_bram_error_raw;
    logic [ELEMENT_INDEX_W-1 : 0] norm_bram_load_addr;
    logic norm_bram_input_wr_en;
    logic norm_bram_gamma_wr_en;
    logic [ELEMENT_INDEX_W-1 : 0] norm_bram_output_rd_addr;
    logic [NORM_WIDTH-1 : 0] norm_bram_output_rd_data;
    logic [31 : 0] legacy_norm_cycle_count;
    logic [31 : 0] bram_norm_cycle_count;
    logic [3 : 0] final_norm_state_debug;

    logic lm_start;
    logic lm_busy;
    logic lm_done;
    logic lm_error;
    logic lm_rd_req_valid;
    logic lm_rd_req_ready;
    logic [ADDR_WIDTH-1 : 0] lm_rd_req_addr;
    logic [15 : 0] lm_rd_req_len_bytes;
    logic lm_rd_rsp_valid;
    logic lm_rd_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] lm_rd_rsp_data;
    logic lm_rd_rsp_last;
    logic lm_wr_req_valid;
    logic lm_wr_req_ready;
    logic [ADDR_WIDTH-1 : 0] lm_wr_req_addr;
    logic [15 : 0] lm_wr_req_len_bytes;
    logic [31 : 0] lm_wr_data;
    logic lm_wr_data_valid;
    logic lm_wr_data_ready;
    logic lm_wr_data_last;
    logic [31 : 0] lm_read_burst_count;
    logic [31 : 0] lm_read_word_count;
    logic [31 : 0] lm_write_word_count;
    logic [TOKEN_ID_WIDTH-1 : 0] lm_best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] lm_best_score_q26;

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]    = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]         = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]        = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_rank[desc_index]         = reader_desc_rank_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index] = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]    = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]       = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]         = reader_desc_dim0_flat[desc_index*32 +: 32];
        end
    endgenerate

    qmap_dot64_reader #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS)
    ) qmap_reader (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(reader_start),
        .i_qmap_base_addr(i_qmap_base_addr),
        .o_busy(),
        .o_done(reader_done),
        .o_error(reader_error),
        .o_mem_req_valid(reader_req_valid),
        .i_mem_req_ready(reader_req_ready),
        .o_mem_req_addr(reader_req_addr),
        .o_mem_req_len_bytes(reader_req_len_bytes),
        .i_mem_rsp_valid(reader_rsp_valid),
        .o_mem_rsp_ready(reader_rsp_ready),
        .i_mem_rsp_data(reader_rsp_data),
        .i_mem_rsp_last(reader_rsp_last),
        .o_header_magic(),
        .o_header_version(),
        .o_header_bytes(),
        .o_descriptor_bytes(),
        .o_descriptor_count(reader_descriptor_count),
        .o_descriptor_capacity(reader_descriptor_capacity),
        .o_descriptor_table_addr(),
        .o_payload_base_addr(),
        .o_image_base_addr(),
        .o_image_bytes(),
        .o_desc_tensor_id_flat(reader_desc_tensor_id_flat),
        .o_desc_role_flat(reader_desc_role_flat),
        .o_desc_dtype_flat(reader_desc_dtype_flat),
        .o_desc_rank_flat(reader_desc_rank_flat),
        .o_desc_flags_flat(reader_desc_flags_flat),
        .o_desc_element_bits_flat(reader_desc_element_bits_flat),
        .o_desc_group_size_flat(reader_desc_group_size_flat),
        .o_desc_scale_tensor_id_flat(reader_desc_scale_tensor_id_flat),
        .o_desc_base_addr_flat(reader_desc_base_addr_flat),
        .o_desc_nbytes_flat(reader_desc_nbytes_flat),
        .o_desc_dim0_flat(reader_desc_dim0_flat),
        .o_desc_dim1_flat(reader_desc_dim1_flat),
        .o_desc_dim2_flat(reader_desc_dim2_flat),
        .o_desc_dim3_flat(reader_desc_dim3_flat),
        .o_desc_aux0_flat(reader_desc_aux0_flat),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(reader_desc_aux2_flat),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    generate
        if (USE_BRAM_STREAMING != 0) begin : gen_bram_final_norm
            assign norm_error =
                norm_bram_error_raw | o_norm_saturation;
            assign legacy_norm_cycle_count = 32'd0;
            assign final_norm_state_debug =
                final_norm_stage_bram.state;

            rmsnorm_bram #(
                .INPUT_SIZE(INPUT_SIZE),
                .IN_WIDTH(HIDDEN_WIDTH),
                .IN_FRAC(HIDDEN_FRAC),
                .GAMMA_WIDTH(GAMMA_WIDTH),
                .GAMMA_FRAC(GAMMA_FRAC),
                .GAMMA_SIGNED(1),
                .INV_RMS_WIDTH(24),
                .INV_RMS_FRAC(16),
                .OUT_WIDTH(NORM_WIDTH),
                .OUT_FRAC(NORM_FRAC),
                .SUM_WIDTH(64),
                .SUM_FRAC(2 * HIDDEN_FRAC),
                .MEAN_SHIFT($clog2(INPUT_SIZE)),
                .RMS_WIDTH(HIDDEN_WIDTH),
                .RMS_FRAC(HIDDEN_FRAC),
                .DIV_NUM_WIDTH(48),
                .DIV_NUM_SHIFT(HIDDEN_FRAC + 16),
                .EPS_Q20(1),
                .ELEMENT_INDEX_W(ELEMENT_INDEX_W)
            ) final_norm_stage_bram (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_input_wr_en(norm_bram_input_wr_en),
                .i_input_wr_addr(norm_bram_load_addr),
                .i_input_wr_data(
                    i_mem_rd_rsp_data[HIDDEN_WIDTH-1 : 0]),
                .i_gamma_wr_en(norm_bram_gamma_wr_en),
                .i_gamma_wr_addr(norm_bram_load_addr),
                .i_gamma_wr_data(
                    i_mem_rd_rsp_data[GAMMA_WIDTH-1 : 0]),
                .i_start(norm_start),
                .o_busy(norm_busy),
                .o_done(norm_done),
                .o_error(norm_bram_error_raw),
                .o_saturation(o_norm_saturation),
                .i_output_rd_addr(norm_bram_output_rd_addr),
                .o_output_rd_data(norm_bram_output_rd_data),
                .o_output_wr_valid(),
                .o_output_wr_addr(),
                .o_output_wr_data(),
                .o_sum_squares(norm_sum_squares),
                .o_mean_square(norm_mean_square),
                .o_inv_rms(norm_inv_rms)
            );
        end
        else begin : gen_legacy_final_norm
            assign norm_bram_error_raw = 1'b0;
            assign norm_bram_output_rd_data = '0;
            assign final_norm_state_debug =
                {2'b00, final_norm_stage.current_state};

            final_rmsnorm_stage #(
                .INPUT_SIZE(INPUT_SIZE),
                .IN_WIDTH(HIDDEN_WIDTH),
                .IN_FRAC(HIDDEN_FRAC),
                .GAMMA_WIDTH(GAMMA_WIDTH),
                .GAMMA_FRAC(GAMMA_FRAC),
                .OUT_WIDTH(NORM_WIDTH),
                .OUT_FRAC(NORM_FRAC)
            ) final_norm_stage (
                .i_clk(i_clk),
                .i_rst_n(i_rst_n),
                .i_start(norm_start),
                .i_input_flat(hidden_flat),
                .i_gamma_flat(gamma_flat),
                .o_busy(norm_busy),
                .o_done(norm_done),
                .o_error(norm_error),
                .o_saturation(o_norm_saturation),
                .o_cycle_count(legacy_norm_cycle_count),
                .o_output_flat(norm_output_flat),
                .o_sum_squares(norm_sum_squares),
                .o_mean_square(norm_mean_square),
                .o_inv_rms(norm_inv_rms)
            );
        end
    endgenerate

    qmap_lm_head_argmax_compute_path #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DESCRIPTOR_SLOTS(DESCRIPTOR_SLOTS),
        .MAX_TILES(MAX_TILES),
        .TILE_COUNT_WIDTH(TILE_COUNT_WIDTH),
        .TILE_ROWS(TILE_ROWS),
        .ROW_PARALLEL(ROW_PARALLEL),
        .INPUT_SIZE(INPUT_SIZE),
        .GROUP_SIZE(GROUP_SIZE),
        .GROUP_COUNT(GROUP_COUNT),
        .GROUP_PARALLEL(GROUP_PARALLEL),
        .ACT_WIDTH(NORM_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH),
        .ROW_ACC_WIDTH(ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH(TOKEN_ID_WIDTH),
        .MEM_DATA_WIDTH(MEM_DATA_WIDTH),
        .MAX_READ_BYTES(MAX_READ_BYTES)
    ) lm_head_path (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(lm_start),
        .i_qmap_base_addr(i_qmap_base_addr),
        .o_busy(lm_busy),
        .o_done(lm_done),
        .o_error(lm_error),
        .o_best_token_id(lm_best_token_id),
        .o_best_score_q26(lm_best_score_q26),
        .o_tiles_started(o_tiles_started),
        .o_tiles_completed(o_tiles_completed),
        .o_mem_read_burst_count(lm_read_burst_count),
        .o_mem_read_word_count(lm_read_word_count),
        .o_mem_write_word_count(lm_write_word_count),
        .o_mem_rd_req_valid(lm_rd_req_valid),
        .i_mem_rd_req_ready(lm_rd_req_ready),
        .o_mem_rd_req_addr(lm_rd_req_addr),
        .o_mem_rd_req_len_bytes(lm_rd_req_len_bytes),
        .i_mem_rd_rsp_valid(lm_rd_rsp_valid),
        .o_mem_rd_rsp_ready(lm_rd_rsp_ready),
        .i_mem_rd_rsp_data(lm_rd_rsp_data),
        .i_mem_rd_rsp_last(lm_rd_rsp_last),
        .o_mem_wr_req_valid(lm_wr_req_valid),
        .i_mem_wr_req_ready(lm_wr_req_ready),
        .o_mem_wr_req_addr(lm_wr_req_addr),
        .o_mem_wr_req_len_bytes(lm_wr_req_len_bytes),
        .o_mem_wr_data(lm_wr_data),
        .o_mem_wr_data_valid(lm_wr_data_valid),
        .i_mem_wr_data_ready(lm_wr_data_ready),
        .o_mem_wr_data_last(lm_wr_data_last),
        .i_mem_wr_done(i_mem_wr_done),
        .i_mem_wr_error(i_mem_wr_error)
    );

    assign effective_final_hidden_base_addr =
        i_final_hidden_base_override_valid ?
        i_final_hidden_base_override_addr :
        desc_base_addr[SLOT_FINAL_HIDDEN][ADDR_WIDTH-1 : 0];
    assign o_effective_final_hidden_base_addr = effective_final_hidden_base_addr;

    assign reader_start = (state == S_READER_START);
    assign norm_start = (state == S_RMS_START);
    assign lm_start = (state == S_LM_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_norm_cycle_count =
        (USE_BRAM_STREAMING != 0) ?
        bram_norm_cycle_count :
        legacy_norm_cycle_count;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data  = i_mem_rd_rsp_data;
    assign reader_rsp_last  = i_mem_rd_rsp_last;

    assign lm_rd_req_ready = (state == S_LM_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign lm_rd_rsp_valid = (state == S_LM_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign lm_rd_rsp_data  = i_mem_rd_rsp_data;
    assign lm_rd_rsp_last  = i_mem_rd_rsp_last;
    assign lm_wr_req_ready = (state == S_LM_WAIT) ? i_mem_wr_req_ready : 1'b0;
    assign lm_wr_data_ready = (state == S_LM_WAIT) ? i_mem_wr_data_ready : 1'b0;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_HIDDEN_REQ)  ? 1'b1 :
        (state == S_GAMMA_REQ)   ? 1'b1 :
        (state == S_LM_WAIT)     ? lm_rd_req_valid :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        (state == S_HIDDEN_REQ)  ? (effective_final_hidden_base_addr + (hidden_chunk_index * MAX_READ_BYTES)) :
        (state == S_GAMMA_REQ)   ? (desc_base_addr[SLOT_FINAL_GAMMA] + (gamma_chunk_index * MAX_READ_BYTES)) :
        (state == S_LM_WAIT)     ? lm_rd_req_addr :
        'd0;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_HIDDEN_REQ)  ? CHUNK_BYTES_U16 :
        (state == S_GAMMA_REQ)   ? CHUNK_BYTES_U16 :
        (state == S_LM_WAIT)     ? lm_rd_req_len_bytes :
        16'd0;

    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_HIDDEN_READ) ? 1'b1 :
        (state == S_GAMMA_READ)  ? 1'b1 :
        (state == S_LM_WAIT)     ? lm_rd_rsp_ready :
        1'b0;

    assign o_mem_wr_req_valid =
        (state == S_NORM_WRITE_REQ) ? 1'b1 :
        (state == S_LM_WAIT)        ? lm_wr_req_valid :
        1'b0;
    assign o_mem_wr_req_addr =
        (state == S_NORM_WRITE_REQ) ? desc_base_addr[SLOT_NORM_OUTPUT] :
        (state == S_LM_WAIT)        ? lm_wr_req_addr :
        'd0;
    assign o_mem_wr_req_len_bytes =
        (state == S_NORM_WRITE_REQ) ? NORM_BYTES_U16 :
        (state == S_LM_WAIT)        ? lm_wr_req_len_bytes :
        16'd0;
    assign o_mem_wr_data =
        (state == S_NORM_WRITE_DATA) ? norm_write_data_word :
        (state == S_LM_WAIT)         ? lm_wr_data :
        32'd0;
    assign o_mem_wr_data_valid =
        (state == S_NORM_WRITE_DATA) ? 1'b1 :
        (state == S_LM_WAIT)         ? lm_wr_data_valid :
        1'b0;
    assign o_mem_wr_data_last =
        (state == S_NORM_WRITE_DATA) ? (norm_write_word_index == (INPUT_SIZE - 1)) :
        (state == S_LM_WAIT)         ? lm_wr_data_last :
        1'b0;

    assign hidden_protocol_error =
        (i_mem_rd_rsp_last != (read_word_index == (CHUNK_WORDS - 1)));
    assign gamma_protocol_error =
        (i_mem_rd_rsp_last != (read_word_index == (CHUNK_WORDS - 1)));
    assign norm_bram_load_addr =
        element_index[ELEMENT_INDEX_W-1 : 0];
    assign norm_bram_input_wr_en =
        (USE_BRAM_STREAMING != 0) &&
        (state == S_HIDDEN_READ) &&
        i_mem_rd_rsp_valid &&
        !hidden_protocol_error;
    assign norm_bram_gamma_wr_en =
        (USE_BRAM_STREAMING != 0) &&
        (state == S_GAMMA_READ) &&
        i_mem_rd_rsp_valid &&
        !gamma_protocol_error;
    assign norm_bram_output_rd_addr =
        ((state == S_NORM_WRITE_DATA) &&
         o_mem_wr_data_valid &&
         i_mem_wr_data_ready &&
         !o_mem_wr_data_last) ?
        (norm_write_word_index[ELEMENT_INDEX_W-1 : 0] + 1'b1) :
        norm_write_word_index[ELEMENT_INDEX_W-1 : 0];

    always @* begin
        element_index = 32'd0;
        norm_write_data_word = 32'd0;
        if (state == S_HIDDEN_READ) begin
            element_index = (hidden_chunk_index * CHUNK_WORDS) + read_word_index;
        end
        else if (state == S_GAMMA_READ) begin
            element_index = (gamma_chunk_index * CHUNK_WORDS) + read_word_index;
        end
        else if (state == S_NORM_WRITE_DATA) begin
            element_index = norm_write_word_index;
        end

        if (state == S_NORM_WRITE_DATA) begin
            if (USE_BRAM_STREAMING != 0) begin
                norm_write_data_word = {
                    {(MEM_DATA_WIDTH-NORM_WIDTH){
                        norm_bram_output_rd_data[NORM_WIDTH-1]}},
                    norm_bram_output_rd_data
                };
            end
            else begin
                norm_write_data_word = {
                    {(MEM_DATA_WIDTH-NORM_WIDTH){
                        norm_output_flat[
                            (element_index*NORM_WIDTH) +
                            (NORM_WIDTH-1)]}},
                    norm_output_flat[
                        element_index*NORM_WIDTH +: NORM_WIDTH]
                };
            end
        end
    end

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count < DESCRIPTOR_SLOTS) ||
            (reader_descriptor_capacity < `QMAP_FINAL_TOKEN_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_LM_HEAD_METADATA) ||
            (desc_tensor_id[SLOT_NORM_OUTPUT] != `QMAP_TENSOR_ID_LM_HEAD_ACTIVATION) ||
            (desc_tensor_id[SLOT_FINAL_HIDDEN] != `QMAP_TENSOR_ID_FINAL_HIDDEN) ||
            (desc_tensor_id[SLOT_FINAL_GAMMA] != `QMAP_TENSOR_ID_FINAL_GAMMA) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_NORM_OUTPUT] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_FINAL_HIDDEN] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_FINAL_GAMMA] != `QMAP_ROLE_PARAMETER) ||
            (desc_dtype[SLOT_NORM_OUTPUT] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_FINAL_HIDDEN] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_FINAL_GAMMA] != `QMAP_DTYPE_I16_Q8_7) ||
            (desc_rank[SLOT_NORM_OUTPUT] != 32'd1) ||
            (desc_rank[SLOT_FINAL_HIDDEN] != 32'd1) ||
            (desc_rank[SLOT_FINAL_GAMMA] != 32'd1) ||
            (desc_element_bits[SLOT_NORM_OUTPUT] != NORM_WIDTH) ||
            (desc_element_bits[SLOT_FINAL_HIDDEN] != HIDDEN_WIDTH) ||
            (desc_element_bits[SLOT_FINAL_GAMMA] != GAMMA_WIDTH) ||
            (desc_dim0[SLOT_NORM_OUTPUT] != INPUT_SIZE) ||
            (desc_dim0[SLOT_FINAL_HIDDEN] != INPUT_SIZE) ||
            (desc_dim0[SLOT_FINAL_GAMMA] != INPUT_SIZE) ||
            (desc_nbytes[SLOT_NORM_OUTPUT] != NORM_BYTES) ||
            (desc_nbytes[SLOT_FINAL_HIDDEN] != HIDDEN_BYTES) ||
            (desc_nbytes[SLOT_FINAL_GAMMA] != GAMMA_BYTES) ||
            (effective_final_hidden_base_addr == '0)) begin
            validate_error = 1'b1;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            if (USE_BRAM_STREAMING == 0) begin
                hidden_flat <= '0;
                gamma_flat <= '0;
            end
            hidden_chunk_index <= 32'd0;
            gamma_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            norm_write_word_index <= 32'd0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_best_token_id <= '0;
            o_best_score_q26 <= '0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count <= 32'd0;
            o_mem_write_word_count <= 32'd0;
            bram_norm_cycle_count <= 32'd0;
        end
        else begin
            o_done <= 1'b0;

            if (o_mem_rd_req_valid && i_mem_rd_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end
            if (i_mem_rd_rsp_valid && o_mem_rd_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end
            if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                o_mem_write_word_count <= o_mem_write_word_count + 1'b1;
            end
            if ((USE_BRAM_STREAMING != 0) &&
                ((state == S_RMS_START) ||
                 (state == S_RMS_WAIT))) begin
                bram_norm_cycle_count <=
                    bram_norm_cycle_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        state <= S_READER_START;
                        if (USE_BRAM_STREAMING == 0) begin
                            hidden_flat <= '0;
                            gamma_flat <= '0;
                        end
                        hidden_chunk_index <= 32'd0;
                        gamma_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        norm_write_word_index <= 32'd0;
                        o_error <= 1'b0;
                        o_best_token_id <= '0;
                        o_best_score_q26 <= '0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        bram_norm_cycle_count <= 32'd0;
                    end
                end

                S_READER_START: begin
                    state <= S_READER_WAIT;
                end

                S_READER_WAIT: begin
                    if (reader_done) begin
                        if (reader_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_VALIDATE;
                        end
                    end
                end

                S_VALIDATE: begin
                    if (validate_error) begin
                        o_error <= 1'b1;
                        state <= S_DONE;
                    end
                    else begin
                        hidden_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        state <= S_HIDDEN_REQ;
                    end
                end

                S_HIDDEN_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_HIDDEN_READ;
                    end
                end

                S_HIDDEN_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (hidden_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            if (USE_BRAM_STREAMING == 0) begin
                                hidden_flat[
                                    element_index*HIDDEN_WIDTH +:
                                    HIDDEN_WIDTH] <=
                                    i_mem_rd_rsp_data[
                                        HIDDEN_WIDTH-1 : 0];
                            end
                            if (i_mem_rd_rsp_last) begin
                                if (hidden_chunk_index == (HIDDEN_CHUNK_COUNT - 1)) begin
                                    gamma_chunk_index <= 32'd0;
                                    read_word_index <= 32'd0;
                                    state <= S_GAMMA_REQ;
                                end
                                else begin
                                    hidden_chunk_index <= hidden_chunk_index + 1'b1;
                                    read_word_index <= 32'd0;
                                    state <= S_HIDDEN_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_GAMMA_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_GAMMA_READ;
                    end
                end

                S_GAMMA_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (gamma_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            if (USE_BRAM_STREAMING == 0) begin
                                gamma_flat[
                                    element_index*GAMMA_WIDTH +:
                                    GAMMA_WIDTH] <=
                                    i_mem_rd_rsp_data[
                                        GAMMA_WIDTH-1 : 0];
                            end
                            if (i_mem_rd_rsp_last) begin
                                if (gamma_chunk_index == (GAMMA_CHUNK_COUNT - 1)) begin
                                    state <= S_RMS_START;
                                end
                                else begin
                                    gamma_chunk_index <= gamma_chunk_index + 1'b1;
                                    read_word_index <= 32'd0;
                                    state <= S_GAMMA_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_RMS_START: begin
                    state <= S_RMS_WAIT;
                end

                S_RMS_WAIT: begin
                    if (norm_done) begin
                        if (norm_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            norm_write_word_index <= 32'd0;
                            state <= S_NORM_WRITE_REQ;
                        end
                    end
                end

                S_NORM_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        norm_write_word_index <= 32'd0;
                        state <= S_NORM_WRITE_PRIME;
                    end
                end

                S_NORM_WRITE_PRIME: begin
                    norm_write_word_index <= 32'd0;
                    state <= S_NORM_WRITE_DATA;
                end

                S_NORM_WRITE_DATA: begin
                    if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                        if (o_mem_wr_data_last) begin
                            state <= S_NORM_WRITE_WAIT;
                        end
                        else begin
                            norm_write_word_index <= norm_write_word_index + 1'b1;
                        end
                    end
                end

                S_NORM_WRITE_WAIT: begin
                    if (i_mem_wr_done || i_mem_wr_error) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            state <= S_LM_START;
                        end
                    end
                end

                S_LM_START: begin
                    state <= S_LM_WAIT;
                end

                S_LM_WAIT: begin
                    if (lm_done) begin
                        if (lm_error) begin
                            o_error <= 1'b1;
                        end
                        else begin
                            o_best_token_id <= lm_best_token_id;
                            o_best_score_q26 <= lm_best_score_q26;
                        end
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    o_error <= 1'b1;
                    state <= S_DONE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
