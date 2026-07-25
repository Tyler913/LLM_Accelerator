`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed LM-head greedy argmax compute path.
//
// The wrapper keeps the existing memory-backed LM-head scheduler intact, but
// sources its control inputs from QMAP descriptors:
//   - final RMSNorm activation descriptor
//   - persistent LM-head Q4 weight descriptor
//   - persistent LM-head Q2.14 scale descriptor
//   - output token/score descriptor
// The weight/scale descriptors expose the runtime scan range through aux2
// (token base) and aux3 (tile count).
module qmap_lm_head_argmax_compute_path #(
    parameter int ADDR_WIDTH       = 64,
    parameter int DESCRIPTOR_SLOTS = 6,
    parameter int MAX_TILES        = 9496,
    parameter int TILE_COUNT_WIDTH = (MAX_TILES <= 1) ? 1 : $clog2(MAX_TILES + 1),
    parameter int TILE_ROWS        = 16,
    parameter int INPUT_SIZE       = 1024,
    parameter int GROUP_SIZE       = 64,
    parameter int GROUP_COUNT      = INPUT_SIZE / GROUP_SIZE,
    parameter int ACT_WIDTH        = 24,
    parameter int ACT_FRAC         = 12,
    parameter int WEIGHT_WIDTH     = 4,
    parameter int SCALE_WIDTH      = 16,
    parameter int SCALE_FRAC       = 14,
    parameter int PARTIAL_WIDTH    = ACT_WIDTH + WEIGHT_WIDTH + $clog2(GROUP_SIZE),
    parameter int SCALED_WIDTH     = PARTIAL_WIDTH + SCALE_WIDTH,
    parameter int ROW_ACC_WIDTH    = SCALED_WIDTH + $clog2(GROUP_COUNT) + 2,
    parameter int TOKEN_ID_WIDTH   = 32,
    parameter int MEM_DATA_WIDTH   = 32,
    parameter int MAX_READ_BYTES   = 1024
)
(
    input  wire logic                              i_clk,
    input  wire logic                              i_rst_n,

    input  wire logic                              i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]           i_qmap_base_addr,

    output logic                                   o_busy,
    output logic                                   o_done,
    output logic                                   o_error,
    output logic [TOKEN_ID_WIDTH-1 : 0]            o_best_token_id,
    output logic signed [ROW_ACC_WIDTH-1 : 0]      o_best_score_q26,
    output logic [31 : 0]                          o_tiles_started,
    output logic [31 : 0]                          o_tiles_completed,
    output logic [31 : 0]                          o_mem_read_burst_count,
    output logic [31 : 0]                          o_mem_read_word_count,
    output logic [31 : 0]                          o_mem_write_word_count,

    output logic                                   o_mem_rd_req_valid,
    input  wire logic                              i_mem_rd_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                o_mem_rd_req_addr,
    output logic [15 : 0]                          o_mem_rd_req_len_bytes,

    input  wire logic                              i_mem_rd_rsp_valid,
    output logic                                   o_mem_rd_rsp_ready,
    input  wire logic [MEM_DATA_WIDTH-1 : 0]       i_mem_rd_rsp_data,
    input  wire logic                              i_mem_rd_rsp_last,

    output logic                                   o_mem_wr_req_valid,
    input  wire logic                              i_mem_wr_req_ready,
    output logic [ADDR_WIDTH-1 : 0]                o_mem_wr_req_addr,
    output logic [15 : 0]                          o_mem_wr_req_len_bytes,

    output logic [31 : 0]                          o_mem_wr_data,
    output logic                                   o_mem_wr_data_valid,
    input  wire logic                              i_mem_wr_data_ready,
    output logic                                   o_mem_wr_data_last,
    input  wire logic                              i_mem_wr_done,
    input  wire logic                              i_mem_wr_error
);

    localparam int SLOT_METADATA   = 0;
    localparam int SLOT_ACTIVATION = 1;
    localparam int SLOT_WEIGHT     = 2;
    localparam int SLOT_SCALE      = 3;
    localparam int SLOT_OUTPUT     = 4;
    localparam int SLOT_EXPECTED   = 5;

    localparam int MEM_DATA_BYTES          = MEM_DATA_WIDTH / 8;
    localparam int ACTIVATION_BYTES        = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int ACTIVATION_CHUNK_BYTES  = MAX_READ_BYTES;
    localparam int ACTIVATION_CHUNK_WORDS  = ACTIVATION_CHUNK_BYTES / MEM_DATA_BYTES;
    localparam int ACTIVATION_CHUNK_COUNT  = ACTIVATION_BYTES / ACTIVATION_CHUNK_BYTES;
    localparam int WEIGHT_ROW_BYTES        = (INPUT_SIZE * WEIGHT_WIDTH) / 8;
    localparam int SCALE_ROW_BYTES         = GROUP_COUNT * (SCALE_WIDTH / 8);
    localparam int OUTPUT_WORDS            = 3;
    localparam logic [15 : 0] ACTIVATION_CHUNK_LEN_BYTES = ACTIVATION_CHUNK_BYTES;
    localparam logic [15 : 0] OUTPUT_BYTES_U16 = OUTPUT_WORDS * MEM_DATA_BYTES;

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_ACT_REQ,
        S_ACT_READ,
        S_SCHED_START,
        S_SCHED_WAIT,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_DONE
    } state_t;

    state_t state;

    logic reader_start;
    logic reader_busy;
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
    logic [31 : 0] reader_header_magic;
    logic [31 : 0] reader_header_version;
    logic [31 : 0] reader_header_bytes;
    logic [31 : 0] reader_descriptor_bytes;
    logic [31 : 0] reader_descriptor_count;
    logic [31 : 0] reader_descriptor_capacity;
    logic [63 : 0] reader_descriptor_table_addr;
    logic [63 : 0] reader_payload_base_addr;
    logic [63 : 0] reader_image_base_addr;
    logic [63 : 0] reader_image_bytes;
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
    logic [31 : 0] desc_flags [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_group_size [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_scale_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux2 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux3 [0 : DESCRIPTOR_SLOTS-1];

    logic [INPUT_SIZE*ACT_WIDTH-1 : 0] activation_flat;
    logic [31 : 0] activation_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] activation_element_index;
    logic [ADDR_WIDTH-1 : 0] activation_req_addr;
    logic expected_activation_last;
    logic activation_protocol_error;

    logic scheduler_start;
    logic scheduler_busy;
    logic scheduler_done;
    logic scheduler_error;
    logic scheduler_req_valid;
    logic scheduler_req_ready;
    logic [ADDR_WIDTH-1 : 0] scheduler_req_addr;
    logic [15 : 0] scheduler_req_len_bytes;
    logic scheduler_rsp_valid;
    logic scheduler_rsp_ready;
    logic [MEM_DATA_WIDTH-1 : 0] scheduler_rsp_data;
    logic scheduler_rsp_last;
    logic [TILE_COUNT_WIDTH-1 : 0] scheduler_tile_count;
    logic [TILE_COUNT_WIDTH-1 : 0] scheduler_current_tile;
    logic [TOKEN_ID_WIDTH-1 : 0] scheduler_best_token_id;
    logic signed [ROW_ACC_WIDTH-1 : 0] scheduler_best_score_q26;
    logic [31 : 0] scheduler_compute_cycle_count;
    logic [31 : 0] scheduler_mem_read_burst_count;
    logic [31 : 0] scheduler_mem_read_word_count;

    logic [1 : 0] write_word_index;
    logic signed [63 : 0] best_score_ext;
    logic [31 : 0] write_data_word;
    logic [31 : 0] scan_base_token;
    logic [31 : 0] scan_tile_count;
    logic [31 : 0] scan_rows;
    logic validate_error;

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]       = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]            = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]           = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_flags[desc_index]           = reader_desc_flags_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index]    = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_group_size[desc_index]      = reader_desc_group_size_flat[desc_index*32 +: 32];
            assign desc_scale_tensor_id[desc_index] = reader_desc_scale_tensor_id_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]       = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]          = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]            = reader_desc_dim0_flat[desc_index*32 +: 32];
            assign desc_dim1[desc_index]            = reader_desc_dim1_flat[desc_index*32 +: 32];
            assign desc_aux0[desc_index]            = reader_desc_aux0_flat[desc_index*32 +: 32];
            assign desc_aux2[desc_index]            = reader_desc_aux2_flat[desc_index*32 +: 32];
            assign desc_aux3[desc_index]            = reader_desc_aux3_flat[desc_index*32 +: 32];
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
        .o_busy(reader_busy),
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
        .o_header_magic(reader_header_magic),
        .o_header_version(reader_header_version),
        .o_header_bytes(reader_header_bytes),
        .o_descriptor_bytes(reader_descriptor_bytes),
        .o_descriptor_count(reader_descriptor_count),
        .o_descriptor_capacity(reader_descriptor_capacity),
        .o_descriptor_table_addr(reader_descriptor_table_addr),
        .o_payload_base_addr(reader_payload_base_addr),
        .o_image_base_addr(reader_image_base_addr),
        .o_image_bytes(reader_image_bytes),
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

    lm_head_argmax_tile_scheduler #(
        .ADDR_WIDTH      (ADDR_WIDTH),
        .MAX_TILES       (MAX_TILES),
        .TILE_COUNT_WIDTH(TILE_COUNT_WIDTH),
        .TILE_ROWS       (TILE_ROWS),
        .INPUT_SIZE      (INPUT_SIZE),
        .GROUP_SIZE      (GROUP_SIZE),
        .GROUP_COUNT     (GROUP_COUNT),
        .ACT_WIDTH       (ACT_WIDTH),
        .ACT_FRAC        (ACT_FRAC),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH),
        .SCALE_WIDTH     (SCALE_WIDTH),
        .SCALE_FRAC      (SCALE_FRAC),
        .PARTIAL_WIDTH   (PARTIAL_WIDTH),
        .SCALED_WIDTH    (SCALED_WIDTH),
        .ROW_ACC_WIDTH   (ROW_ACC_WIDTH),
        .TOKEN_ID_WIDTH  (TOKEN_ID_WIDTH),
        .MEM_DATA_WIDTH  (MEM_DATA_WIDTH),
        .MAX_READ_BYTES  (MAX_READ_BYTES)
    ) scheduler (
        .i_clk                  (i_clk),
        .i_rst_n                (i_rst_n),
        .i_start                (scheduler_start),
        .i_token_base           (scan_base_token[TOKEN_ID_WIDTH-1 : 0]),
        .i_tile_count           (scheduler_tile_count),
        .i_activation_flat      (activation_flat),
        .i_weight_base_addr     (desc_base_addr[SLOT_WEIGHT]),
        .i_scale_base_addr      (desc_base_addr[SLOT_SCALE]),
        .o_mem_req_valid        (scheduler_req_valid),
        .i_mem_req_ready        (scheduler_req_ready),
        .o_mem_req_addr         (scheduler_req_addr),
        .o_mem_req_len_bytes    (scheduler_req_len_bytes),
        .i_mem_rsp_valid        (scheduler_rsp_valid),
        .o_mem_rsp_ready        (scheduler_rsp_ready),
        .i_mem_rsp_data         (scheduler_rsp_data),
        .i_mem_rsp_last         (scheduler_rsp_last),
        .o_busy                 (scheduler_busy),
        .o_done                 (scheduler_done),
        .o_error                (scheduler_error),
        .o_best_token_id        (scheduler_best_token_id),
        .o_best_score_q26       (scheduler_best_score_q26),
        .o_current_tile_index   (scheduler_current_tile),
        .o_tiles_started        (o_tiles_started),
        .o_tiles_completed      (o_tiles_completed),
        .o_compute_cycle_count  (scheduler_compute_cycle_count),
        .o_mem_read_burst_count (scheduler_mem_read_burst_count),
        .o_mem_read_word_count  (scheduler_mem_read_word_count)
    );

    assign o_busy = (state != S_IDLE);
    assign reader_start = (state == S_READER_START);
    assign scheduler_start = (state == S_SCHED_START);
    assign scheduler_tile_count = scan_tile_count[TILE_COUNT_WIDTH-1 : 0];
    assign activation_element_index =
        (activation_chunk_index * ACTIVATION_CHUNK_WORDS) + read_word_index;
    assign activation_req_addr =
        desc_base_addr[SLOT_ACTIVATION] +
        ({{(ADDR_WIDTH-32){1'b0}}, activation_chunk_index} * ACTIVATION_CHUNK_BYTES);
    assign expected_activation_last = (read_word_index == (ACTIVATION_CHUNK_WORDS - 1));
    assign activation_protocol_error =
        (state == S_ACT_READ) && i_mem_rd_rsp_valid &&
        (i_mem_rd_rsp_last != expected_activation_last);

    assign best_score_ext = {{(64-ROW_ACC_WIDTH){o_best_score_q26[ROW_ACC_WIDTH-1]}}, o_best_score_q26};
    assign write_data_word =
        (write_word_index == 2'd0) ? o_best_token_id :
        (write_word_index == 2'd1) ? best_score_ext[31 : 0] :
        best_score_ext[63 : 32];

    assign o_mem_wr_req_valid     = (state == S_WRITE_REQ);
    assign o_mem_wr_req_addr      = desc_base_addr[SLOT_OUTPUT];
    assign o_mem_wr_req_len_bytes = OUTPUT_BYTES_U16;
    assign o_mem_wr_data          = write_data_word;
    assign o_mem_wr_data_valid    = (state == S_WRITE_DATA);
    assign o_mem_wr_data_last     = (write_word_index == (OUTPUT_WORDS - 1));

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_ACT_REQ)     ? 1'b1 :
        (state == S_SCHED_WAIT)  ? scheduler_req_valid :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        (state == S_ACT_REQ)     ? activation_req_addr :
        (state == S_SCHED_WAIT)  ? scheduler_req_addr :
        'd0;
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_ACT_REQ)     ? ACTIVATION_CHUNK_LEN_BYTES :
        (state == S_SCHED_WAIT)  ? scheduler_req_len_bytes :
        16'd0;

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign scheduler_req_ready = (state == S_SCHED_WAIT) ? i_mem_rd_req_ready : 1'b0;

    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_ACT_READ)    ? 1'b1 :
        (state == S_SCHED_WAIT)  ? scheduler_rsp_ready :
        1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data  = i_mem_rd_rsp_data;
    assign reader_rsp_last  = i_mem_rd_rsp_last;
    assign scheduler_rsp_valid = (state == S_SCHED_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign scheduler_rsp_data  = i_mem_rd_rsp_data;
    assign scheduler_rsp_last  = i_mem_rd_rsp_last;

    always @* begin
        scan_base_token = desc_aux2[SLOT_WEIGHT];
        scan_tile_count = desc_aux3[SLOT_WEIGHT];
        scan_rows = desc_aux3[SLOT_WEIGHT] * TILE_ROWS;

        validate_error = 1'b0;
        if ((reader_descriptor_count < DESCRIPTOR_SLOTS) ||
            (reader_descriptor_capacity < `QMAP_LM_HEAD_DESCRIPTOR_CAPACITY) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_LM_HEAD_METADATA) ||
            (desc_tensor_id[SLOT_ACTIVATION] != `QMAP_TENSOR_ID_LM_HEAD_ACTIVATION) ||
            (desc_tensor_id[SLOT_WEIGHT] != `QMAP_TENSOR_ID_LM_HEAD_WEIGHT) ||
            (desc_tensor_id[SLOT_SCALE] != `QMAP_TENSOR_ID_LM_HEAD_SCALE) ||
            (desc_tensor_id[SLOT_OUTPUT] != `QMAP_TENSOR_ID_LM_HEAD_OUTPUT) ||
            (desc_tensor_id[SLOT_EXPECTED] != `QMAP_TENSOR_ID_LM_HEAD_EXPECTED) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_ACTIVATION] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_WEIGHT] != `QMAP_ROLE_Q4_WEIGHT) ||
            (desc_role[SLOT_SCALE] != `QMAP_ROLE_Q4_SCALE) ||
            (desc_role[SLOT_OUTPUT] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_EXPECTED] != `QMAP_ROLE_EXPECTED) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_ACTIVATION] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_WEIGHT] != `QMAP_DTYPE_PACKED_Q4_S4) ||
            (desc_dtype[SLOT_SCALE] != `QMAP_DTYPE_U16_Q2_14) ||
            (desc_dtype[SLOT_OUTPUT] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_EXPECTED] != `QMAP_DTYPE_U32) ||
            (desc_dim0[SLOT_ACTIVATION] != INPUT_SIZE) ||
            (desc_nbytes[SLOT_ACTIVATION] != ACTIVATION_BYTES) ||
            (desc_dim1[SLOT_WEIGHT] != INPUT_SIZE) ||
            (desc_dim1[SLOT_SCALE] != GROUP_COUNT) ||
            (desc_group_size[SLOT_WEIGHT] != GROUP_SIZE) ||
            (desc_element_bits[SLOT_WEIGHT] != WEIGHT_WIDTH) ||
            (desc_element_bits[SLOT_SCALE] != SCALE_WIDTH) ||
            (desc_scale_tensor_id[SLOT_WEIGHT] != `QMAP_TENSOR_ID_LM_HEAD_SCALE) ||
            (desc_aux0[SLOT_WEIGHT] != `QMAP_MATRIX_ID_EMBED_LM_HEAD) ||
            (desc_aux0[SLOT_SCALE] != `QMAP_MATRIX_ID_EMBED_LM_HEAD) ||
            (desc_aux2[SLOT_WEIGHT] != desc_aux2[SLOT_SCALE]) ||
            (desc_aux3[SLOT_WEIGHT] != desc_aux3[SLOT_SCALE]) ||
            (desc_aux3[SLOT_WEIGHT] == 32'd0) ||
            (desc_aux3[SLOT_WEIGHT] > MAX_TILES) ||
            ((desc_aux2[SLOT_WEIGHT] + scan_rows) > desc_dim0[SLOT_WEIGHT]) ||
            ((desc_aux2[SLOT_SCALE] + scan_rows) > desc_dim0[SLOT_SCALE]) ||
            (desc_nbytes[SLOT_WEIGHT] < ({32'd0, desc_dim0[SLOT_WEIGHT]} * WEIGHT_ROW_BYTES)) ||
            (desc_nbytes[SLOT_SCALE] < ({32'd0, desc_dim0[SLOT_SCALE]} * SCALE_ROW_BYTES)) ||
            (desc_dim0[SLOT_OUTPUT] < OUTPUT_WORDS) ||
            (desc_nbytes[SLOT_OUTPUT] < OUTPUT_WORDS * MEM_DATA_BYTES) ||
            (desc_dim0[SLOT_EXPECTED] < OUTPUT_WORDS) ||
            (desc_nbytes[SLOT_EXPECTED] < OUTPUT_WORDS * MEM_DATA_BYTES)) begin
            validate_error = 1'b1;
        end
    end

    always @(posedge i_clk) begin
        if (!i_rst_n) begin
            state                   <= S_IDLE;
            activation_chunk_index  <= 32'd0;
            read_word_index         <= 32'd0;
            activation_flat         <= 'd0;
            write_word_index        <= 2'd0;
            o_done                  <= 1'b0;
            o_error                 <= 1'b0;
            o_best_token_id         <= 'd0;
            o_best_score_q26        <= 'd0;
            o_mem_read_burst_count  <= 32'd0;
            o_mem_read_word_count   <= 32'd0;
            o_mem_write_word_count  <= 32'd0;
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

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        activation_chunk_index <= 32'd0;
                        read_word_index        <= 32'd0;
                        activation_flat        <= 'd0;
                        write_word_index       <= 2'd0;
                        o_error                <= 1'b0;
                        o_best_token_id         <= 'd0;
                        o_best_score_q26        <= 'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count  <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state                  <= S_READER_START;
                    end
                end

                S_READER_START: begin
                    state <= S_READER_WAIT;
                end

                S_READER_WAIT: begin
                    if (reader_done) begin
                        if (reader_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end
                        else begin
                            state <= S_VALIDATE;
                        end
                    end
                end

                S_VALIDATE: begin
                    if (validate_error) begin
                        o_error <= 1'b1;
                        state   <= S_DONE;
                    end
                    else begin
                        activation_chunk_index <= 32'd0;
                        read_word_index        <= 32'd0;
                        state                  <= S_ACT_REQ;
                    end
                end

                S_ACT_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state           <= S_ACT_READ;
                    end
                end

                S_ACT_READ: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (activation_protocol_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end
                        else begin
                            activation_flat[activation_element_index*ACT_WIDTH +: ACT_WIDTH]
                                <= i_mem_rd_rsp_data[ACT_WIDTH-1 : 0];
                            if (expected_activation_last) begin
                                read_word_index <= 32'd0;
                                if (activation_chunk_index == (ACTIVATION_CHUNK_COUNT - 1)) begin
                                    state <= S_SCHED_START;
                                end
                                else begin
                                    activation_chunk_index <= activation_chunk_index + 1'b1;
                                    state                  <= S_ACT_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_SCHED_START: begin
                    state <= S_SCHED_WAIT;
                end

                S_SCHED_WAIT: begin
                    if (scheduler_done) begin
                        if (scheduler_error) begin
                            o_error <= 1'b1;
                            state   <= S_DONE;
                        end
                        else begin
                            o_best_token_id  <= scheduler_best_token_id;
                            o_best_score_q26 <= scheduler_best_score_q26;
                            write_word_index <= 2'd0;
                            state            <= S_WRITE_REQ;
                        end
                    end
                end

                S_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        write_word_index <= 2'd0;
                        state            <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                        if (o_mem_wr_data_last) begin
                            state <= S_WRITE_WAIT;
                        end
                        else begin
                            write_word_index <= write_word_index + 1'b1;
                        end
                    end
                end

                S_WRITE_WAIT: begin
                    if (i_mem_wr_done || i_mem_wr_error) begin
                        if (i_mem_wr_error) begin
                            o_error <= 1'b1;
                        end
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    state  <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
