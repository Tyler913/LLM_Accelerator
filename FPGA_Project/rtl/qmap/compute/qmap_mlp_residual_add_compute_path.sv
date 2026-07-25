`default_nettype none

`include "qmap_defs.svh"

// QMAP-backed final MLP residual-add wrapper for one descriptor-selected layer:
//
//   post_attn_hidden[1024] + down_out[1024] -> layer_out[1024]
//
// The focused arithmetic remains inside mlp_residual_add_stage. This top adds
// QMAP descriptor validation, memory reads/writes, and debug counters.
module qmap_mlp_residual_add_compute_path #(
    parameter int ADDR_WIDTH           = 64,
    parameter int DESCRIPTOR_SLOTS     = 5,
    parameter int NUM_LAYERS           = 28,
    parameter int INPUT_SIZE           = 1024,
    parameter int POST_ATTENTION_WIDTH = 24,
    parameter int POST_ATTENTION_FRAC  = 10,
    parameter int DOWN_WIDTH           = 24,
    parameter int DOWN_FRAC            = 12,
    parameter int OUT_WIDTH            = 24,
    parameter int OUT_FRAC             = 10,
    parameter int MEM_DATA_WIDTH       = 32,
    parameter int MAX_READ_BYTES       = 1024
)
(
    input  wire logic                         i_clk,
    input  wire logic                         i_rst_n,

    input  wire logic                         i_start,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_qmap_base_addr,
    input  wire logic                         i_output_base_override_valid,
    input  wire logic [ADDR_WIDTH-1 : 0]      i_output_base_override_addr,

    output logic                              o_busy,
    output logic                              o_done,
    output logic                              o_error,
    output logic [ADDR_WIDTH-1 : 0]           o_effective_output_base_addr,
    output logic                              o_saturation,
    output logic [31 : 0]                     o_output_count,
    output logic [31 : 0]                     o_stage_cycle_count,
    output logic [31 : 0]                     o_output_write_word_count,
    output logic [31 : 0]                     o_mem_read_burst_count,
    output logic [31 : 0]                     o_mem_read_word_count,
    output logic [31 : 0]                     o_mem_write_req_count,
    output logic [31 : 0]                     o_mem_write_word_count,
    output logic [7 : 0]                      o_state_debug,
    output logic [1 : 0]                      o_read_slot_debug,

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

    localparam int SLOT_METADATA  = 0;
    localparam int SLOT_POST_ATTN = 1;
    localparam int SLOT_DOWN      = 2;
    localparam int SLOT_OUTPUT    = 3;
    localparam int SLOT_EXPECTED  = 4;

    localparam int MEM_DATA_BYTES = MEM_DATA_WIDTH / 8;
    localparam int CHUNK_WORDS    = MAX_READ_BYTES / MEM_DATA_BYTES;
    localparam int VECTOR_BYTES   = INPUT_SIZE * MEM_DATA_BYTES;
    localparam int CHUNK_COUNT    = VECTOR_BYTES / MAX_READ_BYTES;
    localparam int METADATA_WORDS = 11;

    localparam logic [15 : 0] MAX_READ_BYTES_U16 = MAX_READ_BYTES;
    localparam logic [15 : 0] VECTOR_BYTES_U16   = VECTOR_BYTES;

    typedef enum logic [3 : 0] {
        S_IDLE,
        S_READER_START,
        S_READER_WAIT,
        S_VALIDATE,
        S_READ_REQ,
        S_READ_DATA,
        S_STAGE_START,
        S_STAGE_WAIT,
        S_WRITE_REQ,
        S_WRITE_DATA,
        S_WRITE_WAIT,
        S_DONE
    } state_t;

    typedef enum logic [0 : 0] {
        R_POST_ATTN,
        R_DOWN
    } read_slot_t;

    state_t state;
    read_slot_t active_read_slot;

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
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux0_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux1_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux2_flat;
    logic [DESCRIPTOR_SLOTS*32-1 : 0] reader_desc_aux3_flat;

    logic [31 : 0] desc_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_role [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dtype [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_rank [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_flags [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_element_bits [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_group_size [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_scale_tensor_id [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_base_addr [0 : DESCRIPTOR_SLOTS-1];
    logic [63 : 0] desc_nbytes [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_dim1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux0 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux1 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux2 [0 : DESCRIPTOR_SLOTS-1];
    logic [31 : 0] desc_aux3 [0 : DESCRIPTOR_SLOTS-1];

    logic [INPUT_SIZE*POST_ATTENTION_WIDTH-1 : 0] post_attn_hidden_flat;
    logic [INPUT_SIZE*DOWN_WIDTH-1 : 0] down_out_flat;
    logic [INPUT_SIZE*OUT_WIDTH-1 : 0] layer_out_flat;

    logic [31 : 0] read_chunk_index;
    logic [31 : 0] read_word_index;
    logic [31 : 0] write_word_index;
    logic [31 : 0] element_index;
    logic [ADDR_WIDTH-1 : 0] active_read_base_addr;
    logic [31 : 0] write_data_word;
    logic read_protocol_error;
    logic validate_error;
    logic output_base_override_valid_reg;
    logic [ADDR_WIDTH-1 : 0] output_base_override_addr_reg;

    logic stage_start;
    logic stage_busy;
    logic stage_done;
    logic stage_error;
    logic stage_saturation;
    logic [31 : 0] stage_output_count;
    logic [31 : 0] stage_cycle_count;

    genvar desc_index;
    generate
        for (desc_index = 0 ; desc_index < DESCRIPTOR_SLOTS ; desc_index = desc_index + 1) begin : gen_desc_unpack
            assign desc_tensor_id[desc_index]       = reader_desc_tensor_id_flat[desc_index*32 +: 32];
            assign desc_role[desc_index]            = reader_desc_role_flat[desc_index*32 +: 32];
            assign desc_dtype[desc_index]           = reader_desc_dtype_flat[desc_index*32 +: 32];
            assign desc_rank[desc_index]            = reader_desc_rank_flat[desc_index*32 +: 32];
            assign desc_flags[desc_index]           = reader_desc_flags_flat[desc_index*32 +: 32];
            assign desc_element_bits[desc_index]    = reader_desc_element_bits_flat[desc_index*32 +: 32];
            assign desc_group_size[desc_index]      = reader_desc_group_size_flat[desc_index*32 +: 32];
            assign desc_scale_tensor_id[desc_index] = reader_desc_scale_tensor_id_flat[desc_index*32 +: 32];
            assign desc_base_addr[desc_index]       = reader_desc_base_addr_flat[desc_index*64 +: 64];
            assign desc_nbytes[desc_index]          = reader_desc_nbytes_flat[desc_index*64 +: 64];
            assign desc_dim0[desc_index]            = reader_desc_dim0_flat[desc_index*32 +: 32];
            assign desc_dim1[desc_index]            = reader_desc_dim1_flat[desc_index*32 +: 32];
            assign desc_aux0[desc_index]            = reader_desc_aux0_flat[desc_index*32 +: 32];
            assign desc_aux1[desc_index]            = reader_desc_aux1_flat[desc_index*32 +: 32];
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
        .o_desc_dim2_flat(),
        .o_desc_dim3_flat(),
        .o_desc_aux0_flat(reader_desc_aux0_flat),
        .o_desc_aux1_flat(reader_desc_aux1_flat),
        .o_desc_aux2_flat(reader_desc_aux2_flat),
        .o_desc_aux3_flat(reader_desc_aux3_flat)
    );

    mlp_residual_add_stage #(
        .INPUT_SIZE          (INPUT_SIZE),
        .POST_ATTENTION_WIDTH(POST_ATTENTION_WIDTH),
        .POST_ATTENTION_FRAC (POST_ATTENTION_FRAC),
        .DOWN_WIDTH          (DOWN_WIDTH),
        .DOWN_FRAC           (DOWN_FRAC),
        .OUT_WIDTH           (OUT_WIDTH),
        .OUT_FRAC            (OUT_FRAC)
    ) residual_add_stage (
        .i_clk                  (i_clk),
        .i_rst_n                (i_rst_n),
        .i_start                (stage_start),
        .i_post_attn_hidden_flat(post_attn_hidden_flat),
        .i_down_out_flat        (down_out_flat),
        .o_busy                 (stage_busy),
        .o_done                 (stage_done),
        .o_error                (stage_error),
        .o_saturation           (stage_saturation),
        .o_output_count         (stage_output_count),
        .o_cycle_count          (stage_cycle_count),
        .o_layer_out_flat       (layer_out_flat)
    );

    assign reader_start = (state == S_READER_START);
    assign stage_start = (state == S_STAGE_START);
    assign o_busy = (state != S_IDLE) && (state != S_DONE);
    assign o_state_debug = {4'd0, state};
    assign o_read_slot_debug = {1'b0, active_read_slot};
    assign o_effective_output_base_addr =
        (output_base_override_valid_reg === 1'b1) ?
        output_base_override_addr_reg :
        desc_base_addr[SLOT_OUTPUT];

    assign reader_req_ready = (state == S_READER_WAIT) ? i_mem_rd_req_ready : 1'b0;
    assign reader_rsp_valid = (state == S_READER_WAIT) ? i_mem_rd_rsp_valid : 1'b0;
    assign reader_rsp_data  = i_mem_rd_rsp_data;
    assign reader_rsp_last  = i_mem_rd_rsp_last;

    assign o_mem_rd_req_valid =
        (state == S_READER_WAIT) ? reader_req_valid :
        (state == S_READ_REQ)    ? 1'b1 :
        1'b0;
    assign o_mem_rd_req_addr =
        (state == S_READER_WAIT) ? reader_req_addr :
        active_read_base_addr + ({{(ADDR_WIDTH-32){1'b0}}, read_chunk_index} * MAX_READ_BYTES);
    assign o_mem_rd_req_len_bytes =
        (state == S_READER_WAIT) ? reader_req_len_bytes :
        (state == S_READ_REQ)    ? MAX_READ_BYTES_U16 :
        16'd0;
    assign o_mem_rd_rsp_ready =
        (state == S_READER_WAIT) ? reader_rsp_ready :
        (state == S_READ_DATA)   ? 1'b1 :
        1'b0;

    assign o_mem_wr_req_valid = (state == S_WRITE_REQ);
    assign o_mem_wr_req_addr = o_effective_output_base_addr;
    assign o_mem_wr_req_len_bytes = VECTOR_BYTES_U16;
    assign o_mem_wr_data_valid = (state == S_WRITE_DATA);
    assign o_mem_wr_data = write_data_word;
    assign o_mem_wr_data_last = (state == S_WRITE_DATA) && (write_word_index == (INPUT_SIZE - 1));

    assign element_index = (state == S_WRITE_DATA) ?
        write_word_index :
        ((read_chunk_index * CHUNK_WORDS) + read_word_index);

    assign read_protocol_error = (i_mem_rd_rsp_last != (read_word_index == (CHUNK_WORDS - 1)));

    always @* begin
        if (active_read_slot == R_POST_ATTN) begin
            active_read_base_addr = desc_base_addr[SLOT_POST_ATTN];
        end
        else begin
            active_read_base_addr = desc_base_addr[SLOT_DOWN];
        end
    end

    always @* begin
        write_data_word = {
            {8{layer_out_flat[(element_index*OUT_WIDTH) + (OUT_WIDTH-1)]}},
            layer_out_flat[element_index*OUT_WIDTH +: OUT_WIDTH]
        };
    end

    always @* begin
        validate_error = 1'b0;
        if ((reader_descriptor_count != `QMAP_MLP_RESIDUAL_ADD_DESCRIPTOR_COUNT) ||
            (reader_descriptor_capacity != `QMAP_MLP_RESIDUAL_ADD_DESCRIPTOR_CAPACITY) ||
            (DESCRIPTOR_SLOTS != `QMAP_MLP_RESIDUAL_ADD_DESCRIPTOR_COUNT) ||
            (VECTOR_BYTES != `QMAP_MLP_RESIDUAL_ADD_OUTPUT_BYTES) ||
            (MAX_READ_BYTES != `QMAP_MLP_RESIDUAL_ADD_MAX_READ_BYTES) ||
            (desc_tensor_id[SLOT_METADATA] != `QMAP_TENSOR_ID_MLP_RESIDUAL_METADATA) ||
            (desc_tensor_id[SLOT_POST_ATTN] != `QMAP_TENSOR_ID_MLP_RESIDUAL_POST_ATTN) ||
            (desc_tensor_id[SLOT_DOWN] != `QMAP_TENSOR_ID_MLP_RESIDUAL_DOWN) ||
            (desc_tensor_id[SLOT_OUTPUT] != `QMAP_TENSOR_ID_MLP_RESIDUAL_OUTPUT) ||
            (desc_tensor_id[SLOT_EXPECTED] != `QMAP_TENSOR_ID_MLP_RESIDUAL_EXPECTED) ||
            (desc_role[SLOT_METADATA] != `QMAP_ROLE_METADATA) ||
            (desc_role[SLOT_POST_ATTN] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_DOWN] != `QMAP_ROLE_ACTIVATION) ||
            (desc_role[SLOT_OUTPUT] != `QMAP_ROLE_OUTPUT) ||
            (desc_role[SLOT_EXPECTED] != `QMAP_ROLE_EXPECTED) ||
            (desc_dtype[SLOT_METADATA] != `QMAP_DTYPE_U32) ||
            (desc_dtype[SLOT_POST_ATTN] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_DOWN] != `QMAP_DTYPE_I32_Q12_12) ||
            (desc_dtype[SLOT_OUTPUT] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_dtype[SLOT_EXPECTED] != `QMAP_DTYPE_I32_Q14_10) ||
            (desc_rank[SLOT_METADATA] != 32'd1) ||
            (desc_rank[SLOT_POST_ATTN] != 32'd1) ||
            (desc_rank[SLOT_DOWN] != 32'd1) ||
            (desc_rank[SLOT_OUTPUT] != 32'd1) ||
            (desc_rank[SLOT_EXPECTED] != 32'd1) ||
            (desc_element_bits[SLOT_METADATA] != 32'd32) ||
            (desc_element_bits[SLOT_POST_ATTN] != POST_ATTENTION_WIDTH) ||
            (desc_element_bits[SLOT_DOWN] != DOWN_WIDTH) ||
            (desc_element_bits[SLOT_OUTPUT] != OUT_WIDTH) ||
            (desc_element_bits[SLOT_EXPECTED] != OUT_WIDTH) ||
            (desc_group_size[SLOT_METADATA] != 32'd0) ||
            (desc_group_size[SLOT_POST_ATTN] != 32'd0) ||
            (desc_group_size[SLOT_DOWN] != 32'd0) ||
            (desc_group_size[SLOT_OUTPUT] != 32'd0) ||
            (desc_group_size[SLOT_EXPECTED] != 32'd0) ||
            (desc_scale_tensor_id[SLOT_METADATA] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_POST_ATTN] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_DOWN] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_OUTPUT] != `QMAP_NO_TENSOR_ID) ||
            (desc_scale_tensor_id[SLOT_EXPECTED] != `QMAP_NO_TENSOR_ID) ||
            (desc_dim0[SLOT_METADATA] != METADATA_WORDS) ||
            (desc_dim0[SLOT_POST_ATTN] != INPUT_SIZE) ||
            (desc_dim0[SLOT_DOWN] != INPUT_SIZE) ||
            (desc_dim0[SLOT_OUTPUT] != INPUT_SIZE) ||
            (desc_dim0[SLOT_EXPECTED] != INPUT_SIZE) ||
            (desc_dim1[SLOT_METADATA] != 32'd0) ||
            (desc_dim1[SLOT_POST_ATTN] != 32'd0) ||
            (desc_dim1[SLOT_DOWN] != 32'd0) ||
            (desc_dim1[SLOT_OUTPUT] != 32'd0) ||
            (desc_dim1[SLOT_EXPECTED] != 32'd0) ||
            (desc_nbytes[SLOT_METADATA] != (METADATA_WORDS * MEM_DATA_BYTES)) ||
            (desc_nbytes[SLOT_POST_ATTN] != `QMAP_MLP_RESIDUAL_ADD_POST_ATTN_BYTES) ||
            (desc_nbytes[SLOT_DOWN] != `QMAP_MLP_RESIDUAL_ADD_DOWN_BYTES) ||
            (desc_nbytes[SLOT_OUTPUT] != `QMAP_MLP_RESIDUAL_ADD_OUTPUT_BYTES) ||
            (desc_nbytes[SLOT_EXPECTED] != `QMAP_MLP_RESIDUAL_ADD_OUTPUT_BYTES) ||
            ((desc_flags[SLOT_METADATA] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_POST_ATTN] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_DOWN] & `QMAP_TENSOR_F_READ_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_OUTPUT] & `QMAP_TENSOR_F_WRITE_ONLY) == 32'd0) ||
            ((desc_flags[SLOT_EXPECTED] & `QMAP_TENSOR_F_DEBUG_ONLY) == 32'd0) ||
            (desc_aux0[SLOT_METADATA] != `QMAP_STAGE_ID_MLP_RESIDUAL_ADD) ||
            (desc_aux1[SLOT_METADATA] >= NUM_LAYERS) ||
            (desc_aux2[SLOT_METADATA] != INPUT_SIZE) ||
            (desc_aux0[SLOT_POST_ATTN] != `QMAP_STAGE_ID_POST_ATTN_RESIDUAL_NORM) ||
            (desc_aux1[SLOT_POST_ATTN] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux2[SLOT_POST_ATTN] != INPUT_SIZE) ||
            (desc_aux0[SLOT_DOWN] != `QMAP_MATRIX_ID_DOWN_PROJ) ||
            (desc_aux1[SLOT_DOWN] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux2[SLOT_DOWN] != INPUT_SIZE) ||
            (desc_aux0[SLOT_OUTPUT] != `QMAP_STAGE_ID_MLP_RESIDUAL_ADD) ||
            (desc_aux1[SLOT_OUTPUT] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux2[SLOT_OUTPUT] != INPUT_SIZE) ||
            (desc_aux0[SLOT_EXPECTED] != `QMAP_STAGE_ID_MLP_RESIDUAL_ADD) ||
            (desc_aux1[SLOT_EXPECTED] != desc_aux1[SLOT_METADATA]) ||
            (desc_aux2[SLOT_EXPECTED] != INPUT_SIZE) ||
            (o_effective_output_base_addr == '0) ||
            (o_effective_output_base_addr[1:0] != 2'b00)) begin
            validate_error = 1'b1;
        end
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (i_rst_n == 1'b0) begin
            state <= S_IDLE;
            active_read_slot <= R_POST_ATTN;
            read_chunk_index <= 32'd0;
            read_word_index <= 32'd0;
            write_word_index <= 32'd0;
            post_attn_hidden_flat <= '0;
            down_out_flat <= '0;
            output_base_override_valid_reg <= 1'b0;
            output_base_override_addr_reg <= '0;
            o_done <= 1'b0;
            o_error <= 1'b0;
            o_saturation <= 1'b0;
            o_output_count <= 32'd0;
            o_stage_cycle_count <= 32'd0;
            o_output_write_word_count <= 32'd0;
            o_mem_read_burst_count <= 32'd0;
            o_mem_read_word_count <= 32'd0;
            o_mem_write_req_count <= 32'd0;
            o_mem_write_word_count <= 32'd0;
        end
        else begin
            o_done <= 1'b0;

            if (o_mem_rd_req_valid && i_mem_rd_req_ready) begin
                o_mem_read_burst_count <= o_mem_read_burst_count + 1'b1;
            end
            if (i_mem_rd_rsp_valid && o_mem_rd_rsp_ready) begin
                o_mem_read_word_count <= o_mem_read_word_count + 1'b1;
            end
            if (o_mem_wr_req_valid && i_mem_wr_req_ready) begin
                o_mem_write_req_count <= o_mem_write_req_count + 1'b1;
            end
            if (o_mem_wr_data_valid && i_mem_wr_data_ready) begin
                o_mem_write_word_count <= o_mem_write_word_count + 1'b1;
                o_output_write_word_count <= o_output_write_word_count + 1'b1;
            end

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        output_base_override_valid_reg <=
                            (i_output_base_override_valid === 1'b1);
                        output_base_override_addr_reg <= i_output_base_override_addr;
                        active_read_slot <= R_POST_ATTN;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        write_word_index <= 32'd0;
                        o_error <= 1'b0;
                        o_saturation <= 1'b0;
                        o_output_count <= 32'd0;
                        o_stage_cycle_count <= 32'd0;
                        o_output_write_word_count <= 32'd0;
                        o_mem_read_burst_count <= 32'd0;
                        o_mem_read_word_count <= 32'd0;
                        o_mem_write_req_count <= 32'd0;
                        o_mem_write_word_count <= 32'd0;
                        state <= S_READER_START;
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
                        active_read_slot <= R_POST_ATTN;
                        read_chunk_index <= 32'd0;
                        read_word_index <= 32'd0;
                        state <= S_READ_REQ;
                    end
                end

                S_READ_REQ: begin
                    if (i_mem_rd_req_ready) begin
                        read_word_index <= 32'd0;
                        state <= S_READ_DATA;
                    end
                end

                S_READ_DATA: begin
                    if (i_mem_rd_rsp_valid) begin
                        if (read_protocol_error) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            if (active_read_slot == R_POST_ATTN) begin
                                post_attn_hidden_flat[element_index*POST_ATTENTION_WIDTH +: POST_ATTENTION_WIDTH] <=
                                    i_mem_rd_rsp_data[POST_ATTENTION_WIDTH-1 : 0];
                            end
                            else begin
                                down_out_flat[element_index*DOWN_WIDTH +: DOWN_WIDTH] <=
                                    i_mem_rd_rsp_data[DOWN_WIDTH-1 : 0];
                            end

                            if (i_mem_rd_rsp_last) begin
                                if (read_chunk_index == (CHUNK_COUNT - 1)) begin
                                    read_chunk_index <= 32'd0;
                                    read_word_index <= 32'd0;
                                    if (active_read_slot == R_POST_ATTN) begin
                                        active_read_slot <= R_DOWN;
                                        state <= S_READ_REQ;
                                    end
                                    else begin
                                        state <= S_STAGE_START;
                                    end
                                end
                                else begin
                                    read_chunk_index <= read_chunk_index + 1'b1;
                                    read_word_index <= 32'd0;
                                    state <= S_READ_REQ;
                                end
                            end
                            else begin
                                read_word_index <= read_word_index + 1'b1;
                            end
                        end
                    end
                end

                S_STAGE_START: begin
                    state <= S_STAGE_WAIT;
                end

                S_STAGE_WAIT: begin
                    if (stage_done) begin
                        o_saturation <= stage_saturation;
                        o_output_count <= stage_output_count;
                        o_stage_cycle_count <= stage_cycle_count;
                        if (stage_error || (stage_output_count != INPUT_SIZE)) begin
                            o_error <= 1'b1;
                            state <= S_DONE;
                        end
                        else begin
                            write_word_index <= 32'd0;
                            state <= S_WRITE_REQ;
                        end
                    end
                end

                S_WRITE_REQ: begin
                    if (i_mem_wr_req_ready) begin
                        write_word_index <= 32'd0;
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    if (i_mem_wr_data_ready) begin
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
