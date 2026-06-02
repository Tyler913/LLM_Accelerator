`default_nettype none

// Byte-address generator for the per-layer K/V cache region.
//
// Logical layout:
//
//   cache[layer][kv_kind][head][position][dim]
//
// Where:
//
//   kv_kind = 0 for K
//   kv_kind = 1 for V
//
// Address formula:
//
//   offset = layer    * LAYER_STRIDE_BYTES
//          + kv_kind  * KIND_STRIDE_BYTES
//          + head     * HEAD_STRIDE_BYTES
//          + position * POSITION_STRIDE_BYTES
//          + dim      * ELEMENT_BYTES
//
// Default planning uses 28 layers, 8 KV heads, head_dim 128, context 256, and 4 bytes per cache element. The current fixed-point path uses signed 24-bit Q12.12 K/V values padded to 32-bit DDR words.
module kv_cache_addr_gen #(
    parameter int ADDR_WIDTH       = 64,
    parameter int NUM_LAYERS       = 28,
    parameter int NUM_KV_HEADS     = 8,
    parameter int HEAD_DIM         = 128,
    parameter int MAX_CONTEXT      = 256,
    parameter int ELEMENT_BYTES    = 4,
    parameter int LAYER_INDEX_W    = (NUM_LAYERS <= 1) ? 1 : $clog2(NUM_LAYERS),
    parameter int HEAD_INDEX_W     = (NUM_KV_HEADS <= 1) ? 1 : $clog2(NUM_KV_HEADS),
    parameter int POSITION_INDEX_W = (MAX_CONTEXT <= 1) ? 1 : $clog2(MAX_CONTEXT),
    parameter int DIM_INDEX_W      = (HEAD_DIM <= 1) ? 1 : $clog2(HEAD_DIM)
)
(
    input  logic [ADDR_WIDTH-1 : 0]         i_base_addr,
    input  logic [LAYER_INDEX_W-1 : 0]      i_layer_id,
    input  logic                            i_kv_kind,
    input  logic [HEAD_INDEX_W-1 : 0]       i_head_id,
    input  logic [POSITION_INDEX_W-1 : 0]   i_position,
    input  logic [DIM_INDEX_W-1 : 0]        i_dim,

    output logic                            o_valid,
    output logic [ADDR_WIDTH-1 : 0]         o_offset_bytes,
    output logic [ADDR_WIDTH-1 : 0]         o_byte_addr
);

    localparam logic [ADDR_WIDTH-1 : 0] POSITION_STRIDE_BYTES = HEAD_DIM * ELEMENT_BYTES;
    localparam logic [ADDR_WIDTH-1 : 0] HEAD_STRIDE_BYTES     = MAX_CONTEXT * POSITION_STRIDE_BYTES;
    localparam logic [ADDR_WIDTH-1 : 0] KIND_STRIDE_BYTES     = NUM_KV_HEADS * HEAD_STRIDE_BYTES;
    localparam logic [ADDR_WIDTH-1 : 0] LAYER_STRIDE_BYTES    = 2 * KIND_STRIDE_BYTES;

    logic [ADDR_WIDTH-1 : 0] layer_id_ext;
    logic [ADDR_WIDTH-1 : 0] kv_kind_ext;
    logic [ADDR_WIDTH-1 : 0] head_id_ext;
    logic [ADDR_WIDTH-1 : 0] position_ext;
    logic [ADDR_WIDTH-1 : 0] dim_ext;

    logic [ADDR_WIDTH-1 : 0] layer_offset_bytes;
    logic [ADDR_WIDTH-1 : 0] kind_offset_bytes;
    logic [ADDR_WIDTH-1 : 0] head_offset_bytes;
    logic [ADDR_WIDTH-1 : 0] position_offset_bytes;
    logic [ADDR_WIDTH-1 : 0] dim_offset_bytes;

    always @* begin
        layer_id_ext                         = 'd0;
        kv_kind_ext                          = 'd0;
        head_id_ext                          = 'd0;
        position_ext                         = 'd0;
        dim_ext                              = 'd0;
        layer_id_ext[LAYER_INDEX_W-1 : 0]    = i_layer_id;
        kv_kind_ext[0]                       = i_kv_kind;
        head_id_ext[HEAD_INDEX_W-1 : 0]      = i_head_id;
        position_ext[POSITION_INDEX_W-1 : 0] = i_position;
        dim_ext[DIM_INDEX_W-1 : 0]           = i_dim;
    end

    assign layer_offset_bytes    = layer_id_ext * LAYER_STRIDE_BYTES;
    assign kind_offset_bytes     = kv_kind_ext * KIND_STRIDE_BYTES;
    assign head_offset_bytes     = head_id_ext * HEAD_STRIDE_BYTES;
    assign position_offset_bytes = position_ext * POSITION_STRIDE_BYTES;
    assign dim_offset_bytes      = dim_ext * ELEMENT_BYTES;

    assign o_offset_bytes        = layer_offset_bytes + kind_offset_bytes + head_offset_bytes + position_offset_bytes + dim_offset_bytes;
    assign o_byte_addr           = i_base_addr + o_offset_bytes;
    assign o_valid               = (i_layer_id < NUM_LAYERS) && (i_head_id < NUM_KV_HEADS) && (i_position < MAX_CONTEXT) && (i_dim < HEAD_DIM);

endmodule

`default_nettype wire
