`ifndef QMAP_DEFS_SVH
`define QMAP_DEFS_SVH

// QMAP v1 constants used by the first PL-side descriptor reader.
// The format is documented in Source/QMAP_FORMAT.md.

`define QMAP_MAGIC                         32'h5041_4D51
`define QMAP_VERSION                       32'd1

`define QMAP_HEADER_BYTES                  32'd256
`define QMAP_HEADER_FETCH_BYTES            16'd64
`define QMAP_DESCRIPTOR_BYTES              32'd128
`define QMAP_DESCRIPTOR_CAPACITY_DOT64     32'd8
`define QMAP_DESCRIPTOR_COUNT_DOT64        32'd4

`define QMAP_DOT64_BASE_ADDR               64'h0000_0004_1B10_0000
`define QMAP_DOT64_DESCRIPTOR_TABLE_ADDR   64'h0000_0004_1B10_0100
`define QMAP_DOT64_PAYLOAD_BASE_ADDR       64'h0000_0004_1B10_0500
`define QMAP_DOT64_IMAGE_BYTES             64'h0000_0000_0000_0600
`define QMAP_DOT64_ACTIVATION_ADDR         64'h0000_0004_1B10_0500
`define QMAP_DOT64_WEIGHT_ADDR             64'h0000_0004_1B10_0580
`define QMAP_DOT64_SCALE_ADDR              64'h0000_0004_1B10_05A0
`define QMAP_DOT64_EXPECTED_ADDR           64'h0000_0004_1B10_05C0
`define QMAP_DOT64_ACTIVATION_BYTES        64'd128
`define QMAP_DOT64_WEIGHT_BYTES            64'd32
`define QMAP_DOT64_SCALE_BYTES             64'd2
`define QMAP_DOT64_EXPECTED_BYTES          64'd16

`define QMAP_ROW1024_BASE_ADDR             64'h0000_0004_1B20_0000
`define QMAP_ROW1024_DESCRIPTOR_TABLE_ADDR 64'h0000_0004_1B20_0100
`define QMAP_ROW1024_PAYLOAD_BASE_ADDR     64'h0000_0004_1B20_0500
`define QMAP_ROW1024_IMAGE_BYTES           64'h0000_0000_0000_1000
`define QMAP_ROW1024_ACTIVATION_ADDR       64'h0000_0004_1B20_0500
`define QMAP_ROW1024_WEIGHT_ADDR           64'h0000_0004_1B20_0D00
`define QMAP_ROW1024_SCALE_ADDR            64'h0000_0004_1B20_0F00
`define QMAP_ROW1024_EXPECTED_ADDR         64'h0000_0004_1B20_0F40
`define QMAP_ROW1024_ACTIVATION_BYTES      64'd2048
`define QMAP_ROW1024_WEIGHT_BYTES          64'd512
`define QMAP_ROW1024_SCALE_BYTES           64'd32
`define QMAP_ROW1024_EXPECTED_BYTES        64'd8
`define QMAP_ROW1024_MAX_FETCH_BYTES       16'd1024

`define QMAP_QKV_BASE_ADDR                 64'h0000_0004_0008_0000
`define QMAP_QKV_DESCRIPTOR_TABLE_ADDR     64'h0000_0004_0008_0100
`define QMAP_QKV_PAYLOAD_BASE_ADDR         64'h0000_0004_0008_1100
`define QMAP_QKV_DESCRIPTOR_CAPACITY       32'd32
`define QMAP_QKV_DESCRIPTOR_COUNT          32'd12
`define QMAP_QKV_ACTIVATION_BYTES          64'd4096
`define QMAP_QKV_WEIGHT_ROW_BYTES          64'd512
`define QMAP_QKV_SCALE_ROW_BYTES           64'd32
`define QMAP_QKV_OUTPUT_WORD_BYTES         64'd4
`define QMAP_QKV_MAX_READ_BYTES            16'd1024

`define QMAP_ROLE_ACTIVATION               32'd1
`define QMAP_ROLE_Q4_WEIGHT                32'd2
`define QMAP_ROLE_Q4_SCALE                 32'd3
`define QMAP_ROLE_OUTPUT                   32'd4
`define QMAP_ROLE_EXPECTED                 32'd5
`define QMAP_ROLE_METADATA                 32'd6
`define QMAP_ROLE_KV_CACHE                 32'd7
`define QMAP_ROLE_ROPE_TABLE               32'd8
`define QMAP_ROLE_PARAMETER                32'd9

`define QMAP_DTYPE_U32                     32'd5
`define QMAP_DTYPE_I64                     32'd8
`define QMAP_DTYPE_PACKED_Q4_S4            32'd16
`define QMAP_DTYPE_U16_Q2_14               32'd17
`define QMAP_DTYPE_I16_Q4_12               32'd18
`define QMAP_DTYPE_I32_Q12_12              32'd19
`define QMAP_DTYPE_I32_Q14_10              32'd20
`define QMAP_DTYPE_U16_Q8_8                32'd21

`define QMAP_TENSOR_F_ROW_MAJOR            32'h0000_0001
`define QMAP_TENSOR_F_PACKED_Q4_LOW_EVEN   32'h0000_0002
`define QMAP_TENSOR_F_READ_ONLY            32'h0000_0004
`define QMAP_TENSOR_F_WRITE_ONLY           32'h0000_0008
`define QMAP_TENSOR_F_DEBUG_ONLY           32'h0000_0010

`define QMAP_TENSOR_ID_ACTIVATION          32'd1
`define QMAP_TENSOR_ID_WEIGHT              32'd2
`define QMAP_TENSOR_ID_SCALE               32'd3
`define QMAP_TENSOR_ID_EXPECTED            32'd4
`define QMAP_TENSOR_ID_QKV_METADATA        32'd1
`define QMAP_TENSOR_ID_QKV_ACTIVATION      32'd2
`define QMAP_TENSOR_ID_Q_WEIGHT            32'd3
`define QMAP_TENSOR_ID_Q_SCALE             32'd4
`define QMAP_TENSOR_ID_K_WEIGHT            32'd5
`define QMAP_TENSOR_ID_K_SCALE             32'd6
`define QMAP_TENSOR_ID_V_WEIGHT            32'd7
`define QMAP_TENSOR_ID_V_SCALE             32'd8
`define QMAP_TENSOR_ID_Q_OUT               32'd9
`define QMAP_TENSOR_ID_K_OUT               32'd10
`define QMAP_TENSOR_ID_V_OUT               32'd11
`define QMAP_TENSOR_ID_QKV_EXPECTED        32'd12
`define QMAP_NO_TENSOR_ID                  32'hFFFF_FFFF

`define QMAP_MATRIX_ID_Q_PROJ              32'd1
`define QMAP_MATRIX_ID_K_PROJ              32'd2
`define QMAP_MATRIX_ID_V_PROJ              32'd3
`define QMAP_MATRIX_ID_O_PROJ              32'd4
`define QMAP_MATRIX_ID_GATE_PROJ           32'd5
`define QMAP_MATRIX_ID_UP_PROJ             32'd6
`define QMAP_MATRIX_ID_DOWN_PROJ           32'd7
`define QMAP_MATRIX_ID_EMBED_LM_HEAD       32'd8
`define QMAP_Q4_GROUP_SIZE                 32'd64

`endif
