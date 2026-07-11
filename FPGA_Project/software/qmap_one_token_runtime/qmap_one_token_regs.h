#ifndef QMAP_ONE_TOKEN_REGS_H
#define QMAP_ONE_TOKEN_REGS_H

#include <stdint.h>

/*
 * Software-visible register contract for qmap_one_token_control_regs.sv.
 *
 * The RTL decodes byte address bits [11:2] as a 32-bit word index. Keep these
 * offsets byte-addressed for direct AXI4-Lite use from PS software.
 */

#define QOT_MAX_LAYERS   28u
#define QOT_MAX_CONTEXT  256u
#define QOT_TABLE_COUNT  10u

#define QOT_REG_CTRL                 0x000u
#define QOT_REG_STATUS               0x004u
#define QOT_REG_LAYER_START          0x008u
#define QOT_REG_LAYER_COUNT          0x00Cu
#define QOT_REG_POSITION             0x010u
#define QOT_REG_INPUT_TOKEN          0x014u
#define QOT_REG_INPUT_HIDDEN_LO      0x020u
#define QOT_REG_INPUT_HIDDEN_HI      0x024u
#define QOT_REG_OUTPUT_HIDDEN_LO     0x028u
#define QOT_REG_OUTPUT_HIDDEN_HI     0x02Cu
#define QOT_REG_KV_CACHE_LO          0x030u
#define QOT_REG_KV_CACHE_HI          0x034u
#define QOT_REG_FINAL_TAIL_QMAP_LO   0x038u
#define QOT_REG_FINAL_TAIL_QMAP_HI   0x03Cu
#define QOT_REG_FINAL_OVERRIDE_LO    0x040u
#define QOT_REG_FINAL_OVERRIDE_HI    0x044u
#define QOT_REG_FINAL_OVERRIDE_CTRL  0x048u
#define QOT_REG_TABLE_SELECT         0x050u
#define QOT_REG_TABLE_DATA_LO        0x054u
#define QOT_REG_TABLE_DATA_HI        0x058u
#define QOT_REG_TABLE_COMMIT         0x05Cu
#define QOT_REG_OUT_TOKEN            0x060u
#define QOT_REG_OUT_SCORE_LO         0x064u
#define QOT_REG_OUT_SCORE_HI         0x068u
#define QOT_REG_LAYERS               0x06Cu
#define QOT_REG_LAYER_DONE_MASK      0x070u
#define QOT_REG_LAYER_ERROR_MASK     0x074u
#define QOT_REG_LAST_OUTPUT_LO       0x078u
#define QOT_REG_LAST_OUTPUT_HI       0x07Cu
#define QOT_REG_TAIL_HIDDEN_LO       0x080u
#define QOT_REG_TAIL_HIDDEN_HI       0x084u
#define QOT_REG_TAIL_TILES_STARTED   0x088u
#define QOT_REG_TAIL_TILES_COMPLETED 0x08Cu
#define QOT_REG_MEM_RD_REQS          0x090u
#define QOT_REG_MEM_RD_WORDS         0x094u
#define QOT_REG_MEM_WR_REQS          0x098u
#define QOT_REG_MEM_WR_WORDS         0x09Cu

#define QOT_CTRL_START_MASK          0x00000001u
#define QOT_CTRL_CLEAR_STICKY_MASK   0x00000002u

#define QOT_STATUS_BUSY_MASK         0x00000001u
#define QOT_STATUS_DONE_STICKY_MASK  0x00000002u
#define QOT_STATUS_ERROR_STICKY_MASK 0x00000004u
#define QOT_STATUS_COMMAND_ERR_MASK  0x00000008u
#define QOT_STATUS_TOP_ERROR_MASK    0x00000010u
#define QOT_STATUS_TAIL_ERROR_MASK   0x00000020u
#define QOT_STATUS_TAIL_SAT_MASK     0x00000040u
#define QOT_STATUS_STATE_SHIFT       8u
#define QOT_STATUS_STATE_MASK        0x0000FF00u
#define QOT_STATUS_PHASE_SHIFT       16u
#define QOT_STATUS_PHASE_MASK        0x00FF0000u

#define QOT_TABLE_QKV                0u
#define QOT_TABLE_INPUT_NORM         1u
#define QOT_TABLE_ATTN_FRONTEND      2u
#define QOT_TABLE_ATTN_SCORE_VALUE   3u
#define QOT_TABLE_O_PROJ             4u
#define QOT_TABLE_POST_ATTN_NORM     5u
#define QOT_TABLE_MLP_GATE_UP        6u
#define QOT_TABLE_MLP_SILU_MUL       7u
#define QOT_TABLE_MLP_DOWN           8u
#define QOT_TABLE_MLP_RESIDUAL_ADD   9u

#define QOT_TABLE_SELECT_VALUE(table_id, layer_id) \
    ((((uint32_t)(layer_id) & 0xFFu) << 8) | ((uint32_t)(table_id) & 0xFFu))
#define QOT_TABLE_COMMIT_APPLY_MASK  0x00000001u

#define QOT_LAYERS_STARTED(value)    ((uint32_t)(value) & 0x0000FFFFu)
#define QOT_LAYERS_COMPLETED(value)  (((uint32_t)(value) >> 16) & 0x0000FFFFu)

#define QOT_STATUS_HAS_ANY_ERROR(value) \
    (((uint32_t)(value) & (QOT_STATUS_ERROR_STICKY_MASK | \
                           QOT_STATUS_COMMAND_ERR_MASK | \
                           QOT_STATUS_TOP_ERROR_MASK | \
                           QOT_STATUS_TAIL_ERROR_MASK | \
                           QOT_STATUS_TAIL_SAT_MASK)) != 0u)

#endif /* QMAP_ONE_TOKEN_REGS_H */
