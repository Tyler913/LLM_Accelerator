#ifndef QMAP_ONE_TOKEN_RUNTIME_H
#define QMAP_ONE_TOKEN_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#include "qmap_one_token_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef QOT_MMIO_BARRIER
#define QOT_MMIO_BARRIER() do { } while (0)
#endif

enum {
    QOT_OK = 0,
    QOT_ERR_BUSY = -1,
    QOT_ERR_BAD_LAYER = -2,
    QOT_ERR_BAD_TABLE = -3,
    QOT_ERR_BAD_ADDR = -4,
    QOT_ERR_TIMEOUT = -5,
    QOT_ERR_STATUS = -6,
    QOT_ERR_NULL = -7,
    QOT_ERR_BAD_TOKEN = -8,
    QOT_ERR_CAPACITY = -9,
    QOT_ERR_CONTEXT = -10
};

typedef struct qot_layer_qmap_bases {
    uint64_t qkv;
    uint64_t input_norm;       /* Optional. Leave zero for QKV-first layers. */
    uint64_t attn_frontend;
    uint64_t attn_score_value;
    uint64_t o_proj;
    uint64_t post_attn_norm;
    uint64_t mlp_gate_up;
    uint64_t mlp_silu_mul;
    uint64_t mlp_down;
    uint64_t mlp_residual_add;
} qot_layer_qmap_bases_t;

typedef struct qot_run_config {
    uint32_t layer_start;
    uint32_t layer_count;
    uint32_t position;
    uint32_t runtime_context_enable;
    uint32_t input_token_id;
    uint32_t embedding_enable;
    uint64_t embedding_weight_base;
    uint64_t embedding_scale_base;
    uint64_t input_hidden_base;
    uint64_t output_hidden_base;
    uint64_t kv_cache_base;
    uint64_t final_tail_qmap_base;
    uint32_t final_hidden_override_valid;
    uint64_t final_hidden_override_base;

    /* Indexed by absolute layer id. Entries outside the selected layer range
     * are ignored. QOT_MAX_LAYERS entries are expected for full-model code. */
    const qot_layer_qmap_bases_t *layer_qmap_bases;
    uint32_t layer_qmap_base_count;
} qot_run_config_t;

typedef struct qot_result {
    uint32_t status;
    uint32_t token_id;
    int64_t score_q26;
    uint32_t layers_started;
    uint32_t layers_completed;
    uint32_t layer_done_mask;
    uint32_t layer_error_mask;
    uint64_t last_output_base;
    uint64_t tail_hidden_base;
    uint32_t tail_tiles_started;
    uint32_t tail_tiles_completed;
    uint32_t mem_read_reqs;
    uint32_t mem_read_words;
    uint32_t mem_write_reqs;
    uint32_t mem_write_words;
} qot_result_t;

static inline void qot_write32(uintptr_t base, uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)(base + (uintptr_t)offset) = value;
    QOT_MMIO_BARRIER();
}

static inline uint32_t qot_read32(uintptr_t base, uint32_t offset)
{
    uint32_t value = *(volatile const uint32_t *)(base + (uintptr_t)offset);
    QOT_MMIO_BARRIER();
    return value;
}

static inline void qot_write64(uintptr_t base, uint32_t lo_offset,
                               uint32_t hi_offset, uint64_t value)
{
    qot_write32(base, lo_offset, (uint32_t)value);
    qot_write32(base, hi_offset, (uint32_t)(value >> 32));
}

static inline uint64_t qot_read64(uintptr_t base, uint32_t lo_offset,
                                  uint32_t hi_offset)
{
    uint32_t lo = qot_read32(base, lo_offset);
    uint32_t hi = qot_read32(base, hi_offset);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

static inline int64_t qot_read_score_q26(uintptr_t base)
{
    uint64_t raw = qot_read64(base, QOT_REG_OUT_SCORE_LO, QOT_REG_OUT_SCORE_HI);
    if ((raw & (1ull << 55)) != 0ull) {
        raw |= 0xFF00000000000000ull;
    }
    return (int64_t)raw;
}

static inline uint32_t qot_status_state(uint32_t status)
{
    return (status & QOT_STATUS_STATE_MASK) >> QOT_STATUS_STATE_SHIFT;
}

static inline uint32_t qot_status_phase(uint32_t status)
{
    return (status & QOT_STATUS_PHASE_MASK) >> QOT_STATUS_PHASE_SHIFT;
}

static inline void qot_clear_sticky(uintptr_t base)
{
    qot_write32(base, QOT_REG_CTRL, QOT_CTRL_CLEAR_STICKY_MASK);
}

static inline int qot_is_busy(uintptr_t base)
{
    return (qot_read32(base, QOT_REG_STATUS) & QOT_STATUS_BUSY_MASK) != 0u;
}

static inline int qot_check_layer_id(uint32_t layer_id)
{
    return (layer_id < QOT_MAX_LAYERS) ? QOT_OK : QOT_ERR_BAD_LAYER;
}

static inline int qot_check_table_id(uint32_t table_id)
{
    return (table_id < QOT_TABLE_COUNT) ? QOT_OK : QOT_ERR_BAD_TABLE;
}

static inline int qot_addr_is_word_aligned(uint64_t addr)
{
    return addr != 0ull && (addr & 0x3ull) == 0ull;
}

static inline int qot_commit_table(uintptr_t base, uint32_t layer_id,
                                   uint32_t table_id, uint64_t qmap_base)
{
    int rc;

    rc = qot_check_layer_id(layer_id);
    if (rc != QOT_OK) return rc;
    rc = qot_check_table_id(table_id);
    if (rc != QOT_OK) return rc;
    if (!qot_addr_is_word_aligned(qmap_base)) return QOT_ERR_BAD_ADDR;
    if (qot_is_busy(base)) return QOT_ERR_BUSY;

    qot_write32(base, QOT_REG_TABLE_SELECT,
                QOT_TABLE_SELECT_VALUE(table_id, layer_id));
    qot_write64(base, QOT_REG_TABLE_DATA_LO, QOT_REG_TABLE_DATA_HI, qmap_base);
    qot_write32(base, QOT_REG_TABLE_COMMIT, QOT_TABLE_COMMIT_APPLY_MASK);

    return (qot_read32(base, QOT_REG_STATUS) & QOT_STATUS_COMMAND_ERR_MASK) ?
        QOT_ERR_STATUS : QOT_OK;
}

static inline int qot_commit_nonzero_table(uintptr_t base, uint32_t layer_id,
                                           uint32_t table_id, uint64_t qmap_base)
{
    if (qmap_base == 0ull) return QOT_OK;
    return qot_commit_table(base, layer_id, table_id, qmap_base);
}

static inline int qot_layer_tables_have_required(
    const qot_layer_qmap_bases_t *tables)
{
    if (tables == NULL) return QOT_ERR_NULL;
    if (tables->qkv == 0ull ||
        tables->attn_frontend == 0ull ||
        tables->attn_score_value == 0ull ||
        tables->o_proj == 0ull ||
        tables->post_attn_norm == 0ull ||
        tables->mlp_gate_up == 0ull ||
        tables->mlp_silu_mul == 0ull ||
        tables->mlp_down == 0ull ||
        tables->mlp_residual_add == 0ull) {
        return QOT_ERR_BAD_ADDR;
    }
    if (!qot_addr_is_word_aligned(tables->qkv) ||
        (tables->input_norm != 0ull &&
         !qot_addr_is_word_aligned(tables->input_norm)) ||
        !qot_addr_is_word_aligned(tables->attn_frontend) ||
        !qot_addr_is_word_aligned(tables->attn_score_value) ||
        !qot_addr_is_word_aligned(tables->o_proj) ||
        !qot_addr_is_word_aligned(tables->post_attn_norm) ||
        !qot_addr_is_word_aligned(tables->mlp_gate_up) ||
        !qot_addr_is_word_aligned(tables->mlp_silu_mul) ||
        !qot_addr_is_word_aligned(tables->mlp_down) ||
        !qot_addr_is_word_aligned(tables->mlp_residual_add)) {
        return QOT_ERR_BAD_ADDR;
    }
    return QOT_OK;
}

static inline int qot_commit_layer_tables(uintptr_t base, uint32_t layer_id,
                                          const qot_layer_qmap_bases_t *tables)
{
    int rc = qot_layer_tables_have_required(tables);
    if (rc != QOT_OK) return rc;

    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_QKV,
                               tables->qkv)) != QOT_OK) return rc;
    if ((rc = qot_commit_nonzero_table(base, layer_id, QOT_TABLE_INPUT_NORM,
                                       tables->input_norm)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_ATTN_FRONTEND,
                               tables->attn_frontend)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_ATTN_SCORE_VALUE,
                               tables->attn_score_value)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_O_PROJ,
                               tables->o_proj)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_POST_ATTN_NORM,
                               tables->post_attn_norm)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_MLP_GATE_UP,
                               tables->mlp_gate_up)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_MLP_SILU_MUL,
                               tables->mlp_silu_mul)) != QOT_OK) return rc;
    if ((rc = qot_commit_table(base, layer_id, QOT_TABLE_MLP_DOWN,
                               tables->mlp_down)) != QOT_OK) return rc;
    return qot_commit_table(base, layer_id, QOT_TABLE_MLP_RESIDUAL_ADD,
                            tables->mlp_residual_add);
}

static inline int qot_configure_run(uintptr_t base, const qot_run_config_t *cfg)
{
    uint32_t layer;
    uint32_t end_layer;
    int rc;

    if (cfg == NULL) return QOT_ERR_NULL;
    if (cfg->layer_count == 0u || cfg->layer_start >= QOT_MAX_LAYERS) {
        return QOT_ERR_BAD_LAYER;
    }
    if (cfg->position >= QOT_MAX_CONTEXT) return QOT_ERR_CONTEXT;
    end_layer = cfg->layer_start + cfg->layer_count;
    if (end_layer > QOT_MAX_LAYERS || end_layer < cfg->layer_start) {
        return QOT_ERR_BAD_LAYER;
    }
    if (cfg->layer_qmap_bases == NULL || cfg->layer_qmap_base_count < end_layer) {
        return QOT_ERR_NULL;
    }
    if (!qot_addr_is_word_aligned(cfg->input_hidden_base) ||
        !qot_addr_is_word_aligned(cfg->output_hidden_base) ||
        !qot_addr_is_word_aligned(cfg->kv_cache_base) ||
        !qot_addr_is_word_aligned(cfg->final_tail_qmap_base) ||
        (cfg->final_hidden_override_valid != 0u &&
         !qot_addr_is_word_aligned(cfg->final_hidden_override_base))) {
        return QOT_ERR_BAD_ADDR;
    }
    if (cfg->runtime_context_enable != 0u &&
        cfg->input_hidden_base == cfg->output_hidden_base) {
        return QOT_ERR_BAD_ADDR;
    }
    if (cfg->embedding_enable != 0u) {
        if (cfg->input_token_id >= QOT_VOCAB_SIZE) return QOT_ERR_BAD_TOKEN;
        if (!qot_addr_is_word_aligned(cfg->embedding_weight_base) ||
            !qot_addr_is_word_aligned(cfg->embedding_scale_base)) {
            return QOT_ERR_BAD_ADDR;
        }
    }
    if (qot_is_busy(base)) return QOT_ERR_BUSY;

    qot_clear_sticky(base);
    qot_write32(base, QOT_REG_LAYER_START, cfg->layer_start);
    qot_write32(base, QOT_REG_LAYER_COUNT, cfg->layer_count);
    qot_write32(base, QOT_REG_POSITION, cfg->position);
    qot_write32(base, QOT_REG_RUNTIME_CTRL,
                cfg->runtime_context_enable ?
                QOT_RUNTIME_CONTEXT_ENABLE_MASK : 0u);
    qot_write32(base, QOT_REG_INPUT_TOKEN, cfg->input_token_id);
    qot_write32(base, QOT_REG_EMBEDDING_CTRL,
                cfg->embedding_enable ? QOT_EMBEDDING_ENABLE_MASK : 0u);
    qot_write64(base, QOT_REG_EMBED_WEIGHT_LO, QOT_REG_EMBED_WEIGHT_HI,
                cfg->embedding_weight_base);
    qot_write64(base, QOT_REG_EMBED_SCALE_LO, QOT_REG_EMBED_SCALE_HI,
                cfg->embedding_scale_base);
    qot_write64(base, QOT_REG_INPUT_HIDDEN_LO, QOT_REG_INPUT_HIDDEN_HI,
                cfg->input_hidden_base);
    qot_write64(base, QOT_REG_OUTPUT_HIDDEN_LO, QOT_REG_OUTPUT_HIDDEN_HI,
                cfg->output_hidden_base);
    qot_write64(base, QOT_REG_KV_CACHE_LO, QOT_REG_KV_CACHE_HI,
                cfg->kv_cache_base);
    qot_write64(base, QOT_REG_FINAL_TAIL_QMAP_LO, QOT_REG_FINAL_TAIL_QMAP_HI,
                cfg->final_tail_qmap_base);
    qot_write64(base, QOT_REG_FINAL_OVERRIDE_LO, QOT_REG_FINAL_OVERRIDE_HI,
                cfg->final_hidden_override_base);
    qot_write32(base, QOT_REG_FINAL_OVERRIDE_CTRL,
                cfg->final_hidden_override_valid ? 1u : 0u);

    for (layer = cfg->layer_start; layer < end_layer; ++layer) {
        if (cfg->runtime_context_enable != 0u &&
            cfg->layer_qmap_bases[layer].input_norm == 0ull) {
            return QOT_ERR_BAD_ADDR;
        }
        rc = qot_commit_layer_tables(base, layer, &cfg->layer_qmap_bases[layer]);
        if (rc != QOT_OK) return rc;
    }

    return QOT_OK;
}

static inline int qot_start(uintptr_t base)
{
    if (qot_is_busy(base)) return QOT_ERR_BUSY;
    qot_write32(base, QOT_REG_CTRL, QOT_CTRL_START_MASK);
    return QOT_OK;
}

static inline int qot_poll_done(uintptr_t base, uint32_t max_polls,
                                uint32_t *last_status)
{
    uint32_t polls = 0u;

    for (;;) {
        uint32_t status = qot_read32(base, QOT_REG_STATUS);
        if (last_status != NULL) *last_status = status;
        if (QOT_STATUS_HAS_ANY_ERROR(status)) return QOT_ERR_STATUS;
        if ((status & QOT_STATUS_DONE_STICKY_MASK) != 0u) return QOT_OK;
        if (max_polls != 0u && ++polls >= max_polls) return QOT_ERR_TIMEOUT;
    }
}

static inline int qot_read_result(uintptr_t base, qot_result_t *result)
{
    uint32_t layers;

    if (result == NULL) return QOT_ERR_NULL;
    result->status = qot_read32(base, QOT_REG_STATUS);
    result->token_id = qot_read32(base, QOT_REG_OUT_TOKEN);
    result->score_q26 = qot_read_score_q26(base);
    layers = qot_read32(base, QOT_REG_LAYERS);
    result->layers_started = QOT_LAYERS_STARTED(layers);
    result->layers_completed = QOT_LAYERS_COMPLETED(layers);
    result->layer_done_mask = qot_read32(base, QOT_REG_LAYER_DONE_MASK);
    result->layer_error_mask = qot_read32(base, QOT_REG_LAYER_ERROR_MASK);
    result->last_output_base = qot_read64(base, QOT_REG_LAST_OUTPUT_LO,
                                          QOT_REG_LAST_OUTPUT_HI);
    result->tail_hidden_base = qot_read64(base, QOT_REG_TAIL_HIDDEN_LO,
                                          QOT_REG_TAIL_HIDDEN_HI);
    result->tail_tiles_started = qot_read32(base, QOT_REG_TAIL_TILES_STARTED);
    result->tail_tiles_completed = qot_read32(base, QOT_REG_TAIL_TILES_COMPLETED);
    result->mem_read_reqs = qot_read32(base, QOT_REG_MEM_RD_REQS);
    result->mem_read_words = qot_read32(base, QOT_REG_MEM_RD_WORDS);
    result->mem_write_reqs = qot_read32(base, QOT_REG_MEM_WR_REQS);
    result->mem_write_words = qot_read32(base, QOT_REG_MEM_WR_WORDS);
    return QOT_OK;
}

/* Configure, launch, poll, and collect one autoregressive token step. */
static inline int qot_run_token(uintptr_t base, const qot_run_config_t *cfg,
                                uint32_t input_token_id, uint32_t position,
                                uint32_t max_polls, qot_result_t *result)
{
    qot_run_config_t token_cfg;
    int rc;

    if (cfg == NULL || result == NULL) return QOT_ERR_NULL;
    token_cfg = *cfg;
    token_cfg.input_token_id = input_token_id;
    token_cfg.position = position;

    rc = qot_configure_run(base, &token_cfg);
    if (rc != QOT_OK) return rc;
    rc = qot_start(base);
    if (rc != QOT_OK) return rc;
    rc = qot_poll_done(base, max_polls, NULL);
    if (rc != QOT_OK) {
        (void)qot_read_result(base, result);
        return rc;
    }
    return qot_read_result(base, result);
}

/*
 * Consume a prompt token-id sequence and greedily emit max_new_tokens ids.
 * The first emitted id is the LM-head result after the final prompt token.
 * Each later emitted id is fed back through tied-Q4 embedding at the next
 * position. KV cache and QMAP tables remain resident in PL DDR between calls.
 */
static inline int qot_decode_token_ids(
    uintptr_t base,
    const qot_run_config_t *cfg,
    const uint32_t *prompt_token_ids,
    uint32_t prompt_token_count,
    uint32_t max_new_tokens,
    uint32_t *generated_token_ids,
    uint32_t generated_token_capacity,
    uint32_t max_polls_per_token,
    uint32_t *generated_token_count,
    qot_result_t *last_result)
{
    qot_result_t result = {0};
    uint32_t position;
    uint32_t run_count;
    uint32_t prompt_index;
    uint32_t generated_index;
    uint32_t next_token;
    int rc;

    if (generated_token_count != NULL) *generated_token_count = 0u;
    if (cfg == NULL || prompt_token_ids == NULL ||
        generated_token_ids == NULL || last_result == NULL) {
        return QOT_ERR_NULL;
    }
    if (prompt_token_count == 0u || max_new_tokens == 0u) {
        return QOT_ERR_CAPACITY;
    }
    if (generated_token_capacity < max_new_tokens) return QOT_ERR_CAPACITY;
    if (cfg->runtime_context_enable == 0u || cfg->embedding_enable == 0u) {
        return QOT_ERR_CONTEXT;
    }
    if (cfg->final_hidden_override_valid != 0u) {
        return QOT_ERR_CONTEXT;
    }
    /*
     * Validate the complete prompt before launching any hardware work. Returning
     * after a later invalid prompt token would otherwise leave a partially
     * populated KV cache even though the caller received an argument error.
     */
    for (prompt_index = 0u; prompt_index < prompt_token_count; ++prompt_index) {
        if (prompt_token_ids[prompt_index] >= QOT_VOCAB_SIZE) {
            return QOT_ERR_BAD_TOKEN;
        }
    }

    /* prompt_count runs plus one run for every generated token after the first */
    run_count = prompt_token_count + max_new_tokens - 1u;
    if (run_count < prompt_token_count ||
        cfg->position >= QOT_MAX_CONTEXT ||
        run_count > (QOT_MAX_CONTEXT - cfg->position)) {
        return QOT_ERR_CONTEXT;
    }

    position = cfg->position;
    next_token = 0u;
    for (prompt_index = 0u; prompt_index < prompt_token_count; ++prompt_index) {
        rc = qot_run_token(base, cfg, prompt_token_ids[prompt_index], position,
                           max_polls_per_token, &result);
        if (rc != QOT_OK) {
            *last_result = result;
            return rc;
        }
        next_token = result.token_id;
        if (next_token >= QOT_VOCAB_SIZE) {
            *last_result = result;
            return QOT_ERR_BAD_TOKEN;
        }
        ++position;
    }

    generated_token_ids[0] = next_token;
    if (generated_token_count != NULL) *generated_token_count = 1u;
    for (generated_index = 1u; generated_index < max_new_tokens; ++generated_index) {
        if (next_token >= QOT_VOCAB_SIZE) {
            *last_result = result;
            return QOT_ERR_BAD_TOKEN;
        }
        rc = qot_run_token(base, cfg, next_token, position,
                           max_polls_per_token, &result);
        if (rc != QOT_OK) {
            *last_result = result;
            return rc;
        }
        next_token = result.token_id;
        if (next_token >= QOT_VOCAB_SIZE) {
            *last_result = result;
            return QOT_ERR_BAD_TOKEN;
        }
        generated_token_ids[generated_index] = next_token;
        if (generated_token_count != NULL) {
            *generated_token_count = generated_index + 1u;
        }
        ++position;
    }

    *last_result = result;
    return QOT_OK;
}

/*
 * Board bring-up smoke for the AXI-Lite/control seam. This intentionally starts
 * an invalid layer_count=0 request, matching tb_qmap_one_token_axil_top.sv. A
 * passing run reaches done+error through scheduler validation and issues no
 * memory traffic, so it can run before model artifacts are loaded into PL DDR.
 */
static inline int qot_run_no_memory_validation_smoke(uintptr_t base,
                                                     uint32_t max_polls,
                                                     qot_result_t *result)
{
    uint32_t polls;
    qot_result_t local_result;

    if (result == NULL) result = &local_result;
    if (qot_is_busy(base)) return QOT_ERR_BUSY;

    qot_clear_sticky(base);
    qot_write32(base, QOT_REG_LAYER_START, 0u);
    qot_write32(base, QOT_REG_LAYER_COUNT, 0u);
    qot_write32(base, QOT_REG_POSITION, 4u);
    qot_write32(base, QOT_REG_RUNTIME_CTRL, 0u);
    qot_write32(base, QOT_REG_EMBEDDING_CTRL, 0u);
    qot_write64(base, QOT_REG_INPUT_HIDDEN_LO, QOT_REG_INPUT_HIDDEN_HI,
                0x0000000405092540ull);
    qot_write64(base, QOT_REG_OUTPUT_HIDDEN_LO, QOT_REG_OUTPUT_HIDDEN_HI,
                0x0000000415092540ull);
    qot_write64(base, QOT_REG_KV_CACHE_LO, QOT_REG_KV_CACHE_HI,
                0x0000000414100000ull);
    qot_write64(base, QOT_REG_FINAL_TAIL_QMAP_LO, QOT_REG_FINAL_TAIL_QMAP_HI,
                0x0000000405010000ull);
    qot_write32(base, QOT_REG_CTRL, QOT_CTRL_START_MASK);

    for (polls = 0u; max_polls == 0u || polls < max_polls; ++polls) {
        uint32_t status = qot_read32(base, QOT_REG_STATUS);
        if ((status & QOT_STATUS_DONE_STICKY_MASK) != 0u) {
            break;
        }
    }
    if (max_polls != 0u && polls >= max_polls) return QOT_ERR_TIMEOUT;

    qot_read_result(base, result);
    if ((result->status & QOT_STATUS_DONE_STICKY_MASK) == 0u) return QOT_ERR_STATUS;
    if ((result->status & QOT_STATUS_ERROR_STICKY_MASK) == 0u) return QOT_ERR_STATUS;
    if ((result->layer_error_mask & 0x1u) == 0u) return QOT_ERR_STATUS;
    if (result->layers_started != 0u || result->layers_completed != 0u) {
        return QOT_ERR_STATUS;
    }
    if (result->mem_read_reqs != 0u || result->mem_read_words != 0u ||
        result->mem_write_reqs != 0u || result->mem_write_words != 0u) {
        return QOT_ERR_STATUS;
    }

    qot_clear_sticky(base);
    return QOT_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* QMAP_ONE_TOKEN_RUNTIME_H */
