#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qmap_one_token_runtime.h"

static void fill_layer_tables(qot_layer_qmap_bases_t *tables)
{
    uint64_t base = 0x0000000405000000ull;
    uint32_t layer;

    memset(tables, 0, sizeof(*tables) * QOT_MAX_LAYERS);
    for (layer = 0u; layer < QOT_MAX_LAYERS; ++layer) {
        uint64_t layer_base = base + ((uint64_t)layer << 20);
        tables[layer].qkv = layer_base + 0x10000ull;
        tables[layer].input_norm = layer_base + 0x20000ull;
        tables[layer].attn_frontend = layer_base + 0x30000ull;
        tables[layer].attn_score_value = layer_base + 0x40000ull;
        tables[layer].o_proj = layer_base + 0x50000ull;
        tables[layer].post_attn_norm = layer_base + 0x60000ull;
        tables[layer].mlp_gate_up = layer_base + 0x70000ull;
        tables[layer].mlp_silu_mul = layer_base + 0x80000ull;
        tables[layer].mlp_down = layer_base + 0x90000ull;
        tables[layer].mlp_residual_add = layer_base + 0xA0000ull;
    }
}

int main(void)
{
    uint32_t regs[0x1000u / sizeof(uint32_t)] = {0};
    qot_layer_qmap_bases_t tables[QOT_MAX_LAYERS];
    qot_run_config_t cfg;
    qot_result_t result;
    uint32_t prompt[] = {10u, 11u};
    uint32_t generated[3] = {0u, 0u, 0u};
    uint32_t generated_count = 0u;
    uintptr_t base = (uintptr_t)&regs[0];
    int rc;

    fill_layer_tables(tables);
    memset(&cfg, 0, sizeof(cfg));
    cfg.layer_start = 0u;
    cfg.layer_count = 1u;
    cfg.position = 2u;
    cfg.runtime_context_enable = 1u;
    cfg.embedding_enable = 1u;
    cfg.embedding_weight_base = 0x0000000400100000ull;
    cfg.embedding_scale_base = 0x0000000404B30000ull;
    cfg.input_hidden_base = 0x0000000416100000ull;
    cfg.output_hidden_base = 0x0000000416101000ull;
    cfg.kv_cache_base = 0x0000000414100000ull;
    cfg.final_tail_qmap_base = 0x0000000419500000ull;
    cfg.layer_qmap_bases = tables;
    cfg.layer_qmap_base_count = QOT_MAX_LAYERS;

    regs[QOT_REG_STATUS / 4u] = QOT_STATUS_DONE_STICKY_MASK;
    regs[QOT_REG_OUT_TOKEN / 4u] = 42u;
    regs[QOT_REG_LAYERS / 4u] = (1u << 16) | 1u;

    rc = qot_decode_token_ids(base, &cfg, prompt, 2u, 3u,
                              generated, 3u, 8u,
                              &generated_count, &result);
    if (rc != QOT_OK || generated_count != 3u ||
        generated[0] != 42u || generated[1] != 42u || generated[2] != 42u) {
        printf("FAIL: decode sequence rc=%d count=%u tokens=%u,%u,%u\n",
               rc, generated_count, generated[0], generated[1], generated[2]);
        return 1;
    }
    if (regs[QOT_REG_POSITION / 4u] != 5u ||
        regs[QOT_REG_INPUT_TOKEN / 4u] != 42u ||
        regs[QOT_REG_RUNTIME_CTRL / 4u] != QOT_RUNTIME_CONTEXT_ENABLE_MASK) {
        printf("FAIL: final token-step register state is incorrect\n");
        return 1;
    }

    cfg.position = QOT_MAX_CONTEXT - 1u;
    rc = qot_decode_token_ids(base, &cfg, prompt, 2u, 1u,
                              generated, 3u, 8u,
                              &generated_count, &result);
    if (rc != QOT_ERR_CONTEXT) {
        printf("FAIL: context overflow returned %d\n", rc);
        return 1;
    }

    cfg.position = 0u;
    cfg.runtime_context_enable = 0u;
    rc = qot_decode_token_ids(base, &cfg, prompt, 2u, 1u,
                              generated, 3u, 8u,
                              &generated_count, &result);
    if (rc != QOT_ERR_CONTEXT) {
        printf("FAIL: legacy decode guard returned %d\n", rc);
        return 1;
    }

    printf("PASS: qmap one-token host runtime supports persistent token-id decode sequencing.\n");
    return 0;
}
