#include "web_assets.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const qweb_web_asset_t *find_literal(const char *path)
{
    return qweb_web_asset_find(path, strlen(path));
}

int main(void)
{
    const qweb_web_asset_t *root = find_literal("/");
    const qweb_web_asset_t *index = find_literal("/index.html");
    const qweb_web_asset_t *styles = find_literal("/styles.css");
    const qweb_web_asset_t *script = find_literal("/app.js");
    size_t index_number;

    assert(QWEB_WEB_ASSETS_SCHEMA_VERSION == 1u);
    assert(qweb_web_asset_count == 4u);
    assert(strlen(qweb_web_assets_source_sha256) == 64u);

    assert(root != NULL);
    assert(index != NULL);
    assert(styles != NULL);
    assert(script != NULL);
    assert(root->body == index->body);
    assert(root->body_length == index->body_length);
    assert(strcmp(root->etag, index->etag) == 0);
    assert(strcmp(root->mime_type, "text/html; charset=utf-8") == 0);
    assert(strcmp(styles->mime_type, "text/css; charset=utf-8") == 0);
    assert(strcmp(script->mime_type, "text/javascript; charset=utf-8") == 0);

    for (index_number = 0u; index_number < qweb_web_asset_count; ++index_number) {
        const qweb_web_asset_t *asset = &qweb_web_assets[index_number];

        assert(asset->path != NULL);
        assert(asset->path_length == strlen(asset->path));
        assert(asset->mime_type != NULL);
        assert(asset->etag != NULL);
        assert(strlen(asset->etag) == 66u);
        assert(asset->etag[0] == '"');
        assert(asset->etag[65] == '"');
        assert(asset->body != NULL);
        assert(asset->body_length > 0u);
        assert(asset->body[asset->body_length - 1u] == (unsigned char)'\n');
    }

    assert(find_literal("") == NULL);
    assert(find_literal("/missing") == NULL);
    assert(find_literal("/index") == NULL);
    assert(find_literal("/index.html/") == NULL);
    assert(find_literal("/index.html?cache=1") == NULL);
    assert(find_literal("/styles.css#fragment") == NULL);
    assert(find_literal("/web/app.js") == NULL);
    assert(find_literal("/../app.js") == NULL);
    assert(find_literal("/%2e%2e/app.js") == NULL);
    assert(qweb_web_asset_find(NULL, 0u) == NULL);

    {
        static const char embedded_nul[] = {'/', '\0', 'x'};
        assert(qweb_web_asset_find(embedded_nul, sizeof(embedded_nul)) == NULL);
    }

    puts("PASS web asset exact-path host test");
    return 0;
}
