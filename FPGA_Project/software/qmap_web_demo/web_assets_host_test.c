#include "web_assets.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const qweb_web_asset_t *find_literal(const char *path)
{
    return qweb_web_asset_find(path, strlen(path));
}

static int contains_bytes(
    const uint8_t *body,
    size_t body_length,
    const uint8_t *needle,
    size_t needle_length)
{
    size_t offset;

    if (body == NULL || needle == NULL || needle_length > body_length) return 0;
    if (needle_length == 0u) return 1;
    for (offset = 0u; offset <= body_length - needle_length; ++offset) {
        if (body[offset] == needle[0] &&
            memcmp(body + offset, needle, needle_length) == 0) {
            return 1;
        }
    }
    return 0;
}

static int contains_literal(
    const uint8_t *body,
    size_t body_length,
    const char *needle)
{
    return contains_bytes(
        body,
        body_length,
        (const uint8_t *)needle,
        strlen(needle));
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
    assert(!contains_literal(root->body, root->body_length, "href=\"/styles.css\""));
    assert(!contains_literal(root->body, root->body_length, "src=\"/app.js\""));
    assert(contains_literal(
        root->body,
        root->body_length,
        "data-qweb-inline=\"/styles.css\""));
    assert(contains_literal(
        root->body,
        root->body_length,
        "data-qweb-inline=\"/app.js\""));
    assert(contains_bytes(
        root->body,
        root->body_length,
        styles->body,
        styles->body_length));
    assert(contains_bytes(
        root->body,
        root->body_length,
        script->body,
        script->body_length));

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
