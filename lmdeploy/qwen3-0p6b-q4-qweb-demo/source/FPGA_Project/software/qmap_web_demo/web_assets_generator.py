#!/usr/bin/env python3
"""Generate deterministic, fail-closed C assets for the QMAP Web demo."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Iterable, Sequence


SCHEMA_VERSION = 1
MAX_ASSET_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 128 * 1024
HEADER_NAME = "web_assets.h"
SOURCE_NAME = "web_assets.c"
MANIFEST_NAME = "web_assets_manifest.json"

STYLESHEET_REFERENCE = b'  <link rel="stylesheet" href="/styles.css">\n'
SCRIPT_REFERENCE = b'  <script src="/app.js" defer></script>\n'
BODY_CLOSE = b"</body>\n"
INLINE_STYLE_OPEN = b'  <style data-qweb-inline="/styles.css">\n'
INLINE_STYLE_CLOSE = b"  </style>\n"
INLINE_SCRIPT_OPEN = b'  <script data-qweb-inline="/app.js">\n'
INLINE_SCRIPT_CLOSE = b"  </script>\n"


class AssetError(RuntimeError):
    """Raised when source or output paths violate the fixed asset contract."""


@dataclasses.dataclass(frozen=True)
class AssetSpec:
    source_name: str
    symbol: str
    mime_type: str
    routes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AssetPayload:
    spec: AssetSpec
    body: bytes
    sha256: str
    etag: str


ASSET_SPECS: tuple[AssetSpec, ...] = (
    AssetSpec(
        source_name="index.html",
        symbol="index_html",
        mime_type="text/html; charset=utf-8",
        routes=("/", "/index.html"),
    ),
    AssetSpec(
        source_name="styles.css",
        symbol="styles_css",
        mime_type="text/css; charset=utf-8",
        routes=("/styles.css",),
    ),
    AssetSpec(
        source_name="app.js",
        symbol="app_js",
        mime_type="text/javascript; charset=utf-8",
        routes=("/app.js",),
    ),
)


def _strict_directory(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise AssetError(f"{description} must not be a symlink: {path}")
    try:
        mode = path.stat().st_mode
    except FileNotFoundError as exc:
        raise AssetError(f"{description} does not exist: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise AssetError(f"{description} is not a directory: {path}")
    return path.resolve(strict=True)


def validate_project_directory(project_directory: Path | str) -> Path:
    project = _strict_directory(Path(project_directory), "project directory")
    web_directory = project / "web"
    _strict_directory(web_directory, "web asset directory")
    return project


def _validate_source_text(path: Path, body: bytes) -> None:
    if len(body) == 0:
        raise AssetError(f"web asset must not be empty: {path.name}")
    if len(body) > MAX_ASSET_BYTES:
        raise AssetError(
            f"web asset exceeds {MAX_ASSET_BYTES} bytes: {path.name}"
        )
    if body.startswith(b"\xef\xbb\xbf"):
        raise AssetError(f"UTF-8 BOM is forbidden: {path.name}")
    if b"\x00" in body:
        raise AssetError(f"NUL byte is forbidden: {path.name}")
    if b"\r" in body:
        raise AssetError(f"only canonical LF line endings are allowed: {path.name}")
    if not body.endswith(b"\n"):
        raise AssetError(f"web asset must end with one LF: {path.name}")
    try:
        body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AssetError(f"web asset is not strict UTF-8: {path.name}") from exc


def _replace_exactly_once(
    body: bytes,
    needle: bytes,
    replacement: bytes,
    description: str,
) -> bytes:
    count = body.count(needle)
    if count != 1:
        raise AssetError(
            f"index.html must contain exactly one canonical {description}; found {count}"
        )
    return body.replace(needle, replacement, 1)


def compose_index_html(index_html: bytes, styles_css: bytes, app_js: bytes) -> bytes:
    """Build the one-connection boot document from the three canonical sources."""
    if b"</style" in styles_css.lower():
        raise AssetError("styles.css contains an unsafe inline style terminator")
    if b"</script" in app_js.lower():
        raise AssetError("app.js contains an unsafe inline script terminator")

    composed = _replace_exactly_once(
        index_html,
        STYLESHEET_REFERENCE,
        INLINE_STYLE_OPEN + styles_css + INLINE_STYLE_CLOSE,
        "stylesheet reference",
    )
    composed = _replace_exactly_once(
        composed,
        SCRIPT_REFERENCE,
        b"",
        "deferred script reference",
    )
    composed = _replace_exactly_once(
        composed,
        BODY_CLOSE,
        INLINE_SCRIPT_OPEN + app_js + INLINE_SCRIPT_CLOSE + BODY_CLOSE,
        "body closing tag",
    )
    return composed


def load_assets(project_directory: Path | str) -> tuple[AssetPayload, ...]:
    project = validate_project_directory(project_directory)
    web_directory = project / "web"
    expected_names = {spec.source_name for spec in ASSET_SPECS}
    entries = list(web_directory.iterdir())
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"unexpected={','.join(extra)}")
        raise AssetError("web asset inventory drift: " + " ".join(details))

    by_name = {entry.name: entry for entry in entries}
    source_bodies: dict[str, bytes] = {}
    for spec in ASSET_SPECS:
        path = by_name[spec.source_name]
        if path.is_symlink():
            raise AssetError(f"web asset must not be a symlink: {path.name}")
        try:
            mode = path.stat().st_mode
        except FileNotFoundError as exc:
            raise AssetError(f"web asset disappeared while reading: {path.name}") from exc
        if not stat.S_ISREG(mode):
            raise AssetError(f"web asset is not a regular file: {path.name}")
        if path.resolve(strict=True).parent != web_directory.resolve(strict=True):
            raise AssetError(f"web asset escapes its fixed directory: {path.name}")
        body = path.read_bytes()
        _validate_source_text(path, body)
        source_bodies[spec.source_name] = body

    served_bodies = dict(source_bodies)
    served_bodies["index.html"] = compose_index_html(
        source_bodies["index.html"],
        source_bodies["styles.css"],
        source_bodies["app.js"],
    )

    payloads: list[AssetPayload] = []
    total_bytes = 0
    for spec in ASSET_SPECS:
        body = served_bodies[spec.source_name]
        _validate_source_text(web_directory / spec.source_name, body)
        total_bytes += len(body)
        digest = hashlib.sha256(body).hexdigest()
        payloads.append(
            AssetPayload(
                spec=spec,
                body=body,
                sha256=digest,
                etag=f'"{digest}"',
            )
        )
    if total_bytes > MAX_TOTAL_BYTES:
        raise AssetError(f"web asset set exceeds {MAX_TOTAL_BYTES} bytes")
    return tuple(payloads)


def source_set_sha256(payloads: Sequence[AssetPayload]) -> str:
    digest = hashlib.sha256()
    digest.update(f"qweb-web-assets-v{SCHEMA_VERSION}\0".encode("ascii"))
    for payload in payloads:
        spec = payload.spec
        digest.update(spec.source_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(spec.symbol.encode("ascii"))
        digest.update(b"\0")
        digest.update(spec.mime_type.encode("ascii"))
        digest.update(b"\0")
        for route in spec.routes:
            digest.update(route.encode("ascii"))
            digest.update(b"\0")
        digest.update(len(payload.body).to_bytes(8, byteorder="big"))
        digest.update(payload.body)
    return digest.hexdigest()


def _c_string(value: str) -> str:
    encoded = value.encode("ascii")
    pieces = ['"']
    for byte in encoded:
        if byte == 0x22:
            pieces.append(r"\"")
        elif byte == 0x5C:
            pieces.append(r"\\")
        elif 0x20 <= byte <= 0x7E:
            pieces.append(chr(byte))
        else:
            pieces.append(f"\\x{byte:02x}")
    pieces.append('"')
    return "".join(pieces)


def _format_c_bytes(body: bytes) -> str:
    lines = []
    for offset in range(0, len(body), 12):
        chunk = body[offset : offset + 12]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines)


def render_header(source_digest: str) -> bytes:
    text = f"""/* Generated by web_assets_generator.py. Do not edit. */
/* Source set SHA-256: {source_digest} */
#ifndef QWEB_WEB_ASSETS_H
#define QWEB_WEB_ASSETS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern \"C\" {{
#endif

#define QWEB_WEB_ASSETS_SCHEMA_VERSION {SCHEMA_VERSION}u

typedef struct qweb_web_asset {{
    const char *path;
    size_t path_length;
    const char *mime_type;
    const char *etag;
    const uint8_t *body;
    size_t body_length;
}} qweb_web_asset_t;

extern const qweb_web_asset_t qweb_web_assets[];
extern const size_t qweb_web_asset_count;
extern const char qweb_web_assets_source_sha256[65];

/*
 * Match one origin-form path byte-for-byte. Query strings, fragments,
 * percent-encoded aliases, prefixes, and traversal spellings fail closed.
 */
const qweb_web_asset_t *qweb_web_asset_find(
    const char *path,
    size_t path_length);

#ifdef __cplusplus
}}
#endif

#endif /* QWEB_WEB_ASSETS_H */
"""
    return text.encode("utf-8")


def render_source(payloads: Sequence[AssetPayload], source_digest: str) -> bytes:
    lines = [
        "/* Generated by web_assets_generator.py. Do not edit. */",
        f"/* Source set SHA-256: {source_digest} */",
        '#include "web_assets.h"',
        "",
        "#include <string.h>",
        "",
        f"const char qweb_web_assets_source_sha256[65] = {_c_string(source_digest)};",
        "",
    ]
    for payload in payloads:
        symbol = f"qweb_web_asset_{payload.spec.symbol}"
        lines.extend(
            [
                f"static const uint8_t {symbol}[] = {{",
                _format_c_bytes(payload.body),
                "};",
                f"_Static_assert(sizeof({symbol}) == {len(payload.body)}u,",
                f'               "{payload.spec.source_name} length drift");',
                "",
            ]
        )

    lines.append("const qweb_web_asset_t qweb_web_assets[] = {")
    for payload in payloads:
        symbol = f"qweb_web_asset_{payload.spec.symbol}"
        for route in payload.spec.routes:
            lines.extend(
                [
                    "    {",
                    f"        {_c_string(route)},",
                    f"        {len(route.encode('ascii'))}u,",
                    f"        {_c_string(payload.spec.mime_type)},",
                    f"        {_c_string(payload.etag)},",
                    f"        {symbol},",
                    f"        {len(payload.body)}u",
                    "    },",
                ]
            )
    lines.extend(
        [
            "};",
            "",
            "const size_t qweb_web_asset_count =",
            "    sizeof(qweb_web_assets) / sizeof(qweb_web_assets[0]);",
            "",
            "const qweb_web_asset_t *qweb_web_asset_find(",
            "    const char *path,",
            "    size_t path_length)",
            "{",
            "    size_t index;",
            "",
            "    if (path == NULL) return NULL;",
            "    for (index = 0u; index < qweb_web_asset_count; ++index) {",
            "        const qweb_web_asset_t *asset = &qweb_web_assets[index];",
            "",
            "        if (path_length == asset->path_length &&",
            "            memcmp(path, asset->path, path_length) == 0) {",
            "            return asset;",
            "        }",
            "    }",
            "    return NULL;",
            "}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def render_manifest(payloads: Sequence[AssetPayload], source_digest: str) -> bytes:
    assets = []
    for payload in payloads:
        assets.append(
            {
                "body_length": len(payload.body),
                "etag": payload.etag,
                "mime_type": payload.spec.mime_type,
                "routes": list(payload.spec.routes),
                "sha256": payload.sha256,
                "source": f"web/{payload.spec.source_name}",
            }
        )
    manifest = {
        "assets": assets,
        "schema_version": SCHEMA_VERSION,
        "source_set_sha256": source_digest,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render_outputs(project_directory: Path | str) -> dict[str, bytes]:
    payloads = load_assets(project_directory)
    digest = source_set_sha256(payloads)
    return {
        HEADER_NAME: render_header(digest),
        SOURCE_NAME: render_source(payloads, digest),
        MANIFEST_NAME: render_manifest(payloads, digest),
    }


def output_drift(project_directory: Path | str) -> list[str]:
    project = validate_project_directory(project_directory)
    expected = render_outputs(project)
    drift = []
    for name, content in expected.items():
        path = project / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            drift.append(name)
    return drift


def _validate_output_path(project: Path, name: str) -> Path:
    path = project / name
    if path.parent != project:
        raise AssetError(f"generated output escapes project directory: {name}")
    if path.is_symlink():
        raise AssetError(f"generated output must not be a symlink: {path}")
    if path.exists() and not path.is_file():
        raise AssetError(f"generated output is not a regular file: {path}")
    return path


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def generate(project_directory: Path | str) -> str:
    project = validate_project_directory(project_directory)
    rendered = render_outputs(project)
    for name, content in rendered.items():
        path = _validate_output_path(project, name)
        _atomic_write(path, content)
    return json.loads(rendered[MANIFEST_NAME])["source_set_sha256"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the fixed QMAP Web demo asset set."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="qmap_web_demo directory containing the exact web/ inventory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated outputs byte-for-byte without writing",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        project = validate_project_directory(arguments.project_dir)
        if arguments.check:
            drift = output_drift(project)
            if drift:
                print("ERROR generated web asset drift: " + ", ".join(drift), file=sys.stderr)
                return 1
            digest = json.loads((project / MANIFEST_NAME).read_text(encoding="utf-8"))[
                "source_set_sha256"
            ]
            print(f"PASS web assets are current source_sha256={digest}")
            return 0
        digest = generate(project)
        print(f"PASS generated deterministic web assets source_sha256={digest}")
        return 0
    except (AssetError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"ERROR web asset generation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
