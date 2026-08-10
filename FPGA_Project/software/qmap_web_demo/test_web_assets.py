#!/usr/bin/env python3
"""Host tests for deterministic Web UI assets and their generated C table."""

from __future__ import annotations

import contextlib
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import web_assets_generator as generator  # noqa: E402


class _HtmlInventory(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []
        self.input_modes: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id") is not None:
            self.ids.append(values["id"] or "")
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href") or "")
        if tag == "script":
            self.scripts.append(values.get("src") or "")
        if tag == "input" and values.get("name") == "inputMode":
            self.input_modes.append(values.get("value") or "")


def _copy_project(destination: Path) -> Path:
    project = destination / "qmap_web_demo"
    shutil.copytree(ROOT / "web", project / "web")
    return project


class GeneratorContractTests(unittest.TestCase):
    def test_checked_in_outputs_match_exact_sources(self) -> None:
        self.assertEqual(generator.output_drift(ROOT), [])

    def test_render_is_byte_deterministic(self) -> None:
        first = generator.render_outputs(ROOT)
        second = generator.render_outputs(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                generator.HEADER_NAME,
                generator.SOURCE_NAME,
                generator.MANIFEST_NAME,
            },
        )

    def test_manifest_pins_path_mime_length_hash_and_etag(self) -> None:
        rendered = generator.render_outputs(ROOT)
        manifest = json.loads(rendered[generator.MANIFEST_NAME])
        payloads = generator.load_assets(ROOT)
        self.assertEqual(manifest["schema_version"], generator.SCHEMA_VERSION)
        self.assertEqual(
            manifest["source_set_sha256"], generator.source_set_sha256(payloads)
        )
        self.assertEqual(
            [route for asset in manifest["assets"] for route in asset["routes"]],
            ["/", "/index.html", "/styles.css", "/app.js"],
        )
        for item, payload in zip(manifest["assets"], payloads, strict=True):
            self.assertEqual(item["body_length"], len(payload.body))
            self.assertEqual(item["mime_type"], payload.spec.mime_type)
            self.assertEqual(item["sha256"], hashlib.sha256(payload.body).hexdigest())
            self.assertEqual(item["etag"], f'"{item["sha256"]}"')

    def test_generate_then_check_and_detect_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _copy_project(Path(temporary))
            generator.generate(project)
            self.assertEqual(generator.output_drift(project), [])
            with (project / "web" / "app.js").open("ab") as stream:
                stream.write(b"\n")
            self.assertEqual(
                set(generator.output_drift(project)),
                {
                    generator.HEADER_NAME,
                    generator.SOURCE_NAME,
                    generator.MANIFEST_NAME,
                },
            )

    def test_detects_generated_output_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _copy_project(Path(temporary))
            generator.generate(project)
            (project / generator.SOURCE_NAME).write_bytes(b"stale\n")
            self.assertEqual(generator.output_drift(project), [generator.SOURCE_NAME])

    def test_check_mode_never_creates_missing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _copy_project(Path(temporary))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = generator.main(["--project-dir", str(project), "--check"])
            self.assertEqual(result, 1)
            self.assertIn("generated web asset drift", stderr.getvalue())
            for name in (
                generator.HEADER_NAME,
                generator.SOURCE_NAME,
                generator.MANIFEST_NAME,
            ):
                self.assertFalse((project / name).exists())

    def test_rejects_missing_and_unexpected_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _copy_project(Path(temporary))
            (project / "web" / "unexpected.txt").write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(generator.AssetError, "inventory drift"):
                generator.load_assets(project)
            (project / "web" / "unexpected.txt").unlink()
            (project / "web" / "app.js").unlink()
            with self.assertRaisesRegex(generator.AssetError, "inventory drift"):
                generator.load_assets(project)

    def test_rejects_noncanonical_or_non_utf8_content(self) -> None:
        cases = {
            "BOM": b"\xef\xbb\xbfbody\n",
            "NUL": b"body\x00\n",
            "line endings": b"body\r\n",
            "one LF": b"body",
            "strict UTF-8": b"\xff\n",
        }
        for expected_message, content in cases.items():
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temporary:
                    project = _copy_project(Path(temporary))
                    (project / "web" / "app.js").write_bytes(content)
                    with self.assertRaisesRegex(generator.AssetError, expected_message):
                        generator.load_assets(project)

    def test_rejects_nonregular_generated_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = _copy_project(Path(temporary))
            (project / generator.HEADER_NAME).mkdir()
            with self.assertRaisesRegex(generator.AssetError, "not a regular file"):
                generator.generate(project)


class UiContractTests(unittest.TestCase):
    def test_html_has_local_assets_unique_ids_and_both_inputs(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        inventory = _HtmlInventory()
        inventory.feed(html)
        self.assertEqual(inventory.stylesheets, ["/styles.css"])
        self.assertEqual(inventory.scripts, ["/app.js"])
        self.assertEqual(sorted(inventory.input_modes), ["prompt", "tokens"])
        self.assertEqual(len(inventory.ids), len(set(inventory.ids)))
        for required_id in (
            "generateForm",
            "promptInput",
            "tokenInput",
            "maxNewInput",
            "generateButton",
            "jobState",
            "tokenRows",
            "outputText",
        ):
            self.assertIn(required_id, inventory.ids)

    def test_assets_are_offline_only_and_js_uses_exact_api(self) -> None:
        combined = b"\n".join(
            (ROOT / "web" / name).read_bytes()
            for name in ("index.html", "styles.css", "app.js")
        )
        self.assertNotIn(b"http://", combined)
        self.assertNotIn(b"https://", combined)
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        for endpoint in (
            '"/api/health"',
            '"/api/generate"',
            "`/api/generate/${jobId}`",
            "`/api/generate/${jobId}/output`",
        ):
            self.assertIn(endpoint, script)
        self.assertIn('headers: { "Content-Type": "application/json" }', script)
        self.assertIn("max_new_tokens", script)
        self.assertIn("state.activeJobId !== null", script)

    def test_js_avoids_html_injection_sinks(self) -> None:
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        for forbidden in (
            ".innerHTML",
            ".outerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        ):
            self.assertNotIn(forbidden, script)


class HostToolTests(unittest.TestCase):
    def test_node_ui_logic(self) -> None:
        node = shutil.which("node")
        self.assertIsNotNone(node, "node is required for Web UI host tests")
        result = subprocess.run(
            [node or "node", str(ROOT / "web_assets_ui_test.js")],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS Web UI pure logic tests", result.stdout)

    def test_generated_c_exact_path_lookup(self) -> None:
        compiler = shutil.which("gcc")
        self.assertIsNotNone(compiler, "gcc is required for generated C host tests")
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "web_assets_host_test.exe"
            compile_result = subprocess.run(
                [
                    compiler or "gcc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-I",
                    str(ROOT),
                    str(ROOT / generator.SOURCE_NAME),
                    str(ROOT / "web_assets_host_test.c"),
                    "-o",
                    str(executable),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                compile_result.stdout + compile_result.stderr,
            )
            run_result = subprocess.run(
                [str(executable)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stdout + run_result.stderr)
            self.assertIn("PASS web asset exact-path host test", run_result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
