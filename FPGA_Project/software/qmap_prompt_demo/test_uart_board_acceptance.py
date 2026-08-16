from __future__ import annotations

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_uart_board_acceptance as acceptance


PASS_FIXTURE = HERE / "fixtures" / "uart_board_acceptance_pass.txt"


class ScriptedSerial:
    def __init__(
        self,
        *,
        include_echo: bool = True,
        chunk_size: int = 17,
        post_negative: str = "",
    ) -> None:
        startup = "\r\n".join(acceptance.STARTUP_RECORDS) + "\r\n"
        self.buffer = bytearray(startup.encode())
        self.commands: list[str] = []
        self.overlap = False
        self.closed = False
        self.include_echo = include_echo
        self.chunk_size = chunk_size
        self.post_negative = post_negative

    def read(self, size: int) -> bytes:
        if not self.buffer:
            return b""
        count = min(size, self.chunk_size, len(self.buffer))
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        return data

    def write(self, payload: bytes) -> int:
        if self.buffer:
            self.overlap = True
        command = payload.decode("utf-8").rstrip("\r")
        case = acceptance.ACCEPTANCE_CASES[len(self.commands)]
        if command != case.command:
            raise AssertionError(f"unexpected command {command!r}")
        self.commands.append(command)
        echo = f"qot> {command}\r\n" if self.include_echo else ""
        records = list(case.expected_records)
        if case.expected_busy_positions:
            start_index = next(
                index for index, record in enumerate(records) if record.startswith("START ")
            )
            busy = [
                f"BUSY position={position} polls=1000000 status=0x00020501"
                for position in case.expected_busy_positions
            ]
            records[start_index + 1 : start_index + 1] = busy
        response = echo + "\r\n".join(records) + "\r\n"
        if case.name == "reject_out_of_range_id":
            response += self.post_negative
        self.buffer.extend(response.encode("ascii"))
        return len(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class TranscriptAcceptanceTests(unittest.TestCase):
    def test_exact_fixture_passes_all_single_flight_cases(self) -> None:
        report = acceptance.verify_transcript_bytes(PASS_FIXTURE.read_bytes())
        self.assertTrue(report["passed"], report["failure"])
        self.assertTrue(report["single_flight"])
        self.assertEqual(report["ready_observed"], acceptance.READY_RECORD)
        self.assertEqual(report["startup_observed"], list(acceptance.STARTUP_RECORDS))
        self.assertEqual(len(report["cases"]), 8)
        self.assertTrue(all(case["passed"] for case in report["cases"]))
        self.assertEqual(
            [case["name"] for case in report["cases"]][5:7],
            ["text_prompt_first", "text_prompt_repeat"],
        )

    def test_wrong_score_fails_at_exact_record(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(
            b"TOKEN 0 264 1296911292", b"TOKEN 0 264 1296911293", 1
        )
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("text_token_ids", report["failure"])
        self.assertIn("1296911292", report["failure"])

    def test_missing_record_fails_closed(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(b"BYTES 0 2061\n", b"", 1)
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("text_token_ids", report["failure"])

    def test_negative_case_rejects_start_before_error(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(
            b"ERROR PARSE RANGE offset=11",
            b"START prompt=1 max_new=1\nERROR PARSE RANGE offset=11",
        )
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("reject_out_of_range_id", report["failure"])

    def test_missing_ready_fails(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(
            (acceptance.READY_RECORD + "\n").encode("ascii"), b"", 1
        )
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["ready_observed"])
        self.assertIn("before READY", report["failure"])

    def test_startup_instance_must_match_in_order(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(
            b"DDR4 status=0x00000005\n",
            b"TOKENIZER tokens=151669 model_vocab=151936 eos=151643 bytes=3629566\n"
            b"DDR4 status=0x00000005\n",
            1,
        )
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("startup record", report["failure"])

    def test_startup_accepts_fsbl_terminal_carriage_return_prefix(self) -> None:
        next_index, observed = acceptance._advance_startup(
            0, "\rQwen3 text/token prompt demo"
        )
        self.assertEqual(next_index, 1)
        self.assertEqual(observed, "Qwen3 text/token prompt demo")

    def test_fail_before_ready_is_immediate_failure(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(
            b"DDR4 status=0x00000005\n",
            b"DDR4 status=0x00000005\nFAIL injected\n",
            1,
        )
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("failed before READY", report["failure"])

    def test_invalid_utf8_fails_closed(self) -> None:
        report = acceptance.verify_transcript_bytes(
            PASS_FIXTURE.read_bytes().replace(b"Release 2025.1", b"Release \xff", 1)
        )
        self.assertFalse(report["passed"])
        self.assertIn("invalid UTF-8", report["failure"])

    def test_protocol_record_before_command_echo_is_rejected(self) -> None:
        data = PASS_FIXTURE.read_bytes().replace(b"qot> PING\n", b"", 1)
        report = acceptance.verify_transcript_bytes(data)
        self.assertFalse(report["passed"])
        self.assertIn("before command echo", report["failure"])

    def test_trailing_protocol_record_fails(self) -> None:
        report = acceptance.verify_transcript_bytes(
            PASS_FIXTURE.read_bytes() + b"PONG\n"
        )
        self.assertFalse(report["passed"])
        self.assertIn("grace", report["failure"])

    def test_negative_grace_rejects_busy(self) -> None:
        report = acceptance.verify_transcript_bytes(
            PASS_FIXTURE.read_bytes()
            + b"BUSY position=0 polls=1000000 status=0x00020501\n"
        )
        self.assertFalse(report["passed"])
        self.assertIn("grace", report["failure"])

    def test_cli_writes_raw_and_structured_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acceptance"
            return_code = acceptance.main(
                [
                    "--verify-transcript",
                    str(PASS_FIXTURE),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(return_code, 0)
            self.assertEqual(
                (output / "uart_raw.bin").read_bytes(), PASS_FIXTURE.read_bytes()
            )
            report = json.loads((output / "acceptance.json").read_text("utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["artifacts"]["uart_raw_bytes"], PASS_FIXTURE.stat().st_size)
            self.assertEqual(len(report["artifacts"]["uart_raw_sha256"]), 64)

    def test_timeout_arguments_are_strictly_bounded(self) -> None:
        with self.assertRaises(SystemExit):
            acceptance.main(
                [
                    "--verify-transcript",
                    str(PASS_FIXTURE),
                    "--ready-timeout",
                    "0",
                ]
            )

    def test_default_ready_timeout_covers_cold_jtag_runtime_load(self) -> None:
        args = acceptance.build_parser().parse_args(
            ["--verify-transcript", str(PASS_FIXTURE)]
        )
        self.assertEqual(args.ready_timeout, acceptance.MAX_READY_TIMEOUT_SECONDS)

    def test_default_artifacts_do_not_mutate_a_packaged_workbench(self) -> None:
        output = acceptance._default_output_dir()
        self.assertEqual(
            output.parent,
            Path(tempfile.gettempdir()).resolve() / "qot_uart_acceptance",
        )

    def test_live_driver_never_has_two_commands_in_flight(self) -> None:
        fake = ScriptedSerial()
        raw_log = io.BytesIO()
        with mock.patch.object(acceptance, "_open_serial", return_value=fake):
            report = acceptance.run_live(
                "FAKE",
                raw_log,
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=None,
                launcher_log=None,
            )
        self.assertTrue(report["passed"], report["failure"])
        self.assertFalse(fake.overlap)
        self.assertTrue(fake.closed)
        self.assertEqual(fake.commands, [case.command for case in acceptance.ACCEPTANCE_CASES])

    def test_live_driver_requires_command_echo(self) -> None:
        fake = ScriptedSerial(include_echo=False)
        with mock.patch.object(acceptance, "_open_serial", return_value=fake):
            report = acceptance.run_live(
                "FAKE",
                io.BytesIO(),
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=None,
                launcher_log=None,
            )
        self.assertFalse(report["passed"])
        self.assertIn("before command echo", report["failure"])

    def test_live_negative_grace_keeps_lines_from_same_read_chunk(self) -> None:
        fake = ScriptedSerial(
            chunk_size=4096,
            post_negative=(
                "BUSY position=0 polls=1000000 status=0x00020501\r\n"
            ),
        )
        with mock.patch.object(acceptance, "_open_serial", return_value=fake):
            report = acceptance.run_live(
                "FAKE",
                io.BytesIO(),
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=None,
                launcher_log=None,
            )
        self.assertFalse(report["passed"])
        self.assertIn("negative-case grace", report["failure"])

    def test_nonzero_launcher_exit_before_ready_cannot_pass(self) -> None:
        class FailedLauncher:
            pid = 123
            returncode = 9

            def wait(self, timeout: float) -> int:
                return self.returncode

            def poll(self) -> int:
                return self.returncode

            def kill(self) -> None:
                raise AssertionError("completed launcher must not be killed")

        fake = ScriptedSerial()
        with (
            mock.patch.object(acceptance, "_open_serial", return_value=fake),
            mock.patch.object(
                acceptance.subprocess, "Popen", return_value=FailedLauncher()
            ) as popen,
        ):
            report = acceptance.run_live(
                "FAKE",
                io.BytesIO(),
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=HERE / "run_board_smoke.ps1",
                launcher_log=io.BytesIO(),
            )
        self.assertFalse(report["passed"])
        self.assertIn("nonzero return code 9 before READY", report["failure"])
        popen_kwargs = popen.call_args.kwargs
        if os.name == "nt":
            self.assertEqual(
                popen_kwargs["creationflags"], subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            self.assertTrue(popen_kwargs["start_new_session"])

    def test_successful_launcher_exit_allows_uart_ready(self) -> None:
        class SuccessfulLauncher:
            pid = 124
            returncode = 0

            def wait(self, timeout: float) -> int:
                return self.returncode

            def poll(self) -> int:
                return self.returncode

            def kill(self) -> None:
                raise AssertionError("completed launcher must not be killed")

        fake = ScriptedSerial()
        with (
            mock.patch.object(acceptance, "_open_serial", return_value=fake),
            mock.patch.object(
                acceptance.subprocess, "Popen", return_value=SuccessfulLauncher()
            ),
        ):
            report = acceptance.run_live(
                "FAKE",
                io.BytesIO(),
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=HERE / "run_board_smoke.ps1",
                launcher_log=io.BytesIO(),
            )
        self.assertTrue(report["passed"], report["failure"])

    def test_launcher_timeout_is_structured_and_closes_serial(self) -> None:
        class HangingLauncher:
            pid = 2468
            returncode: int | None = None

            def wait(self, timeout: float) -> int:
                raise subprocess.TimeoutExpired("powershell.exe", timeout)

            def poll(self) -> int | None:
                return self.returncode

            def kill(self) -> None:
                raise AssertionError("run_live must delegate tree cleanup")

        launcher = HangingLauncher()
        fake = ScriptedSerial()

        def cleanup(process: HangingLauncher) -> dict[str, object]:
            self.assertIs(process, launcher)
            process.returncode = 1
            return {
                "attempted": True,
                "method": "test-tree-kill",
                "tree_signal_succeeded": True,
                "parent_exited": True,
                "fallback_parent_kill": False,
                "succeeded": True,
                "errors": [],
            }

        with (
            mock.patch.object(acceptance, "_open_serial", return_value=fake),
            mock.patch.object(acceptance.subprocess, "Popen", return_value=launcher),
            mock.patch.object(
                acceptance, "_terminate_launcher_tree", side_effect=cleanup
            ) as terminate,
        ):
            report = acceptance.run_live(
                "FAKE",
                io.BytesIO(),
                ready_timeout=1.0,
                ping_timeout=1.0,
                model_timeout=1.0,
                negative_grace=0.001,
                launcher=HERE / "run_board_smoke.ps1",
                launcher_log=io.BytesIO(),
            )

        self.assertFalse(report["passed"])
        self.assertIn("did not exit within 10.0s", report["failure"])
        self.assertTrue(report["launcher"]["cleanup"]["succeeded"])
        self.assertTrue(fake.closed)
        terminate.assert_called_once_with(launcher)

    @unittest.skipUnless(os.name == "nt", "Windows taskkill contract")
    def test_hung_launcher_taskkills_tree_before_direct_kill(self) -> None:
        events: list[str] = []

        class HungLauncher:
            pid = 4321
            returncode: int | None = None

            def wait(self, timeout: float) -> int:
                events.append(f"wait:{timeout:g}")
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("powershell.exe", timeout)
                return self.returncode

            def poll(self) -> int | None:
                return self.returncode

            def kill(self) -> None:
                events.append("direct-kill")
                self.returncode = -9

        launcher = HungLauncher()

        def taskkill(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            events.append("taskkill")
            self.assertEqual(command[1:], ["/PID", "4321", "/T", "/F"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            launcher.returncode = 1
            return subprocess.CompletedProcess(command, 0, stdout=b"SUCCESS")

        with mock.patch.object(acceptance.subprocess, "run", side_effect=taskkill):
            cleanup = acceptance._terminate_launcher_tree(launcher)

        self.assertTrue(cleanup["succeeded"], cleanup)
        self.assertEqual(cleanup["taskkill_return_code"], 0)
        self.assertIn("taskkill", events)
        self.assertNotIn("direct-kill", events)

    @unittest.skipUnless(os.name == "nt", "Windows taskkill contract")
    def test_taskkill_failure_fails_closed_and_falls_back(self) -> None:
        events: list[str] = []

        class HungLauncher:
            pid = 8765
            returncode: int | None = None

            def wait(self, timeout: float) -> int:
                events.append(f"wait:{timeout:g}")
                if self.returncode is None:
                    raise subprocess.TimeoutExpired("powershell.exe", timeout)
                return self.returncode

            def poll(self) -> int | None:
                return self.returncode

            def kill(self) -> None:
                events.append("direct-kill")
                self.returncode = -9

        launcher = HungLauncher()

        def failed_taskkill(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            events.append("taskkill")
            return subprocess.CompletedProcess(command, 5, stdout=b"denied")

        with mock.patch.object(
            acceptance.subprocess, "run", side_effect=failed_taskkill
        ):
            cleanup = acceptance._terminate_launcher_tree(launcher)

        self.assertFalse(cleanup["succeeded"])
        self.assertTrue(cleanup["fallback_parent_kill"])
        self.assertLess(events.index("taskkill"), events.index("direct-kill"))
        self.assertTrue(any("taskkill returned 5" in error for error in cleanup["errors"]))


if __name__ == "__main__":
    unittest.main()
