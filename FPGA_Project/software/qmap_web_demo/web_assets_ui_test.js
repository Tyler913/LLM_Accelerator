"use strict";

const assert = require("node:assert/strict");
const ui = require("./web/app.js");

assert.deepEqual(ui.parseTokenIds("0, 42  151935\n"), [0, 42, 151935]);
assert.deepEqual(ui.parseTokenIds("0007"), [7]);
assert.throws(() => ui.parseTokenIds(""), /at least one/i);
assert.throws(() => ui.parseTokenIds("-1"), /unsigned integer/i);
assert.throws(() => ui.parseTokenIds("1.5"), /unsigned integer/i);
assert.throws(() => ui.parseTokenIds("151936"), /between 0 and 151935/i);
assert.throws(
  () => ui.parseTokenIds(Array.from({ length: 257 }, (_, index) => index).join(",")),
  /at most 256/i
);

assert.equal(ui.parseMaxNew("1"), 1);
assert.equal(ui.parseMaxNew("256"), 256);
assert.throws(() => ui.parseMaxNew("0"), /1 to 256/i);
assert.throws(() => ui.parseMaxNew("2.0"), /1 to 256/i);
assert.throws(() => ui.parseMaxNew("257"), /1 to 256/i);

assert.deepEqual(
  ui.buildGenerateRequest("prompt", "The future of FPGA is", "", "8"),
  { prompt: "The future of FPGA is", max_new_tokens: 8 }
);
assert.deepEqual(
  ui.buildGenerateRequest("tokens", "ignored", "785,3853,315,89462,374", "2"),
  { tokens: [785, 3853, 315, 89462, 374], max_new_tokens: 2 }
);
assert.throws(() => ui.buildGenerateRequest("prompt", "", "", "1"), /non-empty/i);
assert.throws(
  () => ui.buildGenerateRequest("prompt", "你".repeat(1366), "", "1"),
  /4096 bytes/i
);

const compactBody = ui.serializeGenerateRequest({ prompt: "hello", max_new_tokens: 2 });
assert.deepEqual(JSON.parse(compactBody), { prompt: "hello", max_new_tokens: 2 });
assert.throws(
  () => ui.serializeGenerateRequest({ prompt: "\n".repeat(4096), max_new_tokens: 1 }),
  /8192-byte/i
);

assert.equal(
  ui.decodeOutputBytes(Uint8Array.from([0x20, 0x61, 0xe4, 0xbd, 0xa0])),
  " a你"
);
assert.equal(ui.decodeOutputBytes(Uint8Array.from([0xe4, 0xbd])), "�");
assert.equal(ui.decodeOutputBytes(Uint8Array.from([0xe4, 0xbd, 0xa0])), "你");
assert.equal(ui.isActiveState("queued"), true);
assert.equal(ui.isActiveState("running"), true);
assert.equal(ui.isActiveState("done"), false);
assert.equal(ui.isTerminalState("done"), true);
assert.equal(ui.isTerminalState("error"), true);
assert.equal(ui.isTerminalState("idle"), false);
assert.equal(
  ui.errorMessage({ error: { code: "job_busy", message: "already active" } }, "fallback"),
  "already active"
);
assert.equal(ui.errorMessage({ error: { code: "job_busy" } }, "fallback"), "job_busy");
assert.equal(ui.errorMessage(null, "fallback"), "fallback");

assert.equal(ui.healthFailureAction(true, true, 0, false, false), "retry");
assert.equal(ui.healthFailureAction(true, true, 1, false, false), "offline");
assert.equal(ui.healthFailureAction(false, true, 0, false, false), "offline");
assert.equal(ui.healthFailureAction(true, true, 0, true, false), "offline");
assert.equal(ui.healthFailureAction(true, true, 0, false, true), "stale");

console.log("PASS Web UI pure logic tests");
