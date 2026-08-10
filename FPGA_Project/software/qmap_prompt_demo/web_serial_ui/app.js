"use strict";

const BAUD_RATE = 115200;
const VOCAB_SIZE = 151936;

const ui = {
  connect: document.querySelector("#connectButton"),
  generate: document.querySelector("#generateButton"),
  clear: document.querySelector("#clearButton"),
  help: document.querySelector("#helpButton"),
  inputMode: document.querySelector("#inputMode"),
  promptInput: document.querySelector("#promptInput"),
  promptLabel: document.querySelector("#promptLabel"),
  promptHint: document.querySelector("#promptHint"),
  maxNew: document.querySelector("#maxNewTokens"),
  console: document.querySelector("#console"),
  tokens: document.querySelector("#tokens"),
  generatedText: document.querySelector("#generatedText"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  runState: document.querySelector("#runState"),
  compatibility: document.querySelector("#compatibility"),
};

let port = null;
let writer = null;
let reader = null;
let readTask = null;
let keepReading = false;
let lineBuffer = "";
let runActive = false;
let generatedDecoder = new TextDecoder("utf-8", { fatal: true });
let generatedText = "";

function refreshControls() {
  const connected = port !== null && writer !== null;
  ui.generate.disabled = !connected || runActive;
  ui.help.disabled = !connected || runActive;
}

function setConnection(connected, text) {
  ui.statusDot.classList.toggle("connected", connected);
  ui.statusText.textContent = text;
  ui.connect.textContent = connected ? "断开" : "连接串口";
  if (!connected) runActive = false;
  refreshControls();
}

function setRunActive(active) {
  runActive = active;
  refreshControls();
}

function setRunState(text, kind = "") {
  ui.runState.textContent = text;
  ui.runState.className = `badge ${kind}`.trim();
}

function appendConsole(text) {
  ui.console.textContent += text;
  ui.console.scrollTop = ui.console.scrollHeight;
}

function parseTokenIds(raw) {
  const fields = raw.trim().split(/[\s,]+/).filter(Boolean);
  if (fields.length === 0) throw new Error("请输入至少一个 token ID。");
  if (fields.length > 256) throw new Error("prompt token 数不能超过 256。");
  return fields.map((field) => {
    if (!/^\d+$/.test(field)) throw new Error(`不是十进制 token ID：${field}`);
    const value = Number(field);
    if (!Number.isSafeInteger(value) || value < 0 || value >= VOCAB_SIZE) {
      throw new Error(`token ID 越界：${field}`);
    }
    return value;
  });
}

function resetGeneratedOutput() {
  generatedDecoder = new TextDecoder("utf-8", { fatal: true });
  generatedText = "";
  ui.generatedText.textContent = "等待生成文本……";
}

function decodeHexBytes(hex) {
  if (!/^(?:[0-9a-fA-F]{2})*$/.test(hex)) {
    throw new Error(`板端返回了无效的十六进制字节：${hex}`);
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function handleTokenBytes(clean) {
  const match = clean.match(/^BYTES\s+(\d+)\s+([0-9a-fA-F]*|SPECIAL|UNDECODABLE)$/i);
  if (!match) return false;
  if (/^[0-9a-fA-F]*$/.test(match[2])) {
    generatedText += generatedDecoder.decode(decodeHexBytes(match[2]), {
      stream: true,
    });
  } else if (match[2].toUpperCase() === "UNDECODABLE") {
    throw new Error(`模型输出 token #${match[1]} 没有可解码的 tokenizer 字符串。`);
  }
  ui.generatedText.textContent = generatedText || "（特殊 token，无可见文本）";
  return true;
}

function handleLine(line) {
  const clean = line.replace(/\r$/, "");
  if (handleTokenBytes(clean)) return;
  const positionalToken = clean.match(/^TOKEN\s+(\d+)\s+(\d+)\s+(-?\d+)\b/i);
  const labeledToken = clean.match(/^TOKEN\s+index=(\d+)\s+id=(\d+)\b/i);
  const tokenIndex = positionalToken?.[1] ?? labeledToken?.[1];
  const tokenId = positionalToken?.[2] ?? labeledToken?.[2];
  if (tokenId !== undefined) {
    if (ui.tokens.querySelector(".muted")) ui.tokens.replaceChildren();
    const chip = document.createElement("span");
    chip.className = "token";
    chip.textContent = `#${tokenIndex}  ${tokenId}`;
    if (positionalToken) chip.title = `score_q26=${positionalToken[3]}`;
    ui.tokens.append(chip);
    setRunActive(true);
    setRunState("RUNNING", "running");
  } else if (/^START\b/i.test(clean) || /^PREFILL\b/i.test(clean)) {
    setRunActive(true);
    setRunState("RUNNING", "running");
  } else if (/^DONE\b/i.test(clean)) {
    generatedText += generatedDecoder.decode();
    ui.generatedText.textContent = generatedText || "（没有可见文本）";
    setRunActive(false);
    setRunState("DONE");
  } else if (/^(ERROR|FAIL)\b/i.test(clean)) {
    setRunActive(false);
    setRunState("ERROR", "error");
  } else if (/^(READY|QOT READY)\b/i.test(clean)) {
    setRunActive(false);
    setRunState("READY");
  }
}

async function readLoop() {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  keepReading = true;
  while (port?.readable && keepReading) {
    const activeReader = port.readable.getReader();
    reader = activeReader;
    try {
      while (keepReading) {
        const { value, done } = await activeReader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        appendConsole(text);
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          try {
            handleLine(line);
          } catch (error) {
            appendConsole(`\n[protocol error] ${error.message}\n`);
            setRunActive(false);
            setRunState("ERROR", "error");
          }
        }
      }
    } catch (error) {
      appendConsole(`\n[serial read error] ${error.message}\n`);
      setRunActive(false);
      setRunState("ERROR", "error");
    } finally {
      activeReader.releaseLock();
      if (reader === activeReader) reader = null;
    }
  }
}

async function sendLine(line) {
  if (!writer) throw new Error("串口尚未连接。");
  await writer.write(new TextEncoder().encode(`${line}\r\n`));
  appendConsole(`\n> ${line}\n`);
}

async function connect() {
  if (port) {
    keepReading = false;
    if (reader) await reader.cancel().catch(() => {});
    if (readTask) await readTask.catch(() => {});
    readTask = null;
    if (writer) {
      await writer.close().catch(() => {});
      writer.releaseLock();
      writer = null;
    }
    await port.close();
    port = null;
    lineBuffer = "";
    resetGeneratedOutput();
    setConnection(false, "未连接");
    return;
  }

  port = await navigator.serial.requestPort();
  await port.open({ baudRate: BAUD_RATE, dataBits: 8, stopBits: 1, parity: "none" });
  lineBuffer = "";
  resetGeneratedOutput();
  writer = port.writable.getWriter();
  setConnection(true, "已连接 · 115200");
  readTask = readLoop();
}

ui.connect.addEventListener("click", async () => {
  try {
    await connect();
  } catch (error) {
    appendConsole(`[connect error] ${error.message}\n`);
    setConnection(false, "连接失败");
    port = null;
  }
});

ui.generate.addEventListener("click", async () => {
  try {
    const maxNew = Number(ui.maxNew.value);
    if (!Number.isInteger(maxNew) || maxNew < 1 || maxNew > 256) {
      throw new Error("生成数量必须在 1..256 之间。");
    }
    ui.tokens.replaceChildren();
    resetGeneratedOutput();
    setRunActive(true);
    setRunState("QUEUED", "running");
    if (ui.inputMode.value === "tokens") {
      const ids = parseTokenIds(ui.promptInput.value);
      if (ids.length + maxNew - 1 > 256) {
        throw new Error("prompt_count + max_new - 1 不能超过 256 个运行位置。");
      }
      await sendLine(`TOKENS ${maxNew} ${ids.length} ${ids.join(" ")}`);
    } else {
      const prompt = ui.promptInput.value;
      if (prompt.length === 0) throw new Error("请输入 prompt 文本。");
      if (/[\r\n]/.test(prompt)) {
        throw new Error("首版 UART 文本协议只接受单行 prompt。");
      }
      await sendLine(`PROMPT ${maxNew} ${prompt}`);
    }
  } catch (error) {
    appendConsole(`[request error] ${error.message}\n`);
    setRunActive(false);
    setRunState("ERROR", "error");
  }
});

ui.help.addEventListener("click", () => sendLine("HELP").catch((error) => {
  appendConsole(`[write error] ${error.message}\n`);
}));

ui.clear.addEventListener("click", () => {
  ui.console.textContent = "";
  ui.tokens.innerHTML = '<span class="muted">等待板端输出……</span>';
  resetGeneratedOutput();
  setRunState("IDLE");
});

ui.inputMode.addEventListener("change", () => {
  const tokenMode = ui.inputMode.value === "tokens";
  ui.promptLabel.textContent = tokenMode ? "Prompt token IDs" : "UTF-8 Prompt";
  ui.promptInput.placeholder = tokenMode
    ? "例如：374 或 785 3853 315 89462 374"
    : "例如：The future of FPGA is";
  ui.promptHint.textContent = tokenMode
    ? "使用空格、逗号或换行分隔；每个 ID 必须小于 151936。"
    : "首版串口协议接受单行 UTF-8；tokenize 和 detokenize 都在 PS 端完成。";
});

if (!("serial" in navigator)) {
  ui.compatibility.textContent = "当前浏览器不支持 Web Serial。请使用桌面版 Chrome 或 Edge，并通过 localhost 打开本页面。";
  ui.compatibility.classList.add("error");
  ui.connect.disabled = true;
} else {
  ui.compatibility.textContent = "该 GUI 使用 PC 的 Web Serial，不需要启用板卡网口；请先关闭占用同一 COM 口的 Vitis Serial Monitor。";
}

setConnection(false, "未连接");
resetGeneratedOutput();

navigator.serial?.addEventListener("disconnect", () => {
  void (async () => {
    keepReading = false;
    if (reader) await reader.cancel().catch(() => {});
    if (readTask) await readTask.catch(() => {});
    readTask = null;
    if (writer) writer.releaseLock();
    writer = null;
    port = null;
    lineBuffer = "";
    resetGeneratedOutput();
    setConnection(false, "设备已断开");
    setRunState("DISCONNECTED", "error");
  })();
});
