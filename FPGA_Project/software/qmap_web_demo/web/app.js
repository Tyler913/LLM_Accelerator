(function qwebDemoBootstrap(root) {
  "use strict";

  const LIMITS = Object.freeze({
    maxPromptBytes: 4096,
    maxTokens: 256,
    vocabularySize: 151936,
    maxRequestBytes: 8192,
    maxOutputBytes: 32768,
    requestTimeoutMs: 5000,
    pollIntervalMs: 3000,
    healthIntervalMs: 5000
  });
  const ACTIVE_STATES = new Set(["queued", "running"]);
  const TERMINAL_STATES = new Set(["done", "error"]);
  const utf8Encoder = new TextEncoder();
  const utf8Decoder = new TextDecoder("utf-8", { fatal: false });

  class UiInputError extends Error {}

  class ApiError extends Error {
    constructor(status, code, message) {
      super(message || `HTTP ${status}`);
      this.name = "ApiError";
      this.status = status;
      this.code = code || "http_error";
    }
  }

  function parseMaxNew(value) {
    const text = String(value).trim();
    if (!/^[0-9]+$/.test(text)) {
      throw new UiInputError("Maximum new tokens must be an integer from 1 to 256.");
    }
    const parsed = Number(text);
    if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > LIMITS.maxTokens) {
      throw new UiInputError("Maximum new tokens must be an integer from 1 to 256.");
    }
    return parsed;
  }

  function parseTokenIds(value) {
    const text = String(value).trim();
    if (text.length === 0) {
      throw new UiInputError("Enter at least one token ID.");
    }
    const parts = text.split(/[\s,]+/u).filter(Boolean);
    if (parts.length > LIMITS.maxTokens) {
      throw new UiInputError("At most 256 input token IDs are accepted.");
    }
    return parts.map((part, index) => {
      if (!/^[0-9]+$/.test(part)) {
        throw new UiInputError(`Token ${index + 1} is not an unsigned integer.`);
      }
      const token = Number(part);
      if (!Number.isSafeInteger(token) || token < 0 || token >= LIMITS.vocabularySize) {
        throw new UiInputError(`Token ${index + 1} must be between 0 and 151935.`);
      }
      return token;
    });
  }

  function buildGenerateRequest(mode, prompt, tokenText, maxNewValue) {
    const maxNewTokens = parseMaxNew(maxNewValue);
    if (mode === "prompt") {
      const promptBytes = utf8Encoder.encode(String(prompt));
      if (promptBytes.length === 0) {
        throw new UiInputError("Enter a non-empty prompt.");
      }
      if (promptBytes.length > LIMITS.maxPromptBytes) {
        throw new UiInputError("The UTF-8 prompt exceeds 4096 bytes.");
      }
      return { prompt: String(prompt), max_new_tokens: maxNewTokens };
    }
    if (mode === "tokens") {
      return { tokens: parseTokenIds(tokenText), max_new_tokens: maxNewTokens };
    }
    throw new UiInputError("Choose prompt text or token IDs.");
  }

  function decodeOutputBytes(bytes) {
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    return utf8Decoder.decode(view);
  }

  function serializeGenerateRequest(request) {
    const body = JSON.stringify(request);
    if (utf8Encoder.encode(body).length > LIMITS.maxRequestBytes) {
      throw new UiInputError("The encoded JSON request exceeds the 8192-byte HTTP limit.");
    }
    return body;
  }

  function isActiveState(value) {
    return ACTIVE_STATES.has(String(value));
  }

  function isTerminalState(value) {
    return TERMINAL_STATES.has(String(value));
  }

  function errorMessage(payload, fallback) {
    if (payload && typeof payload === "object" && payload.error &&
        typeof payload.error === "object") {
      if (typeof payload.error.message === "string" && payload.error.message.length > 0) {
        return payload.error.message;
      }
      if (typeof payload.error.code === "string" && payload.error.code.length > 0) {
        return payload.error.code;
      }
    }
    return fallback;
  }

  function healthFailureAction(quiet, connected, failureStreak, protocolError, jobActive) {
    if (jobActive) {
      return "stale";
    }
    if (quiet && connected && failureStreak === 0 && !protocolError) {
      return "retry";
    }
    return "offline";
  }

  const testApi = Object.freeze({
    LIMITS,
    UiInputError,
    ApiError,
    parseMaxNew,
    parseTokenIds,
    buildGenerateRequest,
    serializeGenerateRequest,
    decodeOutputBytes,
    isActiveState,
    isTerminalState,
    errorMessage,
    healthFailureAction
  });
  if (typeof module !== "undefined" && module.exports) {
    module.exports = testApi;
  }
  root.QwebDemoTest = testApi;

  if (typeof document === "undefined") {
    return;
  }

  const elements = {
    connectionPill: document.getElementById("connectionPill"),
    connectionText: document.getElementById("connectionText"),
    refreshHealth: document.getElementById("refreshHealth"),
    form: document.getElementById("generateForm"),
    modeInputs: Array.from(document.querySelectorAll('input[name="inputMode"]')),
    promptPane: document.getElementById("promptPane"),
    tokensPane: document.getElementById("tokensPane"),
    promptInput: document.getElementById("promptInput"),
    promptByteCount: document.getElementById("promptByteCount"),
    tokenInput: document.getElementById("tokenInput"),
    tokenCount: document.getElementById("tokenCount"),
    maxNewInput: document.getElementById("maxNewInput"),
    formError: document.getElementById("formError"),
    generateButton: document.getElementById("generateButton"),
    buttonLabel: document.querySelector(".button-label"),
    submitHint: document.getElementById("submitHint"),
    jobState: document.getElementById("jobState"),
    jobId: document.getElementById("jobId"),
    promptProgress: document.getElementById("promptProgress"),
    generatedCount: document.getElementById("generatedCount"),
    stopReason: document.getElementById("stopReason"),
    progressBar: document.getElementById("progressBar"),
    outputText: document.getElementById("outputText"),
    outputByteCount: document.getElementById("outputByteCount"),
    tokenRows: document.getElementById("tokenRows"),
    runtimeMessage: document.getElementById("runtimeMessage")
  };

  const state = {
    connected: false,
    ready: false,
    remoteState: "unknown",
    activeJobId: null,
    submitting: false,
    healthPending: false,
    pollTimer: null,
    healthTimer: null,
    networkFailures: 0,
    healthFailureStreak: 0,
    expectedMaxNew: 0,
    lastOutputLength: -1
  };

  function currentMode() {
    const selected = elements.modeInputs.find((input) => input.checked);
    return selected ? selected.value : "prompt";
  }

  function setConnection(kind, text) {
    elements.connectionPill.dataset.state = kind;
    elements.connectionText.textContent = text;
  }

  function setRuntimeMessage(text, kind) {
    elements.runtimeMessage.textContent = text;
    if (kind) {
      elements.runtimeMessage.dataset.state = kind;
    } else {
      delete elements.runtimeMessage.dataset.state;
    }
  }

  function showFormError(message) {
    elements.formError.textContent = message;
    elements.formError.hidden = !message;
  }

  function remoteBusy() {
    return isActiveState(state.remoteState) && state.activeJobId === null;
  }

  function refreshControls() {
    const locked = state.submitting || state.activeJobId !== null;
    const canSubmit = state.connected && state.ready && !remoteBusy() && !locked &&
      !state.healthPending;
    elements.generateButton.disabled = !canSubmit;
    elements.promptInput.disabled = locked;
    elements.tokenInput.disabled = locked;
    elements.maxNewInput.disabled = locked;
    elements.modeInputs.forEach((input) => { input.disabled = locked; });
    elements.refreshHealth.disabled = state.healthPending || state.activeJobId !== null;

    if (state.submitting) {
      elements.buttonLabel.textContent = "Submitting…";
      elements.submitHint.textContent = "Reserving the single generation slot.";
    } else if (state.activeJobId !== null) {
      elements.buttonLabel.textContent = "Generation in progress";
      elements.submitHint.textContent = "A second request is disabled until this job finishes.";
    } else {
      elements.buttonLabel.textContent = "Generate on FPGA";
      if (!state.connected) {
        elements.submitHint.textContent = "The board service is unreachable.";
      } else if (!state.ready) {
        elements.submitHint.textContent = "The board service is not ready.";
      } else if (remoteBusy()) {
        elements.submitHint.textContent = "The board is serving another active job.";
      } else {
        elements.submitHint.textContent = "One request at a time; generation runs entirely on the board.";
      }
    }
  }

  async function fetchWithTimeout(path, options) {
    const controller = new AbortController();
    const timeout = root.setTimeout(() => controller.abort(), LIMITS.requestTimeoutMs);
    try {
      return await fetch(path, { ...options, signal: controller.signal, cache: "no-store" });
    } finally {
      root.clearTimeout(timeout);
    }
  }

  async function readJson(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      if (response.ok) {
        throw new ApiError(response.status, "invalid_json", "The board returned malformed JSON.");
      }
    }
    if (!response.ok) {
      const code = payload && payload.error && typeof payload.error.code === "string"
        ? payload.error.code
        : "http_error";
      throw new ApiError(response.status, code, errorMessage(payload, `HTTP ${response.status}`));
    }
    if (!payload || typeof payload !== "object") {
      throw new ApiError(response.status, "invalid_json", "The board returned an empty JSON response.");
    }
    return payload;
  }

  async function apiJson(path, options) {
    const response = await fetchWithTimeout(path, options || { method: "GET" });
    return readJson(response);
  }

  async function apiOutput(jobId) {
    const response = await fetchWithTimeout(`/api/generate/${jobId}/output`, { method: "GET" });
    if (!response.ok) {
      return readJson(response);
    }
    return new Uint8Array(await response.arrayBuffer());
  }

  function updateInputMode() {
    const promptMode = currentMode() === "prompt";
    elements.promptPane.hidden = !promptMode;
    elements.tokensPane.hidden = promptMode;
    showFormError("");
  }

  function updateInputCounts() {
    const promptBytes = utf8Encoder.encode(elements.promptInput.value).length;
    elements.promptByteCount.textContent = `${promptBytes} / ${LIMITS.maxPromptBytes} bytes`;
    elements.promptByteCount.classList.toggle("invalid", promptBytes > LIMITS.maxPromptBytes);
    try {
      const tokens = elements.tokenInput.value.trim().length === 0
        ? []
        : parseTokenIds(elements.tokenInput.value);
      elements.tokenCount.textContent = `${tokens.length} / ${LIMITS.maxTokens} tokens`;
      elements.tokenCount.classList.remove("invalid");
    } catch (_error) {
      elements.tokenCount.textContent = "Invalid token list";
      elements.tokenCount.classList.add("invalid");
    }
  }

  function resetResult() {
    elements.jobState.textContent = "Queued";
    elements.jobState.dataset.state = "queued";
    elements.jobId.textContent = "—";
    elements.promptProgress.textContent = "0 / 0";
    elements.generatedCount.textContent = "0";
    elements.stopReason.textContent = "—";
    elements.progressBar.style.width = "0%";
    elements.outputText.textContent = "";
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = "Waiting for the first generated bytes…";
    elements.outputText.appendChild(placeholder);
    elements.outputByteCount.textContent = "0 bytes";
    elements.tokenRows.replaceChildren();
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "No generated tokens yet.";
    row.appendChild(cell);
    elements.tokenRows.appendChild(row);
    state.lastOutputLength = -1;
  }

  function renderTokens(ids, scores) {
    elements.tokenRows.replaceChildren();
    if (ids.length === 0) {
      const row = document.createElement("tr");
      row.className = "empty-row";
      const cell = document.createElement("td");
      cell.colSpan = 3;
      cell.textContent = "No generated tokens yet.";
      row.appendChild(cell);
      elements.tokenRows.appendChild(row);
      return;
    }
    ids.forEach((tokenId, index) => {
      const row = document.createElement("tr");
      [index, tokenId, scores[index]].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.appendChild(cell);
      });
      elements.tokenRows.appendChild(row);
    });
    elements.tokenRows.lastElementChild.scrollIntoView({ block: "nearest" });
  }

  function validateStatus(payload, expectedJobId) {
    const requiredIntegers = [
      "job_id", "prompt_token_count", "prompt_tokens_consumed",
      "generated_count", "output_length"
    ];
    if (requiredIntegers.some((key) => !Number.isSafeInteger(payload[key]) || payload[key] < 0) ||
        payload.job_id !== expectedJobId || typeof payload.state !== "string" ||
        !Array.isArray(payload.generated_token_ids) ||
        !Array.isArray(payload.generated_scores_q26) ||
        payload.generated_token_ids.length !== payload.generated_count ||
        payload.generated_scores_q26.length !== payload.generated_count) {
      throw new ApiError(200, "invalid_status", "The board returned an inconsistent job status.");
    }
    return payload;
  }

  function renderStatus(status) {
    state.remoteState = status.state;
    elements.jobState.textContent = status.state;
    elements.jobState.dataset.state = status.state;
    elements.jobId.textContent = String(status.job_id);
    elements.promptProgress.textContent = `${status.prompt_tokens_consumed} / ${status.prompt_token_count}`;
    elements.generatedCount.textContent = String(status.generated_count);
    elements.stopReason.textContent = status.stop_reason || "—";
    renderTokens(status.generated_token_ids, status.generated_scores_q26);

    const promptFraction = status.prompt_token_count > 0
      ? status.prompt_tokens_consumed / status.prompt_token_count
      : 0;
    const decodeFraction = state.expectedMaxNew > 0
      ? Math.min(1, status.generated_count / state.expectedMaxNew)
      : 0;
    const progress = Math.min(100, Math.round((promptFraction * 0.35 + decodeFraction * 0.65) * 100));
    elements.progressBar.style.width = `${isTerminalState(status.state) ? 100 : progress}%`;
  }

  async function refreshOutput(jobId, reportedLength, force) {
    if (!force && reportedLength === state.lastOutputLength) {
      return;
    }
    if (reportedLength === 0) {
      elements.outputText.textContent = "";
      const placeholder = document.createElement("span");
      placeholder.className = "placeholder";
      placeholder.textContent = "Waiting for the first generated bytes…";
      elements.outputText.appendChild(placeholder);
      elements.outputByteCount.textContent = "0 bytes";
      state.lastOutputLength = 0;
      return;
    }
    const bytes = await apiOutput(jobId);
    if (bytes.length < reportedLength || bytes.length > LIMITS.maxOutputBytes) {
      throw new ApiError(200, "output_length_mismatch", "The board returned an inconsistent output length.");
    }
    elements.outputText.textContent = decodeOutputBytes(bytes);
    elements.outputByteCount.textContent = `${bytes.length} ${bytes.length === 1 ? "byte" : "bytes"}`;
    elements.outputText.scrollTop = elements.outputText.scrollHeight;
    state.lastOutputLength = bytes.length;
  }

  function schedulePoll(delay) {
    if (state.activeJobId === null) return;
    root.clearTimeout(state.pollTimer);
    state.pollTimer = root.setTimeout(pollJob, delay);
  }

  async function pollJob() {
    if (state.activeJobId === null) return;
    const jobId = state.activeJobId;
    try {
      const payload = validateStatus(
        await apiJson(`/api/generate/${jobId}`, { method: "GET" }),
        jobId
      );
      state.connected = true;
      state.ready = true;
      state.networkFailures = 0;
      setConnection(isActiveState(payload.state) ? "busy" : "online",
                    isActiveState(payload.state) ? "FPGA generating" : "Board online");
      renderStatus(payload);
      await refreshOutput(jobId, payload.output_length, isTerminalState(payload.state));

      if (payload.state === "done") {
        state.activeJobId = null;
        setRuntimeMessage(`Generation complete · ${payload.stop_reason || "stop reason unavailable"}.`, "success");
        refreshControls();
        scheduleHealth(250);
        return;
      }
      if (payload.state === "error") {
        state.activeJobId = null;
        const detail = payload.error && typeof payload.error === "object"
          ? `job ${payload.error.job_code}, session ${payload.error.session_code}, tokenizer ${payload.error.tokenizer_code}`
          : "no error codes returned";
        setRuntimeMessage(`Generation failed: ${detail}.`, "error");
        refreshControls();
        scheduleHealth(250);
        return;
      }
      if (!isActiveState(payload.state)) {
        throw new ApiError(200, "invalid_job_state", `Unexpected job state: ${payload.state}.`);
      }
      setRuntimeMessage(
        payload.state === "queued" ? "Request accepted; waiting for the PL runner." : "PL inference is running; output refreshes after each token.",
        "warning"
      );
      refreshControls();
      schedulePoll(LIMITS.pollIntervalMs);
    } catch (error) {
      if (error instanceof ApiError) {
        state.activeJobId = null;
        state.remoteState = "unknown";
        state.ready = false;
        elements.jobState.textContent = "Error";
        elements.jobState.dataset.state = "error";
        setRuntimeMessage(`Status request failed: ${error.message}`, "error");
        refreshControls();
        scheduleHealth(500);
        return;
      }

      state.connected = false;
      state.networkFailures += 1;
      setConnection("offline", "Connection lost");
      setRuntimeMessage("The board connection dropped. This job remains locked and polling will retry automatically.", "error");
      refreshControls();
      const retryDelay = Math.min(5000, LIMITS.pollIntervalMs * (2 ** Math.min(state.networkFailures, 3)));
      schedulePoll(retryDelay);
    }
  }

  function scheduleHealth(delay) {
    root.clearTimeout(state.healthTimer);
    if (state.activeJobId !== null) return;
    state.healthTimer = root.setTimeout(() => checkHealth(true), delay);
  }

  async function checkHealth(quiet) {
    if (state.healthPending || state.submitting || state.activeJobId !== null) return;
    state.healthPending = true;
    let nextHealthDelay = LIMITS.healthIntervalMs;
    if (!quiet) setConnection("checking", "Checking board");
    refreshControls();
    try {
      const health = await apiJson("/api/health", { method: "GET" });
      if (state.submitting || state.activeJobId !== null) return;
      if (health.service !== "qmap-web" || typeof health.ready !== "boolean" ||
          typeof health.job_state !== "string") {
        throw new ApiError(200, "invalid_health", "The board returned an invalid health response.");
      }
      state.healthFailureStreak = 0;
      state.connected = true;
      state.ready = health.ready;
      state.remoteState = health.job_state;
      if (isActiveState(health.job_state)) {
        setConnection("busy", "Board busy");
        setRuntimeMessage("The board already has an active generation. Wait for it to finish before submitting.", "warning");
      } else if (health.ready) {
        setConnection("online", "Board ready");
        if (!quiet) setRuntimeMessage("Board service and tokenizer are ready.", "success");
      } else {
        setConnection("offline", "Service not ready");
        setRuntimeMessage("The HTTP service answered, but the generation runtime is not initialized.", "error");
      }
    } catch (error) {
      const action = healthFailureAction(
        quiet,
        state.connected,
        state.healthFailureStreak,
        error instanceof ApiError,
        state.submitting || state.activeJobId !== null
      );
      if (action === "stale") return;
      state.healthFailureStreak += 1;
      if (action === "retry") {
        nextHealthDelay = 500;
        return;
      }
      state.connected = false;
      state.ready = false;
      state.remoteState = "unknown";
      setConnection("offline", "Board unavailable");
      if (!quiet) {
        setRuntimeMessage(`Cannot reach the board service: ${error.message}`, "error");
      }
    } finally {
      state.healthPending = false;
      refreshControls();
      scheduleHealth(nextHealthDelay);
    }
  }

  async function submitGeneration(event) {
    event.preventDefault();
    if (state.submitting || state.healthPending || state.activeJobId !== null) return;
    showFormError("");

    let request;
    let requestBody;
    try {
      request = buildGenerateRequest(
        currentMode(),
        elements.promptInput.value,
        elements.tokenInput.value,
        elements.maxNewInput.value
      );
      requestBody = serializeGenerateRequest(request);
    } catch (error) {
      showFormError(error.message);
      return;
    }

    state.submitting = true;
    state.expectedMaxNew = request.max_new_tokens;
    resetResult();
    setRuntimeMessage("Submitting a bounded generation request…", "warning");
    refreshControls();
    try {
      const accepted = await apiJson("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody
      });
      if (!Number.isSafeInteger(accepted.job_id) || accepted.job_id <= 0 ||
          !isActiveState(accepted.state)) {
        throw new ApiError(202, "invalid_acceptance", "The board returned an invalid job acceptance.");
      }
      state.activeJobId = accepted.job_id;
      state.remoteState = accepted.state;
      state.connected = true;
      state.ready = true;
      elements.jobId.textContent = String(accepted.job_id);
      elements.jobState.textContent = accepted.state;
      elements.jobState.dataset.state = accepted.state;
      setConnection("busy", "FPGA generating");
      setRuntimeMessage("Generation accepted; polling the retained job record.", "warning");
    } catch (error) {
      if (!(error instanceof ApiError)) {
        state.connected = false;
        setConnection("offline", "Board unavailable");
      } else if (error.status === 409) {
        state.remoteState = "running";
        setConnection("busy", "Board busy");
      } else {
        state.ready = false;
      }
      setRuntimeMessage(`Generation was not accepted: ${error.message}`, "error");
      showFormError(error.message);
      scheduleHealth(500);
    } finally {
      state.submitting = false;
      refreshControls();
    }
    if (state.activeJobId !== null) {
      schedulePoll(0);
    }
  }

  elements.modeInputs.forEach((input) => input.addEventListener("change", updateInputMode));
  elements.promptInput.addEventListener("input", updateInputCounts);
  elements.tokenInput.addEventListener("input", updateInputCounts);
  elements.form.addEventListener("submit", submitGeneration);
  elements.refreshHealth.addEventListener("click", () => checkHealth(false));
  root.addEventListener("online", () => checkHealth(false));
  root.addEventListener("offline", () => {
    state.connected = false;
    setConnection("offline", "Browser offline");
    refreshControls();
  });

  updateInputMode();
  updateInputCounts();
  refreshControls();
  checkHealth(false);
})(typeof globalThis !== "undefined" ? globalThis : this);
