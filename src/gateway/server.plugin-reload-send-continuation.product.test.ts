import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { expect, it, vi } from "vitest";
import { readRequestText } from "../../test/helpers/gateway-integration-fixtures.js";
import { createManagedHandoffTestBinding } from "../../test/helpers/managed-handoff-isolation.js";
import { startOpenAiResponsesModelServer } from "../../test/helpers/openai-responses-model-fixture.js";
import { writeOpenAiResponsesText } from "../../test/helpers/openai-responses-sse.js";
import {
  installPrivateUpdateHandoffStore,
  writePrivateUpdateHandoffChildGuard,
} from "../../test/helpers/private-update-handoff-store.js";
import {
  awaitGateBeforeSettlement,
  createDeferred,
  withinTest,
} from "../../test/helpers/promise.js";
import { withRuntimePreload } from "../../test/helpers/runtime-preload.js";
import { subagentRuns } from "../agents/subagents/registry/subagent-registry-memory.js";
import { subscribeSubagentRunChanges } from "../agents/subagents/registry/subagent-registry-publication.js";
import { resetSubagentRegistryForTests } from "../agents/subagents/registry/subagent-registry.test-helpers.js";
import type { SubagentRunRecord } from "../agents/subagents/registry/subagent-registry.types.js";
import {
  listSessionPendingInputReceipts,
  listSessionPendingInputs,
} from "../config/sessions/session-accessor.js";
import { readSessionEntryReadOnlyInWorker } from "../config/sessions/session-entry-read-runtime.js";
import { onAgentEvent } from "../infra/agent-events.js";
import { resolvePluginArtifactDeclaredSurface } from "../plugins/capability-artifact.js";
import { resolvePluginCapabilityConsent } from "../plugins/capability-consent.js";
import { computeDeclaredSurfaceHash } from "../plugins/capability-summary.js";
import { getPluginInstance, type PluginInstanceHandle } from "../plugins/plugin-instance-scope.js";
import { watchPluginReplacement } from "../plugins/plugin-replacement-watchers.js";
import { getActivePluginRegistry } from "../plugins/runtime.js";
import { seedInstalledPluginIndex } from "../plugins/test-helpers/installed-plugin-index.js";
import { captureEnv } from "../test-utils/env.js";
import { createOpenClawTestState } from "../test-utils/openclaw-test-state.js";
import { writeToolCall } from "./gateway-provider-fixture.test-support.js";
import { createSubagentModelConfig } from "./server.subagent-stop-settlement.product.test-support.js";
import { disconnectGatewayClient, startGatewayWithClient } from "./test-helpers.e2e.js";

const PLUGIN = "send-continuation-proof";
const TOOL = "send_continuation_probe";
const ROOT = "Create a child, then deliver the original continuation exactly once.";
const CHILD = "Pause for the requester, then finish when the original continuation arrives.";
const MESSAGE = "ORIGINAL-CONTINUATION-91ce";
const DONE = "TARGET-COMPLETED-91ce";
const WAKE =
  "Continue the original task in this conversation. Do not repeat a completed operation.";

type InputItem = { type?: string; role?: string; name?: string; call_id?: string; output?: string };
function output(body: { input?: InputItem[] }, name: string): Record<string, unknown> | undefined {
  const call = body.input?.findLast((item) => item.type === "function_call" && item.name === name);
  const value = body.input?.find(
    (item) => item.type === "function_call_output" && item.call_id === call?.call_id,
  )?.output;
  return value ? (JSON.parse(value) as Record<string, unknown>) : undefined;
}

// External loader fixture: observable tool/hooks only; no task or reload implementation.
async function writePlugin(root: string, hooks: string, generationFile: string) {
  await fs.mkdir(root, { recursive: true });
  await fs.writeFile(
    path.join(root, "package.json"),
    JSON.stringify({ name: PLUGIN, version: "1.0.0", openclaw: { extensions: ["./index.cjs"] } }),
  );
  await fs.writeFile(
    path.join(root, "openclaw.plugin.json"),
    JSON.stringify({
      id: PLUGIN,
      contracts: { tools: [TOOL] },
      configSchema: { type: "object", additionalProperties: false },
    }),
  );
  await fs.writeFile(
    path.join(root, "index.cjs"),
    [
      'const fs = require("node:fs");',
      "module.exports = { id: " + JSON.stringify(PLUGIN) + ", register(api) {",
      "const file = " + JSON.stringify(generationFile) + ";",
      'const generation = fs.existsSync(file) ? Number(fs.readFileSync(file,"utf8")) + 1 : 1;',
      "fs.writeFileSync(file, String(generation));",
      'for (const event of ["agent_end","llm_output"]) api.on(event, (data, ctx) => { fs.appendFileSync(' +
        JSON.stringify(hooks) +
        ', JSON.stringify({ event, generation, runId: ctx.runId ?? data.runId }) + "\\n"); });',
      "api.registerTool({ name: " +
        JSON.stringify(TOOL) +
        ', label: "Probe", description: "Observe the plugin generation", parameters: { type: "object", properties: {} }, async execute() { return { content: [{ type: "text", text: JSON.stringify({ plugin: ' +
        JSON.stringify(PLUGIN) +
        ", generation }) }] }; } });",
      "} };",
    ].join("\n"),
  );
}

it("resumes the same paused task after an external reload without a model retry", async ({
  signal,
}) => {
  const state = await createOpenClawTestState({ label: "plugin-reload-send-continuation" });
  const savedEnv = captureEnv([
    "NODE_OPTIONS",
    "BUN_OPTIONS",
    "OPENCLAW_TEST_PRIVATE_HANDOFF_PATH",
  ]);
  const inheritedHandoff = process.env.OPENCLAW_TEST_PRIVATE_HANDOFF_PATH;
  const handoffDir = inheritedHandoff ? path.dirname(inheritedHandoff) : state.path("handoff");
  const pluginRoot = state.path("external-plugin");
  const hooksFile = state.path("hooks.log");
  const sessionKey = "agent:main:dashboard:" + randomUUID();
  const parentRunId = randomUUID();
  const paused = createDeferred<SubagentRunRecord>();
  const completed = createDeferred<SubagentRunRecord>();
  const sendResult = createDeferred<Record<string, unknown>>();
  const handoffWake = createDeferred<string>();
  const childAnnouncement = createDeferred();
  const failure = createDeferred<never>();
  void failure.promise.catch(() => {});
  let child: { runId: string; sessionKey: string } | undefined;
  let held: PluginInstanceHandle | undefined;
  let reload: Promise<unknown> | undefined;
  let unwatch = () => {};
  let spawned = false;
  let childYielded = false;
  let probed = false;
  let sent = false;
  let sendStarts = 0;
  let seenRootUserItems = 0;
  let effects = 0;
  let rootCalls = 0;
  let childCalls = 0;
  const wakes: string[] = [];
  const rootRuns = new Set<string>();
  const referenceCountsAtEffect: number[] = [];
  const watchTask = () => {
    if (!child) return;
    for (const entry of subagentRuns.values()) {
      if (entry.childSessionKey !== child.sessionKey) continue;
      if (entry.pauseReason === "sessions_yield") paused.resolve(entry);
      if (entry.endedReason === "subagent-complete") completed.resolve(entry);
    }
  };
  const stopTaskWatch = subscribeSubagentRunChanges("projection", watchTask);
  const stopEvents = onAgentEvent((event) => {
    if (event.sessionKey === sessionKey) {
      rootRuns.add(event.runId);
      if (
        event.stream === "tool" &&
        event.data.name === "sessions_send" &&
        event.data.phase === "start"
      ) {
        sendStarts++;
      }
    }
    if (
      event.runId !== parentRunId ||
      event.stream !== "tool" ||
      event.data.name !== "sessions_send" ||
      event.data.phase !== "result"
    )
      return;
    const result = event.data.result as { details?: Record<string, unknown> } | undefined;
    if (result?.details) sendResult.resolve(result.details);
  });
  let gateway: Awaited<ReturnType<typeof startGatewayWithClient>> | undefined;
  let model: Awaited<ReturnType<typeof startOpenAiResponsesModelServer>> | undefined;
  try {
    await fs.mkdir(handoffDir, { recursive: true, mode: 0o700 });
    const directory = await fs.realpath(handoffDir);
    // Reuse the standard child-guard contract, not a launcher-only directory flag.
    // Without a supplied path this case owns its unique store and descendant preload.
    const childEnv = inheritedHandoff
      ? process.env
      : withRuntimePreload(process.env, createManagedHandoffTestBinding(directory).preloadPath);
    const isolation = installPrivateUpdateHandoffStore(directory);
    if (inheritedHandoff) {
      assert.equal(
        Reflect.get(globalThis, Symbol.for("openclaw.test.privateHandoffGuard")),
        isolation.databasePath,
        "The inherited store must have an actual SQLite preload guard",
      );
    }
    const guarded = writePrivateUpdateHandoffChildGuard(
      isolation.databasePath,
      directory,
    )(childEnv);
    process.env.NODE_OPTIONS = guarded.NODE_OPTIONS;
    process.env.BUN_OPTIONS = guarded.BUN_OPTIONS;
    process.env.OPENCLAW_TEST_PRIVATE_HANDOFF_PATH = guarded.OPENCLAW_TEST_PRIVATE_HANDOFF_PATH;
    await writePlugin(pluginRoot, hooksFile, state.path("generation"));
    model = await startOpenAiResponsesModelServer({
      modelId: "reload-send-proof",
      bindError: "Expected private continuation model server",
      onRequest: async (request, response) => {
        try {
          const raw = await readRequestText(request);
          const body = JSON.parse(raw) as { input?: InputItem[] };
          const reply = (text: string) =>
            writeOpenAiResponsesText(response, {
              text,
              messageId: randomUUID(),
              responseId: randomUUID(),
            });
          if (raw.includes(ROOT)) {
            rootCalls++;
            if (!spawned) {
              spawned = true;
              writeToolCall(response, "sessions_spawn", {
                task: CHILD,
                context: "isolated",
                expectsCompletionMessage: true,
                cleanup: "keep",
              });
              return;
            }
            if (!child) {
              const receipt = output(body, "sessions_spawn");
              assert.equal(receipt?.status, "accepted");
              assert.equal(typeof receipt?.runId, "string");
              assert.equal(typeof receipt?.childSessionKey, "string");
              child = {
                runId: receipt.runId as string,
                sessionKey: receipt.childSessionKey as string,
              };
              watchTask();
            }
            if (!probed) {
              await withinTest(paused.promise, signal);
              probed = true;
              writeToolCall(response, TOOL, {});
              return;
            }
            if (!sent) {
              expect(output(body, TOOL)).toMatchObject({ plugin: PLUGIN, generation: 1 });
              const record = getActivePluginRegistry()?.plugins.find((item) => item.id === PLUGIN);
              assert(record, "The actual Gateway must publish the external plugin");
              held = getPluginInstance(record);
              assert(held, "The caller must hold the actual loaded instance");
              expect(held.retentionSnapshot().total).toBeGreaterThan(0);
              const replacement = createDeferred();
              unwatch = watchPluginReplacement(held, () => replacement.resolve());
              assert(gateway);
              reload = gateway.client.request("plugins.reload", {
                plugins: [{ pluginId: PLUGIN }],
              });
              await withinTest(
                awaitGateBeforeSettlement(
                  replacement.promise,
                  reload,
                  "Reload settled without reserving the held instance",
                ),
                signal,
              );
              expect(held.replacementPending).toBe(true);
              sent = true;
              writeToolCall(response, "sessions_send", {
                sessionKey: child.sessionKey,
                message: MESSAGE,
                mode: "resume",
                timeoutSeconds: 0,
              });
              return;
            }
            // A completion can follow the handoff in the same request. Inspect all
            // incoming items, but never recount their history on a later model call.
            const items = body.input ?? [];
            const call = items.findLast(
              (item) => item.type === "function_call" && item.name === "sessions_send",
            );
            const outputIndex = items.findIndex(
              (item) => item.type === "function_call_output" && item.call_id === call?.call_id,
            );
            const users =
              outputIndex < 0
                ? []
                : items.slice(outputIndex + 1).filter((item) => item.role === "user");
            for (const item of users.slice(seenRootUserItems)) {
              const text = JSON.stringify(item);
              for (let count = text.split(WAKE).length - 1; count > 0; count--) {
                wakes.push(text);
                handoffWake.resolve(text);
              }
              if (text.includes(DONE)) childAnnouncement.resolve();
            }
            seenRootUserItems = users.length;
            // Never retry, including on a refusal or either host-originated input.
            reply("Caller finished without retrying the send.");
            return;
          }
          if (raw.includes(CHILD)) {
            childCalls++;
            if (raw.includes(MESSAGE)) {
              effects++;
              referenceCountsAtEffect.push(held?.retentionSnapshot().total ?? -1);
              reply(DONE);
            } else if (!childYielded) {
              childYielded = true;
              writeToolCall(response, "sessions_yield", {
                waitFor: "message",
                acknowledgment: "Waiting for the original continuation.",
              });
            } else {
              reply("NO_REPLY");
            }
            return;
          }
          reply("NO_REPLY");
        } catch (error) {
          failure.reject(error);
          response.destroy(error instanceof Error ? error : undefined);
        }
      },
    });
    const base = createSubagentModelConfig(state.workspaceDir, {
      baseUrl: model.baseUrl + "/v1",
      modelId: "reload-send-proof",
      name: "Reload send proof",
      runtime: true,
    });
    const cfg = {
      ...base,
      skills: { load: { watch: false } },
      plugins: {
        enabled: true,
        allow: [PLUGIN],
        load: { paths: [pluginRoot] },
        slots: { memory: "none" },
        entries: { [PLUGIN]: { enabled: true, hooks: { allowConversationAccess: true } } },
      },
      tools: { profile: "full" as const, codeMode: false, toolSearch: false },
      gateway: { controlUi: { enabled: false } },
    };
    await state.writeConfig(cfg);
    await seedInstalledPluginIndex(
      { [PLUGIN]: { source: "path", sourcePath: pluginRoot, installPath: pluginRoot } },
      { env: process.env, config: cfg },
    );
    await resolvePluginCapabilityConsent({
      config: cfg,
      pluginId: PLUGIN,
      env: process.env,
      acknowledge: {
        reviewToken: computeDeclaredSurfaceHash(
          resolvePluginArtifactDeclaredSurface(pluginRoot, process.env, { config: cfg }),
        ),
      },
    });
    gateway = await startGatewayWithClient({
      cfg,
      configPath: state.configPath,
      token: "reload-send-proof-token",
      scopes: ["operator.admin"],
    });
    await withinTest(gateway.server.startupSettled, signal);
    await gateway.client.request("chat.send", {
      sessionKey,
      message: ROOT,
      idempotencyKey: parentRunId,
      deliver: false,
    });
    const result = await withinTest(Promise.race([sendResult.promise, failure.promise]), signal);
    const callerSettlement = await withinTest(
      gateway.client.request("agent.wait", { runId: parentRunId, timeoutMs: 30_000 }),
      signal,
    );
    expect(callerSettlement).toMatchObject({ status: "ok" });
    assert(reload);
    const reloadResult = await withinTest(reload, signal);
    expect(reloadResult).toMatchObject({ ok: true, restartRequired: false, pluginIds: [PLUGIN] });
    assert(child && held);
    const original = await withinTest(paused.promise, signal);
    // A flat refusal has no future owner to await. Observe its actual settled target,
    // so RED is the lost effect, not merely the missing additive output fields.
    if (result.status === "started") {
      await withinTest(
        Promise.race([
          Promise.all([completed.promise, handoffWake.promise, childAnnouncement.promise]),
          failure.promise,
        ]),
        signal,
      );
    }
    // On the retained path, both expected inputs have reached the model before this snapshot.
    const rootSettlements = await withinTest(
      Promise.all(
        [...rootRuns].map((runId) =>
          gateway!.client.request("agent.wait", { runId, timeoutMs: 30_000 }),
        ),
      ),
      signal,
    );
    for (const settlement of rootSettlements) {
      expect(settlement).toMatchObject({ status: "ok" });
    }
    const target = [...subagentRuns.values()].find(
      (entry) => entry.childSessionKey === child.sessionKey,
    );
    const entry = await readSessionEntryReadOnlyInWorker({
      agentId: "main",
      sessionKey: child.sessionKey,
    });
    assert(entry?.sessionId, "The real spawned child must have its own session");
    const scope = { agentId: "main", sessionKey: child.sessionKey, sessionId: entry.sessionId };
    assert.equal(typeof result.runId, "string");
    const receipts = await listSessionPendingInputReceipts(scope, {
      runIds: [result.runId as string],
    });
    const pending = await listSessionPendingInputs(scope);
    const hooks = (await fs.readFile(hooksFile, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as { event: string; runId?: string });
    const evidence = {
      result,
      reloadResult,
      callerSettlement,
      rootCalls,
      childCalls,
      sendStarts,
      effects,
      oldReferences: held.retentionSnapshot().total,
      target: {
        taskRunId: target?.taskRunId,
        runId: target?.runId,
        pauseReason: target?.pauseReason,
        endedReason: target?.endedReason,
      },
      receipts,
      pending,
      wakes: wakes.length,
    };
    console.log("reload-send-continuation", JSON.stringify(evidence));
    expect(sendStarts).toBe(1);
    expect(held.retentionSnapshot().total).toBe(0);
    expect.soft(effects, "The original continuation must execute without a model retry").toBe(1);
    expect.soft(target).toMatchObject({
      taskRunId: original.taskRunId ?? original.runId,
      endedReason: "subagent-complete",
    });
    expect.soft(target?.pauseReason).toBeUndefined();
    expect.soft(receipts).toHaveLength(1);
    expect(pending).toEqual({ items: [], total: 0 });
    expect.soft(result).toMatchObject({
      status: "started",
      runId: expect.any(String),
      retained: {
        acceptance: "not_accepted",
        reason: "plugin_reload_self_hold",
        plugins: [PLUGIN],
      },
    });
    expect.soft(wakes).toHaveLength(1);
    expect.soft(wakes[0] ?? "").toContain(result.runId as string);
    expect.soft(referenceCountsAtEffect).toEqual([0]);
    expect(
      hooks.filter((entry) => entry.event === "agent_end" && entry.runId === parentRunId),
    ).toHaveLength(1);
    expect(
      hooks.filter((entry) => entry.event === "llm_output" && entry.runId === parentRunId),
    ).toHaveLength(1);
  } finally {
    unwatch();
    stopEvents();
    stopTaskWatch();
    // Close HTTP first to release any fixture request waiting on an aborted test.
    await model?.close();
    if (gateway) {
      await disconnectGatewayClient(gateway.client);
      await gateway.server.close({ reason: "reload send continuation proof complete" });
    }
    if (reload) await Promise.allSettled([reload]);
    await resetSubagentRegistryForTests({ persist: false });
    vi.restoreAllMocks();
    savedEnv.restore();
    await state.cleanup();
  }
});
