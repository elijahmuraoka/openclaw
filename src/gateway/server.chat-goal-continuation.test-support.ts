import { expect, it, type MockInstance } from "vitest";
import { createDeferred } from "../../test/helpers/promise.js";
import * as runs from "../agents/embedded-agent-runner/runs.js";
import type { runEmbeddedAgent as executeEmbeddedAgent } from "../agents/embedded-agent.js";
import { enqueueSessionEventForHost } from "../auto-reply/reply/session-event-handoff.js";
import { loadSessionEntry, patchSessionEntryCore } from "../config/sessions/session-accessor.js";
import { getAgentRunContext } from "../infra/agent-run-registry.js";
import { waitForGatewayActiveWork } from "../infra/gateway-active-work.js";
import type { RespondFn } from "./server-methods/types.js";

export function registerGoalAutomaticContinuationCases(
  scope: () => { agentId: string; sessionKey: string; sessionId: string; storePath: string },
  runEmbeddedAgent: MockInstance<typeof executeEmbeddedAgent>,
  rpc: (
    method: "chat.send" | "sessions.abort",
    params: Record<string, unknown>,
  ) => Promise<MockInstance<RespondFn>>,
  goalStart: (message: string) => Record<string, unknown>,
  userMessages: () => readonly unknown[],
) {
  const result = (sessionId: string, aborted?: boolean) => ({
    payloads: aborted ? [] : [{ text: "More work remains." }],
    meta: {
      durationMs: 0,
      stopReason: aborted ? "aborted" : "end_turn",
      ...(aborted === undefined ? {} : { aborted }),
      agentMeta: { sessionId, provider: "test", model: "test", usage: { input: 1, output: 1 } },
    },
  });
  const expectDrained = async () =>
    expect((await waitForGatewayActiveWork(30_000)).drained).toBe(true);
  const setGoalStatus = (status: "complete" | "paused") =>
    patchSessionEntryCore(scope(), (entry) => ({
      goal: entry.goal ? { ...entry.goal, status } : undefined,
    }));
  const start = async (message: string) =>
    expect((await rpc("chat.send", goalStart(message))).mock.calls[0]?.[0]).toBe(true);
  it("nudges consecutive normal final turns until the same Goal is complete", async () => {
    const completed = createDeferred();
    runEmbeddedAgent.mockImplementation(async (params) => {
      const calls = runEmbeddedAgent.mock.calls.length;
      if (calls === 3) {
        await setGoalStatus("complete");
        completed.resolve();
      }
      return result(params.sessionId);
    });
    await start("Finish a three-turn checklist");
    await completed.promise;
    await expectDrained();
    expect(runEmbeddedAgent).toHaveBeenCalledTimes(3);
    expect(runEmbeddedAgent.mock.calls[1]?.[0].prompt).toContain("Advance");
    expect(runEmbeddedAgent.mock.calls[2]?.[0].prompt).toContain("Advance");
    expect(userMessages()).toHaveLength(1);
    expect(loadSessionEntry(scope())?.goal).toMatchObject({
      status: "complete",
      continuationTurns: 2,
    });
  });

  // A background exec completion is admitted as a session-event turn. Its Goal
  // successor must outlive that event's occurrence, signal, and one-shot delivery,
  // while keeping an explicitly silent completion's delivery restriction.
  const completeExecDuringFirstTurn = (
    onEventTurn: () => Promise<void>,
    event: { deliver?: false } = {},
  ) => {
    const prompts: string[] = [];
    const delivery: Array<false | undefined> = [];
    runEmbeddedAgent.mockImplementation(async (params) => {
      prompts.push(params.prompt);
      delivery.push(getAgentRunContext(params.runId)?.sessionEventDelivery);
      if (runEmbeddedAgent.mock.calls.length === 1) {
        enqueueSessionEventForHost("Background exec completed", {
          ...scope(),
          ...event,
          source: "exec",
          sourceTurnId: "exec:goal-background",
        });
      } else if (runEmbeddedAgent.mock.calls.length >= 2) {
        await onEventTurn();
      }
      return result(params.sessionId);
    });
    return { prompts, delivery };
  };
  it.each([
    { name: "a background exec completion", event: {}, successorDelivery: undefined },
    {
      name: "an explicitly silent completion",
      event: { deliver: false },
      successorDelivery: false,
    },
  ] as const)("keeps pursuing after $name takes the next turn", async (testCase) => {
    const completed = createDeferred();
    const { prompts, delivery } = completeExecDuringFirstTurn(async () => {
      if (runEmbeddedAgent.mock.calls.length === 4) {
        await setGoalStatus("complete");
        completed.resolve();
      }
    }, testCase.event);
    await start("Finish after the transfer");
    await completed.promise;
    await expectDrained();
    expect(runEmbeddedAgent).toHaveBeenCalledTimes(4);
    expect(prompts[1]).toContain("Background exec completed");
    expect(prompts[2]).toContain("Advance");
    expect(prompts[3]).toContain("Advance");
    // Goal successors keep the completion's automatic-delivery restriction.
    expect(delivery).toEqual([
      undefined,
      testCase.successorDelivery,
      testCase.successorDelivery,
      testCase.successorDelivery,
    ]);
    expect(loadSessionEntry(scope())?.goal).toMatchObject({
      status: "complete",
      continuationTurns: 2,
    });
  });

  it.each([
    { name: "a background exec completion", event: {} },
    { name: "an explicitly silent completion", event: { deliver: false } },
  ] as const)("does not continue past a pause made during $name turn", async ({ event }) => {
    completeExecDuringFirstTurn(async () => {
      if (runEmbeddedAgent.mock.calls.length === 2) {
        await setGoalStatus("paused");
      }
    }, event);
    await start("Pause while the transfer reports");
    await expectDrained();
    expect(runEmbeddedAgent).toHaveBeenCalledTimes(2);
    expect(loadSessionEntry(scope())?.goal?.status).toBe("paused");
  });

  it("stops automatic goal pursuit when the operator stops a continuation", async () => {
    const { sessionKey } = scope();
    const secondStarted = createDeferred();
    const released = createDeferred();
    let continuationSignal: AbortSignal | undefined;
    let continuationRunId: string | undefined;
    runEmbeddedAgent.mockImplementation(async (params) => {
      const { sessionId: id, sessionFile: file, replyOperation: operation } = params;
      const second = runEmbeddedAgent.mock.calls.length === 2;
      if (second) {
        continuationRunId = params.runId;
        const runtimeAbort = new AbortController();
        continuationSignal = params.abortSignal
          ? AbortSignal.any([params.abortSignal, runtimeAbort.signal])
          : runtimeAbort.signal;
        const abort = () => {
          runtimeAbort.abort();
          released.resolve();
        };
        // The model is mocked; preserve the runtime's real Stop registration boundary.
        const backend = {
          kind: "embedded" as const,
          runId: params.runId,
          cancel: abort,
          abort,
          isStreaming: () => !runtimeAbort.signal.aborted,
          isAborted: () => runtimeAbort.signal.aborted,
          isCompacting: () => false,
          queueMessage: async () => {},
        };
        operation?.attachBackend(backend);
        runs.attachActiveEmbeddedRun(id, backend, sessionKey, file, "main", operation);
        secondStarted.resolve();
        try {
          await released.promise;
        } finally {
          runs.detachActiveEmbeddedRun(id, backend, sessionKey, file);
          operation?.detachBackend(backend);
        }
      }
      return result(id, continuationSignal?.aborted === true);
    });
    await start("Keep working until stopped");
    try {
      await secondStarted.promise;
      const stopped = await rpc("sessions.abort", { key: sessionKey, runId: continuationRunId });
      expect(stopped.mock.calls[0]?.[0]).toBe(true);
      expect(continuationSignal?.aborted).toBe(true);
    } finally {
      if (continuationSignal?.aborted !== true) {
        await setGoalStatus("paused");
      }
      released.resolve();
    }
    await expectDrained();
    expect(runEmbeddedAgent).toHaveBeenCalledTimes(2);
    expect(loadSessionEntry(scope())?.goal?.status).toBe("active");
  });
}
