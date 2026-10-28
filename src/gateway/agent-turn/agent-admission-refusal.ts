import type { ErrorShape } from "../../../packages/gateway-protocol/src/index.js";
import { readPreparedRuntimeSelfHold } from "../../agents/prepared-model-runtime-admission.js";
import type { PrepareAgentRunDispatchParams } from "./agent-run-admission-types.js";

export type AgentAdmissionStage =
  | "runtime-preparation"
  | "subagent-preparation"
  | "input-preparation"
  | "acceptance";
type AgentAdmissionRefusal = Readonly<{
  acceptance: "not_accepted" | "uncertain";
  stage: AgentAdmissionStage;
}>;
const refusals = new WeakMap<object, AgentAdmissionRefusal>();

/** Admission alone stamps the original guard cause after joining its cleanup. */
export function recordAgentAdmissionRefusal(
  error: unknown,
  stage: AgentAdmissionStage,
  released: boolean,
) {
  const selfHeld = readPreparedRuntimeSelfHold(error);
  if (selfHeld) {
    refusals.set(selfHeld.cause, {
      acceptance: stage === "runtime-preparation" && released ? "not_accepted" : "uncertain",
      stage,
    });
  }
}

/** Read the paired host facts; an error flag or plugin-authored JSON never supplies proof. */
export function readAgentSelfHeldRefusal(error: unknown) {
  const held = readPreparedRuntimeSelfHold(error);
  if (!held) {
    return undefined;
  }
  return {
    ...held,
    admission: refusals.get(held.cause) ?? {
      acceptance: "uncertain" as const,
      stage: "acceptance" as const,
    },
  };
}

export function createAgentPreacceptRefusal(
  params: Pick<PrepareAgentRunDispatchParams, "releaseUnacceptedReservation" | "io">,
  cleanup: (released?: boolean, failure?: string) => Promise<void>,
) {
  const reject = async (error: ErrorShape): Promise<undefined> => {
    let released = false;
    try {
      await cleanup(false, error.message);
      if (readPreparedRuntimeSelfHold(error)) {
        released = params.releaseUnacceptedReservation();
      }
    } finally {
      recordAgentAdmissionRefusal(error, reject.stage, released);
      params.io.emitAcceptance([false, undefined, error]);
    }
    return undefined;
  };
  return Object.assign(reject, { stage: "runtime-preparation" as AgentAdmissionStage });
}
