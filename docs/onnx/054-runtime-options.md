# ONNX Runtime configuration

**Guide ID:** 054  
**Category:** onnx  
**Project:** VoyageAI

## Objective

Set providers, threading and resource limits deliberately.

## Service Context

VoyageAI may evolve from local scripts into a tested prediction service. The service must validate distance inputs, load a verified ONNX model, bound resource use and communicate limitations without presenting predictions as railway guarantees.

## Engineering Guidance

1. Define stable request, response and error contracts.
2. Validate units, ranges, batch size and malformed input.
3. Load and verify one approved model during service startup.
4. Keep model calls behind independently testable domain functions.
5. Bound time, memory, concurrency and retries.
6. Add privacy-safe logs, metrics and trace identifiers.
7. Test native, ONNX and API failure paths.

## Acceptance Criteria

- API and tensor contracts are explicit.
- Invalid and implausible distances are rejected consistently.
- Readiness depends on a valid model artifact.
- Model loading does not repeat per request.
- Requests cannot create unbounded compute or memory work.
- Tests cover parsers, pipeline, ONNX parity and API failures.
- Secrets, data and artifacts have defined trust boundaries.
- Deployment and rollback can select a known model version.

## Verification

Review against train.py, inference.py, requirements.txt and the committed ONNX artifact. Implementation changes should include automated tests, a clean service startup, invalid-input checks and representative inference verification.

## Review Scope

Keep implementation limited to this topic. Separate unrelated model tuning, dependency upgrades and broad refactoring.
