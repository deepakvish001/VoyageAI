# Model promotion workflow

**Guide ID:** 083  
**Category:** mlops  
**Project:** VoyageAI

## Objective

Require evidence and approval before release.

## MLOps Context

VoyageAI requires traceable experiments, versioned model artifacts and responsible prediction delivery. A released model should connect to its code, data, configuration, metrics, limitations and rollback procedure.

## Operational Guidance

1. Define the outcome, accountable owner and evidence required for completion.
2. Link each experiment and artifact to code, data, configuration and metrics.
3. Promote models only through explicit acceptance criteria.
4. Monitor input distribution, latency, failures and labelled quality when available.
5. Keep rollback, backup and incident procedures executable.
6. Bound cloud costs and inference resource use.
7. Communicate intended use, uncertainty and prohibited uses clearly.

## Acceptance Criteria

- Experiments and models have stable identifiers.
- Promotion evidence includes baseline comparison and ONNX parity.
- Released artifacts include metadata and checksums.
- Drift and operational alerts have owners and response guidance.
- Retraining is reproducible and independently reviewed.
- Deployment can roll back to a known artifact.
- Product messaging avoids timetable or operational guarantees.
- Retirement and deprecation remove stale artifacts safely.

## Verification

Review the guide against the current CSV, training script, inference script, dependency pins and ONNX artifact. Implementation changes should demonstrate reproducibility, model identity, representative inference and rollback readiness.

## Review Scope

Keep implementation independently reviewable. Submit unrelated model tuning, platform migrations and formatting changes separately.
