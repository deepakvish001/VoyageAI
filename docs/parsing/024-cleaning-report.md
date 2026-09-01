# Cleaning report

**Guide ID:** 024  
**Category:** parsing  
**Project:** VoyageAI

## Objective

Record rows accepted, rejected and transformed with reasons.

## ML Context

VoyageAI predicts Vande Bharat route travel time from route data, evaluates a Gradient Boosting baseline and exports portable ONNX inference. The current dataset is small and the current model uses distance as its only feature.

## Engineering Guidance

1. Define the data, model or user outcome and measurable completion.
2. Document inputs, units, validation, ownership and failure behaviour.
3. Keep ingestion, transformations, modelling and delivery independently testable.
4. Preserve reproducibility through data versions, configuration and fixed seeds.
5. Avoid leakage and unsupported performance claims.
6. Add privacy-safe diagnostics and explicit artifact provenance.
7. Record limitations, compatibility, rollout and rollback expectations.

## Acceptance Criteria

- Input schema, units and valid ranges are explicit.
- Invalid or missing values have documented behaviour.
- Training and inference transformations remain consistent.
- Data and artifact versions are reproducible.
- Evaluation compares against an appropriate baseline.
- High-risk success and failure paths have verification.
- Documentation avoids operational or passenger guarantees.
- Migration and rollback are defined when artifacts change.

## Verification

Review this guide against VoyageAI/train.py, VoyageAI/inference.py, the CSV schema and the pinned requirements. Implementation changes should run focused tests plus training and ONNX inference in an isolated environment.

## Review Scope

Keep the implementing pull request limited to this topic. Separate unrelated dependency upgrades, broad refactors and formatting-only changes.
