# Model card

**Guide ID:** 040  
**Category:** modeling  
**Project:** VoyageAI

## Objective

Document intended use, limitations, metrics and ethical risks.

## ML Context

VoyageAI uses a small Vande Bharat route dataset, a distance-based Gradient Boosting regressor and ONNX Runtime inference. Improvements must distinguish research evidence from assumptions and preserve training-inference consistency.

## Engineering Guidance

1. Define the modelling question and an appropriate simple baseline.
2. Specify features, transformations, configuration and reproducible seeds.
3. Use evaluation appropriate for small data and report variability.
4. Prevent target leakage and protect holdout evidence.
5. Compare native and ONNX predictions within a stated tolerance.
6. Version artifacts with metadata, metrics and provenance.
7. Document limitations and avoid causal or operational claims.

## Acceptance Criteria

- Feature definitions and units are unambiguous.
- The same transformations apply in training and inference.
- Evaluation includes a meaningful baseline.
- Metrics are reported in interpretable units.
- Data splitting and tuning do not contaminate the holdout.
- ONNX inputs, outputs and compatibility are documented.
- Artifact promotion has measurable acceptance thresholds.
- Rollback to a known model is possible.

## Verification

Review against the current dataset, train.py, inference.py, scikit-learn model and ONNX artifact. Implementation changes should include deterministic tests, evaluation output and native-versus-ONNX parity evidence.

## Review Scope

Keep implementation focused on this topic. Submit unrelated model changes, dependency upgrades and formatting separately.
