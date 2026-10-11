# Model review notes

## Labels and how much to trust results

There are no verified denial or fraud outcomes in this dataset, so labels are simulated, not observed. Each claim is assigned a noisy outcome (clean, denial risk, or FWA risk) based on a generated scenario: ordinary claim, amount inflation, repeat submission, unusual diagnosis/procedure pairing, or provider submission burst. Every scenario can yield a clean label, and even ordinary claims can be adverse. Scenario and label metadata are excluded from model features.

This setup lets us test the pipeline end to end, but metrics only describe recovery of these simulated labels. They are not evidence of real denial/fraud performance, clinical validity, or actual savings. The CMS source is also synthetic; submitted amounts and submission dates are augmented assumptions.

## Feature design: diagnosis and procedure codes

**The current source data and implementation use ICD-9, not ICD-10.** Diagnosis codes are retained as strings and grouped into broad leading-character/three-character categories, preserving ICD-9 V/E prefixes. HCPCS procedure codes are kept as strings and mapped to broad demonstration groups (for example, evaluation/management, radiology, laboratory, and medicine); these are not validated clinical or coverage groupings.

Features include diagnosis/procedure groups, diagnosis/procedure pairing rarity calculated from the warm-up period, and a procedure amount-to-historical-median ratio. Code strings preserve leading zeros. The model also uses claim complexity, member/provider prior activity, repeat-service matches, age, and missingness indicators. Pair rarity is an association signal only; it does not establish medical incompatibility or fraud.

## Class imbalance

The pipeline uses moderate class weights (clean 1, denial 2, FWA 4) and compares weighted logistic regression and weighted histogram gradient boosting. An unweighted boosting model is included as an imbalance ablation. Probabilities are temperature-scaled on a separate, chronological calibration period.

SMOTE was rejected: interpolating synthetic points across mixed numerical and categorical claim/code features would create implausible claims. The labels are already synthetic, so oversampling them would not create more reliable evidence.

## Evaluation and cost comparison

The pipeline uses adverse-outcome average precision (area under the precision-recall curve) and FWA average precision because positive classes are rare. It also reports review precision/recall and FWA senior-review recall at the selected routing policy, plus log loss, multiclass Brier score, calibration, and expected cost per 1,000 claims. The cost policy uses explicit assumed losses, review costs, reviewer detection rates, and daily capacities rather than a universal probability cutoff.

For the current generated demo artifacts, gradient boosting was selected on policy-validation expected cost. On the held-out synthetic test set:

| Measure | Result |
|---|---:|
| Adverse average precision | 0.386 |
| FWA average precision | 0.146 |
| Precision among routed claims | 0.106 |
| Recall among adverse claims routed to review/high-risk | 0.894 |
| FWA recall routed to senior/high-risk | 0.687 |
| Claims routed to review or high-risk | 71.1% |
| Expected cost per 1,000 claims | $27,853 |
| Review nothing expected cost per 1,000 | $58,330 |
| Review everything expected cost per 1,000 | $34,246 |

Under these simulated labels and assumed costs, the selected policy is about $30,477 per 1,000 below reviewing nothing and $6,393 below reviewing everything. Reviewing everything is an unconstrained reference and ignores review capacity. These are not real-world savings. Metrics and costs are sensitive to the synthetic label process and the assumed losses, review effectiveness, and capacity. The held-out test set is for reporting, not model or threshold selection.

## Explainability and reviewer use

For review/high-risk claims, the service returns up to three local reasons based on replacing one feature at a time with a training reference value and measuring the resulting change in predicted loss. Reasons can help reviewers prioritize checks—for example, inspect a high amount relative to the procedure reference, confirm a prior matching submission, or examine recent provider volume.

These are approximate model attributions, not causal explanations or proof of fraud. Correlated features and unrealistic one-feature replacements can distort their apparent contribution. A rare diagnosis/procedure pairing is not proof of incompatibility. Reviewers should use reasons to guide verification, not as an automatic adverse determination.

## Deployment and monitoring

Use a hybrid workflow: score eligible claims in batches on arrival (or a scheduled frequent batch) so one allocator can respect daily review capacity; the FastAPI `/score` endpoint can support interactive single-claim scoring, but it does not enforce shared capacity. `/score-batch` applies capacity only within the submitted request. Production needs a durable, transactional queue and a point-in-time history store that updates as claims are received.

Deploy behind authentication and authorization, validate requests, protect sensitive claim data, and version the model, policy, feature definitions, and code sets. Monitor input schema failures, missingness, unseen/changed codes, claim and score distributions, calibration and performance when adjudicated outcomes mature, subgroup behavior, queue volumes, capacity use, overrides, and reviewer turnaround. Compare these by time and relevant populations; alert on material drift. Maintain effective dates and mappings for ICD and HCPCS releases, measure code novelty and frequency changes, and validate/retrain features when a code-set update changes their meaning.

Capture reviewer decisions and final adjudication as labels with timestamps, reviewer/appeal status, and provenance. Preserve the original model score and decision; do not overwrite predictions with later outcomes. Since only routed claims receive intensive review, audit a sample of auto-approved claims and account for this selection bias before using feedback as training labels. Retrain only after enough adjudicated outcomes accumulate and a time-based challenger evaluation shows reliable improvement; recalibrate as needed, retain a rollback version, and monitor post-release performance.

An audit trail should retain the input/data source reference or a privacy-safe claim key, data and feature snapshot/version, code-set version, model and policy versions, probabilities, selected action, explanation, timestamp, capacity decision, reviewer actions, final outcome, and any override—with access controls, retention rules, and reproducible model/config artifacts. A payer or regulator would expect traceability, reproducibility, change approval, validation evidence, and a clear record of human review and adverse-action rationale.

## What I would do with another week

1. Replace simulated targets with adjudicated denial and investigation outcomes if authorized data is available; otherwise keep the work explicitly as a pipeline demonstration.
2. Validate ICD-9/HCPCS mappings, code-set effective dates, and amount references with domain experts; test missing and previously unseen codes.
3. Add subgroup and temporal robustness checks, uncertainty intervals, and more rigorous probability calibration.
4. Build a transactional shared-capacity queue, append-only audit logging, access controls, and monitoring/alerting around drift and delayed outcomes.
5. Run an end-to-end reproducibility check from raw input to model bundle and API response, documenting exact data versions, assumptions, and limitations.
