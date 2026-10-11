# Implementation guide

## 1. Project flow and layout

Development began with the exploratory design in `notebooks/initial_design.ipynb`. The reusable pipeline was then implemented from that design in `src/`, including data ingestion, feature construction, model fitting, evaluation, routing and explanations. The `src/api.py` module also provides FastAPI endpoints for health checks and single/batch scoring. The automated test suite is in `tests/`.

The command-line pipeline writes generated run products to `artifacts/`. Notebook-specific tables, plots, predictions and its model output are written separately to `outputs/notebook_outputs/`. Keep these destinations distinct: the notebook supports the original exploration/design workflow, while the `src/` package provides the reusable CLI pipeline and API.

## 2. Start with cleaning and the scoring unit

One incoming carrier claim is one prediction. Its procedure/provider fields are repeated across up to 13 line positions. Preserve alignment by suffix: procedure `_1`, provider `_1`, line diagnosis `_1`, amount `_1`. Do not independently aggregate columns and lose which provider performed which service. Codes and IDs stay strings to preserve zeros. Both carrier archives are concatenated. Exact repeated records are ingestion duplicates; repeated service patterns remain potential review signals. Bad dates and empty procedure claims are quarantined, not quietly converted to zeros.

Select a random fixed-seed set of members, then retain all their claims. This preserves histories. Sampling random isolated claims undercounts previous member/provider activity. Even member sampling thins provider histories: interpret provider volumes within the sample, not as the provider's real total workload.

## 3. Analyse before fitting

The pipeline exports missing diagnosis, amount quantiles, line-count statistics, monthly claims and claims per member. Before submission, inspect these outputs and extend EDA with original missing provider/code/amount rates, leading diagnosis/procedure distributions and repeated service patterns. An expensive claim is not automatically an adverse claim. Peer comparisons must compare like procedures and include sparse-group fallback. The displayed amounts are simulated submission amounts. Actual source allowed charges only provide a simulation scale.

## 4. Define outcomes, not queues

A model should estimate clean, denial-risk and FWA-risk probabilities. A queue is a business decision: even a modest risk might merit review if its loss is large. Synthetic labels are necessary because real outcomes are absent. Use multiple noisy mechanisms rather than a deterministic rule. The precise distributions are documented in README and metadata. Noise creates ambiguity; it does not turn generated outcomes into real evidence. Keep `scenario`, `target` and source lineage out of model features.

## 5. Freeze reference statistics and query history as of scoring

Initial 2008 claims create procedure medians and diagnosis/procedure co-occurrence. Those records are warm-up, not training. Every rolling lookup counts dates strictly before the scoring day. This makes it impossible for a future claim or another same-day claim to change an earlier score. Features include member 30-day frequency, provider 7/30-day frequency, provider burst ratio and prior matching submissions. Unknown procedures fall back to the overall warm-up median; unknown providers start with no stored history. No current claim payment or allowed amount is available pre-adjudication, so those fields are excluded.

## 6. Encode, impute and scale appropriately

Numerics get training-fitted median imputation. Logistic regression receives StandardScaler because coefficient optimization and regularization depend on scale. Medical code categories get capped one-hot encoding, with rare/unseen handling. Gradient boosting uses numeric values without scaling because its splits do not require equal units. Both pipelines fit transforms only on training data. ICD-9 category grouping preserves V/E prefixes. HCPCS groups are broad demonstration buckets, not clinical equivalence classes.

## 7. Fit and calibrate

Weighted logistic regression is the baseline. Weighted histogram boosting is the stronger nonlinear candidate; unweighted boosting checks the imbalance choice. No SMOTE is used. Moderate class weights amplify errors on rare outcomes without overwhelming the majority class. Temperature scaling adjusts the probability distribution using a separate naturally distributed temporal set. Inspect the reliability chart: calibration is not automatically good just because a calibrator is fitted. Model/operating policy selection uses a later validation period; test is untouched.

## 8. Route using cost and capacity

Compute the expected cost of each action. Review effectiveness is less than perfect, so review still leaves residual adverse loss. Set capacities and costs in config.json. Batch greedy allocation selects the largest positive loss reductions under the separate review budgets. The selected extra review penalty is validation-chosen. Output daily queue counts to verify workload. Production needs a transactional allocator because independent single-claim API calls cannot share a daily capacity budget safely.

## 9. Evaluate and explain

Average precision is suitable for rare positives; accuracy could reward predicting clean for everything. Precision and recall at available review capacity quantify the operating policy. Auto-approved adverse rate measures the unreviewed risk. Test cost is computed using synthetic labels plus assumed reviewer success, and compared to reviewing everything/nothing and a transparent rules reference. Calibration and sensitivity expose uncertainty in assumed costs. Reasons are ranked by local feature replacement effect on predicted loss, translated into actual claim/history statements. This is association, not proof or causal explanation.

## 10. Serve and test

The FastAPI request contains only submission-time claim facts, not target labels or computed provider statistics. The API uses the persisted feature store and exact offline feature functions. `/score-batch` applies daily capacities to the supplied batch. The immutable snapshot deliberately avoids mutation and provides deterministic repeated scoring. Tests verify leading zeros, schema ingestion, history cutoff, capacity, validation and offline/online parity.

## 11. Review the submission

Run on actual CMS files, inspect `class_balance.csv`, `evaluation.csv`, `calibration_bins.csv`, `sensitivity.csv`, `review_explanations.json` and the generated technical note. Be ready to explain every assumption and why generated outcomes cannot support real fraud-performance claims. Submit the source with README and technical note; remove demo results or clearly mark them if you have not rerun with CMS data.
