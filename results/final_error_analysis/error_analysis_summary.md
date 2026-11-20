# Final False Negative Error Analysis Summary

## Verified facts

- Canonical Final Test false negatives: 2.
- Both records are from `van_Dis_2020`.
- Both records have titles and lack abstracts.
- Both records are `confirmed_non_overlap` with no Train or Validation match.
- Both records remain false negatives at the frozen Validation-selected threshold `0.016082723`.
- No explicit article-level inclusion rationale was found in repository artifacts.
- No causal or qualitative error category has been assigned automatically.

## Thesis-ready factual table

| Case | Review | Abstract available | Probability | Threshold | Probability - threshold | Train/Validation overlap | Notes |
|---|---|---|---:|---:|---:|---|---|
| 1 | van_Dis_2020 | NO | 0.0057630353 | 0.016082723 | -0.0103196877 | None confirmed | Positive label only; no article-level inclusion rationale available |
| 2 | van_Dis_2020 | NO | 0.0156764980 | 0.016082723 | -0.0004062250 | None confirmed | Positive label only; no article-level inclusion rationale available |

## Missing-abstract contingency

| Group | Included articles | False negatives |
|---|---:|---:|
| Abstract available | 460 | 0 |
| Abstract missing | 12 | 2 |

Interpretation is limited to: **Both FN records had missing abstracts.** This does not establish that missing abstracts caused the errors.

## Evidence boundary

Available evidence includes the exact title, identifiers, positive label, frozen model probability and threshold, review-level eligibility criteria, overlap status, and per-review performance. Missing evidence includes article abstracts, article-level inclusion rationales, full texts, and any repository-supported explanation of why the model assigned these probabilities.

## Source integrity

All source hashes were unchanged before and after extraction. No canonical artifact was modified.
