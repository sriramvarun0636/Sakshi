# HumAID text-only transfer · full classifier seed 0

The frozen full-model seed 0 checkpoint was evaluated on the fixed HumAID test role using
tweet text only. The overlap exclusion removes 45 test tweets shared with CrisisMMD,
leaving **8,112** posts: 2,537 from events unseen during CrisisMMD training and 5,575
from events shared by name. No HumAID row was used to train or tune the classifier.
Prediction and metric hashes are recorded in
`runs/humaid/full-s0-20261025/complete.json`; the run uses the official frozen seed 0
checkpoint `full-s0-20261024-062724`.

| Event group | Mapping | n | Informative macro-F1 [95% CI] | Humanitarian macro-F1 [95% CI] |
|---|---|---:|---:|---:|
| Unseen | Strict | 1,845 | 0.675 [0.636, 0.715] | 0.664 [0.637, 0.689] |
| Unseen | Full | 2,537 | 0.532 [0.511, 0.555] | 0.584 [0.565, 0.600] |
| Shared | Strict | 4,591 | 0.670 [0.647, 0.692] | 0.598 [0.584, 0.613] |
| Shared | Full | 5,575 | 0.613 [0.595, 0.631] | 0.574 [0.561, 0.586] |

The **strict** mapping keeps HumAID labels with direct SAKSHI counterparts and excludes
`caution_and_advice`, `sympathy_and_support`, and `requests_or_urgent_needs`. The
**full** mapping retains those rows, mapping the first two to
`other_relevant_information` and urgent needs to
`rescue_volunteering_or_donation_effort`. The strict and full scores therefore use
different denominators and should not be read as a before/after tuning comparison.
Both mappings use 1,000 fixed-seed item bootstrap draws for the displayed intervals.

The full mapping lowers informative macro-F1 particularly on unseen events. This is a
cross-dataset robustness result under an explicit semantic mapping, not a direct
like-for-like comparison with CrisisMMD's official agreed-label test. HumAID has no
image input in this evaluation, so severity and disaster-type heads remain unavailable.
This is one trained seed; multi-seed transfer uncertainty is not measured here.
