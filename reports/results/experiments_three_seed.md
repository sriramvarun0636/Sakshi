# Ablations and single-task baselines · three frozen seeds

Every complete variant is compared with the corresponding full-model seed on identical official test posts.
Δ is variant minus full macro-F1. Its 95% interval resamples test posts together across all three seeds
(1,000 paired draws); seed SD describes variation among the three trained runs.
The interval captures test-post sampling, not uncertainty over all possible training seeds.

## A1 no cross-attention

Frozen seeds: **1/3**.

## A2 no modality gate

Frozen seeds: **1/3**.

## A3 no modality dropout

Frozen seeds: **0/3**.

## A4 flat humanitarian head

Frozen seeds: **1/3**.

## A5 no CIB training

Frozen seeds: **0/3**.

## B4 informative only

Frozen seeds: **0/3**.

## B4 humanitarian only

Frozen seeds: **0/3**.

## B4 severity only

Frozen seeds: **0/3**.

## B4 disaster type only

Frozen seeds: **1/3**.

The experiments share fixed official selectors, but the many reported effects have no multiple-comparison correction.
