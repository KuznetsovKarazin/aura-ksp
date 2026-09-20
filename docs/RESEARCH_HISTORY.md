# Research decisions and retained negative evidence
1. Earlier AURA-HAR work developed data handling, multi-stream baselines and adaptive/temporal alternatives. NTU60 XSub was already used; it is not a fresh test for this release.
2. The AURA-KSP temporal campaign tested skeleton acquisition reductions. K32 did not pass the required gate. Logical read savings alone did not establish useful end-to-end acceleration. `HISTORICAL_TEMPORAL_TRADEOFF.csv` and `K32_INPUT_ONLY_SAVING_BOUND.csv` retain this evidence.
3. Stream screening motivated dropping bone-motion at K64. This was an exploratory choice followed by a frozen quality protocol, not a hypothesis preceding all development.
4. Paired GPU measurements with seed-271828 models passed the 20% latency-saving gate in two cache regimes.
5. A separate quality confirmation and the final frozen official-train / XSet-test experiment followed. Final training included the old validation samples solely for training, without validation checkpoint selection. The selected final checkpoint was epoch 65.
6. Nine final jobs completed: four training, four stream evaluations, one comparison. A single recorded final test event bound the four checkpoints and protocol. The two systems shared their three retained models.
7. Independent read-only audit verified selected final outputs and reproduced fusion, predictions, confusion matrices and all 10,000 bootstrap draws. No retraining or additional test forward was used in auditing or packaging.

Historical negative evidence is retained, while obsolete handoff ZIPs, repeated drafts, provider logs, resumable epoch tensors and caches are excluded. The release is scoped to the paper's AURA-KSP claim, not every earlier AURA-HAR experiment.
