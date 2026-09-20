# AURA-KSP v1.0.0

Code, frozen protocols, trained models and verified results for a controlled comparison of four-stream and three-stream skeleton action recognition on NTU RGB+D 120 XSet.

The artifact includes four final checkpoints, four aligned prediction files, original latency traces, per-class and per-setup tables, and CPU scripts that reproduce the fixed fusion and paired setup-bootstrap analysis. Raw NTU RGB+D source data are not redistributed.

Stream-3 achieved 52,951 correct predictions versus 52,997 for Full-4 on 59,477 examples: a difference of -0.077341 percentage points. The three prespecified aggregate non-inferiority criteria were met. Recall was not preserved uniformly across classes. The unsuccessful temporal K32 branch is included.

The 24.30–24.94% measured latency reduction refers to earlier seed-271828 models on an RTX 4090, batch size 1. The final checkpoints were not timed again. The findings do not establish SOTA performance or generalization across training seeds and hardware.

Download both the repository and research-assets ZIPs with their SHA-256 files. Extract `repository` and `research-assets` side by side, then follow `README.md`.

This research has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan (Grant No. AP23486538 Research and development of a system for recognizing images in video streams based on artificial intelligence).
