# THUNDER benchmark verification

Verified October 17, 2026 against the [official THUNDER leaderboard](https://mics-lab.github.io/thunder/leaderboards/), linked from the [authors’ repository](https://github.com/MICS-Lab/thunder). Source snapshots are cached locally in the ignored `.firecrawl/` directory.

The BACH column of the linear-probing F1-score table reports CONCH 84.4, UNI2-H 84.7, and MUSK 69.9. The zero-shot classification table reports CONCH 56.1 and MUSK 62.5. These match the reference values already recorded in `config/settings.yaml`; no reference score was changed.

These are published benchmark results, not PathQueue measurements. The comparison remains explicitly labeled **same split, different protocol**: our pipeline uses tiles, test-time augmentation, and training-fold model selection. The leaderboard comparison does not establish protocol equivalence, clinical validation, or superiority. PathQueue results carry their own measured confidence intervals; no benchmark interval is invented.
