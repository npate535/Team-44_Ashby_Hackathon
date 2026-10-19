# Resource usage and GPU decision

## Update, Sunday 2026-10-18 afternoon: what the GPU was used for

The morning recommendation below (skip fine-tuning) was reversed at the user's request: use the GPUs to make the
model more accurate. The work is pre-registered and reported in [ACCURACY_ROUND2.md](ACCURACY_ROUND2.md).

- **Laptop, no GPU:** train-only cross-validation showed that larger 512-px regions beat the 256-px tiles for UNI2-h.
  The pre-registered rule adopted `uni2h512_lr`. Test macro-F1 rose from 0.830 to 0.860 (balanced accuracy 0.835 to
  0.871). The published UNI2-h probe's 84.7 lies inside our interval [67.8, 92.8], under a different protocol.
- **Delta GPU (A100, `bfqi-delta-gpu` / `gpuA100x4`), through the guarded `submit.sh`:**
  - Job 22988201: a LoRA fine-tune of UNI2-h, trained fold by fold on round 1's CV folds (train split only).
  - Job 22988274: the final fine-tuned model on all 218 train images, plus calibration/test predictions.
  - Jobs 22988986, 22989463 and 22989465: recipes B and C (fine-tuned features read by the production probe; WiSE-FT weight interpolation), with CV and final models.
  - Script: `~/pq_ft/ft.py` on Delta, mirroring `pipeline/05c_finetune.py`. The LoRA weights stay in private home
    (CC-BY-NC-ND derivative).
- **Access:** Delta was driven through Open OnDemand in the in-app browser after the user signed in. SSH from this Mac
  was not used.

- **Result (details in ACCURACY_ROUND2.md):**
  - **CV:** three fine-tuning recipes (LoRA end-to-end; LoRA + production probe; WiSE-FT weight interpolation) beat
    frozen UNI2-h read through the identical pipeline on cross-validation.
  - **Test:** the gain did not carry over to four-class accuracy. The best fine-tune scored 0.852 vs 0.858–0.860
    macro-F1, inside overlapping CIs. Every fine-tune separates invasive carcinoma perfectly (AUROC 1.000 vs 0.999).
  - **Production:** stays the 512-px probe, `uni2h512_lr`, at 86.0 macro-F1 [67.8, 92.8]. The published UNI2-h
    probe's 84.7 lies inside that interval, under a different protocol.
- **Measured GPU use (`sacct`):**
  - Fine-tuning: 133.8 A100-minutes over five jobs (22988201, 22988274, 22988986, 22989463, 22989465).
  - Event total: about 4.1 A100-hours of the 10 personal GPU-hours.
  - The guard's requested-time cap was raised from 420 to 550 minutes with the user's approval.

---

## Morning snapshot (superseded above)

Snapshot: Sunday 2026-10-18, 09:33 CDT. Compiled from repo records (`data/meta/*.json`, `docs/VERIFICATION.md`,
`docs/BUILD_PLAN.md` §G, `docs/OVERNIGHT.md`). Delta itself was not queried (no live login session), so measured
GPU-hours and the remaining allocation balance are **unverified**. To confirm, log in with `./scripts/delta.sh` and run
`accounts` and `sacct -u rbansal3 -S 2026-10-16 -X`.

## What we have used so far

### Delta GPU
- **300 GPU-minutes requested** (5 h) of the 420-minute cap enforced by `pipeline/jobs/submit.sh` (7 h of Rishabh's
  10 personal GPU-hours). These are Slurm time limits, not measured use; jobs that ended early used less.
- One A100 at a time on `bfqi-delta-gpu` / `gpuA100x4`. Data prep ran on `bfqi-delta-cpu`.
- GPU work: embedding with every encoder, prompt encoding, HoVer-Net cell segmentation.
- No GPU work since the overnight run; later fixes (e.g. tissue QC) re-ran on the laptop from cached embeddings.
- DeltaAI (backup cluster): not used.

### Hosted model servers (CONCH, MUSK, QuiltNet, PLIP)
- Essentially unused. The plan limited them to an equivalence check; no pipeline code calls them, and we found no
  record that the check ran.
- **They shut down today at 6 PM CDT.**

### Lumen
- Tried for case summaries with `gemma-4-31b-it` (656 replies across prompt revisions, cached in `data/llm_cache/`).
  The shipped summaries are written by the deterministic template instead (`backend/summary.py`); the app makes no
  live Lumen calls.

### Shared datasets
| Dataset | Used? | How |
|---|---|---|
| BACH | Yes (core) | Downloaded to `/work/nvme/bfqi/teams/team44/bach`; 10 WSIs used for simulated margins. The PathMMU copy was only inventoried. |
| NCT-CRC-HE-100K | Yes (small) | ~300 tiles per class, cross-organ sanity check of the concept explanations |
| Quilt-1M | Yes | 1,017,708 caption rows scanned, 26,837 breast captions mined for concepts |
| PathGen | Yes | Concept images and captions |
| PathMMU | Inventoried only | HF access was pending; not used |

### Models
- Seven encoders embedded: UNI2-h (production), CONCH, MUSK, H-optimus-0, Virchow2, Phikon-v2, PathGen-CLIP.
  CONCH/MUSK drive the concept explanations and the second-opinion check; Virchow2 and Phikon-v2 are embedded but
  unused. Text prompts encoded with CONCH, MUSK and PathGen-CLIP (1,177 prompts for CONCH).
- HoVer-Net (not an organizer resource): 108,661 nuclei on 1,070 tiles.

### Not used
- Illinois Chat (no trace found).
- The optional fine-tuning job (`finetune` eval is absent).
- ~2 h of the team's GPU cap, the rest of Rishabh's personal GPU-hours, and today's 8 AM–6 PM GPU block.

## Should we use the GPU today?

**We could:** about 8.5 h remain in today's block (ends 6 PM CDT) and ~2 h of the team cap is unrequested.

**We should only if** one job (~1 h, finished well before 6 PM) fixes a weakness judges will notice without breaking
the frozen results.

### Skip: fine-tuning (LoRA)

(Morning numbers, earlier 256-px model; current production numbers are in [FACT_SHEET.md](FACT_SHEET.md).)

- **Small data.** BACH has 400 images; fine-tuning is likely to overfit, and gains would sit inside the already wide
  four-class interval (macro-F1 0.83, CI 0.61–0.93).
- **Invalidates everything downstream.** New embeddings mean redoing calibration, conformal sets, the reliability flag,
  retrieval, all 328 case files, the summaries and the evaluation, on the last day of compute.
- **Little headroom.** Invasive vs rest is already AUROC 0.998.

### Worth doing
1. **Hosted-server equivalence check (no GPU budget; servers close at 6 PM).** Send ~20 tiles to hosted CONCH and
   MUSK and confirm their embeddings match ours. Minutes of work, a solid methods-slide line, and the only item that
   disappears tonight. Must run from inside Delta (one Duo login).
2. **A targeted GPU job only if it answers a question we will present.** The main weakness is in situ vs invasive
   (3 of 30 in situ called invasive). Denser embedding of the BACH WSIs would put more real tissue behind the
   simulated margins, but margins are labeled simulated, so this improves the demo more than the science.
3. **Keep the remainder as insurance** until ~3 PM in case a bug found today needs re-embedding. Unused budget is not
   penalized; a broken result found tomorrow cannot be fixed.

### Highest value, but tight: external validation
Running the frozen model on a second public breast dataset (e.g. BRACS or BreakHis) would turn "works on BACH" into
"works across datasets", strengthening clinical impact and development progress. It is not preloaded, so it needs a
download and license review, a mapping of that dataset's classes to BACH's four, ~1 GPU-hour to embed, and an
evaluation re-run. Feasible before 6 PM only if a teammate starts the download now; otherwise it belongs on the
next-steps slide.

## Recommendation
Run the hosted-server check this morning, hold the GPU budget as insurance until ~3 PM, and use the GPU only if the
team commits to external validation right away.
