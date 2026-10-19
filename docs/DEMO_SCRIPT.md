# 3-minute demo script (real artifacts)

Live site: https://sepia-ashby.xyz. The workspace is shared and may already be running.

**About two minutes before you start talking:**

1. Click **Reset** → **Reset replay**, then **Start replay**.
2. The last planned biopsy, BACH_0167, arrives about 90 s after Start (`sim.max_replay_wall_s: 90`). Once it is in, the queue is complete and stays stable while you talk.

**Locally:** run `make demo-fresh`, which runs `scripts/demo_fresh.sh` and archives the previous workspace database first. Then open `http://127.0.0.1:8000`.

**What is real:** everything below happens on real BACH test images. Frozen sections, margins, arrivals and operating rooms are simulated, and the app says so on every screen. Orders are approved only inside Sepia; nothing is sent to a laboratory system.

Labels and numbers below were checked click by click against the app, the case files and `artifacts/eval/*.json` on 2026-10-19.

| Time | Click | Say |
|---|---|---|
| 0:00 | **Worklist** (replay already complete) | "Biopsies arrive in real time. Sepia sorts them: frozen sections first, then anything where invasive carcinoma can't be excluded, then low-confidence reads, then the rest." Point at the composition bar and the tier groups. |
| 0:15 | Press **/**, type `0167`, click **BACH_0167**, clear the search box. **N** opens the top-risk case (BACH_0128), not this one | "This one arrived last but goes straight into tier 2: its confidence set is {in situ, invasive}." |
| 0:25 | Scroll to **Image evidence** and hover region 5. Then scroll to the **Why** card | "Each region's exact contribution to the invasive score is shaded. On the test set, removing the top three regions lowers the score far more than removing three random ones; that is what makes the heatmap evidence. Mitotic figures and necrosis score high here but failed our AI spot check, 0 of 20 and 3 of 20, so Sepia never gives them as reasons." |
| 0:45 | **Prediction and confidence**, then the **look-alike** table | "Invasive carcinoma is 73% calibrated, but in situ stays in the 90% confidence set. A second model, a MUSK probe, picks the same top category, and 4 of 5 similar reference cases are invasive. The closest in situ look-alike differs mostly in features that failed the spot check, so there is no reliable feature-level reason. That ambiguity is why p63/calponin is drafted." |
| 1:05 | **Follow-up orders**: **Approve orders** → **Confirm approval** → **Sign out…** → **Confirm sign-out** | "The rules drafted both panels; nothing is approved until I confirm, and nothing leaves Sepia." |
| 1:20 | **Oncology inbox**. Point at the category, the set and the summary | "The signed-out result reaches the oncologist with a template summary built only from structured findings; each sentence cites the finding it comes from." |
| 1:35 | **+ Composite** (top bar) → **Open** in the toast → COMP_026. Scroll to the image for the margin strip | "A simulated frozen section from the OR goes straight to the top with a 20-minute timer. Image borders stand in for inked margins; the signal is above threshold at superior, medial and lateral." |
| 1:55 | **Tumor at margin…** (three margins pre-ticked) → **Send: tumor at margin** → **Operating rooms** | "The surgeon's board updates and the timer stops at the turnaround time." |
| 2:10 | **+ Degraded** → **Open** in the toast (BACH_0225_FSIM_resample3) → **Defer to permanent** → confirm | "This one carries a simulated frozen-section artifact. The reliability flag explains itself: the input is unlike the reference images and the second model, MUSK, picks a different category, so the suggestion is to defer." |
| 2:25 | **Model evaluation** (the Summary tiles are at the top) | "On 132 held-out images, all 31 invasive images were called invasive, an exact 95% interval of 88.8 to 100%, and 97% of the rest were not. Four-class macro-F1 is 0.860. Calibration error is 0.055 after temperature scaling. The confidence set covers 91.7% against a 90% target, a method chosen after seeing test results; the pre-registered one covered 72%. Pathologists reading single BACH fields scored 73 to 96%. In a simulated reading day, invasive cases are seen 14 minutes sooner, at a cost of 21 minutes for unflagged ones." |
| 2:50 | **Limitations & data** tab | "BACH is small and curated, the frozen lane is simulated, the order rules need clinical review, and no clinical validation is claimed." |

## Backup paths

- **If BACH_0167 has not arrived:** BACH_0364 is in tier 2 early, but it is not a drop-in.
  - On screen: 70% invasive, set {invasive} only, only ER/PR/HER2/Ki-67 drafted, MUSK disagrees, and 1 of 5 similar cases is invasive.
  - Say only: "Tier 2, 70% invasive, the biomarker panel is drafted."
- **For a three-class uncertain read, use BACH_0087:**
  - On screen: Normal 15%, In situ 43%, Invasive 41%. The large number is P(invasive), 41%. The reliability flag is raised (0.11), and MUSK disagrees.
  - Its look-alike table has one feature that passed the spot check, benign stroma or fat: +1.06 SD here against −1.95 SD in its invasive look-alike.
- **Frozen-section buttons:** each one sends a new case, so a second run picks the next one.
- **Reset:** clears the replay and its decisions (it asks first).

## For questions

- **BACH_0167's reference label is in situ.** The top-1 read is wrong, but the confidence set kept in situ. It is one of the 3 of 30 in situ test images read as invasive.
- **The plain-language tab** is a cached language-model restatement of the evidence. On the live site it is served from the cache only, never generated on the spot.

## Backup recording

An earlier version of this flow is recorded (3 min; local only because it shows BACH imagery): `outputs/demo-video/pathqueue_demo.webm`. It predates the look-alike table, the spot-checked summaries and the approval wording. Re-record it with `scripts/record_demo.py` against a throwaway server.
