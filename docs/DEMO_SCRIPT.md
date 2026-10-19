# 3-minute demo script (real artifacts)

Live site: https://sepia-ashby.xyz. Its review workspace was reset on 2026-10-18 (evening) when the look-alike table and the spot-checked summaries were deployed, so the replay starts from the beginning.

Locally: start from a fresh workspace with `make demo-fresh` (runs `scripts/demo_fresh.sh` and archives the previous workspace database first), then open `http://127.0.0.1:8000`.

Everything below happens on real BACH test images. Frozen sections, margins, arrivals and operating rooms are simulated, and the app says so on every screen.

All numbers below were checked against the deployed case files and `artifacts/eval/*.json` on 2026-10-18.

| Time | Click | Say |
|---|---|---|
| 0:00 | **Start replay** (top bar) | "Biopsies arrive in real time. Sepia sorts them: frozen sections first, then anything where invasive carcinoma can't be excluded, then low-confidence reads, then the rest." Point at the composition bar and the tier groups. |
| 0:20 | Wait for **BACH_0167**, the last planned arrival; it jumps into tier 2. Press **N** / **Open next case**, or click it | "It arrived last but jumps the queue: its confidence set is {in situ, invasive}." |
| 0:35 | Case view: hover the heatmap; open **Why** | "Each region's exact contribution to the invasive score is shaded. Removing the top three regions lowers the score far more than removing random ones; that deletion test is what makes the heatmap evidence. Frequent mitotic figures and necrosis scored high here, but they failed our AI spot check (0 of 20 and 3 of 20 regions acceptable), so Sepia greys them out and does not give them as reasons. What it does report: tumor cell fraction is higher than in 97% of benign reference cases." |
| 0:55 | **Prediction and confidence**, then the **look-alike** table | "Calibrated probability of invasive carcinoma is 73%, but in situ stays in the 90% confidence set. CONCH and MUSK agree on the top class, and 4 of 5 similar reference cases are invasive. The look-alike is the closest in situ training image. The two features that differ most both failed the spot check, so the table says there is no reliable feature-level reason. That ambiguity is why the p63/calponin panel is drafted." |
| 1:10 | **Follow-up orders**: p63/calponin + ER/PR/HER2/Ki-67 → **Order tests** → confirm → **Sign out** → confirm | "Orders are drafted, not placed. Approval is recorded inside Sepia; nothing is sent to a laboratory system." |
| 1:25 | **Oncology inbox** | "The signed-out result reaches the oncologist with a summary written only from structured findings and checked sentence by sentence." |
| 1:40 | **+ Composite** (top bar) → open the tier-1 frozen section (COMP_026) | "A simulated frozen section from the OR goes straight to the top with a 20-minute timer. Image borders stand in for inked margins. The margin signal is above threshold at superior, medial and lateral; the strip on the image shows where." |
| 2:00 | **Tumor at margin…** (margins pre-filled) → **Send** → **Operating rooms** | "The surgeon's board updates and the timer stops at the turnaround time." |
| 2:15 | **+ Degraded** → open it (BACH_0225_FSIM_resample3) → **Defer to permanent** → confirm | "This one carries a simulated frozen-section artifact. The reliability flag explains itself: the input is unlike the training images and the second model disagrees, so the suggestion is to defer to permanent sections." |
| 2:30 | **Model evaluation** | "On 132 held-out images: invasive-vs-rest AUROC 0.999; all 31 of 31 invasive images were called invasive, an exact 95% interval of 88.8 to 100%, and specificity is 97%. Four-class macro-F1 is 0.860 and accuracy 86.4%; the four-class average AUROC is 0.965. Calibration error is 0.055 after temperature scaling. The confidence set covers 91.7% against a 90% target; that method was chosen after viewing test results, and the pre-registered one covered 72.0%. For context, in the BACH paper four pathologists reading single fields scored 73 to 96%. In a simulated reading day, invasive cases are first read after 6.0 minutes instead of 20.2, 14.2 minutes sooner, at a cost of 21.4 minutes for unflagged ones." |
| 2:50 | Scroll to **Limitations & data** | "BACH is small and curated, the frozen lane is simulated, the stain rules need clinical review, and no clinical validation is claimed." |

Backup paths:

- If BACH_0167 has not arrived yet, use any tier-2 case with drafted orders (e.g. BACH_0364).
- For a three-class uncertain read, use BACH_0087:
  - Probabilities: Normal 14.9%, In situ 42.7%, Invasive 40.7%.
  - Its look-alike table has one feature that passed the spot check, benign stroma or fat: +1.06 SD here against −1.95 SD in its invasive look-alike.
- Each simulated frozen-section button sends a new case, so a second run picks the next one.
- **Reset** clears the replay and its decisions (it asks first).
- Once every biopsy has arrived, the replay clock runs at wall time, so the queue stays stable while you talk.

Backup recording of an earlier version of this flow (3 min, local only because it shows BACH imagery): `outputs/demo-video/pathqueue_demo.webm`. It predates the look-alike table and the spot-checked summaries; re-record with `scripts/record_demo.py` against a throwaway server.
