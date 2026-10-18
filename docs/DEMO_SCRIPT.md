# 3-minute demo script (real artifacts)

Start from a fresh workspace: `make demo`, or `make demo-fresh` (runs `scripts/demo_fresh.sh` in the background and archives the previous workspace
database first), open `http://127.0.0.1:8000`. Everything below happens on real BACH test images; frozen sections,
margins, arrivals and operating rooms are simulated, and the app says so on every screen.

| Time | Click | Say |
|---|---|---|
| 0:00 | **Start replay** (top bar) | "Biopsies arrive in real time. Sepia sorts them: frozen sections first, then anything where invasive carcinoma can't be excluded, then low-confidence reads, then the rest." Point at the composition bar and the tier groups. |
| 0:20 | Wait for **BACH_0167** to arrive (planned last arrival; it jumps into tier 2). Press **N** / **Open next case**, or click it | "It arrived last but jumps the queue: its confidence set is {in situ, invasive}." |
| 0:35 | Case view: hover the heatmap; open **Why** | "Each region is scored; the strongest associated features are frequent mitotic figures and tumor necrosis, both above the training average. These are associations from a concept model, not causes and not a Nottingham grade. The AI spot-check badge (AI, not a pathologist) says how often the top regions really show each feature: for these two it is weak, 0 of 20 for mitoses and 3 of 20 for necrosis, so I discount them." |
| 0:55 | **Prediction and confidence** | "65% calibrated probability of invasive carcinoma, but the APS set keeps in situ in play at the 90% level. MUSK agrees, 4 of 5 similar reference cases agree." |
| 1:10 | **Follow-up orders**: p63/calponin + ER/PR/HER2/Ki-67 → **Order tests** → confirm → **Sign out** → confirm | "The agent drafted the myoepithelial panel because of exactly that ambiguity. Nothing is ordered until I approve." |
| 1:25 | **Oncology inbox** | "The signed-out result reaches the oncologist as a report, with a summary written only from structured findings and checked sentence by sentence." |
| 1:40 | **+ Composite** (top bar) → open the tier-1 frozen section (e.g. COMP_026) | "A simulated frozen section from the OR goes straight to the top with a 20-minute timer. Borders stand in for inked margins. The margin signal is above threshold at superior, medial and lateral; the strip on the image shows where." |
| 2:00 | **Tumor at margin…** (margins pre-filled) → **Send** → **Operating rooms** | "The surgeon's board updates and the timer stops at the turnaround time." |
| 2:15 | **+ Degraded** → open it (e.g. BACH_0301_FSIM_resample3) → **Defer to permanent** → confirm | "This one carries a simulated frozen-section artifact. The reliability flag explains itself: input unlike the training images, the second model disagrees. The suggestion is to defer." |
| 2:30 | **Model evaluation** | "Invasive vs rest AUROC 0.999, invasive sensitivity 100% and specificity 97% on 132 held-out images; four-class macro-F1 0.86 after a pre-registered second round on larger 512-pixel regions; ECE 0.055 after temperature scaling; the APS set covers 91.7% against a 90% target, chosen after viewing test results, while the pre-registered LAC covered 72.0%. All 31 invasive images land in the top two tiers. In simulation, invasive cases are seen 14 minutes sooner, at the cost of 25 minutes for unflagged ones." |
| 2:50 | Scroll to **Limitations & data** | "Small public dataset, simulated frozen lane, illustrative stain rules that need clinical review, and no clinical validation claimed." |

Backup paths: if BACH_0167 has not arrived yet, use any tier-2 case with drafted orders (e.g. BACH_0364). Each
simulated frozen-section button sends a new case, so a second run just picks the next one. **Reset** clears the
replay and its decisions (it asks first). Once every biopsy has arrived the replay clock runs at wall time, so the
queue stays stable while you talk.

Backup recording of this flow on the real artifacts (3 min, local only because it shows BACH imagery):
`outputs/demo-video/pathqueue_demo.webm`. Re-record with `scripts/record_demo.py` against a throwaway server.
