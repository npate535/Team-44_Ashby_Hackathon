# PathQueue verification

Verified locally on October 17, 2026 (America/Chicago). This is a research prototype, not a clinically validated system.

## Code and workflows

- Full Python/unit/integration suite: **2,494 passed** in 116.34 seconds.
- Isolated browser suite: **16 passed** in 99.12 seconds, covering desktop/light and phone/dark workflows. These tests use explicitly synthetic temporary workspaces, separate from the real application.
- After the final wording refinements: **64 targeted tests passed** (frontend modules, conformal behavior, evaluation metrics). TypeScript contract consistency passed.
- Real application browser verification used actual BACH-derived artifacts at `http://127.0.0.1:8000/`: tissue images, supporting-region links, measured cell overlays and toggles, coverage disclosure, drafted-order approval, sign-out, reload persistence, and oncology routing.
- Surgical verification followed the specific injected case IDs through the queue: a named-margin call, operating-room routing, stopped timers, a degraded-case deferral, and scored composite margin segments. These are simulated surgical fields and operator test decisions, not clinical margin assessments.
- Actual evaluation charts loaded. One cell-overlay review was recorded explicitly as **AI / unsure**, with no clinical assessment claimed. A 390-pixel phone viewport with dark theme and reduced motion loaded actual cell images without horizontal overflow. No browser page errors occurred in the completed real-data pass.
- Server restart persistence and refusal to silently overwrite a workspace with changed case artifacts are covered by integration tests. Only explicit replay reset clears decisions.

Local evidence is in ignored `outputs/verified-unit-suite.log`, `outputs/verified-browser-suite.log`, `outputs/final-browser/`, and `outputs/real-final-remaining-v3.log`. Evidence screenshots are local because they contain BACH-derived imagery.

## Data and compute

The complete analysis/evaluation pipeline finished with **328 contract-valid application cases**, **zero rejected case files**, and **zero synthetic fixture cases** in the default workspace. These comprise 132 original held-out images, 132 degraded simulation fields, and 64 composite simulation fields. All **901 referenced media assets** were available and passed image decoding checks. Original and simulated populations remain separate in evaluations.

Delta exports were imported using manifest paths, per-file sizes and SHA-256 verification. Seven real encoder outputs were available: CONCH, MUSK, UNI2-H, H-optimus-0, Virchow2, Phikon-v2, and PathGen-CLIP. Production `uni2h_lr` was selected using grouped training cross-validation; the separate `cb_text` model supplies supporting concept associations. Its contributions are not presented as contributions to the production classifier.

Full cell processing completed for **1,070 selected tiles from 364 source cases**, yielding **108,661 nuclei**. Every selected tile had an output. The remaining 36 source cases had no eligible QC regions; measurements are unavailable there. Minimum-count feature exclusions remain missing, rather than becoming zero. In the application, 129 entries have actual cell outputs. Cell types have not been validated by a pathologist.

GPU work used Delta's permitted queue and one GPU at a time. The recorded requested allocation totals 300 of the 420 allowed GPU-minutes; no further jobs were needed. Grounded Lumen generation and its cache were exercised; remaining template text retains its generator attribution. Reviewing precomputed cases requires no live GPU or LLM call.

Before publication, the preceding local artifact workspace and databases were preserved at `artifacts/archive/core-20261018T042840Z`. The final replay is paused with the verification decisions preserved. Licensed imagery, weights, embeddings, credentials, caches, and SQLite records remain outside Git.

## Results and limits

See [RESULTS.md](RESULTS.md) for generated estimates, patient/pseudo-patient grouped confidence intervals, evaluation populations, and protocols. Held-out classification accuracy is **83.3% on 132 original images**. Observed LAC coverage is **67.4%, below its nominal 90% target**; the interface explicitly discloses this shortfall. The theoretical coverage statement requires exchangeability and does not establish coverage for this cohort or an individual case.

The final real model does not produce every exact scenario described in the planning-stage demo story. Its outputs were retained. For example, the two-class ambiguity/order-rule branch is exercised with explicit synthetic workflow fixtures; a matching real prediction was not manufactured.

Arrivals, operating-room assignments, frozen artifacts, image-border margins, and queue-benefit estimates are simulations. Approvals and routed inbox results are persisted locally. Hospital order entry, clinical deployment, expert validation, and a coverage guarantee are not claimed.
