# MUSK_Model_Deep_Dive.Claude

Organizer slide text extracted on 2026-10-17. Image/chart content and spatial relationships are not reproduced. The original deck is in ../originals/.

## Slide 31

MUSK: A Vision–Language Foundation Model for Pathology
Model Deep Dive  ·  2026 Ashby Hackathon

### Speaker notes
Based on Xiang J, Wang X, et al. A vision-language foundation model for precision oncology. Nature 2025;638:769-778. doi:10.1038/s41586-024-08378-w. Sources: main text and figures, Extended Data, Supplementary Information, and the lilab-stanford/MUSK GitHub repository (commit 714b666).

## Slide 32

7
▶ OVERVIEW
MUSK at a Glance
675M
parameters
50M
TCGA image patches
1B
PubMed Central text tokens
1M
image–text pairs
Multimodal transformer with Unified maSKed modeling
Learns from unpaired images and text first, then aligns them with paired data
Logo: github.com/lilab-stanford/MUSK

### Speaker notes
MUSK = Multimodal transformer with Unified maSKed modeling. Stanford (Li lab). Weights: huggingface.co/xiangjx/musk (gated). Code: github.com/lilab-stanford/MUSK. License CC-BY-NC-ND 4.0.

## Slide 33

7
▶ ARCHITECTURE & TRAINING
Two-Step Pretraining
Step 1 learns each modality from huge unpaired data.  Step 2 aligns images and text with ~1M pairs.
Xiang et al., Nature 2025, Fig. 1a

### Speaker notes
Slide counts differ slightly across the paper: Fig. 1a says 32,898 slides, Supplementary Fig. 1 and Methods say 33,898 (22,135 + 11,763), the main Methods says 'nearly 33,000'.

## Slide 34

7
▶ ARCHITECTURE & TRAINING
Backbone: A Multiway Transformer
Every layer shares one self-attention block across modalities
Then switches feed-forward expert by input: V-FFN for image tokens, L-FFN for text tokens
One network encodes image only, text only, or both
Adapted from BEiT-3 (Microsoft torchscale)
Repo config musk_large_patch16_384: 24 layers · hidden 1,024 · 16 heads · FFN 4,096 · 384 px image · 16 px patches · text vocab 64,010
Xiang et al., Nature 2025, Extended Data Fig. 7a; musk/modeling.py

## Slide 35

7
▶ ARCHITECTURE & TRAINING
Step 1: Unified Masked Modeling
Text: masked language
15% of tokens replaced by [MASK]
Predict them from context (cross-entropy)
Images: masked image
40% of 16×16 patches masked
Predict each patch's discrete visual token
Pathology tokenizer: BEiT-v2-style visual tokenizer trained on 5M TCGA images with CTransPath as teacher; natural-image tokenizers gave weaker features
Settings: 1M steps · batch 2,048 images + 2,048 texts · AdamW, peak LR 1.5e-3, cosine, 10k warmup · ~240 h on 64 V100 GPUs
Xiang et al., Nature 2025, Methods; Supplementary Table 16 (right); Supplementary Section 3

## Slide 36

7
▶ ARCHITECTURE & TRAINING
Step 2: Contrastive Alignment
Contrastive loss pulls matching image [CLS] and text [BOS] embeddings together
Auxiliary decoder (4 layers, 50M params) predicts masked text using image tokens as keys and values
Loss: L_align = L_con + 0.1 × L_aux
Settings: 20 epochs · batch 3,072 pairs · peak LR 1e-4 · random resized crop, flips, color jitter · ~10 h on 64 V100s
Note: Methods says 30% of text tokens are masked for L_aux; the figure is labelled 15%
Xiang et al., Nature 2025, Extended Data Fig. 7b; Methods

## Slide 37

7
▶ ARCHITECTURE & TRAINING
Ablations: What Each Design Choice Adds
a  Masked pretraining: +14.7 pts linear probe, +32 pts zero-shot vs. none
b  Pathology data: beats natural images at equal scale, most on retrieval (+16.5 pts text→image)
c  Scale: 50M vs 15M patches gives smaller, steady gains
d  Size: large ≈ base (+1–2 pts); small is clearly worse
Plus: stain/FoV augmentation, pathology tokenizer, decoder and pair filtering each add 1–6 pts
Xiang et al., Nature 2025, Extended Data Fig. 8; Supplementary Tables 13–15, 17

## Slide 38

7
▶ TRAINING DATA
Pretraining Data Sets
Data
Source and size
Text
1,001,800 PubMed Central open-access articles (2000–2023) → ~1B tokens
Images
All TCGA H&E slides: 22,135 frozen + 11,763 FFPE = 33,898 WSIs, 11,577 patients, 33 tumor types → ~50M patches
Pairs
Quilt1M (802K, YouTube teaching videos) + PathAsst/PathCap (207K)
Articles found with “pathology”, “histopathology”, “whole-slide image”, “H&E” and 148 glossary terms
Xiang et al., Nature 2025, Methods; Supplementary Fig. 1 and Methods

## Slide 39

7
▶ TRAINING DATA
Preprocessing for Pretraining
Images
TCGA WSI
H&E, frozen + FFPE
Segment tissue
drop background
Tile
512×512 at 10×, no overlap
Fields of view
random 10×, 20×, 40×
Augment
RandStainNA, flips, gray, jitter
Model input
384×384 → 16×16 patches
Text
PMC articles
full text
Tokenize
SentencePiece, 64K vocab
Mask
15% of tokens
Model input
token sequences
Kept native resolution so crops at several magnifications could be taken during training
Xiang et al., Nature 2025, Methods (masked training settings); Supplementary Methods

## Slide 40

7
▶ TRAINING DATA
Cleaning Image–Text Pairs by Bootstrapping
Train a baseline model on raw Quilt1M
Score every pair by image–text similarity
Drop pairs below 0.1 (~12%): blurry or mismatched captions
Retrain on the cleaned pairs (BLIP-style bootstrap)
Gain: +6.3 pts text→image, +4.9 image→text, +2.2 zero-shot
Xiang et al., Nature 2025, Supplementary Fig. 2 and Table 14 (Ablation 4)

## Slide 41

7
▶ TRAINING DATA
Clinical Reports: GPT-4 Standardization
Why: MUSK-L reads at most 100 tokens; full TCGA reports are longer
How: expert-written questions (size, differentiation, invasion, nodes, margins) turn each report into a short structured summary
Cohort-specific prompts: lung adds PD-L1, TMB, CNS mets; gastro-esophageal adds CPS, MSI, HER2
Xiang et al., Nature 2025, Methods; Supplementary Tables 8–10

## Slide 42

7
▶ INPUTS & OUTPUTS
Input Formats and Sizes
Image
RGB tile (PIL); resize short side to 384 (bicubic), center-crop 384×384
Normalize with mean = std = 0.5 (timm IMAGENET_INCEPTION)
Tensor B×3×384×384, fp16 on GPU → 576 patch tokens + [CLS]
Text
XLM-RoBERTa SentencePiece tokenizer (musk/models/tokenizer.spm)
[BOS] + tokens + [EOS], truncated or padded to 100
IDs B×100 plus padding mask B×100 (1 = pad)
transform = Compose([Resize(384, interpolation=3), CenterCrop(384), ToTensor(),
                     Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)])
txt_ids, pad = utils.xlm_tokenizer(text, tokenizer, max_len=100)
MUSK repo: README.md, demo.ipynb, musk/utils.py

## Slide 43

7
▶ INPUTS & OUTPUTS
Inference Options
Flag
What it does
Set it for
with_head
Applies the 1024×1024 projection learned for image–text alignment
True: zero-shot retrieval and classification
out_norm
L2-normalizes the embedding
True: anything compared by cosine similarity
ms_aug
Adds a 2× view (768 px split into four 384 tiles) and concatenates its mean [CLS] with the 1× [CLS]
True: linear probe and MIL features (images only)
return_global
Returns only [CLS]/[BOS] instead of every token
False: patch- or word-level token maps
vision, language = model(image=x, text_description=ids, padding_mask=pad, with_head=..., out_norm=..., ms_aug=..., return_global=...)
musk/modeling.py (MUSK.forward), musk/utils.py (MultiScaleForward); defaults: ms_aug=False, scales=[1, 2]

## Slide 44

7
▶ INPUTS & OUTPUTS
Output Sizes and Formats
Call
Shape
Notes
Image, ms_aug=False
B × 1024
[CLS] embedding; default image feature
Image, ms_aug=True
B × 2048
1× [CLS] ‖ mean of four 2× [CLS]; head uses first 1024
Image, return_global=False
B × 577 × 1024
[CLS] + 24×24 patch tokens
Text, return_global=True
B × 1024
[BOS] embedding
Text, return_global=False
B × 100 × 1024
one vector per token position
Zero-shot scores
B × N classes
logit_scale · image @ textᵀ, then softmax
Format: model returns a tuple (vision, language) of PyTorch tensors (fp16 if the model is cast); save features yourself as .pt, .npy or .h5
musk/modeling.py; demo.ipynb (2048-dim output with ms_aug=True; example probabilities [0.378, 0.325, 0.297])

## Slide 45

7
▶ INPUTS & OUTPUTS
Using MUSK for Each Task
a  Retrieval: embed images and texts separately; rank by cosine similarity (Recall@K)
b  VQA: concatenate image [CLS] and question [BOS] (2,048 dims); classify over 4,092 answers; whole model fine-tuned 50 epochs
c  Zero-shot: average several prompt templates per class; pick the class with highest similarity
d  Supervised: frozen image embeddings → linear classifier, from 1 to 10 shots up to full training sets
Xiang et al., Nature 2025, Supplementary Fig. 3 and Supplementary Methods

## Slide 46

7
▶ INPUTS & OUTPUTS
From Patches to Slides and Reports
Slides: tile embeddings → gated-attention MIL (512-d, hidden 384) → slide embedding
Reports: GPT-4 summary → MUSK-L
Fusion: concatenate, then a small MLP
Training: 100 epochs · AdamW, LR 1e-3, cosine · class-balanced sampler · early stop after 10 · dropout 0.25 · 5-fold stratified cross-validation
Xiang et al., Nature 2025, Fig. 1b; Methods (melanoma relapse prediction)

## Slide 47

7
▶ APPLICATIONS
Results at a Glance
Task
MUSK
Best comparison
PathMMU image→text R@50
34.4%
CONCH 27.3%
PathVQA accuracy
73.2%
K-PathVQA 68.9%
HER2 status (AUC)
0.826
GigaPath 0.786
Melanoma relapse (AUC)
0.833
CONCH 0.778
Pan-cancer prognosis (c-index)
0.746
CONCH 0.684
NSCLC immunotherapy (AUC)
0.768
PD-L1 0.606
Gastro-eso. immunotherapy (AUC)
0.762
MSI-H 0.616
Xiang et al., Nature 2025, Fig. 1b and Results

## Slide 48

7
▶ APPLICATIONS
Zero-Shot Retrieval and Visual Q&A
Results
PathMMU image→text R@50: 34.4% vs CONCH 27.3%
BookSet image→text R@50: 74.8% vs 71.3%
PathVQA: 73.2%, above the task-specific K-PathVQA (68.9%)
How: frozen MUSK-V and MUSK-L with heads; cosine ranking of 4,265 (BookSet) and 7,774 (PathMMU) pairs. VQA fine-tunes MUSK plus a linear head on 19,654 training questions.
Xiang et al., Nature 2025, Fig. 2; Supplementary Tables 1–2

## Slide 49

7
▶ APPLICATIONS
Patch-Level Image Classification
Results
Zero-shot: +10.5, +27.5, +7.3, +10.1 pts over the next model on 4 data sets
10-shot: best on all 12 benchmarks
Linear probe: 88.2% mean balanced accuracy, +2.2 over CONCH
How: prompt templates averaged per class for zero-shot; logistic/linear heads on frozen multi-scale (2,048-d) embeddings, 10 seeds, images resized to 384.
Xiang et al., Nature 2025, Fig. 3; Extended Data Fig. 2; Supplementary Tables 4–6

## Slide 50

7
▶ APPLICATIONS
Molecular Biomarkers from H&E
Results
HER2: AUC 0.826 vs GigaPath 0.786, CONCH 0.771
Also best for ER, PR (BCNB) and IDH (MUV-IDH)
How: MUSK-V tile features aggregated with AbMIL per slide, fused with MUSK-L embeddings of clinical variables (age, tumor size, grade, location); BCNB 1,057 slides, MUV-IDH 872 slides.
Xiang et al., Nature 2025, Extended Data Fig. 3a; Supplementary Table 7

## Slide 51

7
▶ APPLICATIONS
Melanoma Relapse Prediction
Results
5-year relapse AUC 0.833 (CONCH 0.778)
At 90% sensitivity, ~12% higher specificity
Image + report beats either alone
How: VisioMel: 1,342 patients with diagnostic H&E WSIs and clinical reports; AbMIL slide embedding + MUSK-L report embedding → MLP; 5-fold CV stratified by relapse.
Xiang et al., Nature 2025, Extended Data Fig. 4a–d

## Slide 52

7
▶ APPLICATIONS
Pan-Cancer Prognosis (16 TCGA Cancers)
Results
Mean c-index 0.746 vs stage 0.645, CONCH 0.684
Renal cell carcinoma: 0.887, HR 36.8
Independent of age, sex, stage, grade
How: 7,927 WSIs from 6,602 patients; GPT-4-structured reports; one model per cancer type predicting disease-specific survival; 5-fold CV; KM split at the median risk.
Xiang et al., Nature 2025, Fig. 4; Supplementary Figs. 7–8

## Slide 53

7
▶ APPLICATIONS
Immunotherapy Response: Lung Cancer
Results
Response AUC 0.768 vs PD-L1 0.606
PFS c-index 0.705; HR 2.54 overall
PD-L1-negative tumors: HR 7.38
How: 118 Stanford NSCLC patients on anti-PD-1/PD-L1; pre-treatment H&E + GPT-4-standardized pathology report; AbMIL + report embedding → MLP; 5-fold CV.
Xiang et al., Nature 2025, Fig. 5a–c

## Slide 54

7
▶ APPLICATIONS
Immunotherapy Response: Gastro-Esophageal
Results
Response AUC 0.762 vs MSI-H 0.616
PFS HR 3.49; median 3.6 vs 14.1 months
Stratifies within PD-L1 and MSS/MSI-L groups
How: 101 Stanford patients; same multimodal recipe, with a gastro-esophageal prompt set (CPS, MSI, HER2); 5-fold CV.
Xiang et al., Nature 2025, Extended Data Fig. 5a–c

## Slide 55

7
▶ APPLICATIONS
What the Model Attends To
Lung immunotherapy: responders show dense lymphocyte infiltration and little stroma; non-responders show collagen-rich stroma
Melanoma: relapse cases show ulceration, macrophages, fibrosis and mitoses; non-relapse cases show abundant lymphocytes
Xiang et al., Nature 2025, Fig. 5d and Extended Data Fig. 4e (AbMIL attention, tiles with 85% overlap)

## Slide 56

7
▶ APPLICATIONS
The Recipe Behind These Results
1
Embed tiles:  frozen MUSK-V, multi-scale 2,048-d per tile
2
Summarize reports:  GPT-4 prompts → ≤100 tokens → MUSK-L
3
Pool the slide:  gated-attention MIL → slide embedding
4
Fuse and predict:  concatenate image + text → small MLP
5
Validate:  5-fold stratified CV; AUC, c-index, KM at median
Why it works
The backbone stays frozen, so only small heads are trained on cohorts of 100–1,300 patients.
Reports add information images lack; multimodal beat image-only and text-only in every outcome task.

## Slide 57

7
▶ APPLICATIONS
Caveats and Practical Notes
Evidence
Immunotherapy cohorts: ~220 patients, one center, retrospective
Needs multi-site and prospective validation before clinical use
Practical limits
Text input capped at 100 tokens
Outcome tasks depend on GPT-4 report summaries
Patch encoder: slides need MIL
Access and license
Gated weights: huggingface.co/xiangjx/musk
CC-BY-NC-ND 4.0: non-commercial; models trained on MUSK outputs are restricted too
For the hackathon: frozen MUSK features plus small trained heads is the same recipe the paper used for its outcome models, and it fits on one GPU.
Xiang et al., Nature 2025, Discussion; MUSK repo README (License)

## Slide 58

7
▶ RESOURCES
References and Resources
Xiang J, Wang X, et al. A vision–language foundation model for precision oncology. Nature 2025;638:769–778. doi:10.1038/s41586-024-08378-w
Supplementary Information and Extended Data: nature.com/articles/s41586-024-08378-w
Code and demo notebooks: github.com/lilab-stanford/MUSK (copy in hackathon_resources/pathology_vision_language_models/MUSK)
Weights (gated): huggingface.co/xiangjx/musk
Benchmarks: CLIP Benchmark fork in MUSK/benchmarks; PathMMU at huggingface.co/datasets/jamessyx/PathMMU
Built on: BEiT-3 / torchscale, Quilt1M, PathAsst, CLIP Benchmark, PORPOISE
