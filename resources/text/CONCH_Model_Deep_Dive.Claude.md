# CONCH_Model_Deep_Dive.Claude

Organizer slide text extracted on 2026-10-17. Image/chart content and spatial relationships are not reproduced. The original deck is in ../originals/.

## Slide 31

CONCH: A Vision–Language Foundation Model for Pathology
Model Deep Dive  ·  2026 Ashby Hackathon

### Speaker notes
Based on Lu MY, Chen B, Williamson DFK, et al. A visual-language foundation model for computational pathology. Nat Med 2024;30:863-874. doi:10.1038/s41591-024-02856-4. Sources: main text and figures, Extended Data, Supplementary Information (tables), and the mahmoodlab/CONCH GitHub repository (commit 141cc09).

## Slide 32

7
▶ OVERVIEW
CONCH at a Glance
1.17M
image–caption pairs
448 px
input image size (ViT-B/16)
512
dim shared embedding
14
benchmarks, best on all
CONtrastive learning from Captions for Histopathology
Mahmood lab (Harvard / Mass General Brigham), built on CoCa
Logo: github.com/mahmoodlab/CONCH

### Speaker notes
Only the image and text encoders are in the public weights; the decoder was removed.

## Slide 33

7
▶ ARCHITECTURE & TRAINING
Architecture: CoCa With Three Parts
Image encoder: ViT-B/16 (12 layers, 12 heads, width 768) + two attentional poolers
  Contrast pooler: 1 query → one global image token (512-d)
  Caption pooler: 256 queries → 256 local tokens for the decoder
Text encoder: GPT-style, 12 layers, width 768; appended <CLS> token = caption embedding
Fusion decoder: 12 layers, cross-attention to image tokens, predicts the caption
Lu et al., Nat Med 2024, Fig. 1c (left; radar = Fig. 1d) and Methods; model_configs/conch_ViT-B-16.json

## Slide 34

7
▶ ARCHITECTURE & TRAINING
Pretraining Objective and Settings
Contrastive
Image↔text and text↔image InfoNCE
Aligns global image token with text <CLS>
Captioning
Decoder predicts each caption token from earlier tokens + image tokens
Loss: equal-weighted sum of the two contrastive terms and the captioning term (CoCa)
Settings: 40 epochs · 448×448 images · captions ≤128 tokens · batch 384 × 4 accumulation = 1,536 · AdamW, LR 1e-4, weight decay 0.2, cosine, 250 warmup · learned temperature · fp16 · 8 × A100 80 GB
Lu et al., Nat Med 2024, Methods (visual-language pretraining); Supplementary Table 32

## Slide 35

7
▶ ARCHITECTURE & TRAINING
Step 0: Unimodal Pretraining
Vision: iBOT self-supervision
16M in-house tiles, 256×256 at 20×
21,442 WSIs, 350+ OncoTree cancer subtypes
80 epochs, batch 1,024, 4 × A100
Language: 24-layer GPT
550K MGH surgical-pathology diagnoses + 400K PubMed abstracts + educational text
Reports de-identified with regex; next-word loss
Layers 1–12 → text encoder; 13–24 → decoder
Why: starting from pathology-pretrained towers beats ImageNet or random starts. Swapping in an ImageNet vision encoder lowered zero-shot classification and retrieval; a random text encoder mostly hurt retrieval.
Lu et al., Nat Med 2024, Methods (pretraining unimodal encoders); Supplementary Tables 33–34; Extended Data Fig. 10b

## Slide 36

7
▶ ARCHITECTURE & TRAINING
Ablations: Data and Algorithm Choices
a  Data: human-only (1.17M) beats the unfiltered 1.79M set and the H&E-only 457K set on average
a  Algorithm: CoCa (contrastive + captioning) is best for classification; plain CLIP training is slightly better for retrieval
b  Unimodal pretraining: dropping vision SSL costs the most; dropping text pretraining mostly hurts retrieval
Lu et al., Nat Med 2024, Extended Data Fig. 10

## Slide 37

7
▶ TRAINING DATA
Curating 1.17M Image–Caption Pairs
Sources: EDU (educational notes) and PMC OA (PubMed Central open-access figures)
1  Detect: YOLOv5 finds histology sub-images in figure panels (trained on synthetic panels)
2  Split: a PubMed-pretrained GPT model splits panel captions into sub-captions
3  Match: a CLIP model trained on clean data pairs each sub-image with its best sub-caption
Filter: 1,786,362 pairs → drop non-human → 1,170,647 pairs used for CONCH
Not used: large public slide collections such as TCGA, limiting overlap with public benchmarks
Lu et al., Nat Med 2024, Fig. 1a and Methods (dataset curation); repo README

## Slide 38

7
▶ TRAINING DATA
What the Training Data Covers
19 anatomic topics; GI, bone/soft tissue and lung are largest
~39% H&E (457K); the rest IHC and special stains, which helps on non-H&E images
PMC captions run longer than EDU captions (inset)
Topics assigned by pathologist keyword lists, then 5-NN for the rest
Lu et al., Nat Med 2024, Fig. 1b; Extended Data Fig. 1 (word clouds per topic)

## Slide 39

7
▶ TRAINING DATA
Preprocessing for Pretraining
Images
Figure panel
PMC or EDU
Detect + crop
YOLOv5 sub-images
Human filter
drop animal captions
Resize
short edge, bicubic
Crop / pad
center-crop or zero-pad to 448
Model input
448×448 → 28×28 patches
Text
Caption
full figure legend
Split
GPT sub-captions
Tokenize
byte-level BPE, uncased, 32K
Model input
≤128 tokens
Pretraining used 448 px; downstream tile tasks in the paper resize to 224 px
Lu et al., Nat Med 2024, Methods; repo conch/open_clip_custom (tokenizer, transform)

## Slide 40

7
▶ INPUTS & OUTPUTS
Input Formats and Sizes
Image
PIL image → RGB; resize short side to 448 (bicubic), center-crop 448×448
Normalize with OpenAI CLIP mean/std
Tensor B×3×448×448; force_img_size=224 for speed
Text
Byte-level BPE tokenizer, uncased (vocab 32,007)
Truncate/pad to 127 + 1 reserved for <CLS> = 128
Token IDs B×128 (LongTensor)
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/conch')
img = preprocess(Image.open(path)).unsqueeze(0)     # 1 x 3 x 448 x 448
ids = tokenize(get_tokenizer(), ['an H&E image of invasive ductal carcinoma'])   # 1 x 128
CONCH repo: README.md, notebooks/basics_usage.ipynb, open_clip_custom/transform.py, custom_tokenizer.py, factory.py

## Slide 41

7
▶ INPUTS & OUTPUTS
Inference Options
Call / flag
What it does
Use for
encode_image(proj_contrast=True)
Attention-pooled token × learned 512×512 projection (shared space with text)
Zero-shot classification, retrieval
encode_image(proj_contrast=False)
Attention-pooled token before projection
Linear probe, MIL features
normalize=True / False
L2-normalizes the embedding
True for cosine similarity; False for features
encode_text(ids)
Text <CLS> embedding, projected and normalized
Class prompts, text queries
force_img_size=224
Resize transform and positional embeddings
Faster tiles; the paper's supervised setting
with torch.inference_mode():  feats = model.encode_image(img, proj_contrast=False, normalize=False)
conch/open_clip_custom/coca_model.py (encode_image, encode_text), vision_tower.py (forward_no_head); README

## Slide 42

7
▶ INPUTS & OUTPUTS
Output Sizes and Formats
Call
Shape
Notes
encode_image (either flag)
B × 512
one vector per tile; L2 norm 1 if normalize=True
encode_text
B × 512
normalized by default
Zero-shot scores
B × C
softmax(img @ textᵀ × logit_scale.exp())
model(image, text)
dict
image_features, text_features, logits, labels, logit_scale
Slide features (MI-Zero, ABMIL)
N tiles × 512
one .pt per slide, or .h5 with 'features' + 'coords'
Caption tokens (internal)
B × 256 × 768
decoder input; captioning not in public weights
Format: PyTorch tensors; cache tile embeddings per slide for downstream training
coca_model.py, downstream/zeroshot_path.py, downstream/wsi_datasets.py; README (decoder weights removed)

## Slide 43

7
▶ INPUTS & OUTPUTS
Zero-Shot Classification With Prompts
Tiles: embed each class's prompts, average them, pick the class with highest cosine similarity
Slides (MI-Zero): score every tile, average the top-K tiles per class (K = 1–100), take the highest class
22 templates × several class names per class, written by a pathologist (repo: prompts/*.json)
Lu et al., Nat Med 2024, Fig. 2a,b; Supplementary Tables 35, 38–44; repo zeroshot_path.py

## Slide 44

7
▶ INPUTS & OUTPUTS
Whole-Slide Preprocessing
WSI
OpenSlide
Tissue mask
CLAM: HSV saturation threshold, blur, closing
Tile
256×256 at 10×, contiguous
Resize
224×224
Frozen CONCH
512-d per tile
Cache
.pt / .h5 per slide
Classification
Slide = bag of tile embeddings
MI-Zero top-K pooling (zero-shot) or ABMIL (supervised)
Segmentation
224 px tiles, 75% overlap, highest magnification
Average overlapping tile scores → per-pixel class
Lu et al., Nat Med 2024, Methods (WSI processing; extending zero-shot transfer to WSIs)

## Slide 45

7
▶ APPLICATIONS
Results at a Glance
Task
CONCH
Next best
BRCA subtyping, zero-shot
91.3%
BiomedCLIP 55.3%
NSCLC subtyping, zero-shot
90.7%
PLIP 78.7%
CRC100k tissue, zero-shot
79.1%
PLIP 67.4%
Brain tumors, 30 classes (ABMIL)
68.2%
CTransPath 61.4%
Text→image retrieval, mean recall
44.0%
BiomedCLIP 26.7%
Tumor segmentation Dice (DigestPath)
0.615
BiomedCLIP 0.446
Gleason fine-tune, quad. κ
0.872
CTransPath 0.813
Lu et al., Nat Med 2024, Fig. 1d; Results; Supplementary Tables 1–31

## Slide 46

7
▶ APPLICATIONS
Zero-Shot Slide and Tile Classification
Results
Slides: BRCA 91.3%, RCC 90.2%, NSCLC 90.7%
Tiles: CRC100k 79.1%, WSSS4LUAD 71.9%, SICAP κ 0.690
Others near chance on BRCA (51–55%)
How: no training. Pathologist-written prompts ensembled per class; MI-Zero top-K pooling over 256 px tiles at 10× for slides; balanced accuracy on held-out TCGA sets (150–225 slides).
Lu et al., Nat Med 2024, Fig. 2c (left panel); Supplementary Tables 1–7

## Slide 47

7
▶ APPLICATIONS
Supervised Use of Frozen Embeddings
Results
ABMIL: BRCA 86.7%, RCC 94.2%, NSCLC 93.3%
Linear probe: CRC100k 93.8%, ties CTransPath
8 labels/class beat other models' 64 (BRCA)
How: frozen 512-d tile embeddings at 224 px. Slides: gated ABMIL, 20 epochs, AdamW 1e-4, class-balanced sampling. Tiles: logistic regression (L-BFGS). Few-shot: 1–512 labels/class, 5 draws each.
Lu et al., Nat Med 2024, Fig. 2d and Fig. 3 (shown); Supplementary Tables 15–19, 36

## Slide 48

7
▶ APPLICATIONS
Rare Disease: 30 Brain Tumor Types
Results
Zero-shot 37.1% vs 3.3% chance (+17 over BiomedCLIP)
ABMIL 68.2%, +6.8 over CTransPath
Others need ~4× more labels
How: EBRAINS atlas, 2,319 slides, 50/25/25 split; ABMIL on frozen CONCH tiles with early stopping on validation loss; prompts from Supplementary Tables 42–44.
Lu et al., Nat Med 2024, Extended Data Fig. 7; Supplementary Tables 20–21

## Slide 49

7
▶ APPLICATIONS
Cross-Modal Retrieval
Results
Text→image mean recall 44.0% (BiomedCLIP 26.7%)
Source A: 68.8% mean recall
TCGA LUAD: +5.3 points, not significant
How: embed images and captions separately (projected, normalized); rank by cosine similarity; Recall@1/5/10 on 797, 1,755 and 165 held-out pairs.
Lu et al., Nat Med 2024, Fig. 4a; Supplementary Tables 22–27

## Slide 50

7
▶ APPLICATIONS
Searching Images With Words
Examples: LUAD growth patterns (left) and free-text morphology queries over 321K held-out tiles (right); numbers are cosine similarity
Lu et al., Nat Med 2024, Fig. 4b,c and Extended Data Fig. 8

## Slide 51

7
▶ APPLICATIONS
Zero-Shot Tumor Segmentation
Results
SICAP prostate Dice 0.601 (PLIP 0.549)
DigestPath colon Dice 0.615 (BiomedCLIP 0.446)
High sensitivity; false positives sit next to tumor
How: tile slides at 224 px with 75% overlap; classify each tile with tumor vs normal prompts; average overlapping scores and assign each pixel its top class.
Lu et al., Nat Med 2024, Fig. 5; Supplementary Tables 28–29

## Slide 52

7
▶ APPLICATIONS
Heatmaps: Where the Model Looks
Color = cosine similarity of each tile to the predicted class prompt
High-similarity regions match the pathologist's tumor outline (blue) in breast IDC (left) and three RCC subtypes (right)
Interpretable for free: the slide score is just the mean of the top-K tiles
Lu et al., Nat Med 2024, Fig. 2e and Extended Data Fig. 3

## Slide 53

7
▶ APPLICATIONS
Captioning and Fine-Tuning
Results
Captioning: METEOR 0.195 vs GIT-large 0.125
Gleason fine-tune: κ 0.872 vs CTransPath 0.813
With 1% of labels: κ 0.662 vs 0.595
How: captioning fine-tunes the full model with only the captioning loss on 558 pathologist-cleaned pairs, top-K=50 sampling. Gleason: end-to-end fine-tune on 189K ROIs at 448 px, 8 × A100. Needs the decoder, which is not public.
Lu et al., Nat Med 2024, Extended Data Fig. 9; Supplementary Tables 30–31, 37

## Slide 54

7
▶ APPLICATIONS
The Recipe Behind These Results
1
Write prompts:  several class names × 22 templates, averaged per class
2
Tile the slide:  CLAM mask, 256 px tiles at 10×, resize to 224
3
Embed once:  frozen CONCH → 512-d per tile, cached
4
Score or train:  zero-shot cosine + top-K pooling, or ABMIL / linear probe
5
Evaluate:  held-out sets; balanced accuracy, κ, Dice, Recall@K; bootstrap CIs
Why it works
Language gives labels for free: a new task needs prompts, not annotations.
Embeddings are reused across tasks, so one GPU pass per slide serves zero-shot, retrieval and supervised heads.

## Slide 55

7
▶ APPLICATIONS
Caveats and Practical Notes
Limits
Zero-shot weak on many-class rare tasks (37% on 30 classes)
Not tested across stains/scanners/sites
Practical
Text capped at 128 tokens
MI-Zero suits mutually exclusive classes, not Gleason sums
Prompt wording matters (Extended Data Fig. 2)
Access and license
Gated weights: huggingface.co/MahmoodLab/conch
CC-BY-NC-ND 4.0; no decoder, so no captioning
For the hackathon: start with zero-shot prompts as a baseline, then train a linear probe or ABMIL on cached 512-d embeddings; both fit on one GPU.
Lu et al., Nat Med 2024, Discussion; CONCH repo README and LICENSE

## Slide 56

7
▶ RESOURCES
References and Resources
Lu MY, Chen B, Williamson DFK, et al. A visual-language foundation model for computational pathology. Nat Med 2024;30:863–874. doi:10.1038/s41591-024-02856-4
Preprint: arXiv:2307.12914 · Supplementary Information and Extended Data: nature.com/articles/s41591-024-02856-4
Code and notebooks: github.com/mahmoodlab/CONCH (copy in hackathon_resources/pathology_vision_language_models/CONCH)
Weights (gated): huggingface.co/MahmoodLab/conch
Related: MI-Zero (Lu et al., CVPR 2023), CoCa (Yu et al., 2022), CLAM, ABMIL; successor slide model TITAN uses CONCH v1.5
