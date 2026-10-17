# 2026_Ashby_Hackathon_Opening_Session

Organizer slide text extracted on 2026-10-17. Image/chart content and spatial relationships are not reproduced. The original deck is in ../originals/.

## Slide 1

2026 Ashby Prize in Computational Science Hackathon
Opening Session  ·  Saturday, October 17, 2026

### Speaker notes
Opening session for hack day 1. Outline: staff intro, logistics, domain overview, initial goal, Delta, shared files, tutorials, other compute, timeline, help.

## Slide 2

‹#›
▶ WELCOME
Who Is Here to Help This Weekend
Charles Blatti
Research Scientist, NCSA Visual Analytics
Kun Wang
Assistant Professor, Comparative Biosciences & Bioengineering
Vlad Kindratenko
Assistant Director, CAII @ NCSA
Josh Allen
Research Programmer, NCSA Genomics
Pam Joop
Office Administrator 
Shirui Luo
Research Scientist, CAII @ NCSA
Tani Adeniba
Research Program Manager

### Speaker notes
Introduce the organizers present today. Fill in the placeholder with volunteers and experts on site.

## Slide 3

‹#›
▶ WELCOME
Thank You for Your Patience
~180
students confirmed
~60
teams
>3×
larger than we planned for
This is our largest hackathon by far. Rooms, GPU queues and staff time will be stretched, and some things will go wrong.
Please give us, and each other, some grace. Tell us on Slack when something breaks and we will fix it as fast as we can.

## Slide 4

‹#›
▶ LOGISTICS
Rooms: First Come, First Served
Room assignments handed out this morning
Rooms may be shared: be respectful of other teams
Staff are near the first-floor main entrances
Source: Illinois Facility Access Maps

### Speaker notes
Hand out room assignments first come, first served. Stress that rooms are shared.

## Slide 5

‹#›
▶ LOGISTICS
A Day at the Hackathon
8:30 am
Doors open, light breakfast
9:00 am
Start-of-day meeting on Teams
12:00 pm
Lunch (pizza)
1:00 pm
Work continues, snacks
4:00 pm
End-of-day meeting on Teams
5:00 pm
In-person support ends
Start- and end-of-day meetings happen on Microsoft Teams: announcements, blockers, and quick check-ins. Join from anywhere.
Same for Saturday and Sunday.
All meetings will use the Teams Meeting link from the invite.

## Slide 6

‹#›
▶ LOGISTICS
Food, Building and Courtesy
Food
Served on the NCSA first floor
Breakfast 8:30, lunch at noon, afternoon snacks
Trash
Use the bins outside the rooms
Leave shared rooms as you found them, erase boards
Building
Doors locked all weekend; staff at the main entrances, 8:30 to 5:00
Issues: #building-logistics on Slack
NCSA is a working building on Monday. Thank you for keeping it clean.

## Slide 7

‹#›
▶ LOGISTICS
Team Changes Are Expected
Team A
2 people
Team B
2 people
New team
4 people
Teammates or whole teams withdrawing? Post in #team-updates so we can update team lists
Short-handed? Look for a merge at the current box team list: post idea or skills you need
https://uofi.box.com/s/hod5wsihamzcjydvdwnsgzpp4f79bj4x 

## Slide 8

‹#›
▶ DOMAIN OVERVIEW
What Is Digital Clinical Pathology?
https://www.pathpresenter.com/digital-pathology-basics-what-is-digital-pathology/ 
Glass slides become whole-slide images (WSIs), read on screen by pathologists and by algorithms

## Slide 9

‹#›
▶ DOMAIN OVERVIEW
Digital Pathology Is a Diverse Field
https://github.com/lingxitong/Awesome-AI4DigitalPathology - nearly 500 references

## Slide 10

‹#›
▶ DOMAIN OVERVIEW
How Slides Become Data: The Scanner
Scanner optics over a glass slide (Patel et al. 2021, Fig. 2)
0.25 µm
per pixel at 40× (0.5 µm at 20×)
> 2 GB
a typical slide file
Gigapixel
images, stored as tiled pyramids
Patel A et al. J Pathol Inform 2021;12:50. doi:10.4103/jpi.jpi_66_21  ·  
Janowczyk A et al. Virchows Arch 2024 (Swiss recommendations)  ·  
Herrmann MD et al. J Pathol Inform 2018;9:37

## Slide 11

‹#›
▶ DOMAIN OVERVIEW
How Slides Become Data: File Formats
WSI pyramid tiled into frames (Herrmann et al. 2018, Fig. 2)
Every WSI is multi-resolution pyramid of compressed tiles.
Kind
Formats
Read with
Vendor
SVS (Aperio), NDPI (Hamamatsu), MRXS (3DHistech), SCN (Leica)
OpenSlide, Bio-Formats
Clinical standard
DICOM WSI (TCGA and CPTAC in NCI Imaging Data Commons)
wsidicom, highdicom, Slim
Open research
OME-TIFF, OME-Zarr (chunked, cloud-ready)
Bio-Formats, zarr
Model-ready
Image tiles (PNG/JPEG) and embedding arrays
PIL, h5py, PyTorch
Goode A et al. J Pathol Inform 2013 (OpenSlide)  ·  Moore J et al. Histochem Cell Biol 2023 (OME-Zarr)  ·  Fedorov A et al. Radiographics 2023 (IDC)

## Slide 12

‹#›
▶ DOMAIN OVERVIEW
Vision + Language Foundation Models
Muneer, et al. From classical machine learning to emerging foundation models: review on multimodal data integration for cancer research. Artif Intell Rev (2026). https://doi.org/10.1007/s10462-026-11522-9 
Many different ways to spatially assay a tissue
Different 2D and 3D imaging assays
Spatial Omics assays
Focus models that combine 2D microscopic imaging with clinical data text

## Slide 13

‹#›
▶ DOMAIN OVERVIEW
Vision + Language Pathology Models
https://pmc.ncbi.nlm.nih.gov/articles/PMC13081565/figure/F1/ 

## Slide 14

‹#›
▶ DOMAIN OVERVIEW
Why Pair Images with Text?
Labels for free
Captions and reports already describe what is on the slide
No pixel-level annotation needed to train
One model, many tasks
Shared embedding space for images and text
Zero-shot classification by writing a prompt
Search images with words, and the reverse
Closer to the clinic
Pathologists think and report in language
Can draft captions and reports, or answer questions
Result: a reusable encoder you can query with images, text, or both

### Speaker notes
Purpose of vision-language foundation models. Sources: CONCH (Lu et al., Nat Med 2024), PRISM (Shaikovski et al., 2024), MUSK (Xiang et al., Nature 2025).

## Slide 15

‹#›
▶ DOMAIN OVERVIEW
Pretraining Vision-Language Models
https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649/figure/F1/ 
From the MUSK Model

## Slide 16

‹#›
▶ DOMAIN OVERVIEW
Tile-Level vs Whole-Slide (WSI) Models
https://pmc.ncbi.nlm.nih.gov/articles/PMC13081565/figure/F2/ 
Also known as Patch-Level models

## Slide 17

‹#›
▶ DOMAIN OVERVIEW
Where the Training Data Comes From
CONCH training data: 1.17M image–caption pairs curated from PubMed and education sources (Lu et al., Nat Med 2024, Fig. 1a,b)
Medical Twitter: 208K pairs (PLIP)
YouTube teaching videos: ~1M pairs from 1,087 hours (Quilt-1M)
PubMed figures + captions: 1.17M pairs (CONCH)
LLM-generated captions: 1.6M pairs from TCGA tiles (PathGen)
Unpaired corpora: 50M images + 1B text tokens (MUSK)
Slides + clinical reports: 335K WSIs (TITAN), 587K WSIs (PRISM)

## Slide 18

‹#›
▶ DOMAIN OVERVIEW
Inference Tasks with Foundation Models
https://pmc.ncbi.nlm.nih.gov/articles/PMC12295649/figure/F1/ 
From the MUSK Model
Image or Text to Embedding
Similarity between Image Pairs, Text pairs
Cross Modal Retrieval
Zero-Shot Classification of Images or Text
Visual question answering
Supervised Classification or prediction 
Image Captioning
Image segmentation and labeling

## Slide 19

‹#›
▶ DOMAIN OVERVIEW
Applications: One Model, Many Clinical Tasks
CONCH vs. PLIP, BiomedCLIP and OpenAI CLIP on 14 benchmarks (Lu et al., Nat Med 2024, Fig. 1d)
Zero-shot subtyping of lung, kidney and breast cancers (TCGA NSCLC, RCC, BRCA)
Tissue classification of colorectal patches (CRC100k)
Cross-modal retrieval: find similar cases from an image or a description
Segmentation and captioning without task-specific training
Outcomes: MUSK adds prognosis and immunotherapy-response prediction
Reports and chat: TITAN, PRISM, HistoGPT draft reports; PathChat and Quilt-LLaVA answer questions

## Slide 20

‹#›
▶ DOMAIN OVERVIEW
The Model Landscape at a Glance
Model
Year
Level
Trained on
Weights
PLIP
2023
Patch
208K medical-Twitter image–text pairs
Open
QuiltNet
2023
Patch
~1M pairs from YouTube teaching videos
Open
CONCH
2024
Patch
1.17M image–caption pairs
Gated
MUSK
2025
Patch
50M images + 1B text tokens, then 1M pairs
Gated
Quilt-LLaVA
2024
Patch (chat)
107K instruction Q&A pairs
Open
PathChat
2024
ROI (chat)
456K+ instructions
Not released
TITAN
2025
Slide
335,645 WSIs + reports + synthetic captions
Gated
PRISM
2024
Slide
587,196 WSIs with clinical reports
Gated
Highlighted: preloaded on Delta for this hackathon. Details and links: hackathon_resources/pathology_vision_language_models/README.md

## Slide 21

‹#›
Your Challenge
Study foundation models trained on pathology images and related medical text
Leverage and augment them for a specific clinical application your team chooses
Aim to make clinical pathology more precise, reproducible, or interpretable
Show working progress and explain why it matters to pathologists and patients
Multi-Modal Pathology Models: Exploring Practical Applications for Improving Clinical Workflows
▶ INITIAL GOAL

## Slide 22

‹#›
▶ INITIAL GOAL
Keep It Weekend-Sized for Shared Resources
Tiles, not slides
Work on patches (e.g. 224–512 px), not whole slides
One WSI holds thousands of patches
Frozen weights
Use MUSK, CONCH, or other models as-is
Embed once, then train small heads or prompt zero-shot
Reasonable Data Sizes 
Probably avoid high resolution spatial omics and 3D imaging
Can start from the preloaded datasets

### Speaker notes
Constraints for the initial goal: tiles/patches over WSIs, frozen foundation model weights, avoid data-intensive modalities. Teams can stretch later once something works.

## Slide 23

‹#›
Similar case retrieval from image and free-text description
Report assistant that flags ambiguous terms and suggests improvements
Education tool that shows sets of example contrasts between two conditions
Your own clinical question
From Models to Clinical Impact
▶ INITIAL GOAL
MUSK
CONCH
NCSA hosted open-weight LLM models
Other models you have access to
Example Ideas
TCGA slides + reports
PathMMU image based multiple choice
Quilt-1M images from Youtube Educational videos
PathGen combined dataset
Download or available on NCSA compute
Pretrained Models
Application Datasets

## Slide 24

‹#›
Delta and DeltaAI
NCSA's GPU clusters for AI research
▶ DELTA
What's Ready for You
CONCH, MUSK, QuiltNet, and PLIP: hosted servers, plus shared weights for your own job.
Paired image–text datasets (PathGen, Quilt-1M, CRC, PathMMU) preloaded in /projects/bfqi/shared/datasets
Delta GPU nodes for inference, embeddings, and training small classifiers
Limited DeltaAI access
Access through SSH or Open OnDemand (next slides)

## Slide 25

‹#›
▶ DELTA
Access Through SSH
Open a terminal on your laptop (macOS/Linux Terminal, Windows PowerShell)
Log in with your NCSA username, password and NCSA Duo two-factor
You land on a login node: edit files and submit jobs there, but run nothing heavy
Copy files with scp or rsync, or use the OnDemand file browser
Shared ACCESS allocation [Projects]
Manage your NCSA identity [Documentation]
# Delta
ssh <username>@login.delta.ncsa.illinois.edu
# check your allocations
accounts
# should see hackathon project
# bfqi-delta-cpu and bfqi-delta-gpu
quota
# should see storage use and limits
Delta [Architecture] [Login] [OnDemand] [Documentation] [Chat Help]
DeltaAI [Architecture] [Login] [OnDemand] [Documentation] [Chat Help]

## Slide 26

‹#›
▶ DELTA
Access Through Open OnDemand
A browser portal to Delta, no terminal needed: https://openondemand.delta.ncsa.illinois.edu/ 
Log into ACCESS, then log in with your NCSA username, password and NCSA Duo two-factor
Browse and upload files, open a shell
Best starting point for medical students and first-time cluster users
Launch interactive apps like Jupyter Lab on CPU nodes

## Slide 27

‹#›
▶ DELTA
Live Check: Log In to OnDemand Now
1
Open openondemand.delta.ncsa.illinois.edu 
2
Log into ACCESS, then log in with your NCSA username, password and NCSA Duo two-facto
3
Open a shell (Clusters > Delta Shell Access) and type accounts to confirm hackathon allocation
4
Stuck? Post on Slack #compute-issues with a screenshot
IMPORTANT  You should see “bfqi-delta-gpu” and “bfqi-delta-cpu” when you run the command ‘accounts’. 

## Slide 28

‹#›
▶ DELTA
Live Check: Free Delta Nodes
1
sinfo -s
Allocated (A): Completely in use.
Idle (I): Free and available for jobs.
Other (O): Down, draining, or under maintenance.
Total (T): Total nodes in that partition

## Slide 29

‹#›
▶ DELTA
OnDemand Apps
Request only the GPUs and wall time you need, otherwise start with CPUs
Name of account: 
Partition: 
Duration of Job: 
Number of CPUs: 
Amount of RAM:
Leave idle Jupyter sessions holding a GPU (end them, or scancel)

## Slide 30

‹#›
▶ DELTA
Be a Good Steward of Shared GPUs
Do
Run work as Slurm jobs on compute nodes
Request only the GPUs and wall time you need, otherwise start with CPUs
Compute embeddings once; share them in your team folder
Don't
Train or run inference on login nodes
Leave idle Jupyter sessions holding a GPU (end them, or scancel)
Download the same large dataset again and again
16 reserved A100 GPUs shared by ~60 teams: one GPU per team at a time, short jobs.
100G in your home /u/<username> directory to start
Issues* /projects/bfqi/<username> for data and outputs; team folders in /projects/bfqi/teams/<team>.

## Slide 31

‹#›
▶ DELTA
Hackathon Compute Budget
16 reserved A100 GPUs shared by ~60 teams: one GPU per team at a time, short jobs.

## Slide 32

‹#›
▶ DELTA
File Storage Limits Overview
100G in your home /u/<username> directory
500G* in shared /projects/bfqi/<username> directory to start
500G+ space available in node-local /tmp/ for small, fast I/O
Issues* with 1000G in shared scratch space /work/hdd/bfqi/<username>, 500G* in shared /work/nvme/bfqi/<username>
`quota`

## Slide 33

‹#›
▶ SHARED FILES
Model Repositories: MUSK and CONCH
CONCH
Mahmood lab, Nat Med 2024
GitHub: mahmoodlab/CONCH
HF: MahmoodLab/CONCH
MUSK
Li lab, Nature 2025
GitHub: lilab-stanford/MUSK
HF: xiangjx/musk
# CONCH and MUSK are gated: each person must
# accept the terms themselves
#  1. request access on each HF page (fast with .edu)
#  2. on Delta: huggingface-cli login
#  3. pass conch or musk as the model; the weights
#     download to your home dir on first use
CONCH and MUSK licenses forbid sharing copies: get your own access before using them, including through the hosted servers.
Code and examples: github.com/ncsa/ashby-hackathon-2026

## Slide 34

‹#›
▶ SHARED FILES
File Formats, Preprocessing and Options
Slide
.svs / DICOM
Tissue mask
drop background
Tiles
224–512 px, 20×
Frozen encoder
MUSK or CONCH
Embeddings
.h5 / .pt
Your model
probe, retrieval, LLM
Preloaded datasets start at the tile step
Grey steps are only needed if you bring your own slides
Options: tile size and magnification, stain normalization, caching embeddings vs. computing on the fly
The shared copies start at the tile step: 
CRC = .tif tiles in one folder per class; 
Quilt-1M = .jpg images plus a lookup CSV of captions; PathGen = JSON captions plus a 2k-slide sample; PathMMU = per-source folders (ARCH, BACH, MHIST, CRC, Osteo). 
No precomputed embeddings.

## Slide 35

‹#›
▶ SHARED FILES
Preloaded Datasets
PathGen-1.6M
1.6M image–caption pairs
Captions written by a multi-agent LLM pipeline over TCGA tiles
Sun et al., arXiv:2407.00203
Quilt-1M
~1M image–text pairs
From 1,087 hours of YouTube histopathology teaching videos
Ikezogwo et al., NeurIPS 2023
NCT-CRC-HE-100K
100,000 H&E patches, 224 × 224 px
9 colorectal tissue classes
Kather et al., PLoS Med 2019; zenodo.org/record/1214456
/projects/bfqi/shared/datasets/
      quilt/                                                          # Quilt-1M
      pathgen_dataset/                                     # PathGen-1.6M
      NCT-CRC-HE-100K/data-samples/         # CRC, 9 class folders
      pathmmu/                                                 # PathMMU# Read in place, no need to copy

### Speaker notes
PathGen-1.6M citation: Sun Y et al. PathGen-1.6M: 1.6 million pathology image-text pairs generation through multi-agent collaboration. arXiv:2407.00203 (2024); verify the final venue. NCT-CRC-HE-100K: Kather JN et al. PLoS Med 2019;16(1):e1002730, doi:10.1371/journal.pmed.1002730.

## Slide 36

‹#›
▶ TUTORIALS
Slurm Job: Running Inference
Write a batch script, submit it with sbatch, check it with squeue -u $USER
Example: embed a folder of tiles with a pathology model and save to CSV (rerun to resume)
One GPU is plenty for patch-level inference
No --reservation flag needed. Keep jobs to an hour or two; jobs left running Saturday night must finish by 7:59am Sunday.
# one-time setup (login node)
git clone https://github.com/ncsa/ashby-hackathon-2026.git
cd ashby-hackathon-2026
module load pytorch-conda/2.8
pip install --user "open_clip_torch>=2.24"
# submit: <tiles dir> <output csv> <model>
sbatch --account=bfqi-delta-gpu submit.sbatch \
    <DATASET_DIR>/TUM  tum_embeddings.csv \
    openclip:wisdomik/QuiltNet-B-32
Full script and options: For CONCH or MUSK: request access on their Hugging Face pages (fast with an .edu address), run huggingface-cli login, then use conch or musk as the model.

## Slide 37

‹#›
▶ TUTORIALS
Slurm Job: Training a Classifier
Train a small head (logistic regression or MLP) on frozen embeddings
Runs in seconds on CPU; no GPU needed; compare against zero-shot prompts
Split by patient or slide, not by tile, so tiles from one patient never sit in both train and test (CRC-100K has no patient IDs, so a random split overstates accuracy)
 #!/bin/bash
 #SBATCH --account=bfqi-delta-cpu
 #SBATCH --partition=cpu
 #SBATCH --cpus-per-task=4
 #SBATCH --mem=16G
 #SBATCH --time=00:30:00
 module load pytorch-conda/2.8
 python train_probe.py embeddings.csv

## Slide 38

‹#›
▶ TUTORIALS
Querying a Served Model
Your notebook
Delta login node or OnDemand
HTTP request
image or text
Model service
GPU on Delta
Response
embeddings or answers
No GPU job of your own needed: send requests to a model Josh is serving 
Good for quick experiments: embeddings and zero-shot classification with text prompts
Call it from inside Delta; copy embeddings onto your laptop afterward if you like
Where: an A40 node under reservation sup-30434. Host, port and API key get posted on Slack (#compute-issues) 
Models: CONCH, MUSK, QuiltNet-B-32, and PLIP, one endpoint each. Same API for all four.
Auth: Authorization: Bearer <key>. Endpoints are /embed and /classify.
Example: STUDENT_GUIDE.md from the repo.
Rate limits: there are none; requests are batched together on the server.

### Speaker notes
conch: http://gpub055.delta.ncsa.illinois.edu:8000  key: [REDACTED_API_KEY]
musk: http://gpub055.delta.ncsa.illinois.edu:8001  key: [REDACTED_API_KEY]
quiltnet: http://gpub055.delta.ncsa.illinois.edu:8002  key: [REDACTED_API_KEY]
plip: http://gpub055.delta.ncsa.illinois.edu:8003  key: [REDACTED_API_KEY]

## Slide 39

‹#›
Illinois Chat
▶ OTHER COMPUTE
Campus LLM + RAG platform, free with your NetID
Ask about papers, models, and code
https://chat.illinois.edu/multimodal-pathology/chat 
Make your own channels and use APIs to access Open Models
chat.illinois.edu
Illinois Chat: AI Assistant

## Slide 40

‹#›
More Local, Open Models
▶ OTHER COMPUTE
Illinois Computes LLM platform, log in with NCSA ID
Start with 10 coins (usage varies per model by size) and slowly recharges every hour. If you run out, post with full name in #compute-issues.
Browser-based chat interface or point your own applications to single Lumen API key which can access all models
https://lumen.ncsa.illinois.edu/help/ 
Connect your Lumen account to opencode for university resources for agentic coding: Setup Guide 
See examples and guidance for using Delta and the currently running models here in the hackathon guide.
Lumen: AI Model Gateway

## Slide 41

‹#›
▶ OTHER COMPUTE
Agentic Coding with opencode
opencode is an open-source coding agent for the terminal
Point it at Lumen to use university-hosted open-weight LLMs at no cost to you
Install on Delta as HPC-GPT, so the agent can work next to your data and jobs.
Review what it writes before you run it on shared GPUs
See hackathon guide to get started with the actively running models
# On Delta: no install or API key needed
module load hpc-gpt
hpc-gpt        # run in your project folder
To install opencode on your laptop with your own lumen key, see HACKATHON_COMPUTE.md

## Slide 42

‹#›
▶ OTHER COMPUTE
New AI Research Resources
Most of these resources are brand new or have doubled in capacity in the last year.
Thanks to all of the IT support people who have made these resources a possibility

## Slide 43

‹#›
▶ TIMELINE
Hackathon 
Working
Schedule
Dates may shift with the number of teams; watch Slack.
Hack Days
Sat–Sun, Oct 17–18
Sat Oct 17 & Sun Oct 18: Hack days at NCSA, 8:30 am–5:00 pm
Wed Oct 21: Submit summary text, slides, and availability
Oct 22–26: Judging of submissions by rubric
Oct TBD: Top 10* live presentations, awards announced
Nov TBD: Awards ceremony
Access Ends
Mon, Oct 19
Submissions Due
Wed, Oct 21

### Speaker notes
Confirm the exact end time of resource access on Monday, Oct 19.

## Slide 44

‹#›
▶ TIMELINE
Project Submission
Submit by Wed, October 21
A form with your project summary (abstract) and team availability for presentations
A stand-alone presentation (PPT or PDF): aims, methods, results, challenges overcome and remaining, references. No more than 20 main slides.
Top 10* Live Presentations
About 8-minute talk plus 4 minutes of judge questions; location and schedule posted next week

## Slide 45

‹#›
How You Will Be Judged
▶ TIMELINE
Innovation & creativity: novel use of AI, ML, and information science tools or datasets to advance an interesting clinical issue
Development progress: effective use of compute, existing literature, and tools
Clinical impact: relevance and potential effect your solution could have on clinical practice
Communication: quality and clarity of your final summary and oral presentation

## Slide 46

‹#›
Help Lives in Slack
▶ HELP
Slack invite link 
#all-hackathon: for announcements
#domain-discussion: pathology and clinical questions
#compute-issues: Delta, OnDemand, Slurm, Lumen
#useful-resources: pinned links, tutorials and papers
#team-updates and #building-logistics for the admin questions and updates
Harassment and abuse will not be tolerated and result in expulsion
Sign up if you haven't yet

## Slide 47

‹#›
Most Important…
▶ HELP
Meet new people
Learn and teach new tools and tricks
Get familiar with campus resources
Play in a emerging and important domain
…and be Patient and Kind
Have Fun…

## Slide 48

Questions?
Event website
Box Resources
Ask anytime on Slack
blatti@illinois.edu
‹#›
Science
Charles Blatti  ·  blatti@illinois.edu 
Kun Wang  ·  kwang222@illinois.edu
Technical
Volodymyr Kindratenko  ·  kindrtnk@illinois.edu 
Josh Allen  ·  jallen17@illinois.edu 
Administrative
Tani Adeniba (CCIL) - tda3@illinois.edu 
Pam Joop  (NCSA) - pjoop@illinois.edu 

## Slide 49

‹#›
Later Reminders About Wrapping Up
▶ TIMELINE
Ends Sunday 6pm: Reserved GPU nodes, preloaded model servers, Lumen resets, 
Ends Wednesday:?
Ends in two weeks: access to project credits
Ends in one month: access to project file storage, slack history
Plan to keep: Illinois Chat, Box folder resources, 
