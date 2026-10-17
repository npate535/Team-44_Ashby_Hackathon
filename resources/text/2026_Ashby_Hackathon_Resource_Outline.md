# Organizer resource outline (Oct 12; older logistics may be superseded)

Extracted 2026-10-17. Current compute guide and Oct 17 Slack updates take precedence for compute/logistics.

2026 Ashby Prize in Computational Science Hackathon
Multi-Modal Pathology Models: Exploring Practical Applications for Improving Clinical Workflows
Hack days: Sat & Sun, Oct 17–18, 8:30 am–5:00 pm at NCSA  ·  Event website  ·  Box folder  ·  Slack
BEFORE SATURDAY, OCT 17: TO-DO LIST
Tonight: confirm your participation with the Confirmation Form.
By Wednesday, Oct 14: create an ACCESS account with your institutional email, log in once, then submit your ACCESS ID through this Google Form to be added to the hackathon compute allocation.
By Wednesday, Oct 14: request access to the gated Hugging Face models and datasets (approval can take a day or two; see below).
Join Slack and open the Box folder for student materials, team lists, room lists, and this week’s recording. Read a review paper or two.
Meet as a team: choose a clinical question, split roles, and sketch a plan. Try out Illinois Chat and Lumen.
By Friday: we will add you to the hackathon compute account, send Teams events, and post NCSA room assignments.
Saturday 8:30 am: arrive with your laptop and charger, working access, a team plan, and a first idea.
KEY DATES
Sat–Sun, Oct 17–18
Hack days at NCSA, 8:30 am–5:00 pm
Wed, Oct 21
Submit project abstract and team availability (form) and a stand-alone presentation (PPT or PDF, ≤ 20 main slides)
Oct 22–26
Judging of submissions by rubric
Oct TBD
Top 10* live presentations (about 8 min talk + 4 min judge questions); awards announced
Nov TBD
Awards ceremony
Schedule and submission details may be revised to accommodate the number of teams; watch Slack for updates.
HACK-DAY LOGISTICS (SAT & SUN)
8:30 am start (light breakfast)  ·  9:00 am orientation on Teams  ·  12:00 pm lunch (pizza)  ·  1:00 pm work continues (snacks)  ·  4:00 pm team briefing on Teams  ·  5:00 pm in-person support ends.
NCSA doors are locked for the weekend; staff near the first-floor main entrances will let you in from 8:30 am to 5:00 pm. Use Slack for building issues.
NCSA space (NCSA floor map) is limited: find your own team space if you can, otherwise tell us.
YOUR CHALLENGE, JUDGING, AND PRIZES
Study foundation models trained on pathology images and medical text, then leverage and augment them for a clinical application your team chooses, aiming to make pathology more precise, reproducible, or interpretable.
Example ideas: similar-case retrieval from an image plus free-text description; a report assistant that flags ambiguous terms; an education tool showing example contrasts between two conditions; or your own clinical question.
Judging: innovation & creativity · development progress · clinical impact · communication.
Prizes: $3,000 / $1,500 / $750 for the top three teams. Only current UIUC students can receive prize money; teams are strongly encouraged to include a Siebel School of Computing and Data Science student.
COMPUTE: DELTA AND DELTAAI (NCSA GPU CLUSTERS)
CONCH and MUSK weights and several paired image–text datasets are preloaded; GPU nodes are available for inference, embeddings, and supervised learning. More instructions to follow.
Delta: Documentation · Chat help     DeltaAI: Documentation · Chat help
MODELS AND DATASETS (GATED HUGGING FACE ACCESS)
Request access and agree to non-commercial terms on each page (allow a day or two):
Models: MUSK · CONCH · PathGen-CLIP     Dataset: PathGen
Highlighted application datasets (download or use the NCSA copies): PathGen, TCGA slides, PathMMU, and Quilt-1M.
AI ASSISTANTS FOR YOUR TEAM
Illinois Chat: campus LLM + RAG platform, free with your NetID. Ask our hackathon pathology chatbot about papers, models, and code.
Lumen: NCSA’s AI model gateway. Browser chat or a single API key for all hosted open models (coins recharge hourly). Connect it to opencode for agentic coding with this setup guide.
BACKGROUND READING RESOURCES
Awesome-AI4DigitalPathology (≈500 curated references)
Li et al. Multi-modal foundation models for computational pathology: a survey. TMLR (2025).
Xiang et al. A vision–language foundation model for precision oncology (MUSK). Nature (2025).
Lu et al. A visual-language foundation model for computational pathology. (CONCH). Nat Med (2024).
Muneer et al. From classical machine learning to emerging foundation models. Artif Intell Rev (2026).
CONTACTS
First place to ask questions: the hackathon Slack workspace.
Science
Charles Blatti
blatti@illinois.edu

Kun Wang
kwang222@illinois.edu
Technical
Volodymyr Kindratenko
kindrtnk@illinois.edu

Josh Allen
jallen17@illinois.edu
Administrative
Tani Adeniba (CCIL)
tda3@illinois.edu

Pam Joop (NCSA)
pjoop@illinois.edu
Organized by NCSA, CAII, the Cancer Center at Illinois, and the Cancer Digital Insights Working Group.

## Links
https://uofi.box.com/s/vsl57eshv213e6gzcovj01umzt61wp69
https://chat.illinois.edu/Delta-Documentation/chat
https://arxiv.org/abs/2401.16355
mailto:blatti@illinois.edu
https://huggingface.co/xiangjx/musk
https://pmc.ncbi.nlm.nih.gov/articles/PMC13081565/
mailto:jallen17@illinois.edu
https://cancer.illinois.edu/
https://uofi.box.com/s/vsl57eshv213e6gzcovj01umzt61wp69
https://facilityaccessmaps.fs.illinois.edu/archibus/schema/ab-products/essential/workplace/index.html?blId=0564&flId=01
https://chat.illinois.edu/multimodal-pathology/chat
https://huggingface.co/datasets/jamessyx/PathGen
https://docs.google.com/document/d/1Lk66GkbMH30M4GdnlcgiPHiLACx96cYs4pLZCUQ9Mr4/edit?tab=t.jp6or9kmw1ou
https://doi.org/10.1007/s10462-026-11522-9
mailto:kwang222@illinois.edu
https://www.ncsa.illinois.edu/
https://lumen.ncsa.illinois.edu/help/
https://huggingface.co/jamessyx/PathGen-CLIP
https://chat.illinois.edu/
https://pubmed.ncbi.nlm.nih.gov/38504017/
https://account.access-ci.org/register
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/
https://opencode.ai/download
mailto:pjoop@illinois.edu
https://forms.illinois.edu/sec/871865309
https://chat.illinois.edu/multimodal-pathology/chat
https://huggingface.co/MahmoodLab/CONCH
https://arxiv.org/abs/2306.11207
https://lumen.ncsa.illinois.edu/help/
https://doi.org/10.1038/s41586-024-08378-w
mailto:tda3@illinois.edu
https://publish.illinois.edu/cancerdigitalinsights/
https://docs.ncsa.illinois.edu/systems/delta/en/latest/
https://huggingface.co/datasets/jamessyx/PathGen
https://github.com/lingxitong/Awesome-AI4DigitalPathology
https://ai.ncsa.illinois.edu/
https://chat.illinois.edu/DeltaAI-Documentation/chat
mailto:kindrtnk@illinois.edu
https://cancer.illinois.edu/2026-ashby-prize-in-computational-science-hackathon/