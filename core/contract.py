"""Case JSON contract: pydantic v2 models for plan §7 + CELLS_SPEC §7.2 + BUILD_PLAN §A13.

This contract lets the pipeline (step 08), backend, agent and frontend work in parallel (plan §7). Change it only
together with fixtures/*.json and frontend/src/types.ts (plan §16); tests/test_contract.py checks all three.

    Case        a case JSON on disk (artifacts/cases/<case_id>.json); may carry `eval_only` (ground truth)
    CasePublic  what the API returns: identical fields minus `eval_only`, which it rejects outright
    to_public() validate a Case and return the CasePublic dict (JSON-ready); every API response goes through it

Every model forbids unknown fields, so drift fails loudly. Cross-field rules (enforced on validation):
  * lane "surgical" => simulated is true, margin is present, flag.tier == 1; lane "biopsy" => no margin, not
    simulated, no simulated_variant, flag.tier in 2..4.
  * simulated_variant "degraded" needs `degradation`; any variant needs source_case_ids (the case(s) it was built from).
  * prediction: probs align with classes and sum to 1 (+-0.01); conformal_set uses those classes.
  * tiles are the QC-kept contract-grid tiles: idx = row * cols + col, x = col * tile_size, y = row * tile_size,
    unique idx, quality.qc_tiles_kept == len(tiles); concepts[].top_tiles and cells.tiles[].idx are listed tiles;
    tiles[].contribution only with prediction.attribution (whose target is a prediction class).
  * findings ids are unique F<n>; each finding's data carries its kind's keys (FINDING_REQUIRED_KEYS); summary
    sentences cite at least one existing finding.
  * margin: per_margin has all four margins and equals the max p of that margin's segments; involved == margins with
    per_margin > threshold (except "defer", an unreliable read, which lists no involved margins but may keep its
    threshold for display); "clear" has no involved margin, "tumor_at_margin" has at least one; each segment touches
    its own image edge (superior: y = 0, inferior: y + size = height, medial: x = 0, lateral: x + size = width).
  * cells: status "ok" needs model and mpp_model; other statuses carry no tiles or features.
  * agent: "drafted" has orders, "none" has none.
  * contrast_case.detail (optional; null when features are missing): one row per differs_most_in concept in the same
    order; delta = query_z - contrast_z; reliable iff the concept's spot check has acceptable / reviewed >=
    SPOTCHECK_RELIABLE_MIN (never with no spot check); query_region is a contract-grid index.
  * concept findings may carry data.spotcheck ({acceptable, reviewed} or null) and data.reliable, always together,
    with the same rule (step 08 always writes both; backend.summary never gives an unreliable concept as a reason).

Wording (plan §16): fields hold suggestions, never diagnoses; the UI says "suggested", "associated with",
"simulated", "research use only".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core import schemas

SCHEMA_VERSION = "1.0"
# A concept counts as a reason in the look-alike comparison only when at least this share of its reviewed spot-check
# regions was acceptable. Fixed before looking at which concepts pass (judge-feedback plan T4); do not tune.
SPOTCHECK_RELIABLE_MIN = 0.75


def spotcheck_passed(spot: Any) -> bool:
    """The fixed reliability rule: a spot check with reviewed > 0 and acceptable / reviewed >= SPOTCHECK_RELIABLE_MIN.
    `spot` is {acceptable, reviewed}, a SpotcheckCount, or None (no spot check: never reliable)."""
    if spot is None:
        return False
    ok, n = (spot["acceptable"], spot["reviewed"]) if isinstance(spot, Mapping) else (spot.acceptable, spot.reviewed)
    return int(n) > 0 and int(ok) / int(n) >= SPOTCHECK_RELIABLE_MIN

# ---- vocabularies (mirrored as union types in frontend/src/types.ts) ---------------------------------------------------
Lane = Literal["surgical", "biopsy"]
FlagReason = Literal[
    "aggressive_top",
    "aggressive_in_conformal_set",
    "set_size_gt_1",
    "empty_set",
    "low_reliability",
    "insufficient_tissue",
    "aging_promoted",
    "surgical_frozen_section",
]
MarginName = Literal["superior", "inferior", "medial", "lateral"]
MarginCall = Literal["clear", "tumor_at_margin", "defer"]
FindingKind = Literal["prediction", "concept", "retrieval", "quality", "cells", "margin"]
AgentStatus = Literal["none", "drafted", "approved", "edited_and_approved", "deferred_closer_look", "dismissed"]
OrderType = Literal["stain", "recut", "deeper_levels", "new_sample"]
ConceptSource = Literal["text", "cav", "cell"]
ConceptDirection = Literal["up", "down", "context"]
CellsStatus = Literal["ok", "disabled", "failed"]
CellType = Literal["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
SimulatedVariant = Literal["degraded", "composite"]
DegradeType = Literal["blur", "jpeg", "stain", "resample"]

ENUMS = {
    "Lane": Lane, "FlagReason": FlagReason, "MarginName": MarginName, "MarginCall": MarginCall,
    "FindingKind": FindingKind, "AgentStatus": AgentStatus, "OrderType": OrderType, "ConceptSource": ConceptSource,
    "ConceptDirection": ConceptDirection, "CellsStatus": CellsStatus, "CellType": CellType,
    "SimulatedVariant": SimulatedVariant, "DegradeType": DegradeType,
}
FLAG_REASONS: tuple[str, ...] = get_args(FlagReason)
MARGIN_NAMES: tuple[str, ...] = get_args(MarginName)

# Keys each finding kind must carry in `data` (extra keys allowed; numbers quoted by summaries must appear here).
FINDING_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "prediction": ("top", "p", "set"),
    "concept": ("concept", "contribution", "tiles"),
    "retrieval": ("agree", "of", "label"),
    "quality": ("reliability_flag",),
    "cells": ("feature", "value", "ref_percentile", "tiles"),
    "margin": ("suggested_call", "involved"),
}

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegInt = Annotated[int, Field(ge=0)]
PosInt = Annotated[int, Field(gt=0)]
CaseId = Annotated[str, Field(pattern=schemas.CASE_ID_PATTERN)]
ImageRef = Annotated[str, Field(pattern=schemas.IMAGE_REF_PATTERN)]
CellRef = Annotated[str, Field(pattern=schemas.CELL_REF_PATTERN)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- blocks ------------------------------------------------------------------------------------------------------------
class ImageInfo(_Model):
    display: ImageRef                 # display JPEG, e.g. "images/BACH_0123.jpg" (downscaled; coordinates stay in source px)
    width: PosInt                     # source image width in pixels
    height: PosInt
    tile_size: PosInt                 # contract-grid tile edge (512)
    grid: tuple[PosInt, PosInt]       # [rows, cols], e.g. [3, 4]


class Quality(_Model):
    qc_tiles_kept: NonNegInt          # QC-passing contract-grid tiles (== len(tiles))
    ood_percentile: Prob              # OOD distance percentile vs clean calibration cases
    reliability: Prob                 # 1 - P(error) from the reliability combiner
    reliability_flag: bool


class Attribution(_Model):
    """How tiles[].contribution decomposes the read: constant + sum of the contributions of every listed tile equals
    log P(target) - mean_k log P(k) of the calibrated probabilities (pipeline/05d_tile_contrib.py)."""
    method: Literal["exact_linear"]
    target: NonEmptyStr               # class whose centered logit is decomposed (the aggressive class)
    constant: float                   # centered intercept / temperature
    unit: Literal["centered calibrated logit"]


class Prediction(_Model):
    model: NonEmptyStr                # production model name (settings model_choice)
    classes: list[NonEmptyStr] = Field(min_length=2)
    probs: list[Prob]                 # calibrated, aligned with classes
    risk_score: Prob                  # calibrated P(aggressive class)
    conformal_alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    conformal_set: list[str]          # may be empty (flagged as reason empty_set)
    attribution: Optional[Attribution] = None   # null when the read's model has no per-tile decomposition

    @model_validator(mode="after")
    def _consistent(self) -> "Prediction":
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("prediction.classes must be unique")
        if len(self.probs) != len(self.classes):
            raise ValueError(f"prediction.probs has {len(self.probs)} values for {len(self.classes)} classes")
        if not math.isclose(sum(self.probs), 1.0, abs_tol=0.01):
            raise ValueError(f"prediction.probs must sum to 1 (got {sum(self.probs):.4f})")
        unknown = [c for c in self.conformal_set if c not in self.classes]
        if unknown or len(set(self.conformal_set)) != len(self.conformal_set):
            raise ValueError(f"prediction.conformal_set must hold unique prediction classes; unknown {unknown}")
        if self.attribution is not None and self.attribution.target not in self.classes:
            raise ValueError(f"prediction.attribution.target {self.attribution.target!r} is not a prediction class")
        return self


class ConceptComponents(_Model):
    text_z: Optional[float] = None    # text-prompt score (z)
    cav_z: Optional[float] = None     # caption-derived concept-vector score (z); null when not caption-consistent


class Concept(_Model):
    id: NonEmptyStr                   # settings concepts.ids, or a cell feature id when source == "cell"
    label: NonEmptyStr
    value: float                      # case-level value (z units for text/cav; physical units for cell)
    contribution: float               # signed push toward (+) / away from (-) the aggressive class
    direction: ConceptDirection       # expected direction (sanity reference from config); not the learned sign
    spread: Optional[Annotated[float, Field(ge=0.0)]] = None   # prompt-ensemble spread (text/cav)
    top_tiles: list[NonNegInt]        # tile idx that drove the value (highlighted in the UI)
    source: ConceptSource = "text"    # text = text prompts; cav = text + caption-derived vector; cell = measured
    components: Optional[ConceptComponents] = None
    caption_consistent: Optional[bool] = None   # concept vector passed held-out AUROC (not "validated")
    value_std: Optional[float] = None  # case value standardized against train cases (SD units): the concept model's
                                       # input; sign = above (+) / below (-) the average training case
    frac_std: Optional[float] = None   # the concept's second input, share of regions above concepts.z_threshold,
                                       # standardized against train cases (SD units); the push sums both inputs


class Tile(_Model):
    idx: NonNegInt
    row: NonNegInt
    col: NonNegInt
    x: NonNegInt                      # source pixels
    y: NonNegInt
    risk: Prob                        # tile-level P(aggressive class) (heatmap)
    concepts: dict[str, float]        # concept id -> tile z
    attention: Optional[Annotated[float, Field(ge=0.0)]] = None   # ABMIL attention when that model is used
    contribution: Optional[float] = None   # region's additive term (prediction.attribution); null when not pooled


class SimilarCase(_Model):
    case_id: CaseId                   # train-split reference case (its label may be shown)
    label: NonEmptyStr
    similarity: Annotated[float, Field(ge=-1.0, le=1.0)]
    image: ImageRef


class SpotcheckCount(_Model):
    acceptable: NonNegInt             # reviewed top regions judged to show the concept
    reviewed: PosInt                  # top regions reviewed (eval/spotcheck.json summary; AI or human, see that file)

    @model_validator(mode="after")
    def _consistent(self) -> "SpotcheckCount":
        if self.acceptable > self.reviewed:
            raise ValueError(f"spotcheck acceptable {self.acceptable} exceeds reviewed {self.reviewed}")
        return self


class ContrastDetail(_Model):
    """One differs_most_in concept: case-level feature (c__top<k>) z-scored on train cases, for both cases."""
    concept: NonEmptyStr              # settings concepts.ids
    label: NonEmptyStr                # concept bank display label
    query_z: float                    # this case, SD of training cases
    contrast_z: float                 # the contrast case, same scale
    delta: float                      # query_z - contrast_z (from unrounded values; |delta| is the ranking key)
    spotcheck: Optional[SpotcheckCount]   # null when the concept has no reviewed spot-check regions
    reliable: bool                    # spot check passed (acceptable / reviewed >= SPOTCHECK_RELIABLE_MIN)
    query_region: Optional[NonNegInt]     # contract-grid idx of this case's highest tile score; null when unknown
    contrast_region: Optional[NonNegInt]  # same for the contrast case

    @model_validator(mode="after")
    def _consistent(self) -> "ContrastDetail":
        passed = spotcheck_passed(self.spotcheck)
        if self.reliable != passed:
            raise ValueError(f"contrast detail {self.concept}: reliable must be {passed} (spot check "
                             f"{self.spotcheck.model_dump() if self.spotcheck else None}, cut-off {SPOTCHECK_RELIABLE_MIN})")
        if abs(self.delta - (self.query_z - self.contrast_z)) > 0.02:
            raise ValueError(f"contrast detail {self.concept}: delta {self.delta} != query_z - contrast_z")
        return self


class ContrastCase(_Model):
    case_id: CaseId                   # nearest reference with a different label
    label: NonEmptyStr
    similarity: Annotated[float, Field(ge=-1.0, le=1.0)]
    differs_most_in: list[str]        # concept ids with the largest |delta z|
    image: Optional[ImageRef] = None  # optional thumbnail (defaults to images/<case_id>.jpg in the UI)
    detail: Optional[list[ContrastDetail]] = None   # per differs_most_in concept; null when features are missing

    @model_validator(mode="after")
    def _consistent(self) -> "ContrastCase":
        if self.detail is not None and [d.concept for d in self.detail] != self.differs_most_in:
            raise ValueError("contrast_case.detail must list the differs_most_in concepts in the same order")
        return self


class Agreement(_Model):
    conch_musk_same_top: Optional[bool] = None
    neighbor_agreement: Optional[Prob] = None   # share of the top-k references with the predicted class


class MarginSegment(_Model):
    margin: MarginName                # corner squares appear once per adjacent margin
    x: NonNegInt                      # source pixels
    y: NonNegInt
    size: PosInt
    p: Prob                           # fine-scale P(carcinoma) in the segment


class Margin(_Model):
    suggested_call: MarginCall
    involved: list[MarginName]
    per_margin: dict[MarginName, Prob]         # max segment p per margin
    strip_um: Optional[Annotated[float, Field(gt=0.0)]] = None   # strip depth in µm (strip_px x mpp)
    threshold: Optional[Prob] = None           # margin involved iff per_margin > threshold
    segments: list[MarginSegment] = []
    involved_length_um: Optional[dict[MarginName, Annotated[float, Field(ge=0.0)]]] = None

    @model_validator(mode="after")
    def _consistent(self) -> "Margin":
        missing = [m for m in MARGIN_NAMES if m not in self.per_margin]
        if missing:
            raise ValueError(f"margin.per_margin needs all four margins; missing {missing}")
        if len(set(self.involved)) != len(self.involved):
            raise ValueError("margin.involved must not repeat margins")
        if self.suggested_call == "clear" and self.involved:
            raise ValueError("margin call 'clear' cannot list involved margins")
        if self.suggested_call == "tumor_at_margin" and not self.involved:
            raise ValueError("margin call 'tumor_at_margin' needs at least one involved margin")
        if self.suggested_call == "defer" and (self.involved or self.involved_length_um):
            raise ValueError("margin call 'defer' (unreliable read) lists no involved margins; per_margin and "
                             "segments still carry the probabilities for display")
        if self.threshold is not None and self.suggested_call != "defer":
            above = {m for m in MARGIN_NAMES if self.per_margin[m] > self.threshold}
            if set(self.involved) != above:
                raise ValueError(f"margin.involved {self.involved} must equal margins above threshold {sorted(above)}")
        if self.segments:
            for name in MARGIN_NAMES:
                ps = [s.p for s in self.segments if s.margin == name]
                if not ps:
                    raise ValueError(f"margin.segments has no {name} segment")
                if not math.isclose(max(ps), self.per_margin[name], abs_tol=1e-6):
                    raise ValueError(f"margin.per_margin[{name}] = {self.per_margin[name]} is not the max segment p {max(ps)}")
        if self.involved_length_um and not set(self.involved_length_um) <= set(self.involved):
            raise ValueError("margin.involved_length_um may only list involved margins")
        return self


class CellTile(_Model):
    idx: NonNegInt                    # contract-grid tile idx
    overlay: CellRef                  # "cells/<case_id>_<idx>_overlay.png"
    counts: dict[CellType, NonNegInt]


class CellFeature(_Model):
    value: Optional[float]            # null when below the minimum nucleus counts
    unit: str                         # "" for ratios, "per mm²", "µm²"
    ref_percentile: Optional[Prob]    # percentile among train Normal/Benign cases
    contribution: Optional[float]     # coefficient x standardized value when the model uses cell features


class CellsBlock(_Model):
    status: CellsStatus               # absent or {"status": "disabled"} when cells.enabled is false
    model: Optional[NonEmptyStr] = None
    mpp_model: Optional[Annotated[float, Field(gt=0.0)]] = None
    tiles: list[CellTile] = []
    features: dict[str, CellFeature] = {}
    flags: dict[str, bool] = {}

    @model_validator(mode="after")
    def _consistent(self) -> "CellsBlock":
        if self.status == "ok" and (self.model is None or self.mpp_model is None):
            raise ValueError("cells status 'ok' needs model and mpp_model")
        if self.status != "ok" and (self.tiles or self.features):
            raise ValueError(f"cells status '{self.status}' must not carry tiles or features")
        return self


class Flag(_Model):
    flagged: bool                     # biopsy with a model/quality flag (core.tiers.flag_for)
    tier: Annotated[int, Field(ge=1, le=4)]
    reasons: list[FlagReason]


class Finding(_Model):
    id: Annotated[str, Field(pattern=r"^F[1-9][0-9]*$")]
    kind: FindingKind
    data: dict[str, Any]

    @model_validator(mode="after")
    def _keys(self) -> "Finding":
        missing = [k for k in FINDING_REQUIRED_KEYS[self.kind] if k not in self.data]
        if missing:
            raise ValueError(f"finding {self.id} ({self.kind}) data is missing {missing}")
        spot_keys = {"spotcheck", "reliable"} & set(self.data)
        if self.kind == "concept" and spot_keys:
            if len(spot_keys) < 2:
                raise ValueError(f"finding {self.id} (concept) data needs spotcheck and reliable together")
            spot = self.data["spotcheck"]
            if spot is not None:
                if not isinstance(spot, Mapping):
                    raise ValueError(f"finding {self.id} spotcheck must be {{acceptable, reviewed}} or null")
                SpotcheckCount.model_validate(spot)
            reliable = self.data["reliable"]
            if not isinstance(reliable, bool) or reliable != spotcheck_passed(spot):
                raise ValueError(f"finding {self.id} ({self.data.get('concept')}): reliable must be "
                                 f"{spotcheck_passed(spot)} for spot check {spot} (cut-off {SPOTCHECK_RELIABLE_MIN})")
        return self


class SummarySentence(_Model):
    text: NonEmptyStr
    findings: list[str] = Field(min_length=1)   # cited finding ids


class Summary(_Model):
    generator: Annotated[str, Field(pattern=r"^(template|lumen:[A-Za-z0-9._\-]+(\+template)?)$")]   # template, LLM, or LLM
                                                  # with required template sentences appended (backend.summary)
    sentences: list[SummarySentence]


class Order(_Model):
    type: OrderType
    items: list[NonEmptyStr] = Field(min_length=1)
    rule_id: NonEmptyStr              # config/stain_rules.yaml rule that drafted it
    rationale: NonEmptyStr


class BlockedAction(_Model):
    action: NonEmptyStr               # e.g. "draft_stains"
    reason: NonEmptyStr               # guard shown in the UI (e.g. never in the frozen lane)
    rule_id: Optional[str] = None


class Agent(_Model):
    status: AgentStatus
    orders: list[Order] = []
    blocked: list[BlockedAction] = []

    @model_validator(mode="after")
    def _consistent(self) -> "Agent":
        if self.status == "drafted" and not self.orders:
            raise ValueError("agent status 'drafted' needs at least one order")
        if self.status == "none" and self.orders:
            raise ValueError("agent status 'none' cannot carry orders")
        return self


class Degradation(_Model):
    type: DegradeType
    severity: Annotated[int, Field(ge=1)]


class EvalOnly(BaseModel):
    """Ground truth: never shown in the worklist or case view, never returned by the API."""

    model_config = ConfigDict(extra="allow")
    true_label: str
    split: str


# ---- the case ------------------------------------------------------------------------------------------------------------
class CaseBase(_Model):
    schema_version: Literal["1.0"]
    synthetic: bool = False           # true for fixtures and data_synthetic outputs
    case_id: CaseId
    lane: Lane
    simulated: bool                   # true for the simulated surgical lane
    simulated_variant: Optional[SimulatedVariant] = None
    degradation: Optional[Degradation] = None     # required when simulated_variant == "degraded"
    source_case_ids: list[CaseId] = []            # source case(s) of a simulated variant
    arrived_offset_s: Annotated[float, Field(ge=0.0)]   # arrival offset on the simulator clock
    image: ImageInfo
    quality: Quality
    prediction: Prediction
    concepts: list[Concept]
    tiles: list[Tile]
    similar_cases: list[SimilarCase]
    contrast_case: Optional[ContrastCase]
    agreement: Agreement
    margin: Optional[Margin]
    cells: Optional[CellsBlock] = None
    flag: Flag
    findings: list[Finding]
    summary: Summary
    agent: Agent

    @model_validator(mode="after")
    def _cross_field(self) -> "CaseBase":
        if self.lane == "surgical":
            if not self.simulated:
                raise ValueError("surgical lane cases must be simulated (simulated frozen section)")
            if self.margin is None:
                raise ValueError("surgical lane cases need a margin read")
            if self.flag.tier != 1:
                raise ValueError("surgical lane cases are flag tier 1")
        else:
            if self.margin is not None:
                raise ValueError("biopsy cases have margin null")
            if self.simulated or self.simulated_variant is not None:
                raise ValueError("biopsy cases are not simulated")
            if self.flag.tier == 1:
                raise ValueError("flag tier 1 is reserved for the surgical lane")
        if (self.simulated_variant == "degraded") != (self.degradation is not None):
            raise ValueError("degradation is required for simulated_variant 'degraded' and only then")
        if (self.simulated_variant is not None) != bool(self.source_case_ids):
            raise ValueError("source_case_ids are required for a simulated_variant and only then")
        self._check_tiles()
        self._check_findings()
        if self.margin is not None:
            self._check_margin_geometry()
        return self

    def _check_tiles(self) -> None:
        rows, cols = self.image.grid
        size = self.image.tile_size
        seen: set[int] = set()
        for tile in self.tiles:
            ok = (tile.row < rows and tile.col < cols and tile.idx == tile.row * cols + tile.col
                  and tile.x == tile.col * size and tile.y == tile.row * size)
            if not ok:
                raise ValueError(f"tile {tile.idx} does not match the {rows}x{cols} grid of {size}-px tiles")
            if tile.idx in seen:
                raise ValueError(f"tile {tile.idx} is listed twice")
            seen.add(tile.idx)
        if self.prediction.attribution is None and any(t.contribution is not None for t in self.tiles):
            raise ValueError("tiles[].contribution needs prediction.attribution")
        if self.quality.qc_tiles_kept != len(self.tiles):
            raise ValueError(f"quality.qc_tiles_kept = {self.quality.qc_tiles_kept} but {len(self.tiles)} tiles are listed")
        ids = [c.id for c in self.concepts]
        if len(set(ids)) != len(ids):
            raise ValueError("concept ids must be unique")
        for row in (self.contrast_case.detail or []) if self.contrast_case else []:
            if row.query_region is not None and row.query_region >= rows * cols:
                raise ValueError(f"contrast_case.detail {row.concept} query_region {row.query_region} is outside the "
                                 f"{rows}x{cols} grid")
        for concept in self.concepts:
            stray = [i for i in concept.top_tiles if i not in seen]
            if stray:
                raise ValueError(f"concept {concept.id} top_tiles {stray} are not listed tiles")
        if self.cells is not None:
            stray = [t.idx for t in self.cells.tiles if t.idx not in seen]
            if stray:
                raise ValueError(f"cells.tiles idx {stray} are not listed tiles")

    def _check_findings(self) -> None:
        ids = [f.id for f in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError("finding ids must be unique")
        for sentence in self.summary.sentences:
            unknown = [f for f in sentence.findings if f not in ids]
            if unknown:
                raise ValueError(f"summary sentence cites unknown findings {unknown}")

    def _check_margin_geometry(self) -> None:
        width, height = self.image.width, self.image.height
        edge = {
            "superior": lambda s: s.y == 0,
            "inferior": lambda s: s.y + s.size == height,
            "medial": lambda s: s.x == 0,
            "lateral": lambda s: s.x + s.size == width,
        }
        for seg in self.margin.segments:
            if seg.x + seg.size > width or seg.y + seg.size > height or not edge[seg.margin](seg):
                raise ValueError(f"{seg.margin} segment at ({seg.x}, {seg.y}) does not sit on the {seg.margin} edge")


class Case(CaseBase):
    eval_only: Optional[EvalOnly] = None

    def public(self) -> "CasePublic":
        return CasePublic.model_validate(self.model_dump(exclude={"eval_only"}))


class CasePublic(CaseBase):
    """API shape: Case minus eval_only. Passing eval_only raises (extra fields are forbidden)."""


# Models mirrored field for field in frontend/src/types.ts (tests/test_contract.py).
PUBLIC_MODELS = (
    ImageInfo, Quality, Attribution, Prediction, ConceptComponents, Concept, Tile, SimilarCase, SpotcheckCount,
    ContrastDetail, ContrastCase, Agreement,
    MarginSegment, Margin, CellTile, CellFeature, CellsBlock, Flag, Finding, SummarySentence, Summary, Order,
    BlockedAction, Agent, Degradation, CasePublic,
)


def to_public(case: Mapping[str, Any] | Case) -> dict[str, Any]:
    """Validate a case (dict or Case) and return its public JSON-ready dict without eval_only."""
    model = case if isinstance(case, Case) else Case.model_validate(case)
    return model.public().model_dump(mode="json")
