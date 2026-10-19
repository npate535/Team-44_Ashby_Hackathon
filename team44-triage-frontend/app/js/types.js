// JSDoc typedefs mirroring the Case JSON contract. Keep in sync with core/contract.py + frontend/src/types.ts
// (converted from frontend/src/types.ts; tests/test_frontend_static.py checks every interface is present).
// Research prototype — not for clinical use. The API returns CasePublic (never eval_only).

/** @typedef {"surgical" | "biopsy"} Lane */
/** @typedef {"aggressive_top" | "aggressive_in_conformal_set" | "set_size_gt_1" | "empty_set" | "low_reliability" | "insufficient_tissue" | "aging_promoted" | "surgical_frozen_section"} FlagReason */
/** @typedef {"superior" | "inferior" | "medial" | "lateral"} MarginName */
/** @typedef {"clear" | "tumor_at_margin" | "defer"} MarginCall */
/** @typedef {"prediction" | "concept" | "retrieval" | "quality" | "cells" | "margin"} FindingKind */
/** @typedef {"none" | "drafted" | "approved" | "edited_and_approved" | "deferred_closer_look" | "dismissed"} AgentStatus */
/** @typedef {"stain" | "recut" | "deeper_levels" | "new_sample"} OrderType */
/** @typedef {"text" | "cav" | "cell"} ConceptSource */
/** @typedef {"up" | "down" | "context"} ConceptDirection */
/** @typedef {"ok" | "disabled" | "failed"} CellsStatus */
/** @typedef {"neoplastic" | "inflammatory" | "connective" | "dead" | "epithelial"} CellType */
/** @typedef {"degraded" | "composite"} SimulatedVariant */
/** @typedef {"blur" | "jpeg" | "stain" | "resample"} DegradeType */

/**
 * @typedef {Object} ImageInfo
 * @property {string} display
 * @property {number} width
 * @property {number} height
 * @property {number} tile_size
 * @property {[number, number]} grid
 */

/**
 * @typedef {Object} Quality
 * @property {number} qc_tiles_kept
 * @property {number} ood_percentile
 * @property {number} reliability
 * @property {boolean} reliability_flag
 */

/**
 * @typedef {Object} Attribution
 * @property {"exact_linear"} method
 * @property {string} target
 * @property {number} constant
 * @property {"centered calibrated logit"} unit
 */

/**
 * @typedef {Object} Prediction
 * @property {string} model
 * @property {string[]} classes
 * @property {number[]} probs
 * @property {number} risk_score
 * @property {number} conformal_alpha
 * @property {string[]} conformal_set
 * @property {Attribution | null} [attribution]
 */

/**
 * @typedef {Object} ConceptComponents
 * @property {number | null} [text_z]
 * @property {number | null} [cav_z]
 */

/**
 * @typedef {Object} Concept
 * @property {string} id
 * @property {string} label
 * @property {number} value
 * @property {number} contribution
 * @property {ConceptDirection} direction
 * @property {number | null} [spread]
 * @property {number[]} top_tiles
 * @property {ConceptSource} [source]
 * @property {ConceptComponents | null} [components]
 * @property {boolean | null} [caption_consistent]
 */

/**
 * @typedef {Object} Tile
 * @property {number} idx
 * @property {number} row
 * @property {number} col
 * @property {number} x
 * @property {number} y
 * @property {number} risk
 * @property {Record<string, number>} concepts
 * @property {number | null} [attention]
 * @property {number | null} [contribution]
 */

/**
 * @typedef {Object} SimilarCase
 * @property {string} case_id
 * @property {string} label
 * @property {number} similarity
 * @property {string} image
 */

/**
 * @typedef {Object} SpotcheckCount
 * @property {number} acceptable
 * @property {number} reviewed
 */

/**
 * @typedef {Object} ContrastDetail
 * @property {string} concept
 * @property {string} label
 * @property {number} query_z
 * @property {number} contrast_z
 * @property {number} delta
 * @property {SpotcheckCount | null} spotcheck
 * @property {boolean} reliable
 * @property {number | null} query_region
 * @property {number | null} contrast_region
 */

/**
 * @typedef {Object} ContrastCase
 * @property {string} case_id
 * @property {string} label
 * @property {number} similarity
 * @property {string[]} differs_most_in
 * @property {string | null} [image]
 * @property {ContrastDetail[] | null} [detail]
 */

/**
 * @typedef {Object} Agreement
 * @property {boolean | null} [conch_musk_same_top]
 * @property {number | null} [neighbor_agreement]
 */

/**
 * @typedef {Object} MarginSegment
 * @property {MarginName} margin
 * @property {number} x
 * @property {number} y
 * @property {number} size
 * @property {number} p
 */

/**
 * @typedef {Object} Margin
 * @property {MarginCall} suggested_call
 * @property {MarginName[]} involved
 * @property {Partial<Record<MarginName, number>>} per_margin
 * @property {number | null} [strip_um]
 * @property {number | null} [threshold]
 * @property {MarginSegment[]} [segments]
 * @property {Partial<Record<MarginName, number>> | null} [involved_length_um]
 */

/**
 * @typedef {Object} CellTile
 * @property {number} idx
 * @property {string} overlay
 * @property {Partial<Record<CellType, number>>} counts
 */

/**
 * @typedef {Object} CellFeature
 * @property {number | null} value
 * @property {string} unit
 * @property {number | null} ref_percentile
 * @property {number | null} contribution
 */

/**
 * @typedef {Object} CellsBlock
 * @property {CellsStatus} status
 * @property {string | null} [model]
 * @property {number | null} [mpp_model]
 * @property {CellTile[]} [tiles]
 * @property {Record<string, CellFeature>} [features]
 * @property {Record<string, boolean>} [flags]
 */

/**
 * @typedef {Object} Flag
 * @property {boolean} flagged
 * @property {number} tier
 * @property {FlagReason[]} reasons
 */

/**
 * @typedef {Object} Finding
 * @property {string} id
 * @property {FindingKind} kind
 * @property {Record<string, unknown>} data
 */

/**
 * @typedef {Object} SummarySentence
 * @property {string} text
 * @property {string[]} findings
 */

/**
 * @typedef {Object} Summary
 * @property {string} generator
 * @property {SummarySentence[]} sentences
 */

/**
 * @typedef {Object} Order
 * @property {OrderType} type
 * @property {string[]} items
 * @property {string} rule_id
 * @property {string} rationale
 */

/**
 * @typedef {Object} BlockedAction
 * @property {string} action
 * @property {string} reason
 * @property {string | null} [rule_id]
 */

/**
 * @typedef {Object} Agent
 * @property {AgentStatus} status
 * @property {Order[]} [orders]
 * @property {BlockedAction[]} [blocked]
 */

/**
 * @typedef {Object} Degradation
 * @property {DegradeType} type
 * @property {number} severity
 */

/**
 * @typedef {Object} CasePublic
 * @property {"1.0"} schema_version
 * @property {boolean} [synthetic]
 * @property {string} case_id
 * @property {Lane} lane
 * @property {boolean} simulated
 * @property {SimulatedVariant | null} [simulated_variant]
 * @property {Degradation | null} [degradation]
 * @property {string[]} [source_case_ids]
 * @property {number} arrived_offset_s
 * @property {ImageInfo} image
 * @property {Quality} quality
 * @property {Prediction} prediction
 * @property {Concept[]} concepts
 * @property {Tile[]} tiles
 * @property {SimilarCase[]} similar_cases
 * @property {ContrastCase | null} contrast_case
 * @property {Agreement} agreement
 * @property {Margin | null} margin
 * @property {CellsBlock | null} [cells]
 * @property {Flag} flag
 * @property {Finding[]} findings
 * @property {Summary} summary
 * @property {Agent} agent
 */

export {};
