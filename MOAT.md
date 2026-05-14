# Colony Counter Moat Documentation

This file documents the scientific and engineering judgment encoded into Colony Counter. Each entry represents a piece of domain expertise that a generic developer with API access could not replicate without microbiology bench experience or significant scientific consultation.

Add new entries when the analysis pipeline is updated. Date each entry. Reference the file and code section where the judgment lives.

The goal of this file is twofold:
1. A growing record of what makes Colony Counter defensible against generic competitors (ImageJ plugins, ChatGPT-vision wrappers, off-the-shelf computer vision tools).
2. Internal documentation so the reasoning behind each piece of judgment is preserved.

---

## 2026-05-13 — Image retention for accuracy validation

### What changed
File: `app.py`
Commit: `08d6b59`

Added a backend image retention pipeline. Every plate image uploaded to `/analyze` is now stored in a private Supabase bucket (`colony-uploads`) with a row written to the `colony_analyses` table containing the colony count, density channel, MIME-type-aware filename extension, and metadata. The storage is wrapped in a best-effort try/except so the user still receives their count if storage fails.

### Why this is moat work, not just engineering
This is the foundation for validating model accuracy against real-world bench images. The original training set consisted of plates Chioma photographed herself in her own lab, which represents a single imaging setup (one camera, one lighting condition, one plate type, one colony morphology). Real users will upload images taken with different cameras, different lighting, different plate media, and different bacterial species. Without retention, the model cannot be validated against this real-world variation.

The validation set built from these retained images becomes the basis for:
- Confirming that the LOW/MEDIUM/HIGH density channels are routing correctly
- Identifying systematic over- or under-counting on specific image types
- Eventually publishing accuracy benchmarks alongside competitor tools
- Justifying paid pricing with validated accuracy data

### The judgment encoded
- Storage is temporary and purposeful. Once accuracy is validated and Chioma is confident in the model, image storage will be turned off. This is not a permanent data collection program.
- The MIME type detection (`_resolve_content_type`) preserves the actual upload format (PNG stays PNG, TIFF stays TIFF) rather than re-encoding everything as JPEG. Microscopy images and high-fidelity plate photographs often arrive as PNG or TIFF, and degrading them at storage time would compromise the validation set.
- The silent-failure guard means the user-facing analysis is never blocked by storage problems. The model's accuracy validation is a backend concern, not a user concern.

---

## 2026-04 to 2026-05 — Density-channel architecture (v1.9.x)

### What changed
File: `app.py`
Commits: `5514f7b` (v1.9.1), `82ae529` (v1.9.2)

Refactored the colony detection pipeline into three density channels: LOW, MEDIUM, and HIGH. Each channel uses different thresholds, watershed parameters, and noise filters optimized for its target plate density.

### Why this is moat work, not just engineering
A single set of detection parameters cannot handle the full range of plate densities a microbiologist encounters. Sparse plates (under 30 colonies) and confluent plates (over 300 colonies) have fundamentally different counting failure modes:

- LOW density: edge artifacts and noise pixels get mistakenly counted as colonies, leading to overcounting
- MEDIUM density: most-tuned channel, the default for typical research workflows
- HIGH density: touching and overlapping colonies need watershed splitting and circularity filters to count as separate organisms

A generic computer vision pipeline without bench experience would either use one set of parameters across all images (poor accuracy at the extremes) or require the user to manually classify their plate density (poor UX). The density-channel architecture solves this by pre-scanning the image to choose the right channel automatically.

### The judgment encoded
- score_threshold_fraction calibrated per channel: 0.19 hardcoded for LOW (was inheriting 0.15-0.17 from the auto-detection that caused overcounting), 0.1275 for MEDIUM, separate for HIGH
- watershed_split_trigger calibrated: 250 for LOW, 80 for MEDIUM (smaller because medium plates have more touching colonies)
- min_circularity calibrated: 0.40 for LOW, 0.259 for MEDIUM
- Edge/center/runt noise filtering applied per channel to remove artifacts that resemble colonies at the wrong density

---

## 2026-03 to 2026-04 — Structured microbiologist prompting strategy

### What changed
Initial prototype built with Claude Vision API using a quadrant-by-quadrant counting approach with morphology assessment and confidence flagging.

### Why this is moat work, not just engineering
GPT-4 Vision had been tried on similar tasks (the Microglia Morphology Analyzer) and produced unreliable counts because it tried to count the whole image at once and would hallucinate. A bench microbiologist counting colonies on an agar plate does not look at the whole plate at once. They scan systematically, often by quadrant or by region, and they make explicit confidence judgments ("clear colony," "probable colony," "ambiguous, looking again").

Encoding this systematic counting strategy into the prompting approach produced meaningfully more accurate counts than a naive "how many colonies are on this plate" prompt. This is methodology, not engineering: it required knowing how microbiologists actually count plates manually before designing the prompts.

### The judgment encoded
- Quadrant-by-quadrant analysis to mirror manual counting practice
- Morphology assessment (color, size, shape) to distinguish true colonies from artifacts (water droplets, agar bubbles, dust)
- Confidence flagging on ambiguous regions
- Explicit handling of common artifacts that confound automated counting: condensation droplets, agar surface defects, edge effects near the plate rim

---

## How to contribute to this file

When you encode new scientific judgment into the Colony Counter pipeline, add a dated section to this file. Each section should include:

1. **Date** in YYYY-MM-DD format
2. **What changed**: the file, commit hash if available, and a one-paragraph description of the change
3. **Why this is moat work, not just engineering**: the domain expertise required to make this decision. If a generic developer could have made this change by reading documentation, it is not moat work and does not belong here.
4. **The judgment encoded**: specific, numerically precise claims that a competitor would need to discover independently. Vague statements like "we tuned the algorithm" are not useful. Specific statements like "score_threshold_fraction = 0.19 for LOW density to avoid overcounting from edge artifacts" are.

Future categories of moat work that will likely accumulate here:

- Species-specific colony morphology recognition (E. coli vs. Staphylococcus vs. yeast)
- Plate media handling (blood agar vs. MacConkey vs. selective media with different optical properties)
- Mixed-culture detection where multiple species are present on the same plate
- Contamination flagging (mold, fungal overgrowth, bacteriophage plaques)
- Confluence detection where colony counts are unreliable because colonies have merged
- Integration of biological context (incubation time, dilution series position, expected counts based on the protocol)

Each of these areas is something a microbiologist knows but a generic developer does not. Document each piece as you encode it.
