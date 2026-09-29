# Steps for the coding agent: adapt LUMIN-v2-Search-Agent's mining pipeline for vision queries

Goal: produce authentic (non-template) natural-language retrieval queries for the vision
project's `queries.csv`, using the same anchor -> harvest -> extract pipeline already built for
the KG project, pointed at MSL/HiRISE visual classes instead of PDS4 schema concepts.

Repo: C:\Users\sokhn\OneDrive\Documents\GitHub\lumin-v2-core\lumin_mining, working in `lumin_mining/`.
`vision_anchors.py` (already drafted, see file in paper_mining/) will go in that same folder.

## Step 1 — Verify the MSL class-ID mapping before anything else

`vision_anchors.py` uses class NAMES (APXS, Drill, Ground, ...), confirmed against the Deep Mars
paper (Wagstaff et al. 2018) that introduced the MSL dataset. But the actual manifest.csv used
for baseline testing uses numeric class_label IDs, and I have not verified which ID maps to
which name.

- Download `msl_synset_words-indexed.txt` from https://zenodo.org/record/1049137 (this ships
  with the dataset and is the authoritative ID -> name mapping).
- Build a small `MSL_ID_TO_NAME` dict from it and confirm it against a few images you can
  visually check (e.g. does ID 0 actually show what the file claims?).
- Do NOT assume alphabetical order matches the numeric IDs — confirm from the actual file.

## Step 2 — Reuse ads_harvest.py as-is

No changes needed. It's already domain-agnostic — it takes a query string and returns ADS docs,
regardless of what domain the query is about. Import `harvest_anchor` and `search` directly.

Needs `ADS_API_KEY` in `.env` (same as the KG pipeline already requires) 

## Step 3 — Write `llm_extract_vision.py`, adapted from `llm_extract.py`

Same shape (excerpt in, candidate phrases out, explicit/implicit flag), different question. The
KG version asks "does this map to a PDS4 field." This version should ask something closer to:

> Given this excerpt and the visual class it was found under, does the excerpt describe what
> [class] actually looks like — its shape, texture, color, size, or visual context? Extract the
> descriptive phrase or sentence if so.

Keep the explicit/implicit distinction from the original — it's a good anti-circularity
safeguard and should carry over unchanged: explicit means the excerpt is clearly describing the
visual appearance, implicit means the term is just used in passing without real descriptive
content. Only explicit extractions should go into the query set without a human double-check.

Needs `ANTHROPIC_API_KEY` in `.env` (same as the KG pipeline).

## Step 4 — Write `mine_vision_queries.py`, adapted from `mine_dataset.py`

Same orchestration shape (loop over anchors, harvest, extract, write output), but:

- Import from `vision_anchors.py` instead of `anchors.py` (no `parse_summary` step needed —
  `HIRISE_ANCHORS` / `MSL_ANCHORS` / `ALL_ANCHORS` are already plain Python lists).
- Output schema should match the vision project's `queries.csv` directly, not the KG project's
  `id/tier/target_concept_or_alias` schema:

  ```
  query_text,class_label,mission,query_type
  ```

  `query_text` = the cleaned sentence (reuse `clean_sentence()` from `mine_dataset.py` verbatim,
  it's generic). `class_label` = the mapped numeric ID from Step 1, not the anchor's name string.
  `mission` = the anchor's `.mission`. `query_type` = `"authentic"` always, for every row this
  script produces.
- Keep the same two-file safety net the KG version has: write a `vision_candidate_terms.csv`
  alongside the queries, so there's a human-reviewable trail of what was found and rejected, not
  just a black box that spat out a CSV.

## Step 5 — Run it, per mission, and sanity-check before trusting it

```bash
python mine_vision_queries.py --mission HiRISE
python mine_vision_queries.py --mission MSL
```

Expect these to behave very differently. HiRISE classes are real geomorphology terms with an
actual descriptive literature — expect a reasonable hit rate. MSL classes are mostly rover
hardware — if mining comes back thin or empty for most MSL classes, that is itself a real,
reportable finding (the same MSL/HiRISE asymmetry that's shown up in every result so far might be
baked into the literature, not just into the models), not a sign the script is broken. Don't
force it by loosening the query-quality guards just to get more rows out of MSL.

## Step 6 — Human spot-check before merging into the real queries.csv

Same discipline the original pipeline documents: read a sample of the mined rows yourself before
they become part of the actual eval set, especially anything flagged implicit rather than
explicit. A wrong or off-target "authentic" query is worse than no query at all here, since the
whole point of this dataset is that it's trustworthy.

## Step 7 — Report back what mining actually found

Specifically worth flagging back to the team:
- How many authentic queries were mined per class, for both missions
- Which classes (if any) came back completely empty
- Whether the MSL thinness hypothesis from this doc held up or didn't
