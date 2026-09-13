# Strict six-camera production: Phase A read-only audit

Date: 2026-09-13 UTC

Status: Phase A complete. No formal dataset was created or modified, and no GPU
render was started.

## Decision

The existing implementation is a useful base, but it is not sufficient for the
requested production dataset. A targeted code change is required before any
pilot. The existing publisher, hash checks, no-op rule, atomic directory swap,
multi-asset compositor, and dynamic GPU queue should be reused. STORM, CVAC,
DCN, and the accepted Gaussian assets do not need retraining.

The main missing production capability is a strict, authority-enriched planner
that keeps the STORM context and official nuScenes timing identity, rejects
cross-`sample_token` rings, and renders each missing six-camera background once
per STORM context rather than once per group or asset.

## 1. Task interpretation

The unit of release is one synchronized group, not an individual triplet. A
group has one split, scene, target frame/sample identity, STORM context, frozen
checkpoint tuple, and global asset union. Each of the six cameras receives its
own visible subset of that union. A visible view uses the audited removal and
same-identity insertion path; an invisible view uses the same-camera unedited
STORM target as `input` and records an explicit no-op. Every released group has
exactly 18 flat PNG files in the fixed nuScenes ring order.

The formal output remains:

```text
/mnt/ojc/workplace2/dataset/nusc_pair_6cam/
  README.md
  train/
  val/
  metadata/train_groups.jsonl
  metadata/val_groups.jsonl
  metadata/audit/
```

Nothing is eligible for that path until the pilot and full group-level audit
pass.

## 2. Repository, source, GPU, and storage state

- `DriveHarm_sixcam` is clean on `main` at `c3c8553`, aligned with
  `origin/main`.
- The protected original `DriveHarm` repository was not changed. It contains a
  user-owned untracked `PROGRESS_20260904.md`, which must remain untouched.
- `/mnt/ojc/workplace2/dataset/nusc_pair_6cam` does not exist.
- `/mnt/ojc/workplace2/sixcam_run` is a live symlink into the provenance
  archive. The historical four-group six-camera release was not found, although
  the earlier 35-triplet regression pilot and its final 25 accepted triplets are
  still archived.
- Current sizes: `nusc_pair` 35 GB; archived `sixcam_run` 309 GB.
- `/mnt` currently has about 196 GB free and 207 million free inodes. A copied
  full release will not fit safely; same-filesystem hardlinks plus bounded
  staging are mandatory.
- At the final Phase A check, GPU 0 and GPU 1 were idle (1 MiB used, 0%); GPUs
  2--7 had active memory/utilization. No process was killed and no GPU job was
  started.

## 3. Formal `nusc_pair` audit

The audit read every released file, recomputed SHA-256, decoded it with Pillow,
and checked role membership, mode, and dimensions.

| Split | Triplets | PNG files | Unique scenes | Role alignment | Decode / RGB / 512x288 | SHA-256 vs authority |
|---|---:|---:|---:|---|---|---|
| train | 51,132 | 153,396 | 616 | exact | 153,396 / 153,396 | 153,396 / 153,396 |
| val | 9,399 | 28,197 | 110 | exact | 28,197 / 28,197 | 28,197 / 28,197 |
| total | 60,531 | 181,593 | - | no missing/extra role | zero failures | zero mismatches |

There are no duplicate sample IDs. The three basename sets are identical in
each split. The full scan took 112.6 seconds.

Authority manifests:

- train:
  `/mnt/ojc/generated_asset_library/authorities/train/nusc_pair_release_v9_records.jsonl`
  (51,132 rows, SHA-256
  `77edae0473ea26a44c926e12cd5a3f49b78292f53fd9fe5b50423a8d39ea3f6f`)
- val:
  `/mnt/ojc/generated_asset_library/provenance/sixcam_run/nusc_pair_val_reaudit_20260902/retained_release_records.jsonl`
  (9,399 rows, SHA-256
  `f6e905b41226969eb1c2337edf8ae93cf2e9f74b35b27b42757189bff6b1a759`)

Official nuScenes split replay found zero train/val scene overlap and zero scene
assigned to the wrong split. The accepted data covers 616 of 700 official train
scenes and 110 of 150 official val scenes.

## 4. Identity and lineage recoverability

The sample ID makes `scene`, STORM window/context digest, `combination_id`,
target frame, and camera recoverable for 100% of train and val records.
`sample_token` is not stored directly in either final release record.

- Rich production metadata joins exactly to 46,707/51,132 train records
  (91.35%) and 9,399/9,399 val records (100%).
- Every one of the 25,878 distinct `(camera exposure timestamp, camera)` keys in
  those joined rows matched the official nuScenes `sample_data` table and thus
  recovered both `sample_data_token` and `sample_token`.
- Val must use the repaired metadata authority. The earlier val metadata agrees
  on all GT and target hashes but only 1,601/9,399 input hashes; the repaired
  metadata agrees on all three roles for all 9,399 rows.
- The remaining 4,425 train records come from several historical authority
  classes rather than the current `local_exact_union` metadata. Their image and
  asset hashes are authoritative, but complete exposure/sample lineage must be
  fused from their original manifests or reconstructed from the official
  trajectory before they can enter a strict group.

A separate six-camera timing replay found a material synchronization issue:
some camera exposure sets assigned to one STORM target frame straddle two
official nuScenes `sample_token` values. These rings must be rejected, not
silently grouped by scene/frame.

## 5. Existing six-camera capacity

There are three different counts, because they answer different questions.

1. The current `sixcam.py`, which collapses by `(scene, frame)`, exactly
   reproduces the historical train result: 23 complete unambiguous scene-frames,
   267 candidates, 260 publishable groups, and 7 exclusions. It cannot process
   the final val authority schema.
2. Preserving the STORM context finds 30 train context-frames with all six
   baseline views and 333 groups whose 18 source files are structurally ready.
3. Official token replay rejects 15 of those groups. Therefore the current
   fully evidenced, strict directly publishable count is **318 train groups and
   0 val groups**.

The audited candidate calculation starts only from asset unions already present
in the accepted train/val releases; it does not invent new assets.

| Class | Train groups | Val groups | Total groups |
|---|---:|---:|---:|
| Official timing confirmed coherent | 43,247 | 8,762 | **52,009** |
| Historical train lineage still needing trajectory resolution | 4,352 | 0 | **4,352** |
| Known cross-`sample_token`, hard exclude | 2,128 | 551 | **2,679** |
| Pre-timing eligible pool | 49,727 | 9,313 | 59,040 |

Before timing replay, 317 train candidates were already excluded: 244 had
conflicting accepted inputs, 39 had ambiguous backgrounds inside a strict
context/camera slot, and 34 lacked explicit visibility evidence. Twelve train
rows have an asset signature not covered by the currently loaded visibility
aliases and are not used to create candidates.

The defensible production forecast is therefore:

- 318 groups are immediately strict and directly reusable;
- 52,009 groups are confirmed synchronized candidates before new-background
  image QC;
- 56,361 is the maximum current candidate ceiling if all 4,352 old-lineage
  candidates are recovered and synchronized;
- the final delivered count may be lower because a failed rendered background
  or visible edit quarantines the whole group.

## 6. Reuse and render deficit

For the confirmed synchronized pool:

| Split | Candidate groups | Missing unique baseline camera-frames | Contexts to render once | Missing visible exact-composition views |
|---|---:|---:|---:|---:|
| train | 43,247 | 40,015 | 2,711 | 406 |
| val | 8,762 | 13,189 | 878 | 50 |
| total | **52,009** | **53,204** | **3,589** | **456** |

Resolving the old train lineage ceiling would add at most 6,835 unique baseline
camera-frames in 385 contexts and 181 visible exact-composition views. Known
cross-token candidates account for another 2,613 baseline camera-frames and are
excluded from the render plan.

Existing accepted pairs and baselines remain the primary source and are
hardlinked, never rerendered. Across the pre-timing pool, the planner can reuse
50,415 unique accepted train sample IDs and all 9,399 accepted val sample IDs;
one accepted triplet can serve several group placements without duplicating
content.

The archived single-asset runs contain complete asset identity, PLY hashes,
official pose/exposure, removal support, RGB/alpha/depth layers, and occlusion
receipts. A pre-timing scan of 653 missing visible tasks found 61 whose complete
layers were technically CPU-recomposable. Another 411 had historical output but
failed a renderer gate or, for val, was later quarantined; these must not be
silently restored. The remaining tasks lack a suitable context result. The
strict planner must repeat this classification after timing filtering, then
either compose from accepted layers, render a new visible view, or exclude the
whole group.

The archived runs do not save invisible cameras' STORM baseline/GT outputs.
That is why a baseline-only extraction mode is necessary. The 53,204 missing
camera-frames must be generated as 3,589 context jobs, each loading STORM once
and exporting the needed six-camera target/GT views. Rendering per group or per
camera would waste orders of magnitude of work.

All inspected production results bind the same checkpoint tuple:

- STORM:
  `141d66c31b0e74a2d7674a37882826ee99168afc8037dd0a36482f5827ba3fbd`
- CVAC:
  `256cf70203af71609fe2fd2fa854a243311aaacf11f74e75192b1f552c9f3ca8`
- DCN:
  `eaedc491eda4200ebc728831a211980696b07a0b324ba6290abdd92b7005d08b`

## 7. Current code: reusable and missing capabilities

Reusable now:

- correct fixed six-camera ring order;
- exactly 18 flat names per group;
- hardlink/copy materialization with source and destination hash checks;
- invisible-view `input=target` behavior and replay audit;
- sibling staging plus atomic directory replacement;
- multi-asset far-to-near composition with independent alpha/depth/occlusion
  receipts;
- asynchronous dynamic GPU queue, checkpoint hash binding, shard logs, and
  resume validation.

Missing or unsafe for this production target:

- grouping currently keys baselines by only `(scene, frame, camera)`, so it can
  collapse distinct STORM windows/contexts;
- group identity omits split, official sample/target identity, instance tokens,
  PLY hashes, and checkpoint tuple;
- no hard gate verifies six-camera official timing/sample coherence;
- final val records use `selected_obj_ids` and omit top-level scene/frame, which
  the current parser does not normalize; the current command raises
  `ValueError` on val;
- visibility is indexed without frozen context/source authority and cannot
  represent the multiple historical train authority classes safely;
- no baseline-only all-camera exporter or context-level backfill manifest;
- no root-level train/val/metadata atomic release, group quarantine, or complete
  per-view lineage/occlusion receipt;
- current tests validate the historical 260-group path, not the strict context,
  token, multi-authority, val, and backfill contracts.

All 12 existing unit tests pass in the existing
`nuscenes_localization_v2_splatad` environment. The first attempted environment
was missing the declared `openai` dependency; this was an environment selection
issue, not a code-test failure.

## 8. Pilot design

After the targeted code change, run a 12-group pilot (216 formal PNG files) on
GPU 0 and GPU 1 only, one worker per GPU initially:

- 8 train groups and 4 val groups;
- at least 2 direct-reuse groups and at least 6 groups requiring newly exported
  six-camera backgrounds;
- single-asset and multi-asset unions;
- near, medium, and far depth strata from the stored per-asset median depth;
- no occlusion, verified scene-depth occlusion, and visually confirmed
  vehicle/fence/tree/pole foreground cases;
- assets visible in multiple cameras and assets visible in only one camera;
- at least four groups with multiple explicit no-op cameras;
- at least one archived-layer CPU recomposition candidate;
- an intentionally cross-token fixture that must be rejected before render.

The pilot is accepted only if every group passes all 18-file checks, all visible
views have a valid nonzero edit and full identity/geometry/occlusion receipts,
all invisible views are byte-identical `input=target`, all six official timing
bindings are coherent, and manual inspection accepts all 216 files. Contact
sheets are generated outside the formal dataset. A failure quarantines the
entire group.

The pilot also benchmarks one context with one asset versus several assets so
the full-run ETA is measured rather than inferred.

## 9. GPU and disk estimate

Historical production time was 341.78 GPU-hours for 12,726 train single-asset
context jobs (mean 96.68 seconds) and 104.43 GPU-hours for 2,825 val jobs (mean
133.08 seconds). Applying those full-job means to the context-level baseline
plan gives a conservative estimate:

- confirmed pool: about 105 GPU-hours;
- including the unresolved train ceiling: about 116 GPU-hours;
- GPU 0+1 ideal wall time: about 53--58 hours;
- GPU 0+1 practical wall time with 15% orchestration/retry allowance: about
  61--67 hours;
- eventual eight-GPU practical wall time: about 15--17 hours, provided the GPUs
  are actually free.

These are conservative full-pipeline bounds. The proposed baseline-only mode
skips asset removal/insertion and should be faster; the 12-group pilot supplies
the authoritative throughput estimate before bulk production.

At the observed mean PNG size (201,324 bytes), 56,361 groups would contain
1,014,498 logical image names and about 204 GB of apparent image content. The
52,009 confirmed pool would contain 936,162 names and about 188 GB. A copied
publication is unsafe with 196 GB free. Hardlink publication reuses existing
content; confirmed missing baseline GT+target files are approximately 21.4 GB,
and the maximum ceiling is approximately 24.2 GB. Allowing metadata, selected
new visible inputs, logs, pilot sheets, and bounded staging, reserve 35--50 GB
of incremental disk and retain at least a 100 GB safety margin.

## 10. Planned targeted changes

No production code was changed in Phase A. Phase B should create an isolated
feature branch and modify only these areas:

- `driveharm/sixcam.py`: normalize both authorities, preserve context, build a
  checkpoint/asset-hash-bound group ID, publish train+val+metadata atomically,
  and quarantine whole groups;
- new `driveharm/sixcam_index.py`: authority fusion, official sample-data/token
  lookup, visibility replay, strict candidate/deficit manifests, and stable
  resume hashes;
- `driveharm/storm_adapter.py`: baseline-only context export plus optional exact
  visible-view backfill, with GT/target/checkpoint receipts;
- `driveharm/render.py`: reuse its dynamic queue but bind resume validation to
  the context render contract and requested output membership;
- `driveharm/cli.py`: strict index, pilot, backfill, publish, and independent
  audit entry points;
- `tests/test_sixcam.py` plus new production-contract tests for context
  separation, cross-token rejection, val schema, per-camera asset subsets,
  no-op equality, multi-asset depth order, idempotent resume, and atomic
  quarantine/release;
- `README.md` and a small checked-in configuration template documenting exact
  commands, checkpoints, counts, recovery, audit, and metadata schema.

`DriveHarm_standalone` does not need to be copied wholesale. Its runtime
bootstrap should be imported only if the pilot proves the existing external
STORM profile is insufficient.

## 11. Go/no-go

**Go for Phase B targeted implementation; no-go for immediate bulk GPU.**

The existing accepted assets/checkpoints/pairs should be reused. The first GPU
action after code/tests is the 12-group GPU 0+1 pilot. Full production remains
blocked on pilot acceptance and a measured baseline-only throughput/disk check.
