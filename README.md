# DriveHarm SixCam

DriveHarm SixCam is the isolated six-camera release extension of DriveHarm. It
retains the complete train-quality STORM asset re-insertion pipeline and adds a
batch publisher that groups already audited camera triplets into synchronized
nuScenes rings. The original DriveHarm repository is not modified. This
repository contains code and documentation only; assets, checkpoints, images
and run receipts stay outside Git.

The code is organized as a reusable pair-production core with a frozen train
production profile. Dataset adapters provide either the native train capacity
manifest or normalized observation JSONL; renderer implementations satisfy one
small executable/result contract; GPU topology, split destination and release
sources are runtime inputs. New splits or compatible datasets therefore reuse
planning, review, composition, full audit and atomic publication instead of
copying the pipeline.

The immutable upstream asset generator is intentionally outside this project.
DriveHarm starts from its completed, SHA-256-bound hand-off and never changes
how PLY assets are produced.

The authoritative behavior is the production flow that created
`nusc_pair/train`. The validation split is not used as a generation template.
One bug found while auditing validation data is retained only as a regression
check at the final quality gate.

## Six-camera batch release

One published group is one scene and one target frame with exactly 18 flat PNG
files in this fixed ring order:

1. `CAM_FRONT`
2. `CAM_FRONT_RIGHT`
3. `CAM_BACK_RIGHT`
4. `CAM_BACK`
5. `CAM_BACK_LEFT`
6. `CAM_FRONT_LEFT`

Every camera contributes `gt`, `input` and `target`, named as
`{group_id}__{CAMERA_NAME}__{role}.png`. An asset is inserted only in cameras
where the frozen visibility manifest explicitly marks it usable. In every
other camera, `input` is the exact unedited `target`; absence of an accepted
pair row is never interpreted as invisibility.

The publisher requires an unambiguous `gt/target` baseline in all six cameras.
For each visible camera it also requires an already audited pair whose exact
canonical asset set matches the assets visible in that camera. A missing or
conflicting view excludes the complete group; partial 18-image groups are never
published. Single- and multi-asset groups use the same rule.

```bash
driveharm-sixcam sixcam-release \
  --source-root /data/nusc_pair/train \
  --records /data/accepted_records.jsonl \
  --visibility-manifest /data/single_asset_jobs.json \
  --destination /data/sixcam/flat \
  --receipt-root /data/sixcam/receipts \
  --category car --category truck --category bus --category van \
  --maximum-groups 0 --materialize hardlink --workers 32

driveharm-sixcam sixcam-audit \
  --dataset-root /data/sixcam/flat \
  --groups /data/sixcam/receipts/groups.jsonl \
  --source-records /data/accepted_records.jsonl \
  --visibility-manifest /data/single_asset_jobs.json \
  --output-root /data/sixcam/audit --workers 32
```

`--maximum-groups 0` means all complete groups. Repeat `--scene` or
`--category` to select a subset. Publication uses a temporary sibling and one
atomic rename. An existing destination is refused unless `--replace` is
explicit; replacement preserves the previous release as a timestamped sibling.

The independent audit checks all 18 files per group, exact camera order and
names, PNG/RGB/512x288 contracts, hashes and flat-directory membership. It also
replays source-record identity and content bindings plus per-asset visibility:
visible views must reference an exact audited pair and make an effective edit;
invisible views must reference no pair and be byte-identical to `target`.

To generate new capacity rather than regroup existing accepted rows, first run
the retained train pipeline below so that every chosen scene/frame has six
unambiguous baseline views and every visible camera has an accepted exact pair.
Then run `sixcam-release` and `sixcam-audit`. This keeps size, orientation,
grounding, identity, broken/doubled-asset and physical-occlusion decisions in
the same train gates instead of introducing a second quality policy.

## Pair definition

- `gt`: aligned real sensor RGB, retained for real-domain evaluation.
- `input`: actor-removed STORM background plus the exact canonical Gaussian
  asset, composed back-to-front with physically verified foreground occlusion.
- `target`: unedited STORM render from the identical checkpoint, camera and
  exposure timestamp.

An asset visible in any of the six cameras is eligible. Camera 0 has no special
status and is not required.

Planning and storage are camera-triplet based: one visible camera produces one
`gt/input/target` row. An asset visible in all six cameras therefore produces
six aligned rows, or 18 images in total (six per role), not 12 images.

## Production stages

1. Verify the upstream asset manifest hash, every PLY hash, `obj_id`, official
   `instance_token`, category, official dimensions and canonical `+X` axis.
2. Read the train `exact_asset_windows.jsonl`, validate its temporal windows,
   and build deterministic single-asset plus compatible multi-asset render
   opportunities for every visible camera.
3. Review each unique asset UID once, then propagate its decision to every
   frame/camera/combination. Bounded asynchronous workers use the official
   OpenAI Python client and a strict JSON schema. Hash-bound completed decisions
   are reused; request errors remain pending and are requested again.
4. Verify the exact STORM, CVAC and DCN artifact hashes, split accepted jobs
   across a shared async shard queue. Eight GPUs and two workers per GPU are the
   train production profile; each worker pulls several smaller shards so a fast
   card does not sit idle behind another card's long tail. Jobs from one scene
   window stay together to reuse STORM context, and hash-valid completed shards
   resume without another GPU pass.
5. Compose premultiplied asset layers far-to-near. A foreground mask is applied
   only when an independently bound official instance is distinct from and
   strictly nearer than the inserted target, or when a hash-bound static region
   has enough strictly-nearer scene-depth seeds. This covers vehicles, poles,
   branches and fences without treating the target's own mask as an occluder.
   Target-instance support is diagnostic only and can never erase physical
   ordering.
   The real camera timestamp and normalized STORM time are bound in every row;
   actor removal must be effective and unchanged outside its exact edit mask.
6. Apply the same two geometry proof routes as the validated train release:
   an independently passed strict renderer gate, or the area-aware numeric
   recovery limits below. Both still require official 3-D ground lock,
   15% minimum complete-asset visibility and hard rejection of broken, doubled
   or clearly reversed assets. Color, brand, trim and ordinary lighting
   differences are audit signals, not rejection criteria.
   Multi-asset edit overlap is capped at the train value of 2% of the smaller
   edit mask, and both removal and insertion must change at least 20 pixels.
7. Review every rendered triplet, then independently check every PNG, content
   hash, triplet membership, duplicate signature, identity receipt, geometry
   receipt and occlusion receipt. Every candidate is excluded from acceptance.
8. Materialize only the accepted manifest into a temporary sibling directory,
   verify hashes again, and atomically activate the release.

## Train parity matrix

| train production capability | implementation |
|---|---|
| immutable original-asset boundary | `contracts.py` verifies the hand-off, exact manifest and every PLY hash; asset generation is untouched |
| temporal accepted windows and capacity | `planning.py` reads train `exact_asset_windows.jsonl` and validates context/target frames |
| all six cameras, no front-only condition | every `visible_target_frame_camera_key` becomes an opportunity |
| single and multi-asset variants | deterministic subsets are planned once per window, never repeat an official instance, and are projected only where every selected actor is visible; multi-asset masks must pass 2% overlap |
| exact identity and canonical heading | `obj_id`, `instance_token`, category, PLY hash, official dimensions and `+X` are cross-bound |
| identity visual review | `review.py` uses bounded async workers and strict JSON-schema decisions |
| STORM/CVAC/DCN consistency | `render.py` verifies the artifact contract and job/frame/camera binding, then batches across eight GPUs |
| actor removal and same-domain pair | `compose.py` binds camera/STORM time and requires a hash-bound usable cleanup receipt, exact edit mask, STORM target-quality pass and effective removal |
| scale, pose, grounding and integrity | area-aware train limits plus official dimensions, bottom lock, orientation and broken/doubled checks |
| physical foreground occlusion | distinct official instance or hash-bound static depth region, independent mask, strictly nearer depth, 4%/16-pixel materiality and direct alpha clipping |
| complete asset visibility | full alpha is retained unless physical foreground evidence exists; visible fraction must be at least 15% |
| complete release audit | `audit.py` checks every image, record, role membership, hash, duplicate, identity, geometry, occlusion and visual decision |
| old/new audited union | `release.py` merges repeated source/manifest pairs, excludes duplicate IDs/content, verifies again and atomically activates |
| direct candidate removal | `quarantine` moves all three matching files together with a recoverable hash receipt |

The frozen geometry limits are:

| projected pixels | IoU | center px | width/height ratio | yaw |
|---:|---:|---:|---:|---:|
| `<100` | 0.30 | 12 | 0.70–1.40 | 25° |
| `<400` | 0.33 | 11 | 0.72–1.37 | 25° |
| `<1600` | 0.36 | 10 | 0.75–1.33 | 24° |
| `>=1600` | 0.40 | 10 | 0.78–1.30 | 22° |

Foreground occlusion uses the train release's 4% materiality floor with a
minimum of 16 pixels. This catches the val-only regression in which a distinct
nearer official instance was incorrectly suppressed by target-instance mask
protection. It also catches material restoration of occlusion already verified
by the renderer. The final visual gate remains mandatory because sparse fence
wires and semi-transparent foliage can be under-resolved by a depth map; a
triplet that visibly overwrites such foreground structure is rejected rather
than repaired by a speculative image-space rule.

## External contracts

The asset hand-off JSON points to the immutable exact-asset JSON used by train,
or to an equivalent normalized JSONL manifest:

```json
{
  "status": "complete",
  "manifest": "/data/drivelab_asset.json",
  "manifest_sha256": "<sha256>",
  "quarantine_manifest": "/data/asset_quarantine.json",
  "quarantine_manifest_sha256": "<sha256>"
}
```

The native train registry `bridge_summary.json` is also accepted directly; its
`outputs.exact_asset_manifest.path/sha256` fields are interpreted as the same
immutable hand-off. A capacity summary carrying
`exact_asset_manifest`/`exact_asset_manifest_sha256` is accepted as well.

The existing train shape (`global_uid`, `obj_id`, `instance_token`, `category`,
`asset_path`, `size_xyz`, and the canonical hash/axis object) is accepted
directly. Official `wlh` dimensions come from each train capacity candidate and
are converted to renderer `length,width,height` order. The original identity
`review_manifest.jsonl` supplies the source view, canonical views, heading view
and their hashes. The quarantine fields are optional; when supplied, they are
hash-verified and excluded before any identity review or rendering.

An adapter for another compatible source emits one row per visible actor with
`scene_name`, `window_id`, `frame_index`, `obj_id`, `instance_token`,
`area_pixels`, `visibility_level`, `visible_cameras`, official dimensions and
hash-bound review images. Visibility may contain any subset of camera `0`–`5`.
The planner groups variants by official instance, applies the train deterministic
seed and 256-combination window cap, and never combines two variants of the same
instance.

The render contract binds code/checkpoint artifacts before any GPU starts:

```json
{
  "artifacts": {
    "storm": {"path": "/models/storm.ckpt", "sha256": "<sha256>"},
    "cvac": {"path": "/models/cvac.ckpt", "sha256": "<sha256>"},
    "dcn": {"path": "/models/dcn.ckpt", "sha256": "<sha256>"}
  }
}
```

The company renderer is an executable accepting:

```text
--jobs MANIFEST --output-root DIRECTORY --results RESULTS_JSONL --gpu 0
```

DriveHarm exposes exactly one physical GPU to each process through
`CUDA_VISIBLE_DEVICES`. Each result row must have `status=complete`, matching
`sample_id`, job hash, checkpoint hashes, hash-bound `real_gt`,
`storm_baseline`, `actor_removed`, exact removal-mask paths and one or more
exact-identity asset layers. Layer receipts carry projection
metrics plus independently verified foreground masks and official-instance
depth decisions.

### Frozen train STORM adapter

`driveharm-storm-renderer` is the thin production adapter for the existing
train renderer. It maps a DriveHarm row back to the exact
`scene/obj_id/frame/camera` opportunity, verifies `instance_token`, PLY hash,
official dimensions and canonical `+X`, runs the unchanged STORM/CVAC/DCN
renderer, then converts its full premultiplied layer and depth evidence into the
small result contract above. The profile is runtime JSON and is not committed:

```json
{
  "repo_root": "/code/storm_dataset",
  "python": "/envs/storm/bin/python",
  "job_root": "/data/train/jobs",
  "data_root": "/data/storm_adapter",
  "annotation_list": "/data/storm_adapter/scene_list/train_annotations.txt",
  "raw_nuscenes_root": "/data/nuScenes",
  "storm_checkpoint": "/models/storm.pth",
  "cvac_checkpoint": "/models/cvac.pth",
  "dcn_checkpoint": "/models/dcn.pth"
}
```

Pass it through the generic scheduler with two renderer arguments:

```bash
driveharm render \
  --jobs /data/run/accepted_jobs.jsonl \
  --output-root /data/run/03_render \
  --renderer "$(command -v driveharm-storm-renderer)" \
  --renderer-arg=--profile \
  --renderer-arg=/data/train_storm_profile.json \
  --render-contract /data/render_contract.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 1 --shards-per-worker 4
```

`--reuse-legacy` is only for rebuilding adapter metadata from already complete,
strictly validated renderer outputs; it refuses to proceed if any frozen result
is missing and never stands in for rendering.

## Installation and review server

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

For a local multimodal review service, start vLLM with the company-approved
model and reserve 90% of each selected GPU's memory:

```bash
vllm serve /models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen3-VL-8B-Instruct \
  --gpu-memory-utilization 0.90
```

The review implementation uses `AsyncOpenAI`, awaits
`client.chat.completions.create`, and requires JSON-schema output. Set the key
in an environment variable; do not place credentials in manifests or command
history.

## End-to-end run

```bash
export OPENAI_API_KEY=local
driveharm run \
  --asset-post /data/asset_post.json \
  --observations /data/exact_asset_windows.jsonl \
  --identity-manifest /data/review_manifest.jsonl \
  --renderer /company/bin/storm_pair_renderer \
  --render-contract /data/render_contract.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2 \
  --shards-per-worker 4 \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen3-VL-8B-Instruct \
  --review-concurrency 16 \
  --work-root /data/driveharm_run \
  --destination /data/nusc_pair/train
```

Every stage is also available separately through `plan`, `review`, `render`,
`compose`, `audit`, `quarantine`, and `release`. Run `driveharm COMMAND --help`
for its exact arguments.

For a multi-batch train union, repeat `--source-root` and
`--accepted-records` in matching order on the `release` command. Equal triplet
content is retained once; conflicting content under the same sample ID stops
publication.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the native train exact-asset/capacity/identity shapes,
all-camera planning, single and combined jobs, checkpoint and job binding,
area-aware orientation rejection, 2% edit overlap, exact-mask/nearer-occluder
ordering, actor-removal locality, composition, independent audit, publication,
and recoverable triplet quarantine. They also enforce exactly 18 flat files per
six-camera group, explicit invisible-camera no-ops, source/visibility replay,
the compact source-file count and asynchronous OpenAI JSON-schema contract.

## Verified reference release

The implementation preserves the policies used for
`/mnt/ojc/workplace2/dataset/nusc_pair/train`: 51,132 triplets from 616 official
train scenes, with all 153,396 images decoded and hash-checked after release.
The frozen receipt is
`/mnt/ojc/workplace2/sixcam_run/nusc_pair_release_v9_official_train_val_v1/summary.json`.
The four area bins above were also compared directly against the original train
producer function and matched exactly. The distinct-nearer occluder case is an
additional regression test and does not change the train generation stages or
their reasonable thresholds.

## Real pilot evidence

### Six-camera grouping pilot

The isolated batch publisher was run against the formal 51,132-row train
release. The source contains 10,366 scene/frame keys; 23 have complete,
unambiguous six-camera baselines. Across those frames, 267 candidate asset sets
were found and 260 form complete groups; seven are correctly excluded because
a visible camera lacks an exact accepted pair. Restricting to
car/truck/bus/van leaves 186 complete vehicle groups.

The small real pilot published four groups (72 PNGs) from `scene-0067`, with
five genuinely edited visible views and 19 invisible no-op views. Every image
passed decode, shape, hash, camera/role membership, upstream source-record and
visibility replay. All five edited views were also inspected at original
resolution; no identity swap, reversal, broken/doubled asset or clear
foreground-occlusion violation was found. This pilot deliberately does not
materialize the full available batch.

### Original camera-triplet pilot

A 35-row isolated pilot covered all six cameras, 12 train scenes, car/bus/van,
clear views and 13 renderer-declared occlusion cases. Eight scheduler slots
completed 35/35 real STORM renders. Against each render's own frozen output,
all 35 `gt` and `target` images were pixel-identical; DriveHarm `input` differed
only at premultiplied alpha rounding edges (mean MAE 0.0037/255, worst PSNR
64.62 dB).

The structural audit checked all 105 images. Full-frame plus asset-centred
visual review then rejected 10 rows: eight views of one car behind wire fencing,
one chain-link case, and one tree mask that left an implausibly truncated car.
The remaining 25 triplets were materialized into an isolated release and all 75
published images passed an independent second audit with zero duplicates or
candidates. This is evidence that the pipeline preserves train-quality rows and
excludes clear historical occlusion failures; it is not permission to skip the
visual gate in a larger run.
