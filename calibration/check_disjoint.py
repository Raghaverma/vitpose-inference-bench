#!/usr/bin/env python3
"""Stage 6 Gate A prerequisite: prove the INT8 calibration corpus
(calibration/samples/*.npy) was never also used as a golden/validation
fixture (golden/*.npy).

    source .venv/bin/activate && python3 -m calibration.check_disjoint

Why this matters
------------------
An engine "validated" against data it was also calibrated on would
silently inflate its own accuracy numbers: INT8 entropy calibration tunes
per-tensor quantization ranges to whatever activation statistics the
calibration corpus produces, so if a later validation sample's input is
*also* a calibration sample, that validation run partly measures "how well
did calibration fit the calibration data" rather than "does the INT8
engine generalize." Gate A's calibration corpus and Stage 1-4's golden
fixtures must be provably disjoint before either one's numbers mean
anything together -- this script is that proof, run once, checked in.

Two independent checks, both must pass
-----------------------------------------
1. EXACT-BYTES -- is any calibration sample's pixel content byte-identical
   to any golden fixture's pixel content?

   Hashed as RAW ARRAY BYTES (`np.load(...).tobytes()`), NOT whole-.npy-file
   bytes. Whole-file sha256 (this repo's usual `sha256_file()` convention,
   see conversion/build_engine.py) hashes the `.npy` header too, which
   encodes shape/dtype/numpy-version metadata that has nothing to do with
   the actual pixel content. That means whole-file hashing would (a) call
   two files "disjoint" even when they hold IDENTICAL pixel data, if one
   was saved with a squeezed shape or a different numpy version's header
   padding than the other, and (b) only ever catch a literal byte-for-byte
   file copy. Content identity, not file identity, is the actual risk this
   check exists to catch, so array bytes are the correct comparison unit.
   (Whole-file sha256 IS still computed for every calibration sample below,
   but only to cross-check against calibration/manifest.json's own
   recorded per-sample "sha256" field -- a cheap staleness guard, not the
   disjointness determination itself.)

   golden/distinct_crops.npy and golden/distinct_pytorch_fp16_outputs.npy
   each bundle 16 logically-independent slots in one file (slot 0 real,
   slots 1-15 synthetic -- see golden/build_distinct_batch.py). Hashing
   either file as a single whole-array blob could never coincidentally
   collide with a single calibration sample's hash anyway, since the two
   have different total byte counts by construction -- that would make the
   check trivially "pass" for a reason that has nothing to do with actual
   content disjointness. So every slot along a >1-length leading dimension
   is ALSO hashed individually, in addition to the whole-array hash. This
   is a strict superset of "hash the raw array bytes of every golden/*.npy"
   (every named file is still hashed whole too), not a departure from it.

2. PROVENANCE -- exact-byte disjointness (check 1) can pass while both
   corpora still share the same underlying real-world image. Right now
   this repo has exactly ONE real source photo (samples/sample.jpg,
   gitignored), reduced to ONE detected+preprocessed crop
   (golden/person_crop.npy), and calibration/manifest.json's entire 96-
   sample corpus is synthetic augmentation of that same crop
   (calibration/build_calibration_corpus.py). So golden/person_crop.npy is
   the ultimate source of BOTH the golden fixtures AND the calibration
   corpus -- check 1 passing proves no single file was reused verbatim, it
   does NOT prove the two corpora probe independent real-world image
   content. This is a real, already-documented limitation of this
   first-pass synthetic corpus (see manifest.json's "content_source_note"),
   not a bug in this script or in the corpus build -- there is no second
   real photo in this repo yet, so this check is EXPECTED to fire its
   warning right now. The warning does not fail the build for that reason,
   but this script must NEVER silently omit printing it.

Exit code
----------
0 only if check 1 finds zero array-content collisions AND check 2's
provenance warning was printed (regardless of which branch -- match or no
match -- printed it; the guarantee is that provenance got checked and
reported, not a specific outcome). Any array-content collision, or any
structural problem (missing corpus, golden/*.npy set doesn't match what
this script was written against, manifest/on-disk hash mismatch), raises
SystemExit naming the exact colliding path(s) + hash -- never a silent
partial pass, matching this repo's refuse-and-exit discipline throughout
(see baseline.py, golden/build_distinct_batch.py, conversion/build_engine.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline import REPO_ROOT
from conversion.build_engine import sha256_file

CALIBRATION_DIR = REPO_ROOT / "calibration"
SAMPLES_DIR = CALIBRATION_DIR / "samples"
MANIFEST_PATH = CALIBRATION_DIR / "manifest.json"
GOLDEN_DIR = REPO_ROOT / "golden"

# The exact golden/*.npy fixture set this script was written against (see
# the "Key existing files" this script's Gate A task specified). Not a
# glob-and-trust: if a new golden/*.npy fixture is added later without
# updating this list, refuse rather than silently leave it unchecked --
# an unenumerated file must not be able to make this script's "checked
# every golden/*.npy" claim false while still printing PASS.
EXPECTED_GOLDEN_FILES = [
    "person_crop.npy",
    "distinct_crops.npy",
    "distinct_pytorch_fp16_outputs.npy",
    "pytorch_fp16_output.npy",
]


def array_sha256(arr: np.ndarray) -> str:
    """sha256 of an array's raw content bytes -- NOT the .npy file's bytes.
    See module docstring, check 1, for why file-bytes is the wrong
    comparison unit for this script's actual question."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def golden_slot_hashes(path: Path) -> list[tuple[str, str]]:
    """[(slot_label, array_sha256), ...] for one golden/*.npy file: the
    whole-array hash, PLUS one hash per slot if the leading dim is a stack
    of >1 independently-meaningful slots (golden/build_distinct_batch.py's
    convention for distinct_crops.npy / distinct_pytorch_fp16_outputs.npy).
    See module docstring, check 1, for why slots need to be hashed
    individually rather than just as one whole-array blob."""
    arr = np.load(path)
    hashes = [(f"{path.name} (whole array)", array_sha256(arr))]
    if arr.ndim >= 1 and arr.shape[0] > 1:
        for i in range(arr.shape[0]):
            hashes.append((f"{path.name}[{i}]", array_sha256(arr[i])))
    return hashes


def check_exact_bytes() -> dict:
    """Returns the parsed manifest (so check_provenance doesn't re-read
    the file) if check 1 passes; raises SystemExit otherwise."""
    print("[check_disjoint] --- Check 1: EXACT-BYTES ---")

    if not SAMPLES_DIR.is_dir():
        raise SystemExit(
            f"[check_disjoint] REFUSING: {SAMPLES_DIR} does not exist -- run "
            f"calibration/build_calibration_corpus.py first.")
    sample_paths = sorted(SAMPLES_DIR.glob("*.npy"))
    if not sample_paths:
        raise SystemExit(
            f"[check_disjoint] REFUSING: no .npy files under {SAMPLES_DIR} -- nothing to "
            f"check disjointness against.")
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"[check_disjoint] REFUSING: {MANIFEST_PATH} does not exist -- run "
                          f"calibration/build_calibration_corpus.py first.")

    actual_golden = sorted(p.name for p in GOLDEN_DIR.glob("*.npy"))
    if actual_golden != sorted(EXPECTED_GOLDEN_FILES):
        raise SystemExit(
            f"[check_disjoint] REFUSING: golden/*.npy on disk is {actual_golden}, but this "
            f"script was written against EXPECTED_GOLDEN_FILES={sorted(EXPECTED_GOLDEN_FILES)}. "
            f"A changed golden fixture set means this script's coverage claim ('checked every "
            f"golden/*.npy') would be false if it proceeded anyway -- update "
            f"EXPECTED_GOLDEN_FILES only after confirming the new/removed file(s) are "
            f"accounted for.")

    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_sha_by_path = {s["path"]: s["sha256"] for s in manifest["samples"]}

    # Calibration side: whole-file hash (cross-checked against the
    # manifest's own recorded per-sample hash -- a staleness guard, not
    # the disjointness comparison) AND raw array-content hash (the actual
    # comparison unit, per module docstring).
    sample_array_hash: dict[str, str] = {}
    for p in sample_paths:
        file_hash = sha256_file(p)
        rel_path = f"calibration/samples/{p.name}"
        recorded = manifest_sha_by_path.get(rel_path)
        if recorded is None:
            raise SystemExit(
                f"[check_disjoint] REFUSING: {p} exists on disk but {MANIFEST_PATH} has no "
                f"entry for {rel_path} -- manifest is stale relative to the samples directory. "
                f"Regenerate the corpus rather than trust an incomplete manifest.")
        if recorded != file_hash:
            raise SystemExit(
                f"[check_disjoint] REFUSING: {p} has sha256 {file_hash}, but {MANIFEST_PATH} "
                f"records {recorded} for {rel_path} -- the manifest is stale or this sample "
                f"file was modified after the manifest was written. Regenerate the corpus "
                f"rather than trust a mismatched manifest.")
        sample_array_hash[str(p)] = array_sha256(np.load(p))
    print(f"[check_disjoint] hashed {len(sample_paths)} calibration sample file(s); all "
          f"{len(sample_paths)} on-disk whole-file hashes match calibration/manifest.json's "
          f"recorded per-sample sha256 (staleness guard OK).")

    # Golden side: raw array-content hash, per file AND per slot for the
    # two batch-shaped fixtures (see golden_slot_hashes()).
    golden_hash_to_labels: dict[str, list[str]] = {}
    golden_slot_count = 0
    for name in EXPECTED_GOLDEN_FILES:
        for label, h in golden_slot_hashes(GOLDEN_DIR / name):
            golden_hash_to_labels.setdefault(h, []).append(f"golden/{label}")
            golden_slot_count += 1
    print(f"[check_disjoint] hashed {len(EXPECTED_GOLDEN_FILES)} golden/*.npy file(s), "
          f"{golden_slot_count} whole-file-or-slot array(s) total "
          f"({len(golden_hash_to_labels)} distinct content hashes).")

    # The actual disjointness determination: set-intersect calibration
    # sample array-content hashes against golden array-content hashes.
    collisions = []
    for sample_path, a_hash in sample_array_hash.items():
        if a_hash in golden_hash_to_labels:
            for label in golden_hash_to_labels[a_hash]:
                collisions.append((sample_path, label, a_hash))

    if collisions:
        lines = "\n".join(f"    {samp}  <->  {label}  (sha256 {h})"
                           for samp, label, h in collisions)
        raise SystemExit(
            f"[check_disjoint] REFUSING: {len(collisions)} calibration sample(s) are "
            f"byte-identical (by array content) to a golden fixture -- the calibration "
            f"corpus was (at least partially) also used as a validation fixture, which would "
            f"silently inflate any accuracy number measured against that overlap:\n{lines}")

    print(f"[check_disjoint] PASS -- 0 array-content collisions between "
          f"{len(sample_array_hash)} calibration samples and {len(golden_hash_to_labels)} "
          f"distinct golden array-content hashes.")
    return manifest


def check_provenance(manifest: dict) -> None:
    """Never raises SystemExit on its own (a shared-source-image warning is
    an expected, documented limitation right now -- see module docstring,
    check 2) -- but the caller's exit-code policy requires this function to
    have actually run and printed before exiting 0."""
    print("[check_disjoint] --- Check 2: PROVENANCE ---")

    person_crop_path = GOLDEN_DIR / "person_crop.npy"
    actual_hash = sha256_file(person_crop_path)
    manifest_hash = manifest.get("source_image_sha256")
    if manifest_hash is None:
        raise SystemExit(
            f"[check_disjoint] REFUSING: {MANIFEST_PATH} has no 'source_image_sha256' field "
            f"-- this check cannot run without it, and this script must never silently skip "
            f"the provenance warning (see module docstring's exit-code policy).")

    env_path = GOLDEN_DIR / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    env_hash = env.get("source_image_sha256")  # golden/env.json does not currently record this
                                                 # field at all (checked directly, not assumed);
                                                 # an explicit .get() rather than a silent
                                                 # KeyError-catch, so a future env.json revision
                                                 # that DOES add it gets picked up automatically.
    if env_hash is not None and env_hash != actual_hash:
        raise SystemExit(
            f"[check_disjoint] REFUSING: {env_path} records source_image_sha256={env_hash}, "
            f"which does not match the actual sha256 of {person_crop_path} "
            f"({actual_hash}) -- golden/env.json itself is inconsistent with the file it "
            f"describes. Not proceeding with a provenance comparison against a source that "
            f"disagrees with itself.")

    if manifest_hash == actual_hash:
        print(
            "[check_disjoint] WARNING -- PROVENANCE COLLISION: calibration/manifest.json's "
            f"source_image_sha256 ({manifest_hash}) matches the actual sha256 of "
            f"{person_crop_path} ({actual_hash}). The ENTIRE calibration corpus and the "
            "ENTIRE golden fixture set derive from the SAME one real photo -- exact-byte "
            "disjointness (Check 1) passing does NOT mean these two corpora probe "
            "independent real-world image content; it only means no single sample file was "
            "reused verbatim. This repo has exactly one real source photo "
            "(samples/sample.jpg, gitignored) right now, so this collision is EXPECTED and "
            "is not itself a bug in this script or in the corpus build -- but it MUST be "
            "resolved (calibration built from a genuinely distinct real capture source: a "
            "public pose-dataset crop and/or the user's own private sports footage, per "
            "calibration/manifest.json's own content_source_note) before any accuracy number "
            "measured on this calibration corpus / INT8 engine pair can be trusted as a real "
            "generalization claim rather than a same-photo tautology. Do not read Check 1's "
            "PASS above as 'these corpora are independent' -- they are not, yet.")
    else:
        note = (" (golden/env.json has no source_image_sha256 field of its own to "
                 "cross-check; the actual on-disk hash of person_crop.npy is the "
                 "authoritative comparison here.)" if env_hash is None else "")
        print(
            f"[check_disjoint] source_image_sha256 in manifest ({manifest_hash}) does NOT "
            f"match the actual sha256 of {person_crop_path} ({actual_hash}) -- calibration "
            f"corpus's source image is provably independent of this golden fixture's "
            f"source.{note}")

    print("[check_disjoint] provenance check executed and reported above -- required for "
          "exit 0 regardless of which branch printed.")


def main() -> None:
    manifest = check_exact_bytes()
    check_provenance(manifest)
    print(
        "[check_disjoint] EXIT 0 -- exact-bytes check passed AND the provenance check ran and "
        "printed its finding (see above). A printed provenance WARNING is expected right now "
        "and does not fail the build on its own -- see module docstring's exit-code policy -- "
        "but it is never silently omitted.")


if __name__ == "__main__":
    main()
