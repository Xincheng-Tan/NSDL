"""
LOCUS-Bench instantiation wrapper (P7.2, docs/benchmark.md §9).

Wraps the official ``generate_prompt.instantiate_task`` (untouched) with the
three sampling knobs of benchmark.md §4.2/§4.3:

  1. **Balanced realization sampling** — a min-count sampler over
     ``state_realization_spec.realizations`` guarantees per-(family, cell)
     realization counts differ by at most 1 (strictly better than the
     §4.3 tolerance ``max share ≤ 1/|R| + 0.03``). GT expansion per
     realization is already done inside ``locus_task_pool.make_template``.
  2. **Distractor injection** (S2/S3 scenes only) — 1..3 extra assets that
     the GT never touches:
       * ``clutter``: neutral objects (never task targets) on open surfaces;
       * ``extra_present_same_tag`` (F3/F5 only): an unchosen same-tag item
         *already at the destination* — a legal "found candidate ≠ target"
         signal under the quantifier semantics ("all fruits" / "anything
         already stored can stay"). Never placed off-destination, which
         would contradict the closed-world GT.
     Family-aware forbidden areas: F5's cleared surface (mask5) never hosts
     clutter (an item there would fall in the instruction's scope).
  3. **Meta accounting** — every instance carries
     ``meta = {bench_family, cell, state_realization, paraphrase_id,
     mask_map, opaque_container, instantiate_seed, distractors, ...}``.
     Masks are pinned by the wrapper (recorded in ``mask_map``) so the
     §7-3 vocabulary cross-coverage audit is computable offline.
     ``state_realization`` lives ONLY in meta — never in
     description/task_name (audited in tests).

``allowed_paraphrase_ids`` supports the §5.2 paraphrase-holdout split
(train restricted to a prefix of paraphrase ids; holdout uses the rest).

Reproducibility: one ``random.Random(seed)`` drives everything; the official
``instantiate_task``'s use of the global ``random`` module is bridged by
re-seeding it with a recorded ``instantiate_seed`` per instance.
"""

import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from contextlib import redirect_stdout

from locus_task_pool import (
    FAMILY_CELLS, enumerate_realizations, iter_cells, layouts_for_masks,
    make_template, weighted_mask_choice,
)
from generate_prompt import instantiate_task

__all__ = [
    "RealizationSampler",
    "BridgeVariantSampler",
    "MaskVocabSampler",
    "STRATIFY_COUNT_FACTOR",
    "audit_mask_keys",
    "stratified_mask_keys",
    "instantiate_locus_task",
    "instantiate_counterfactual_pair",
    "generate_locus_cell",
    "generate_locus_batch",
    "realization_balance_report",
]

# objects present in every layout yaml and never used as task targets
NEUTRAL_CLUTTER = ["hammer", "rock", "shoe", "towel", "clothes iron",
                   "indoor plant"]
FRUIT_TAGS = ["apple", "pear", "peach", "banana", "tomato"]

# families whose quantifier semantics admit a same-tag extra item at the
# destination ("all fruits" / "anything already stored can stay")
SAME_TAG_EXTRA_FAMILIES = {"F3", "F5"}


class RealizationSampler:
    """Min-count balanced sampler over a cell's realization enumeration."""

    def __init__(self):
        self._counts = {}

    def next(self, family, s, m, rng):
        reals = enumerate_realizations(family, s, m)
        key = (family, s, m)
        counts = self._counts.setdefault(key, [0] * len(reals))
        lo = min(counts)
        idx = rng.choice([i for i, c in enumerate(counts) if c == lo])
        counts[idx] += 1
        return reals[idx]


# A plan row whose instance count is below this multiple of a vocab list's
# size cannot cover that list by chance: with n draws from |V| values the
# expected number of unseen values is |V|(1-1/|V|)^n, which only becomes
# small once n >> |V|. At n = 3|V| roughly 5% of the vocab is still expected
# to be missing (|V|=9, n=27 -> 0.32 values), and the F9 OOD cells sit far
# below even that (n=20, |V|=9). See docs/benchmark.md §9 P7.5c.
STRATIFY_COUNT_FACTOR = 3


def audit_mask_keys(tpl):
    """Mask keys the §7-3 vocabulary cross-coverage audit actually reads.

    `locus_validate._vocab_coverage` walks `state_realization_spec`'s
    candidates and records the `(candidate value, destination value)` pair,
    so only the candidates' own `mask_key` and their `dest_mask` — when that
    dest is itself a mask list rather than a literal fixture name — enter the
    coverage matrix. Restricting stratification to these keys keeps every
    other mask draw byte-identical.
    """
    keys = set()
    for cand in tpl["state_realization_spec"]["candidates"]:
        keys.add(cand["mask_key"])
        dest = cand.get("dest_mask")
        if dest and dest.startswith("mask") and dest in tpl:
            keys.add(dest)
    return keys


def stratified_mask_keys(family, s, m, count,
                         factor=STRATIFY_COUNT_FACTOR):
    """Audit-relevant mask keys of one cell that `count` draws cannot cover.

    Returned as a set so callers (and tests) can predict exactly which rows a
    `MaskVocabSampler` changes. A key with a single-value vocab is never
    included: there is nothing to stratify and pinning it must keep consuming
    rng identically.
    """
    tpl = make_template(family, s, m, random.Random(0))
    return {key for key in audit_mask_keys(tpl)
            if len(tpl.get(key, [])) > 1
            and count < factor * len(tpl[key])}


class MaskVocabSampler:
    """Deterministic round-robin over a mask vocab, for small plan rows.

    §7-3 wants every `(candidate, destination)` pair in the *train grid* to
    be exercised, but the audit counts pairs per family across all splits,
    and a family's only cell naming some destination can be an OOD cell whose
    instance count is pinned by `--n-ood` (F9's `sink` destination lives only
    in `F9_s1m4`, n=20 vs a 9-value candidate vocab). Sampling i.i.d. there
    leaves holes no amount of train ramping can fill — the holes are a
    property of the 20-draw lottery, not of the data.

    So for those rows the value is drawn min-count-first, exactly like
    `RealizationSampler` handles realizations: ties broken by `rng.choice`,
    which keeps the row seed-reproducible while guaranteeing the vocab is
    covered once the row reaches |V| instances. Weights are deliberately NOT
    honoured here: `MASK_VOCAB_WEIGHTS` exists to push three detector classes
    over the §6.2 box gate, which is a *train*-split concern, and no train row
    is small enough to be stratified (verified in tests). Skewing a 20-draw
    OOD row would only re-open the holes this sampler closes.

    Draws are transactional. An instance can be rejected by the SimEnv gate
    and re-instantiated, which would otherwise spend vocab budget on a value
    that never reaches the batch (the realization sampler dodges this by
    being called once outside the retry loop; the mask pin happens inside
    `instantiate_locus_task`, so it cannot). `next` therefore stages its
    increments, and the caller must `commit()` a kept instance or
    `rollback()` a rejected one.
    """

    def __init__(self):
        self._counts = {}
        self._pending = []

    def next(self, family, s, m, key, values, rng):
        counts = self._counts.setdefault(
            (family, s, m, key), [0] * len(values))
        staged = self._staged(family, s, m, key, len(values))
        lo = min(c + p for c, p in zip(counts, staged))
        idx = rng.choice([i for i in range(len(values))
                          if counts[i] + staged[i] == lo])
        self._pending.append(((family, s, m, key), idx))
        return values[idx]

    def _staged(self, family, s, m, key, n):
        out = [0] * n
        for pending_key, idx in self._pending:
            if pending_key == (family, s, m, key):
                out[idx] += 1
        return out

    def commit(self):
        for key, idx in self._pending:
            self._counts[key][idx] += 1
        self._pending = []

    def rollback(self):
        self._pending = []

    def counts(self):
        """Committed per-(cell, key) draw counts; for audits and tests."""
        return {key: list(counts) for key, counts in self._counts.items()}


class BridgeVariantSampler:
    """Min-count sampler for F9-M2's atomic carrier components."""

    VARIANTS = ("load", "push", "unload")

    def __init__(self):
        self._counts = {}

    def next(self, family, s, m, rng):
        if family != "F9" or m != 2:
            return None
        key = (family, s, m)
        counts = self._counts.setdefault(key, [0] * len(self.VARIANTS))
        lo = min(counts)
        idx = rng.choice([i for i, count in enumerate(counts)
                          if count == lo])
        counts[idx] += 1
        return self.VARIANTS[idx]


def _existing_types(init_pos):
    return {k.rsplit('_', 1)[0] for k in init_pos
            if not (k.startswith('R') and k[1:].isdigit())}


def _dest_area(spec, mask_map):
    """Destination position-area of the cell's candidates, if it is an open
    area (containers like plate/cabinet are not valid clutter positions)."""
    for cand in spec["candidates"]:
        dest = mask_map.get(cand.get("dest_mask", ""), None)
        if dest in ("table", "rack"):
            return dest
    return None


def _inject_distractors(inst, family, s, mask_map, spec, rng):
    if s not in (2, 3):
        return []
    init_pos = inst["init_pos"]
    existing = _existing_types(init_pos)
    dest = _dest_area(spec, mask_map)

    if family == "F5":
        # mask5 (the cleared surface) is in the instruction's scope — an
        # untouched item there would contradict the GT
        areas = ["kitchen island area"] + ([dest] if dest else [])
    else:
        areas = ["kitchen work area", "kitchen island area"] + \
                ([dest] if dest else [])

    distractors = []
    # The F3-S2 bridge stays deliberately bounded: k=2 and at most one
    # neutral distractor. Same-tag extras are an S3-only extrapolation.
    budget = 1 if family == "F3" and s == 2 else rng.randint(1, 3)

    same_tag_allowed = family in SAME_TAG_EXTRA_FAMILIES and s >= 3
    if same_tag_allowed and dest and rng.random() < 0.5:
        pool = [f for f in FRUIT_TAGS
                if f not in mask_map.values() and f not in existing]
        if family == "F5":
            # F5's tag scope is "items on the surface", not fruits; reuse
            # its own item vocab minus chosen values
            pool = [v for v in ("bread", "apple", "spoon", "peach")
                    if v not in mask_map.values() and v not in existing]
        if pool:
            name = rng.choice(pool)
            distractors.append(
                {"name": name, "pos": dest, "kind": "extra_present_same_tag"})
            existing.add(name)
            budget -= 1

    clutter_pool = [c for c in NEUTRAL_CLUTTER if c not in existing]
    for name in rng.sample(clutter_pool, min(budget, len(clutter_pool))):
        distractors.append(
            {"name": name, "pos": rng.choice(areas), "kind": "clutter"})

    idx = len(init_pos)
    for d in distractors:
        init_pos[f'{d["name"]}_{idx}'] = [d["pos"]]
        idx += 1
    return distractors


def _stable_u64(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    # Keep metadata parquet-safe (Arrow infers signed int64 for nested
    # structs) while retaining 63 deterministic hash bits.
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & \
        ((1 << 63) - 1)


_BASE_TOPOLOGY = {
    ("F1", 0): "single_fetch",
    ("F1", 1): "serial_fetch",
    ("F1", 2): "parallel_fetch",
    ("F2", 0): "single_or_dual_fetch",
    ("F2", 1): "serial_conditional_fetch",
    ("F2", 2): "parallel_conditional_fetch",
    ("F3", 0): "single_quantified_fetch",
    ("F3", 1): "serial_quantified_fetch",
    ("F3", 2): "parallel_quantified_fetch",
    ("F4", 1): "single_fallback_retrieve",
    ("F4", 2): "parallel_serve_and_dish",
    ("F4", 3): "shared_cabinet_serve_and_stow",
    ("F5", 1): "serial_clear",
    ("F5", 2): "parallel_clear",
    ("F5", 3): "shared_cabinet_clear",
    ("F6", 2): "parallel_set_table",
    ("F6", 3): "cabinet_set_table",
    ("F7", 1): "serial_device_prep",
    ("F7", 2): "parallel_device_prep",
    ("F8", 2): "independent_deposit_twin",
    ("F8", 3): "shared_open_deposits",
    ("F9", 2): "local_carrier_components",
    ("F9", 3): "load_push_subchain",
    ("F9", 4): "three_robot_carrier_relay",
    ("F10", 0): "single_inspection",
}

_BASE_OPEN_CELLS = {
    ("F4", 3), ("F5", 3), ("F6", 3), ("F8", 3),
}


def _base_mechanical_signature(family, m, inst):
    """M-axis signature with S4's state-required open removed."""
    return {
        "m_level": m,
        "topology": _BASE_TOPOLOGY[(family, m)],
        "n_robots": len(inst["robots"]),
        "base_open": (family, m) in _BASE_OPEN_CELLS,
    }


def instantiate_locus_task(family, s, m, rng, realization=None,
                           paraphrase_id=None, allowed_paraphrase_ids=None,
                           inject_distractors=True, allowed_layouts=None,
                           counterfactual_pair_seed=None,
                           bridge_variant=None,
                           mask_sampler=None, stratify_keys=frozenset()):
    """One fully-instantiated LOCUS sample with meta accounting.

    ``realization`` should come from ``RealizationSampler`` (or
    ``enumerate_realizations``); ``None`` falls back to the generator's
    default (max-structure) realization. ``allowed_layouts`` restricts the
    template's layout pool (P7.4 layout holdout); if the intersection is
    empty the family simply doesn't support the restriction and the full
    pool is kept (recorded in ``meta.layout_restricted``).

    A second, non-negotiable narrowing happens after the masks are pinned:
    a mask naming a fixture that only some layouts instantiate (``oven``)
    restricts the pool to those layouts, because a sample whose instruction
    names an absent fixture is un-groundable. Unlike the holdout restriction
    this one cannot be waived — an empty intersection raises (recorded in
    ``meta.fixture_layout_masks``).
    """
    if family == "F9" and m == 2 and bridge_variant is None:
        bridge_variant = rng.choice(("load", "push", "unload"))
    probe = make_template(
        family, s, m, random.Random(0), bridge_variant=bridge_variant)
    n_par = len(probe["description"])
    if paraphrase_id is None:
        ids = (list(allowed_paraphrase_ids)
               if allowed_paraphrase_ids is not None else range(n_par))
        paraphrase_id = rng.choice(list(ids))
    if not (0 <= paraphrase_id < n_par):
        raise ValueError(f"paraphrase_id {paraphrase_id} out of range "
                         f"[0, {n_par}) for {family} s{s}m{m}")

    tpl = make_template(
        family, s, m, rng, realization=realization or None,
        bridge_variant=bridge_variant)
    spec = tpl["state_realization_spec"]

    layout_restricted = False
    if allowed_layouts is not None:
        keep = [l for l in tpl["layout_idx"] if l in set(allowed_layouts)]
        if keep:
            tpl["layout_idx"] = keep
            layout_restricted = True

    # pin masks so mask_map is recordable (vocab-coverage audit, §7-3).
    # Sampling is weighted by MASK_VOCAB_WEIGHTS: three detector classes are
    # single-list in the train grid and miss the §6.2 >=20 detector-val box
    # gate under uniform choice (docs/YOLO.md §6.2). Uniform lists still go
    # through rng.choice, so cells with no weighted class are unchanged.
    # `stratify_keys` (P7.5c) swaps the i.i.d. draw for a min-count round
    # robin on the few audit-relevant keys whose row is too small to cover
    # its vocab by chance -- see MaskVocabSampler. Both paths consume exactly
    # one rng draw per key, so a row's OTHER samples are unaffected by whether
    # a given key is stratified.
    mask_map = {}
    for key in sorted(k for k in tpl if k.startswith("mask")):
        if key in stratify_keys and mask_sampler is not None:
            value = mask_sampler.next(family, s, m, key, tpl[key], rng)
        else:
            value = weighted_mask_choice(rng, tpl[key])
        tpl[key] = [value]
        mask_map[key] = value
    tpl["description"] = [tpl["description"][paraphrase_id]]

    # Masks are pinned, so the layout pool can now be narrowed to layouts that
    # actually instantiate the named fixtures (`oven` exists in 1/6/8 only, see
    # FIXTURE_MASK_LAYOUTS). Order matters: narrowing BEFORE instantiate_task
    # picks the layout is what makes the sample groundable. Narrowing the pool
    # rather than re-rolling the mask keeps the mask distribution and the rng
    # consumption of unaffected samples byte-identical.
    fixture_layouts, restricting_masks = layouts_for_masks(
        mask_map.values(), tpl["layout_idx"])
    if restricting_masks and not fixture_layouts:
        raise ValueError(
            f"{family} s{s}m{m} masks {restricting_masks} admit no layout "
            f"within {tpl['layout_idx']} -- the sample cannot be rendered "
            f"groundably (see FIXTURE_MASK_LAYOUTS)")
    if restricting_masks:
        tpl["layout_idx"] = fixture_layouts

    seed = rng.getrandbits(32)
    random.seed(seed)   # official instantiate_task uses the global random
    with redirect_stdout(io.StringIO()):
        inst = instantiate_task(tpl)

    distractors = (_inject_distractors(inst, family, s, mask_map, spec, rng)
                   if inject_distractors else [])

    hidden_fallback = (
        family == "F4" and spec["realization"].get("mask1") == "cabinet")
    state_entailed_open = bool(s == 4 and hidden_fallback)

    pair_id = None
    pair_basis = None
    if family == "F4" and s == 4:
        # Every key here is counterfactual-invariant. In particular,
        # state_realization is intentionally excluded.
        pair_basis = {
            "family": family,
            "s": s,
            "m": m,
            "pair_seed": counterfactual_pair_seed,
            "paraphrase_id": paraphrase_id,
            "mask_map": mask_map,
            "layout_id": inst["layout_id"],
            "instantiate_seed": seed,
        }
        pair_id = f"cfp-{_stable_u64(pair_basis):016x}"
    style_seed = _stable_u64({
        "purpose": "render-style",
        "pair": pair_basis,
        "instantiate_seed": None if pair_basis else seed,
    })

    inst["meta"] = {
        "bench_family": family,
        "cell": [s, m],
        "state_realization": spec["realization"],
        "n_realizations": len(spec["realizations"]),
        "paraphrase_id": paraphrase_id,
        "n_paraphrases": n_par,
        "mask_map": mask_map,
        "opaque_container": tpl["opaque_container"],
        "instantiate_seed": seed,
        "style_seed": style_seed,
        "distractors": distractors,
        "device_variant": tpl.get("device_variant"),
        "empty_plan": not inst["ground_truth"],
        "layout_restricted": layout_restricted,
        # which named fixture masks narrowed the layout pool (groundability)
        "fixture_layout_masks": restricting_masks,
        "state_entailed_open": state_entailed_open,
        "base_mechanical_signature": _base_mechanical_signature(
            family, m, inst),
        "bridge_kind": tpl.get("bridge_kind"),
        "bridge_variant": tpl.get("bridge_variant"),
        "counterfactual_pair_id": pair_id,
        "observation_tool": (
            {"name": "detect", "args": {"query": {"tag": "fruit"}}}
            if family == "F3" else None),
    }
    return inst


def instantiate_counterfactual_pair(family, s, m, rng, **kwargs):
    """Instantiate the two F4-S4 realizations from one invariant seed.

    The returned cabinet/counter samples share masks, paraphrase, layout,
    camera/style seed, robot permutation and non-target randomness. Only
    target placement and the entailed GT branch differ.
    """
    if family != "F4" or s != 4:
        raise ValueError("counterfactual pairing is defined only for F4-S4")
    # §7.10 requires the pair to share its masks; both halves get the same
    # fresh Random(pair_seed), so an i.i.d. draw gives them identical values.
    # A min-count sampler would NOT: the first half's draw increments the
    # count, so the second half sees a different min-count set and picks a
    # different value -- silently breaking pair invariance. No F4-S4 row is
    # small enough to be stratified (asserted in tests), so refuse loudly
    # rather than let a future plan change corrupt the pairing audit.
    if kwargs.get("stratify_keys"):
        raise ValueError(
            "stratified mask sampling is incompatible with F4-S4 "
            "counterfactual pairing (§7.10 mask invariance): "
            f"{sorted(kwargs['stratify_keys'])}")
    pair_seed = rng.getrandbits(64)
    out = []
    for state in ("cabinet", "counter"):
        out.append(instantiate_locus_task(
            family, s, m, random.Random(pair_seed),
            realization={"mask1": state},
            counterfactual_pair_seed=pair_seed,
            **kwargs))
    return tuple(out)


def generate_locus_cell(family, s, m, n, rng, sampler=None,
                        variant_sampler=None, **kwargs):
    sampler = sampler or RealizationSampler()
    variant_sampler = variant_sampler or BridgeVariantSampler()
    fixed_variant = kwargs.pop("bridge_variant", None)
    if family == "F4" and s == 4:
        out = []
        while len(out) < n:
            out.extend(instantiate_counterfactual_pair(
                family, s, m, rng, **kwargs))
        return out[:n]
    out = []
    for _ in range(n):
        out.append(instantiate_locus_task(
            family, s, m, rng,
            realization=sampler.next(family, s, m, rng),
            bridge_variant=(fixed_variant or
                            variant_sampler.next(family, s, m, rng)),
            **kwargs))
    return out


def generate_locus_batch(n_per_cell, seed=0, families=None, **kwargs):
    """Balanced batch over every (family, cell); realization counts within a
    cell differ by at most 1."""
    rng = random.Random(seed)
    sampler = RealizationSampler()
    variant_sampler = BridgeVariantSampler()
    out = []
    for fam, s, m in iter_cells():
        if families and fam not in families:
            continue
        cell_instances = generate_locus_cell(
            fam, s, m, n_per_cell, rng, sampler=sampler,
            variant_sampler=variant_sampler, **kwargs)
        for inst in cell_instances:
            inst["meta"]["sample_index"] = len(out)
            out.append(inst)
    return out


def realization_balance_report(instances):
    """Per-(family, cell) realization distribution + §4.3 audit key."""
    per = defaultdict(Counter)
    for inst in instances:
        meta = inst["meta"]
        key = f'{meta["bench_family"]}_s{meta["cell"][0]}m{meta["cell"][1]}'
        per[key][json.dumps(meta["state_realization"], sort_keys=True)] += 1
    report = {}
    for key, counter in sorted(per.items()):
        total = sum(counter.values())
        n_classes = len(counter)
        max_share = max(counter.values()) / total
        report[key] = {
            "total": total,
            "n_realization_classes": n_classes,
            "max_share": round(max_share, 4),
            "tolerance": round(1.0 / n_classes + 0.03, 4),
            "ok": max_share <= 1.0 / n_classes + 0.03 + 1e-9,
            "counts": dict(counter),
        }
    return report


if __name__ == "__main__":
    batch = generate_locus_batch(n_per_cell=8, seed=0)
    report = realization_balance_report(batch)
    n_bad = sum(1 for v in report.values() if not v["ok"])
    n_empty = sum(1 for i in batch if i["meta"]["empty_plan"])
    n_distr = sum(len(i["meta"]["distractors"]) for i in batch)
    print(f"{len(batch)} instances, balance violations={n_bad}, "
          f"empty_plan={n_empty}, distractors={n_distr}")
