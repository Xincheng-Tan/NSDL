"""W3 follow-up: paired per-target reading of the evidence-conditional audit.

Consumes the ``targets_7b__*.jsonl`` files written by
``scripts/audit_w3_evidence_conditional.py`` plus the first [GOAL] of each
trace, and answers three questions the per-condition table cannot:

1. On the SAME targets, does supplying the required fact (yolo-absent ->
   full-oracle-present) change the first judgment?  (paired McNemar)
2. What is a wrong "active" judgment made of: an explicit skip, a target
   omitted from a non-empty [GOAL], or an all-empty [GOAL]?  How much of that
   moves with the evidence channel?
3. Is the full-oracle drop on active targets a judgment failure or a
   relation-vocabulary shift?  Split present targets by whether their GT
   relation is one the SFT corpus ever demonstrates (``on table`` /
   ``in cabinet``) or one only the oracle backend emits (``at kitchen work
   area`` / ``on rack`` / ...).

Usage:
    python scripts/audit_w3_paired_targets.py \
        --audit-dir results/lnsdl/audits/w3_evidence_conditional \
        --bucket-root results/lnsdl/v3_3/locus
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from math import comb
from pathlib import Path

CONDITIONS = {
    "off": ("bracket_off", "_bracket_off"),
    "yolo": ("yolo+crop", ""),
    "query": ("bracket_oracle", "_bracket_oracle"),
    "full": ("bracket_oracle_full", "_bracket_oracle_full"),
}
# W4 (plan 序 34, 2026-09-11): vocabulary-and-scope-matched oracle rows. Optional --
# sections 1-5 are unchanged when their target files are absent.
W4_CONDITIONS = {
    "cropoff": ("cropoff", "_cropoff"),
    "yolovocab": ("bracket_oracle_yolovocab", "_bracket_oracle_yolovocab"),
    "yolovocab_full": ("bracket_oracle_yolovocab_full", "_bracket_oracle_yolovocab_full"),
}
MAINROW = "locus_g1_mainrow_v341_cap524288_7b_seed42"
TRAINED_RELATION_FORMS = {"on table", "in cabinet"}
_GOAL_RE = re.compile(r"\[GOAL\]\s*(\{.*?\})\s*(?=\[|$)", re.S)


def _load_targets(audit_dir: Path, name: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in open(audit_dir / f"targets_7b__{name}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[(r["stem"], r["target"])] = r
    return rows


def _first_goal(bucket_dir: str, stem: str) -> dict | None:
    hits = glob.glob(f"{bucket_dir}/*/{stem}.json")
    if not hits:
        return None
    trace = json.load(open(hits[0], encoding="utf-8"))["agentic_eval"]["trace"]
    for turn in trace.get("assistant_turns") or []:
        text = turn if isinstance(turn, str) else json.dumps(turn)
        m = _GOAL_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _acc(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "–"
    k = sum(r["correct"] for r in rows)
    return f"{k / n:.3f} ({k}/{n})"


def _mcnemar(a: dict, b: dict, keys: list) -> tuple[int, int, float]:
    ab = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    ba = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    n = ab + ba
    if n == 0:
        return ab, ba, 1.0
    m = min(ab, ba)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)
    return ab, ba, p


def _vocab(row: dict) -> str:
    forms = {rel.split(" ", 1)[1] for rel in row["gt_relations"]}
    if not forms:
        return "none"
    if forms <= TRAINED_RELATION_FORMS:
        return "trained"
    if forms & TRAINED_RELATION_FORMS:
        return "mixed"
    return "untrained"


def _error_kind(target: str, goal: dict | None) -> str:
    if goal is None:
        return "no_goal"
    skip = {str(x).lower() for x in goal.get("skip") or []}
    active = {str(x).lower() for x in goal.get("active") or []}
    if target in skip:
        return "explicit_skip"
    if not skip and not active:
        return "empty_goal"
    return "omitted_from_nonempty_goal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="results/lnsdl/audits/w3_evidence_conditional")
    ap.add_argument("--bucket-root", default="results/lnsdl/v3_3/locus")
    ap.add_argument("--out", default=None, help="markdown path (default <audit-dir>/paired_summary.md)")
    args = ap.parse_args()
    audit_dir = Path(args.audit_dir)
    T = {c: _load_targets(audit_dir, spec[0]) for c, spec in CONDITIONS.items()}
    keys = sorted(set.intersection(*(set(t) for t in T.values())))
    W4 = {c: _load_targets(audit_dir, spec[0]) for c, spec in W4_CONDITIONS.items()
          if (audit_dir / f"targets_7b__{spec[0]}.jsonl").is_file()}
    out: list[str] = ["# W3 paired per-target reading (7B main-row ckpt, `__phisched_v2` caliber)", ""]
    out.append(f"paired targets: {len(keys)}")

    out += ["", "## 1. Same-target facet-A across conditions (paired McNemar vs full)", "",
            "| condition | facet-A | b (cond right / full wrong) | c (cond wrong / full right) | p |", "|---|---|---:|---:|---|"]
    for c in ("off", "yolo", "query", "full"):
        b, cc, p = _mcnemar(T[c], T["full"], keys)
        out.append(f"| {c} | {_acc([T[c][k] for k in keys])} | {b} | {cc} | {p:.4f} |")

    out += ["", "## 2. Strata by (fact present under yolo, fact present under full) — same targets", "",
            "| stratum | n | off | yolo | full | yolo→full b/c |", "|---|---:|---|---|---|---|"]
    strata = collections.defaultdict(list)
    for k in keys:
        strata[(T["yolo"][k]["relation_seen"], T["full"][k]["relation_seen"])].append(k)
    for subset_name, pred in (("all", lambda r: True), ("gt_active", lambda r: r["gt_active"]), ("gt_skip", lambda r: not r["gt_active"])):
        for st, ks in sorted(strata.items()):
            ks2 = [k for k in ks if pred(T["full"][k])]
            if not ks2:
                continue
            b, cc, _ = _mcnemar(T["yolo"], T["full"], ks2)
            out.append(f"| {subset_name} · yolo={st[0]} full={st[1]} | {len(ks2)} | {_acc([T['off'][k] for k in ks2])} | "
                       f"{_acc([T['yolo'][k] for k in ks2])} | {_acc([T['full'][k] for k in ks2])} | {b}/{cc} |")

    out += ["", "## 3. Present ∧ active targets under full, split by GT-relation vocabulary", "",
            "| vocabulary | n | full | same targets under yolo | under off |", "|---|---:|---|---|---|"]
    grp = collections.defaultdict(list)
    for k in keys:
        r = T["full"][k]
        if r["relation_seen"] and r["gt_active"]:
            grp[_vocab(r)].append(k)
    for v, ks in sorted(grp.items()):
        out.append(f"| {v} | {len(ks)} | {_acc([T['full'][k] for k in ks])} | {_acc([T['yolo'][k] for k in ks])} | {_acc([T['off'][k] for k in ks])} |")
    grp = collections.defaultdict(list)
    for k in keys:
        r = T["query"][k]
        if r["relation_seen"] and r["gt_active"]:
            grp[_vocab(r)].append(k)
    out += ["", "query-scope oracle, present ∧ active by vocabulary:", ""]
    for v, ks in sorted(grp.items()):
        out.append(f"- {v}: n={len(ks)} query {_acc([T['query'][k] for k in ks])} / same targets yolo {_acc([T['yolo'][k] for k in ks])}")

    out += ["", "## 4. Anatomy of wrong ACTIVE first judgments (target is gt_active, model did not mark it active)", "",
            "| condition | wrong active | explicit skip | omitted from non-empty GOAL | all-empty GOAL | by family |", "|---|---:|---:|---:|---:|---|"]
    for c, (_, suffix) in CONDITIONS.items():
        bucket_dir = f"{args.bucket_root}/{MAINROW}{suffix}"
        wrong = [k for k in keys if T[c][k]["gt_active"] and T[c][k]["parsed"] and not T[c][k]["correct"]]
        goals: dict[str, dict | None] = {}
        kinds = collections.Counter()
        fam = collections.defaultdict(collections.Counter)
        for stem, target in wrong:
            if stem not in goals:
                goals[stem] = _first_goal(bucket_dir, stem)
            kind = _error_kind(target, goals[stem])
            kinds[kind] += 1
            fam[T[c][(stem, target)]["family"]][kind] += 1
        fam_txt = "; ".join(f"{f}: " + ", ".join(f"{kk}={vv}" for kk, vv in sorted(cnt.items())) for f, cnt in sorted(fam.items()))
        out.append(f"| {c} | {len(wrong)} | {kinds['explicit_skip']} | {kinds['omitted_from_nonempty_goal']} | {kinds['empty_goal']} | {fam_txt} |")

    out += ["", "## 5. Population confound check for the headline 'present ∧ active' numbers", ""]
    pa_full = [k for k in keys if T["full"][k]["relation_seen"] and T["full"][k]["gt_active"]]
    pa_yolo = [k for k in keys if T["yolo"][k]["relation_seen"] and T["yolo"][k]["gt_active"]]
    out.append(f"- present∧active under full (n={len(pa_full)}): full {_acc([T['full'][k] for k in pa_full])}; the SAME targets under yolo {_acc([T['yolo'][k] for k in pa_full])}, under off {_acc([T['off'][k] for k in pa_full])}")
    out.append(f"- present∧active under yolo (n={len(pa_yolo)}): yolo {_acc([T['yolo'][k] for k in pa_yolo])}; the SAME targets under full {_acc([T['full'][k] for k in pa_yolo])}")
    out.append("- ⇒ '.533 vs .789' in the per-condition table compares two different target populations; the paired numbers are the ones above.")

    if "yolovocab" in W4 and "cropoff" in W4:
        out += ["", "## 6. W4 (序 34): vocabulary- and scope-matched oracle vs crop-off YOLO (same targets)", "",
                "`yolovocab` = oracle, query scope, relations restricted to the word forms YOLO emits "
                "(`on table` / `in cabinet` / `on sink` / `on toaster`; counter / rack / carrier placements silent). "
                "Its matched YOLO partner is `cropoff` (no crop channel in any oracle mode), so the only difference "
                "between the two arms is whether the relation facts are correct.", "",
                "| pair | n | A | B | b (A right / B wrong) | c (A wrong / B right) | p |", "|---|---:|---|---|---:|---:|---|"]
        pairs = [("yolovocab vs cropoff", "yolovocab", "cropoff", W4, W4),
                 ("yolovocab vs query (gt vocab)", "yolovocab", "query", W4, T),
                 ("yolovocab vs yolo+crop", "yolovocab", "yolo", W4, T),
                 ("cropoff vs yolo+crop", "cropoff", "yolo", W4, T)]
        if "yolovocab_full" in W4:
            pairs += [("yolovocab_full vs full (gt vocab)", "yolovocab_full", "full", W4, T),
                      ("yolovocab_full vs yolovocab", "yolovocab_full", "yolovocab", W4, W4),
                      ("yolovocab_full vs cropoff", "yolovocab_full", "cropoff", W4, W4)]
        for name, a, b, TA, TB in pairs:
            ks = sorted(set(TA[a]) & set(TB[b]))
            bb, cc, p = _mcnemar(TA[a], TB[b], ks)
            out.append(f"| {name} | {len(ks)} | {_acc([TA[a][k] for k in ks])} | {_acc([TB[b][k] for k in ks])} | {bb} | {cc} | {p:.4f} |")
        out += ["", "### 6a. yolovocab vs cropoff by (fact present under cropoff, fact present under yolovocab)", "",
                "| stratum | n | cropoff | yolovocab | b/c | p |", "|---|---:|---|---|---|---|"]
        ks_all = sorted(set(W4["yolovocab"]) & set(W4["cropoff"]))
        strata = collections.defaultdict(list)
        for k in ks_all:
            strata[(W4["cropoff"][k]["relation_seen"], W4["yolovocab"][k]["relation_seen"])].append(k)
        for subset_name, pred in (("all", lambda r: True), ("gt_active", lambda r: r["gt_active"]), ("gt_skip", lambda r: not r["gt_active"])):
            for st, ks in sorted(strata.items()):
                ks2 = [k for k in ks if pred(W4["yolovocab"][k])]
                if not ks2:
                    continue
                bb, cc, p = _mcnemar(W4["yolovocab"], W4["cropoff"], ks2)
                out.append(f"| {subset_name} · cropoff={st[0]} yolovocab={st[1]} | {len(ks2)} | {_acc([W4['cropoff'][k] for k in ks2])} | "
                           f"{_acc([W4['yolovocab'][k] for k in ks2])} | {bb}/{cc} | {p:.4f} |")
        out += ["", "### 6b. Present ∧ active targets under yolovocab, by GT-relation vocabulary (all should be `trained`/`none`)", "",
                "| vocabulary | n | yolovocab | same targets under cropoff | under query (gt vocab) |", "|---|---:|---|---|---|"]
        grp = collections.defaultdict(list)
        for k in ks_all:
            r = W4["yolovocab"][k]
            if r["relation_seen"] and r["gt_active"]:
                grp[_vocab(r)].append(k)
        for v, ks in sorted(grp.items()):
            out.append(f"| {v} | {len(ks)} | {_acc([W4['yolovocab'][k] for k in ks])} | {_acc([W4['cropoff'][k] for k in ks])} | "
                       f"{_acc([T['query'][k] for k in ks if k in T['query']])} |")
        out += ["", "### 6c. Anatomy of wrong ACTIVE first judgments (W4 conditions)", "",
                "| condition | wrong active | explicit skip | omitted from non-empty GOAL | all-empty GOAL |", "|---|---:|---:|---:|---:|"]
        for c, (_, suffix) in W4_CONDITIONS.items():
            if c not in W4:
                continue
            bucket_dir = f"{args.bucket_root}/{MAINROW}{suffix}"
            wrong = [k for k in ks_all if k in W4[c] and W4[c][k]["gt_active"] and W4[c][k]["parsed"] and not W4[c][k]["correct"]]
            goals: dict[str, dict | None] = {}
            kinds = collections.Counter()
            for stem, target in wrong:
                if stem not in goals:
                    goals[stem] = _first_goal(bucket_dir, stem)
                kinds[_error_kind(target, goals[stem])] += 1
            out.append(f"| {c} | {len(wrong)} | {kinds['explicit_skip']} | {kinds['omitted_from_nonempty_goal']} | {kinds['empty_goal']} |")

    text = "\n".join(out) + "\n"
    out_path = Path(args.out) if args.out else audit_dir / "paired_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
