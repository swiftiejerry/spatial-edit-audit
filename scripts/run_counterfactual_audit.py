"""Run paired counterfactual gate interventions (native vs disable_damage vs relabel) across S6 and S8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_edit_audit import (  # noqa: E402
    GateConfig,
    apply_reference_intervention,
    evaluate_multigate_prototype,
    generate_main_case,
    load_section_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired S6/S8 counterfactual gate audit.")
    parser.add_argument("--replicates", type=int, default=20, help="Replicates per stratum per section (max 20).")
    args = parser.parse_args()

    ref_dir = ROOT / "data" / "reference"
    baselines = {b: load_section_baseline(ref_dir / f"{b}_baseline.npz") for b in ("B1", "B2", "B3", "B4")}

    summary = {"S6": {"native": 0, "disable_damage": 0, "relabel": 0, "n": 0},
               "S8": {"native": 0, "disable_damage": 0, "relabel": 0, "n": 0}}
    cfg_native = GateConfig(disable_damage_gate=False)
    cfg_no_dmg = GateConfig(disable_damage_gate=True)

    for _bg, bl in baselines.items():
        for stratum, relabel_mode in ((6, "three_way_split"), (8, "homogeneous")):
            key = f"S{stratum}"
            for rep in range(1, int(args.replicates) + 1):
                case = generate_main_case(bl, block=2, stratum=stratum, rep=rep, fit_i_case=True)
                t_id = case["truth"]["positions"]["T"]
                r_nat = evaluate_multigate_prototype(bl, case["X_Q"], case["I_case"], config=cfg_native)
                r_no = evaluate_multigate_prototype(bl, case["X_Q"], case["I_case"], config=cfg_no_dmg)
                r_rel = evaluate_multigate_prototype(
                    bl,
                    case["X_Q"],
                    case["I_case"],
                    reference_labels=apply_reference_intervention(bl, t_id, relabel_mode),
                    config=cfg_native,
                )
                summary[key]["n"] += 1
                summary[key]["native"] += int(t_id in r_nat["accepted_ids"])
                summary[key]["disable_damage"] += int(t_id in r_no["accepted_ids"])
                summary[key]["relabel"] += int(t_id in r_rel["accepted_ids"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
