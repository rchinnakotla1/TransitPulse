"""Phase 4: evaluate the LLM narratives on a sample of real test events.

* 20 test events from the SHAP sample, spread over the probability range (deciles), x 3 audiences = 60 narratives
* automatic checks per narrative: faithfulness (no invented numbers, top factor named), length, source (llm / template fallback)
* writes ``reports/xai/narrative_eval.csv`` and an empty human-rating rubric ``narrative_rubric.csv``
  (clarity / correctness / usefulness, 1-5) so a person can rate them - we deliberately do not use an LLM as judge.

Run: ``python -m src.xai.evaluate_narratives``
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.encoding import feature_list
from src.xai import llm_client, narrator
from src.xai.shap_analysis import XAI_DIR, threshold


def sample_events(mode: str = "planning", n: int = 20, seed: int = 5) -> pd.DataFrame:
    d = pd.read_parquet(XAI_DIR / f"shap_{mode}.parquet")
    d["bin"] = pd.qcut(d.prob, 10, labels=False, duplicates="drop")
    return d.groupby("bin", group_keys=False).sample(n // 10, random_state=seed)


def facts_for(row: pd.Series, mode: str) -> dict:
    feats = feature_list(mode)
    sv = pd.Series({f: row[f"shap__{f}"] for f in feats})
    return narrator.facts_from_shap(row, sv, float(row.base_value), threshold(mode), mode)


def main(mode: str = "planning"):
    st = llm_client.health_check(force=True)
    print("OpenAI:", st.code, st.message)
    ev = sample_events(mode)
    rows = []
    for i, r in ev.iterrows():
        f = facts_for(r, mode)
        for aud in narrator.AUDIENCES:
            out = narrator.llm_explanation(f, aud, use_llm=st.ok)
            chk = narrator.faithfulness(out["text"], f)
            rows.append({"event": i, "audience": aud, "source": out["source"], "status": out["status"],
                         "ok": chk["ok"], "invented_numbers": ",".join(chk["invented_numbers"]),
                         "top_factor_mentioned": chk["top_factor_mentioned"], "words": len(out["text"].split()),
                         "prob": round(float(r.prob), 3), "actual": int(r.delayed), "text": out["text"],
                         "rejected_text": out.get("rejected_text", ""),
                         "template": narrator.template_explanation(f, aud), "facts": json.dumps(f)})
    df = pd.DataFrame(rows)
    df.to_csv(XAI_DIR / "narrative_eval.csv", index=False)
    rub = df[["event", "audience", "source", "text"]].assign(clarity_1to5="", correctness_1to5="", usefulness_1to5="",
                                                             rater="", notes="")
    rub.to_csv(XAI_DIR / "narrative_rubric.csv", index=False)
    summ = df.groupby(["audience", "source"]).agg(n=("ok", "size"), faithful=("ok", "mean"), words=("words", "mean"))
    rej = (df.status == "unfaithful").mean()
    print(summ.round(2)); print(f"rejected by fact check: {rej:.1%}")
    return df


if __name__ == "__main__":
    main()
