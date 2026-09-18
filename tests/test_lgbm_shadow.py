"""LGBM 影子训练门控测试（2026-09-18 更新：全样本训练 + 双基线影子验证）

不依赖 lightgbm（本地无 libomp）——只测门控与数据重建路径；
真训练路径由注入的 trainer 桩覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.learning.lgbm_shadow import build_training_rows, shadow_train


def _make_daily(tmp_path: Path, n: int = 3, v2_from: int = 0):
    """构造 n 天 × 每天 10 场的 daily predictions + 账本记录。

    v2_from 及之后的日期记录带 chain=v2（模拟洁净期）。
    """
    recs = []
    for d_idx in range(n):
        date = f"2026-09-{d_idx + 1:02d}"
        dd = tmp_path / "daily" / date
        dd.mkdir(parents=True)
        preds = []
        for m in range(10):
            mid = f"{date}_测试{m:03d}"
            preds.append({
                "match_id": mid,
                "elo_home": 1500 + m * 5,
                "elo_away": 1500 - m * 5,
                "handicap": -0.5,
                "home_xg": 1.4 + m * 0.05,
                "away_xg": 1.1,
                "djyy_model_prob": {"home": 0.5, "draw": 0.25, "away": 0.25},
            })
            rec = {"match_id": mid, "date": date, "actual_idx": m % 3,
                   "final_prob": [0.4, 0.3, 0.3],
                   "market_fair": [0.42, 0.28, 0.30]}
            if d_idx >= v2_from:
                rec["chain"] = "v2"
            recs.append(rec)
        (dd / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False))
    return recs


class _FakeTrainer:
    """桩：记录 train 调用，predict 返回均匀分布"""

    def __init__(self):
        self.train_calls = 0
        self.saved = False

    def train(self, X, y, eval_features=None, eval_labels=None):
        self.train_calls += 1

    def predict_single(self, feats):
        return [0.34, 0.33, 0.33]

    def save(self):
        self.saved = True


class _PerfectTrainer(_FakeTrainer):
    """桩：按特征里的 elo_home 编码还原标签（构造性完美预测）→ 触发 ready"""

    def predict_single(self, feats):
        # _make_daily 中 label = m%3, elo_home = 1500+m*5 → m = (elo-1500)/5
        m = int((feats["elo_home"] - 1500) / 5)
        label = m % 3
        return [1.0 if i == label else 0.0 for i in range(3)]


def test_build_training_rows_from_daily(tmp_path):
    recs = _make_daily(tmp_path, n=2)
    rows = build_training_rows(recs, tmp_path / "daily")
    assert len(rows) == 20
    assert rows[0]["label"] in (0, 1, 2)
    assert "elo_diff" in rows[0]["features"]
    assert all(r["_v2"] for r in rows)  # v2_from=0 → 全部洁净


def test_below_min_samples_not_trained(tmp_path):
    recs = _make_daily(tmp_path, n=2)  # 20 行 < 300
    fake = _FakeTrainer()
    status = shadow_train(recs, recs, tmp_path / "daily",
                          tmp_path / "lgbm_model.txt", trainer=fake,
                          config={"min_train_samples": 300})
    assert status["trained"] is False
    assert status["ready"] is False
    assert fake.train_calls == 0
    assert "300" in status["reason"]


def test_trains_and_evaluates_dual_baseline(tmp_path):
    recs = _make_daily(tmp_path, n=60)  # 600 行 ≥ 300
    fake = _FakeTrainer()
    status = shadow_train(recs, recs, tmp_path / "daily",
                          tmp_path / "lgbm_model.txt", trainer=fake,
                          config={"min_train_samples": 300, "holdout_frac": 0.3})
    assert status["trained"] is True
    assert fake.train_calls == 2  # 切分训练 + 全量重训
    assert fake.saved is True
    assert status["holdout_n"] > 0
    assert status["holdout_brier_lgbm"] is not None
    # 均匀预测 vs final/market 都不占优 → ready False
    assert status["ready"] is False
    assert "未双基线显著" in status["reason"]


def test_perfect_model_becomes_ready(tmp_path):
    recs = _make_daily(tmp_path, n=60)
    pt = _PerfectTrainer()
    status = shadow_train(recs, recs, tmp_path / "daily",
                          tmp_path / "lgbm_model.txt", trainer=pt,
                          config={"min_train_samples": 300, "holdout_frac": 0.3})
    assert status["trained"] is True
    assert status["ready"] is True, status["reason"]
    assert status["delta_vs_fusion_v2"] > 0.1
    assert status["delta_vs_market"] > 0.1


def test_v2_only_fusion_comparison(tmp_path):
    # 只有最后5天是 v2: 对照融合只统计 v2 子集 → 样本小, 不会 ready
    recs = _make_daily(tmp_path, n=60, v2_from=55)
    fake = _FakeTrainer()
    status = shadow_train(recs, [r for r in recs if r.get("chain") == "v2"],
                          tmp_path / "daily", tmp_path / "lgbm_model.txt",
                          trainer=fake, config={"min_train_samples": 300})
    assert status["trained"] is True
    assert status["ready"] is False


def test_trainer_exception_is_contained(tmp_path):
    recs = _make_daily(tmp_path, n=60)

    class _Boom(_FakeTrainer):
        def train(self, X, y, **kw):
            raise RuntimeError("boom")

    status = shadow_train(recs, recs, tmp_path / "daily",
                          tmp_path / "lgbm_model.txt", trainer=_Boom(),
                          config={"min_train_samples": 300})
    assert status["trained"] is False
    assert "boom" in status["reason"] or "训练异常" in status["reason"]
