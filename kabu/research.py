"""ルールの探索・検証・採用判定。

流れ（ファミリーごと）:
  1. パラメータ全組み合わせを全期間でバックテストする。
  2. 「学習期間」(oos_start より前) の成績だけでパラメータを 1 つ選ぶ。
     - 1 点だけ突出した組み合わせを避けるため、パラメータごとの平均的な良さで選ぶ。
  3. 選んだ 1 つを「未使用データ」(oos_start 以降) で採点する。未使用データは選択に一切使わない。
  4. ウォークフォワード（それまでの全データで選ぶ → 翌年で試す、を毎年繰り返す）でも採点する。
     これは「毎年ルールを更新し続けたら実際どうなったか」の再現になる。
  5. 採用基準をすべて満たしたものだけを採用する。
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data.prices import Panel
from .data.universe import load_universe
from .engine import Arrays, Result, Setup, run_backtest
from .indicators import Features
from .metrics import monthly_pnl, summarize, window, yearly_table
from .strategies import FAMILIES, Family, Params, build_setup, param_key

MIN_TRADES_PER_YEAR = 15  # これ未満しか取引がない組み合わせは統計的に評価できないので選ばない


@dataclass
class Context:
    cfg: Config
    panel: Panel
    features: Features
    arr: Arrays
    shortable: np.ndarray

    @classmethod
    def create(cls, cfg: Config, panel: Panel) -> "Context":
        uni = load_universe(cfg).set_index("code")
        shortable = np.array([bool(uni["shortable"].get(c, False)) for c in panel.codes])
        feats = Features(panel, cfg.universe["min_turnover"], cfg.universe["min_price"])
        return cls(cfg, panel, feats, Arrays.from_panel(panel), shortable)

    def setup(self, family: str, params: Params, rule_id: str | None = None) -> Setup:
        s = build_setup(family, params, self.features, self.shortable, rule_id)
        s.alloc = self.cfg.rule_alloc   # 実際に持てる建玉の大きさで検証する（買える銘柄が変わるため）
        return s

    def backtest(self, setups: list[Setup], start: str | None = None, end: str | None = None) -> Result:
        cfg = self.cfg
        return run_backtest(self.arr, setups, cfg.capital, cfg.max_leverage, cfg.costs, cfg.lot_size,
                            start or cfg.data["backtest_start"], end, max_exposure=cfg.exposure_cap)


# ----------------------------------------------------------------------
# 全組み合わせの評価（並列）
# ----------------------------------------------------------------------
_CTX: Context | None = None


def _eval_combo(job: tuple[str, Params]) -> dict:
    family, params = job
    ctx = _CTX
    res = ctx.backtest([ctx.setup(family, params)])
    m = monthly_pnl(res.equity)
    tr = res.trades
    period = tr["entry_date"].dt.to_period("M") if len(tr) else pd.PeriodIndex([], freq="M")
    pnl = tr["pnl"].to_numpy() if len(tr) else np.array([])
    gp = pd.Series(np.where(pnl > 0, pnl, 0.0), index=period).groupby(level=0).sum().reindex(m.index, fill_value=0.0)
    gl = pd.Series(np.where(pnl <= 0, -pnl, 0.0), index=period).groupby(level=0).sum().reindex(m.index, fill_value=0.0)
    n = pd.Series(1, index=period).groupby(level=0).sum().reindex(m.index, fill_value=0)
    return {"key": param_key(params), "params": params, "pnl": m.to_numpy(), "gp": gp.to_numpy(),
            "gl": gl.to_numpy(), "n": n.to_numpy(), "months": m.index}


def _warm_features(ctx: Context, fam: Family) -> None:
    """fork 前に必要な指標をすべて計算しておく（子プロセスで重複計算しないため）。"""
    base = {k: v[0] for k, v in fam.grid.items()}
    for k, values in fam.grid.items():
        for v in values:
            ctx.setup(fam.key, {**base, k: v})


def evaluate_grid(ctx: Context, fam: Family, workers: int | None = None) -> dict[str, dict]:
    global _CTX
    _CTX = ctx
    _warm_features(ctx, fam)
    jobs = [(fam.key, p) for p in fam.combos()]
    workers = workers or min(8, os.cpu_count() or 1)
    if workers > 1 and hasattr(os, "fork"):
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_eval_combo, jobs, chunksize=4)
    else:
        out = [_eval_combo(j) for j in jobs]
    return {r["key"]: r for r in out}


# ----------------------------------------------------------------------
# 選択
# ----------------------------------------------------------------------
def _slice_stats(r: dict, lo: pd.Period | None, hi: pd.Period | None) -> dict:
    """[lo, hi) の月の成績。"""
    months = r["months"]
    sel = np.ones(len(months), dtype=bool)
    if lo is not None:
        sel &= months >= lo
    if hi is not None:
        sel &= months < hi
    pnl = r["pnl"][sel]
    n_m = len(pnl)
    trades = int(r["n"][sel].sum())
    gp, gl = float(r["gp"][sel].sum()), float(r["gl"][sel].sum())
    mean = float(pnl.mean()) if n_m else 0.0
    std = float(pnl.std(ddof=1)) if n_m > 1 else 0.0
    return {
        "months": n_m, "trades": trades, "mean": mean,
        "tstat": mean / (std / math.sqrt(n_m)) if std > 0 else 0.0,
        "pf": gp / gl if gl > 0 else (9.99 if gp > 0 else 0.0),
        "gp": gp, "gl": gl, "pnl": pnl,
    }


def _objective(st: dict) -> float:
    years = max(st["months"] / 12.0, 1e-9)
    if st["trades"] < MIN_TRADES_PER_YEAR * years:
        return -math.inf
    return st["tstat"]


def select_params(fam: Family, grid: dict[str, dict], lo: pd.Period | None, hi: pd.Period | None) -> dict | None:
    """期間 [lo, hi) の成績でパラメータを 1 つ選ぶ。

    「全組み合わせの中の 1 位」は偶然の当たりを拾いやすいので使わない。
    代わりに、パラメータごとに「その値を使った全組み合わせの成績の中央値」が最も良い値を選び、
    それらを組み合わせる。数百通りの中央値で決まるので、偶然の 1 点に左右されにくい。
    """
    own = {k: _objective(_slice_stats(r, lo, hi)) for k, r in grid.items()}
    valid = {k: v for k, v in own.items() if v != -math.inf}
    if not valid:
        return None
    chosen: Params = {}
    for name, values in fam.grid.items():
        best_v, best_m = None, -math.inf
        for val in values:
            xs = [valid[k] for k in valid if grid[k]["params"][name] == val]
            if not xs:
                continue
            m = float(np.median(xs))   # 中央値: 1 点だけ突出した組み合わせに引っ張られない
            if m > best_m:
                best_v, best_m = val, m
        chosen[name] = best_v if best_v is not None else values[0]
    key = param_key(chosen)
    if key not in valid:
        # 選んだ組み合わせ自体の取引が少なすぎる場合は、有効な中で隣接込みの成績が最良のものにする
        def smooth(k: str) -> float:
            nb = [valid.get(param_key(q), 0.0) for q in fam.neighbors(grid[k]["params"])]
            return 0.5 * valid[k] + 0.5 * (float(np.mean(nb)) if nb else valid[k])
        key = max(valid, key=smooth)
    nb = [own[param_key(q)] for q in fam.neighbors(grid[key]["params"])]
    nb = [x if x != -math.inf else 0.0 for x in nb]
    return {"key": key, "params": grid[key]["params"], "score": valid[key],
            "neighbor_pass": float(np.mean([x > 0 for x in nb])) if nb else 1.0,
            "neighbors": len(nb),
            "grid_positive": float(np.mean([v > 0 for v in valid.values()]))}


def walk_forward(fam: Family, grid: dict[str, dict], cfg: Config) -> dict:
    """毎年「それまでの全データで選ぶ → 翌年で試す」を繰り返した通算成績（最初の学習は N 年分）。"""
    v = cfg.validation
    months = next(iter(grid.values()))["months"]
    first, last = months[0].year, months[-1].year
    train, test = int(v["wf_train_years"]), int(v["wf_test_years"])
    pnl_parts, gp, gl, trades, folds = [], 0.0, 0.0, 0, []
    for y in range(first + train, last + 1, test):
        lo, hi = None, pd.Period(f"{y}-01", "M")
        sel = select_params(fam, grid, lo, hi)
        if sel is None:
            folds.append({"year": y, "params": None, "pnl": 0.0, "trades": 0})
            n_test = int(((months >= hi) & (months < pd.Period(f"{y + test}-01", "M"))).sum())
            pnl_parts.append(np.zeros(n_test))
            continue
        st = _slice_stats(grid[sel["key"]], hi, pd.Period(f"{y + test}-01", "M"))
        pnl_parts.append(st["pnl"])
        gp += st["gp"]; gl += st["gl"]; trades += st["trades"]
        folds.append({"year": y, "params": sel["params"], "pnl": float(st["pnl"].sum()),
                      "trades": st["trades"], "pf": st["pf"]})
    pnl = np.concatenate(pnl_parts) if pnl_parts else np.array([])
    n = len(pnl)
    mean = float(pnl.mean()) if n else 0.0
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    return {
        "months": n, "trades": trades, "monthly_mean": mean,
        "profit_factor": gp / gl if gl > 0 else 0.0,
        "positive_months": float((pnl > 0).mean()) if n else 0.0,
        "tstat": mean / (std / math.sqrt(n)) if std > 0 else 0.0,
        "positive_years": int(sum(1 for f in folds if f["pnl"] > 0)), "years": len(folds),
        "folds": folds,
    }


# ----------------------------------------------------------------------
# 採点
# ----------------------------------------------------------------------
def evaluate_rule(ctx: Context, family: str, params: Params, since: str | None = None) -> dict:
    """1 つのルールを全期間で走らせ、学習期間 / 未使用データ / 全期間 / 直近 の成績をまとめる。

    since を渡すと、その日以降（= このルールを台帳に載せた後の、本当に未知だったデータ）の成績も付ける。
    """
    cfg = ctx.cfg
    res = ctx.backtest([ctx.setup(family, params)])
    oos = cfg.validation["oos_start"]
    cap, tgt = cfg.capital, cfg.monthly_target
    is_t, is_e = window(res.trades, res.equity, None, oos)
    oos_t, oos_e = window(res.trades, res.equity, oos, None)
    recent_n = int(cfg.validation["suspend_recent_months"])
    recent_start = (res.equity.index[-1].to_period("M") - (recent_n - 1)).start_time
    rc_t, rc_e = window(res.trades, res.equity, str(recent_start.date()), None)
    m = monthly_pnl(res.equity)
    forward = None
    if since:
        fw_t, fw_e = window(res.trades, res.equity, since, None)
        if len(fw_e):
            forward = summarize(fw_t, fw_e, cap, tgt)
            forward["since"] = since
    return {
        "family": family, "params": params, "forward": forward,
        "is": summarize(is_t, is_e, cap, tgt),
        "oos": summarize(oos_t, oos_e, cap, tgt),
        "full": summarize(res.trades, res.equity, cap, tgt),
        "recent": summarize(rc_t, rc_e, cap, tgt),
        "yearly": yearly_table(res.trades, res.equity),
        "monthly": [{"month": str(k), "pnl": float(x)} for k, x in m.items()],
        "exit_reasons": res.trades["reason"].value_counts().to_dict() if len(res.trades) else {},
    }


def check_gates(ev: dict, wf: dict | None, neighbor_pass: float | None, cfg: Config) -> list[dict]:
    """採用基準の判定。各基準について {name, value, threshold, ok} を返す。"""
    v = cfg.validation
    oos, full = ev["oos"], ev["full"]
    annual = full["monthly_mean"] * 12
    dd_years = full["max_drawdown"] / annual if annual > 0 else 99.0
    gates = [
        {"core": True, "name": "未使用データでの取引回数", "value": oos["trades"], "threshold": v["min_trades_oos"],
         "ok": oos["trades"] >= v["min_trades_oos"], "fmt": "int"},
        {"core": True, "name": "未使用データのプロフィットファクター", "value": oos["profit_factor"],
         "threshold": v["min_profit_factor_oos"], "ok": oos["profit_factor"] >= v["min_profit_factor_oos"], "fmt": "x"},
        {"core": False, "name": "未使用データでプラスだった月の割合（取引があった月のうち）", "value": oos["positive_active_months"],
         "threshold": v["min_positive_months"], "ok": oos["positive_active_months"] >= v["min_positive_months"],
         "fmt": "pct"},
        {"core": True, "name": "月次損益の t値（全期間）", "value": full["tstat"], "threshold": v["min_tstat"],
         "ok": full["tstat"] >= v["min_tstat"], "fmt": "x"},
        {"core": False, "name": "最大ドローダウンが平均年間利益の何年分か", "value": dd_years,
         "threshold": v["max_drawdown_years"], "ok": dd_years <= v["max_drawdown_years"],
         "fmt": "x", "upper": True},
    ]
    if wf is not None:
        gates.append({"core": True, "name": "ウォークフォワード通算のプロフィットファクター", "value": wf["profit_factor"],
                      "threshold": v["min_profit_factor_wf"], "ok": wf["profit_factor"] >= v["min_profit_factor_wf"],
                      "fmt": "x"})
    if neighbor_pass is not None:
        gates.append({"core": True, "name": "近傍パラメータが学習期間で黒字だった割合", "value": neighbor_pass,
                      "threshold": v["min_neighbor_pass"], "ok": neighbor_pass >= v["min_neighbor_pass"], "fmt": "pct"})
    return gates


def tier_of(gates: list[dict]) -> str | None:
    """採用基準の結果から段階を決める。

    active : すべての基準を満たす → 実際の資金で運用してよい
    watch  : 統計的な優位性の基準（core）は満たすが、リスク面などの基準が未達 → 仮想売買で観察
    None   : 不採用
    """
    if all(g["ok"] for g in gates):
        return "active"
    if all(g["ok"] for g in gates if g.get("core")):
        return "watch"
    return None


def neighbor_pass_for(fam: Family, grid: dict[str, dict], params: Params, hi: pd.Period) -> float | None:
    """あるパラメータの「隣」が学習期間で黒字だった割合。"""
    if param_key(params) not in grid:
        return None
    vals = []
    for q in fam.neighbors(params):
        r = grid.get(param_key(q))
        if r is None:
            continue
        o = _objective(_slice_stats(r, None, hi))
        vals.append(o if o != -math.inf else 0.0)
    return float(np.mean([x > 0 for x in vals])) if vals else 1.0


def health_check(ev: dict, cfg: Config) -> dict:
    """採用中ルールの健康診断（直近の成績が崩れていないか）。"""
    v = cfg.validation
    rc, full = ev["recent"], ev["full"]
    reasons = []
    if rc["trades"] >= 10 and rc["profit_factor"] < v["suspend_profit_factor"]:
        reasons.append(f"直近 {v['suspend_recent_months']} か月のプロフィットファクターが {rc['profit_factor']:.2f}"
                       f"（基準 {v['suspend_profit_factor']} 未満）")
    hist_dd = full["max_drawdown"]
    if hist_dd > 0 and rc["max_drawdown"] > v["suspend_dd_multiple"] * hist_dd:
        reasons.append("直近のドローダウンが過去最大の想定を大きく超えた")
    return {"ok": not reasons, "reasons": reasons}


def research_family(ctx: Context, fam: Family, workers: int | None = None) -> dict:
    """1 ファミリーの探索〜採点。"""
    cfg = ctx.cfg
    grid = evaluate_grid(ctx, fam, workers)
    oos_p = pd.Period(cfg.validation["oos_start"][:7], "M")
    sel = select_params(fam, grid, None, oos_p)
    wf = walk_forward(fam, grid, cfg)
    out = {"family": fam.key, "combos_tested": len(grid), "walk_forward": wf, "candidate": None}
    if sel is None:
        out["note"] = "学習期間に十分な取引回数のある組み合わせがありませんでした。"
        return out
    ev = evaluate_rule(ctx, fam.key, sel["params"])
    gates = check_gates(ev, wf, sel["neighbor_pass"], cfg)
    out["candidate"] = {
        "params": sel["params"], "selection_score": sel["score"], "neighbor_pass": sel["neighbor_pass"],
        "evidence": ev, "gates": gates, "passed": all(g["ok"] for g in gates), "tier": tier_of(gates),
    }
    out["_grid"] = grid
    return out


def rule_objective(ev: dict) -> float:
    """ルール同士を比べるときの物差し（全期間の月次 t値）。"""
    return float(ev["full"]["tstat"])
