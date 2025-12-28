"""
Rates Risk Toolkit v2.1 - Production Grade
Jaymin Patil | Fixed Income Analyst Portfolio
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq  # robust 1D root finder

from data_loader import fetch_us_treasury_curve_today, build_india_gsec_curve, write_yield_curves_csv


# ---------- Data loaders (CSV → objects) ----------

def load_portfolio_from_csv(path: str = "data/bonds_portfolio.csv") -> List[Bond]:
    """
    Read bonds_portfolio.csv and return a list of Bond objects.
    Uses pandas.read_csv for robustness. [web:110]
    """
    df = pd.read_csv(path)
    bonds: List[Bond] = []
    for _, row in df.iterrows():
        bonds.append(
            Bond(
                bond_id=str(row["bond_id"]),
                face_amt_cr=float(row["face_amt_cr"]),
                coupon_rate=float(row["coupon_rate"]),
                maturity_years=float(row["maturity_years"]),
                freq=int(row.get("freq", 2)),
            )
        )
    return bonds


def load_curve_from_csv(
    path: str = "data/yield_curves.csv",
    curve_name: str = "US_TSY",
) -> Dict[float, float]:
    """
    Read yield_curves.csv and return {tenor: rate} for the selected curve.
    """
    df = pd.read_csv(path)
    sub = df[df["curve_name"] == curve_name]
    curve: Dict[float, float] = {
        float(row["tenor"]): float(row["rate"]) for _, row in sub.iterrows()
    }
    if not curve:
        raise ValueError(f"No rows found for curve_name='{curve_name}' in {path}")
    return curve


# ---------- Data structures ----------

@dataclass
class Bond:
    """Immutable bond definition"""
    bond_id: str
    face_amt_cr: float
    coupon_rate: float
    maturity_years: float
    freq: int = 2  # payments per year


# ---------- Pricer ----------

class BondPricer:
    """Production bond pricing engine with validation + YTM solver"""

    def __init__(self, day_count: str = "act/365"):
        self.day_count = day_count
        self.day_count_fraction = 365.25

    def validate_inputs(self, face_amt: float, coupon_rate: float, maturity_years: float):
        """Basic sanity checks."""
        if face_amt <= 0:
            raise ValueError("Face amount must be positive")
        if coupon_rate < 0 or coupon_rate > 0.20:
            raise ValueError("Coupon rate unrealistic")
        if maturity_years <= 0 or maturity_years > 30:
            raise ValueError("Maturity between 0-30 years")

    def bond_cashflows(self, bond: Bond, settle_date: str = "2025-12-26") -> pd.DataFrame:
        """Generate a simple fixed-coupon cashflow schedule."""
        self.validate_inputs(bond.face_amt_cr, bond.coupon_rate, bond.maturity_years)

        settle = pd.to_datetime(settle_date)
        maturity = settle + DateOffset(years=int(bond.maturity_years))

        # integer number of coupon periods
        periods = int(bond.maturity_years * bond.freq)

        payment_dates = pd.date_range(
            end=maturity,
            periods=periods + 1,
            freq=f"{int(365.25 // bond.freq)}D"
        )[1:]

        coupon_per_period = (bond.coupon_rate / bond.freq) * bond.face_amt_cr

        rows = []
        for i, date in enumerate(payment_dates):
            days_to_payment = (date - settle).days
            principal = bond.face_amt_cr if i == len(payment_dates) - 1 else 0.0
            total_cf = coupon_per_period + principal
            rows.append(
                {
                    "payment_date": date.strftime("%Y-%m-%d"),
                    "days_to_payment": days_to_payment,
                    "time_fraction": days_to_payment / self.day_count_fraction,
                    "coupon_payment": round(coupon_per_period, 6),
                    "principal_payment": principal,
                    "total_cf": total_cf,
                }
            )

        return pd.DataFrame(rows)

    def discount_factors(
        self,
        spot_curve: Dict[float, float],
        target_times: np.ndarray,
        interp_method: str = "cubic",
    ) -> np.ndarray:
        """Interpolate spot curve and return continuous-compounding discount factors."""
        tenors = np.array(sorted(spot_curve.keys()), dtype=float)
        spot_rates = np.array([spot_curve[t] for t in tenors], dtype=float)

        if interp_method == "cubic" and len(tenors) >= 3:
            interpolator = CubicSpline(tenors, spot_rates, extrapolate=True)
            r_t = interpolator(target_times)
        else:
            interpolator = interp1d(
                tenors,
                spot_rates,
                kind="linear",
                fill_value="extrapolate",
            )
            r_t = interpolator(target_times)

        return np.exp(-r_t * target_times)

    def price_bond(
        self,
        cashflows_df: pd.DataFrame,
        spot_curve: Dict[float, float],
        settle_date: str,
    ) -> Dict[str, float]:
        """Price bond via discounted cashflows and simple accrued interest."""
        settle = pd.to_datetime(settle_date)

        if "days_to_payment" not in cashflows_df.columns:
            cashflows_df["days_to_payment"] = (
                pd.to_datetime(cashflows_df["payment_date"]) - settle
            ).dt.days

        cashflows_df["time_fraction"] = (
            cashflows_df["days_to_payment"] / self.day_count_fraction
        )

        # Discount factors and clean price
        dfactors = self.discount_factors(
            spot_curve,
            cashflows_df["time_fraction"].values,
        )
        clean_price = float(np.sum(cashflows_df["total_cf"].values * dfactors))

        # Accrued interest approximation: linear between last and next coupon
        payment_dates = pd.to_datetime(cashflows_df["payment_date"])
        next_coupon = payment_dates.iloc[0]
        if len(payment_dates) > 1:
            last_coupon = next_coupon - (payment_dates.iloc[1] - payment_dates.iloc[0])
        else:
            last_coupon = next_coupon - pd.Timedelta(days=365.25 / 2)

        days_between = (next_coupon - last_coupon).days
        days_since = max(0, min((settle - last_coupon).days, days_between))
        coupon_amt = float(cashflows_df["coupon_payment"].iloc[0])
        accrued = coupon_amt * days_since / days_between

        dirty_price = clean_price + accrued

        # Solve flat YTM off dirty price
        ytm = self.solve_ytm_flat(cashflows_df, dirty_price)

        return {
            "clean_price": round(clean_price, 6),
            "dirty_price": round(dirty_price, 6),
            "accrued_interest": round(accrued, 6),
            "ytm": round(ytm, 6),
        }

    # ---------- YTM solver (flat curve) ----------

    def _pv_from_flat_ytm(
        self,
        cashflows_df: pd.DataFrame,
        ytm: float,
        freq: int,
    ) -> float:
        """Present value of bond cashflows under flat yield (discrete compounding)."""
        # Assume equal spacing; use index as period count
        cf = cashflows_df["total_cf"].values
        n = len(cf)
        y_per = ytm / freq
        pv = 0.0
        for t in range(1, n + 1):
            pv += cf[t - 1] / ((1 + y_per) ** t)
        return pv

    def solve_ytm_flat(
        self,
        cashflows_df: pd.DataFrame,
        target_dirty: float,
        freq: int = 2,
    ) -> float:
        """Solve for flat YTM (discrete) that matches target_dirty."""
        # Root function: PV(ytm) - target_price = 0
        def f(y):
            return self._pv_from_flat_ytm(cashflows_df, y, freq) - target_dirty

        # Bracket between 0% and 20%
        ytm = brentq(f, 0.0, 0.20)
        return ytm


# ---------- Risk engine ----------

class RiskEngine:
    """Duration, DV01, convexity, and portfolio aggregation."""

    def __init__(self, pricer: BondPricer):
        self.pricer = pricer

    def full_risk_profile(
        self,
        bond: Bond,
        spot_curve: Dict[float, float],
        settle_date: str = "2025-12-26",
    ) -> Dict[str, float]:
        """Complete risk profile for one bond."""
        cfs = self.pricer.bond_cashflows(bond, settle_date)
        base = self.pricer.price_bond(cfs, spot_curve, settle_date)
        base_price = base["clean_price"]

        # 1bp parallel bumps
        shift = 0.0001
        up_curve = {k: v + shift for k, v in spot_curve.items()}
        down_curve = {k: v - shift for k, v in spot_curve.items()}

        up_price = self.pricer.price_bond(cfs, up_curve, settle_date)["clean_price"]
        down_price = self.pricer.price_bond(cfs, down_curve, settle_date)["clean_price"]

        # Modified duration (finite difference)
        mod_duration = -(up_price - down_price) / (2 * base_price * shift)

        # Convexity (finite difference)
        convexity = (up_price + down_price - 2 * base_price) / (
            base_price * (shift ** 2)
        )

        # DV01 (per 1bp)
        dv01_cr = abs(mod_duration * base_price * 0.0001)

        # Example: 2y and 5y key rate durations
        krd_map = self.key_rate_durations(bond, spot_curve, key_tenors=[2.0, 5.0, 10.0], settle_date=settle_date)

        return {
            "clean_price": base_price,
            "dirty_price": base["dirty_price"],
            "ytm": base["ytm"],
            "mod_duration": round(mod_duration, 6),
            "convexity": round(convexity, 6),
            "dv01_cr": round(dv01_cr, 6),
            "krd_2y": krd_map.get(2.0, np.nan),
            "krd_5y": krd_map.get(5.0, np.nan),
            "krd_10y": krd_map.get(10.0, np.nan),

        }
    
    
    def key_rate_durations(
        self,
        bond: Bond,
        spot_curve: Dict[float, float],
        key_tenors: List[float],
        settle_date: str = "2025-12-26",
        shift: float = 0.0001,
    ) -> Dict[float, float]:
        """
        Compute key rate duration for selected curve tenors.
        For each key tenor:
          - bump ONLY that tenor up/down by 'shift'
          - reprice bond
          - apply KRD formula: (P- - P+) / (2 * Δy * P0)  [per CFA/industry convention]. [web:80][web:83]
        Returns:
          {tenor: krd_value}
        """
        # Base cashflows and price
        cfs = self.pricer.bond_cashflows(bond, settle_date)
        base_price = self.pricer.price_bond(cfs, spot_curve, settle_date)["clean_price"]

        krd = {}
        for k in key_tenors:
            if k not in spot_curve:
                # skip tenors not present in the curve
                continue

            # Build up/down curves where ONLY this tenor is shifted
            up_curve = {t: (r + shift if t == k else r) for t, r in spot_curve.items()}
            down_curve = {t: (r - shift if t == k else r) for t, r in spot_curve.items()}

            p_up = self.pricer.price_bond(cfs, up_curve, settle_date)["clean_price"]
            p_down = self.pricer.price_bond(cfs, down_curve, settle_date)["clean_price"]

            # Key rate duration (per 1 change in yield, using standard formula). [web:80][web:83]
            krd_k = (p_down - p_up) / (2 * shift * base_price)
            krd[k] = round(krd_k, 6)

        return krd
    
    def scenario_pnl_bond(
        self,
        bond: Bond,
        base_curve: Dict[float, float],
        scenarios: Dict[str, Dict[float, float]],
        settle_date: str = "2025-12-26",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute price/P&L for a single bond under multiple curve scenarios.
        scenarios: {scenario_name: scenario_curve_dict}
        Returns:
          {scenario_name: {"price": ..., "abs_pnl": ..., "pct_pnl": ...}}
        """
        cfs = self.pricer.bond_cashflows(bond, settle_date)
        base_price = self.pricer.price_bond(cfs, base_curve, settle_date)["clean_price"]

        out = {}
        for name, scen_curve in scenarios.items():
            scen_price = self.pricer.price_bond(cfs, scen_curve, settle_date)["clean_price"]
            abs_pnl = scen_price - base_price
            pct_pnl = abs_pnl / base_price
            out[name] = {
                "price": round(scen_price, 6),
                "abs_pnl": round(abs_pnl, 6),
                "pct_pnl": round(pct_pnl, 6),
            }
        return out

    def scenario_pnl_portfolio(
        self,
        portfolio: List[Bond],
        base_curve: Dict[float, float],
        scenarios: Dict[str, Dict[float, float]],
        settle_date: str = "2025-12-26",
    ) -> pd.DataFrame:
        """
        Aggregate scenario P&L at portfolio level.
        Returns DataFrame with columns: scenario, port_price, abs_pnl_cr, pct_pnl
        """
        rows = []

        for name, scen_curve in scenarios.items():
            port_base_mv = 0.0
            port_scen_mv = 0.0

            for b in portfolio:
                cfs = self.pricer.bond_cashflows(b, settle_date)
                base_price = self.pricer.price_bond(cfs, base_curve, settle_date)["clean_price"]
                scen_price = self.pricer.price_bond(cfs, scen_curve, settle_date)["clean_price"]

                # market values in Cr (price already in Cr terms)
                port_base_mv += base_price
                port_scen_mv += scen_price

            abs_pnl = port_scen_mv - port_base_mv
            pct_pnl = abs_pnl / port_base_mv

            rows.append(
                {
                    "scenario": name,
                    "port_base_mv_cr": round(port_base_mv, 6),
                    "port_scen_mv_cr": round(port_scen_mv, 6),
                    "abs_pnl_cr": round(abs_pnl, 6),
                    "pct_pnl": round(pct_pnl, 6),
                }
            )

        return pd.DataFrame(rows)






def build_scenarios(base_curve: Dict[float, float]) -> Dict[str, Dict[float, float]]:
    """
    Construct simple term-structure scenarios:
      - parallel +50bp / -50bp
      - bear steepener (long rates up more)
      - bull flattener (short up, long down)
    These are standard FI term-structure views. [web:85][web:89]
    """
    scen = {}

    # Parallel shifts
    scen["parallel_+50bp"] = {t: r + 0.005 for t, r in base_curve.items()}
    scen["parallel_-50bp"] = {t: r - 0.005 for t, r in base_curve.items()}

    # Bear steepener: long end +75bp, short end +25bp
    scen["bear_steepener"] = {
        t: (r + 0.0025 if t <= 2.0 else r + 0.0075) for t, r in base_curve.items()
    }

    # Bull flattener: short end +75bp, long end +25bp
    scen["bull_flattener"] = {
        t: (r + 0.0075 if t <= 2.0 else r + 0.0025) for t, r in base_curve.items()
    }

    return scen

def main():
    """
    Production workflow:
    1. Fetch live US/India curves (FRED + scrape)
    2. Load 10-bond portfolio from CSV
    3. Compute per-bond + portfolio risks (DV01, KRDs, scenarios)
    4. Export CSVs for Excel dashboard
    """
    os.makedirs("outputs", exist_ok=True)

    pricer = BondPricer()
    risk_engine = RiskEngine(pricer)


    # LIVE DATA: Fetch + write curves
    print("📡 Fetching live curves...")
    us_curve = fetch_us_treasury_curve_today()
    india_curve = build_india_gsec_curve()
    write_yield_curves_csv(us_curve, india_curve)
    print(f"✅ US 10Y: {us_curve[10]:.1%} | India 10Y: {india_curve[10]:.1%}")

    # Load US curve from fresh CSV
    base_curve = load_curve_from_csv(curve_name="US_TSY")


    # Portfolio (from CSV)
    portfolio = load_portfolio_from_csv(path="data/bonds_portfolio.csv")
    print(f"📂 Loaded portfolio with {len(portfolio)} bonds.")


    # Per-bond risk
    rows = []
    for b in portfolio:
        rp = risk_engine.full_risk_profile(b, base_curve)
        rp["bond_id"] = b.bond_id
        rp["face_amt_cr"] = b.face_amt_cr
        rp["coupon_rate"] = b.coupon_rate
        rp["maturity_years"] = b.maturity_years
        # market value in Cr (approx = clean price, since face is in Cr)
        rp["market_value_cr"] = rp["clean_price"]
        rows.append(rp)

    risk_df = pd.DataFrame(rows)

    # ---- portfolio-level summary ----
    total_mv = risk_df["market_value_cr"].sum()
    total_dv01 = risk_df["dv01_cr"].sum()
    port_dur = (risk_df["mod_duration"] * risk_df["market_value_cr"]).sum() / total_mv
    port_conv = (risk_df["convexity"] * risk_df["market_value_cr"]).sum() / total_mv

    # Portfolio key rate durations (2y and 5y)
    port_krd_2y = (risk_df["krd_2y"] * risk_df["market_value_cr"]).sum() / total_mv
    port_krd_5y = (risk_df["krd_5y"] * risk_df["market_value_cr"]).sum() / total_mv
    port_krd_10y = (risk_df["krd_10y"] * risk_df["market_value_cr"]).sum() / total_mv

    risk_df["dv01_pct"] = risk_df["dv01_cr"] / total_dv01 

    # ----- Scenario analysis -----
    scenarios = build_scenarios(base_curve)
    scen_df = risk_engine.scenario_pnl_portfolio(portfolio, base_curve, scenarios)

    print("\n📉 Scenario P&L (portfolio):")
    print(scen_df)

    # ----- Portfolio summary for Excel (1 row) -----
    portfolio_summary = pd.DataFrame([
        {
            "total_mv_cr": total_mv,
            "port_duration": port_dur,
            "port_dv01_cr": total_dv01,
            "port_convexity": port_conv,
            "port_krd_2y": port_krd_2y,
            "port_krd_5y": port_krd_5y,
            "port_krd_10y": port_krd_10y,
            "as_of_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        }
    ])

    portfolio_summary.to_csv("outputs/portfolio_summary.csv", index=False)

    # Print summary
    print(f"\n✅ Portfolio MV: {total_mv:.2f} Cr")
    print(f"✅ Portfolio DV01: -{total_dv01:.4f} Cr per 1bp")
    print(f"✅ Portfolio Duration: {port_dur:.4f} years")
    print(f"✅ Portfolio Convexity: {port_conv:.4f}")
    print(f"✅ Portfolio KRD 2Y: {port_krd_2y:.4f}")
    print(f"✅ Portfolio KRD 5Y: {port_krd_5y:.4f}")
    print(f"✅ Portfolio KRD 10Y: {port_krd_10y:.4f}")


    print("\n✅ Per-bond risks computed for Excel")


    # Export
    risk_df.to_csv("outputs/risk_raw.csv", index=False)
    scen_df.to_csv("outputs/scenario_raw.csv", index=False)

    pd.DataFrame(
        [
            {
                "bond_id": b.bond_id,
                "face_amt_cr": b.face_amt_cr,
                "coupon_rate": b.coupon_rate,
                "maturity_years": b.maturity_years,
                "freq": b.freq,
            }
            for b in portfolio
        ]
    ).to_csv("outputs/portfolio_raw.csv", index=False)

    print("\n💾 Saved: outputs/risk_raw.csv and outputs/portfolio_raw.csv")
    return risk_df
