import os
import pandas as pd
from rates_risk import (
    BondPricer, RiskEngine, load_portfolio_from_csv, 
    load_curve_from_csv, build_scenarios, 
    fetch_us_treasury_curve_today, build_india_gsec_curve, 
    write_yield_curves_csv
)

def main():
    """
    Production workflow:
    1. Fetch live US/India curves (FRED + scrape)
    2. Load 10-bond portfolio from CSV
    3. Compute per-bond + portfolio risks (DV01, KRDs, scenarios)
    4. Export CSVs for Excel dashboard
    """

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    pricer = BondPricer()
    risk_engine = RiskEngine(pricer)

    # LIVE DATA: Fetch + write curves
    print("Fetching live curves...")
    us_curve = fetch_us_treasury_curve_today()
    india_curve = build_india_gsec_curve()
    write_yield_curves_csv(us_curve, india_curve)
    print(f"US 10Y: {us_curve[10]:.1%} | India 10Y: {india_curve[10]:.1%}")

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
        rp["market_value_cr"] = rp["clean_price"]
        rows.append(rp)

    risk_df = pd.DataFrame(rows)

    # Portfolio-level summary
    total_mv = risk_df["market_value_cr"].sum()
    total_dv01 = risk_df["dv01_cr"].sum()
    port_dur = (risk_df["mod_duration"] * risk_df["market_value_cr"]).sum() / total_mv
    port_conv = (risk_df["convexity"] * risk_df["market_value_cr"]).sum() / total_mv
    port_krd_2y = (risk_df["krd_2y"] * risk_df["market_value_cr"]).sum() / total_mv
    port_krd_5y = (risk_df["krd_5y"] * risk_df["market_value_cr"]).sum() / total_mv
    port_krd_10y = (risk_df["krd_10y"] * risk_df["market_value_cr"]).sum() / total_mv

    risk_df["dv01_pct"] = risk_df["dv01_cr"] / total_dv01 

    # Scenario analysis
    scenarios = build_scenarios(base_curve)
    scen_df = risk_engine.scenario_pnl_portfolio(portfolio, base_curve, scenarios)

    print("\n Scenario P&L (portfolio):")
    print(scen_df)

    # Portfolio summary for Excel
    portfolio_summary = pd.DataFrame([{
        "total_mv": total_mv,
        "port_duration": port_dur,
        "port_dv01": total_dv01,
        "port_convexity": port_conv,
        "port_krd_2y": port_krd_2y,
        "port_krd_5y": port_krd_5y,
        "port_krd_10y": port_krd_10y,
        "as_of_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }])
    portfolio_summary.to_csv("outputs/portfolio_summary.csv", index=False)

    # Print summary
    print(f"\n Portfolio MV: {total_mv:.2f} Cr")
    print(f" Portfolio DV01: -{total_dv01:.4f} Cr per 1bp")
    print(f" Portfolio Duration: {port_dur:.4f} years")
    print(f" Portfolio Convexity: {port_conv:.4f}")
    print(f" Portfolio KRD 2Y: {port_krd_2y:.4f}")
    print(f" Portfolio KRD 5Y: {port_krd_5y:.4f}")
    print(f" Portfolio KRD 10Y: {port_krd_10y:.4f}")

    print("\n Per-bond risks computed for Excel")

    # Export
    risk_df.to_csv("outputs/risk_raw.csv", index=False)
    scen_df.to_csv("outputs/scenario_raw.csv", index=False)
    
    pd.DataFrame([{
        "bond_id": b.bond_id,
        "face_amt_cr": b.face_amt_cr,
        "coupon_rate": b.coupon_rate,
        "maturity_years": b.maturity_years,
        "freq": b.freq,
    } for b in portfolio]).to_csv("outputs/portfolio_raw.csv", index=False)

    print("\n Saved: outputs/risk_raw.csv and outputs/portfolio_raw.csv")
    return risk_df

if __name__ == "__main__":
    main()