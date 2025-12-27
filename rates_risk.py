"""
Rates Risk Toolkit v2.0 - Production Grade
Jaymin Patil | Fixed Income Analyst Portfolio
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pandas.tseries.offsets import DateOffset
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Bond:
    """Immutable bond definition"""
    bond_id: str
    face_amt_cr: float
    coupon_rate: float
    maturity_years: float
    freq: int = 2

class BondPricer:
    """Production bond pricing engine with validation"""
    
    def __init__(self, day_count: str = 'act/365'):
        self.day_count = day_count
        self.day_count_fraction = 365.25
        
    def validate_inputs(self, face_amt: float, coupon_rate: float, maturity_years: float):
        """Input validation (industry standard)"""
        if face_amt <= 0:
            raise ValueError("Face amount must be positive")
        if coupon_rate < 0 or coupon_rate > 0.2:
            raise ValueError("Coupon rate unrealistic")
        if maturity_years <= 0 or maturity_years > 30:
            raise ValueError("Maturity between 0-30 years")
    
    def bond_cashflows(self, bond: Bond, settle_date: str = "2025-12-26") -> pd.DataFrame:
        """Enhanced cashflow generator with proper conventions"""
        self.validate_inputs(bond.face_amt_cr, bond.coupon_rate, bond.maturity_years)
        
        settle = pd.to_datetime(settle_date)
        maturity = settle + DateOffset(years=int(bond.maturity_years))
        
        # Backward generation with exact semi-annual periods
        periods = int(bond.maturity_years * bond.freq)
        payment_dates = pd.date_range(end=maturity, periods=periods+1, freq=f'{365.25//bond.freq}D')[1:]
        
        coupon_per_period = (bond.coupon_rate / bond.freq) * bond.face_amt_cr
        
        cashflows = []
        for i, date in enumerate(payment_dates):
            days_to_payment = (date - settle).days
            coupon = coupon_per_period
            principal = bond.face_amt_cr if i == len(payment_dates)-1 else 0
            cashflows.append({
                'payment_date': date.strftime('%Y-%m-%d'),
                'days_to_payment': days_to_payment,
                'time_fraction': days_to_payment / self.day_count_fraction,
                'coupon_payment': round(coupon, 2),
                'principal_payment': principal,
                'total_cf': coupon + principal
            })
        
        return pd.DataFrame(cashflows).round(4)
    
    def discount_factors(self, spot_curve: Dict[float, float], target_times: np.ndarray, 
                        interp_method: str = 'cubic') -> np.ndarray:
        """Industry-grade interpolation (cubic spline default)"""
        tenors = np.array(sorted(spot_curve.keys()))
        spot_rates = np.array([spot_curve[t] for t in tenors])
        
        if interp_method == 'cubic':
            interpolator = CubicSpline(tenors, spot_rates, extrapolate=True)
            interpolated_spots = interpolator(target_times)
        else:
            interpolator = interp1d(tenors, spot_rates, kind='linear', fill_value='extrapolate')
            interpolated_spots = interpolator(target_times)
        
        # Continuous compounding discount factors
        return np.exp(-interpolated_spots * target_times)
    
    def price_bond(self, cashflows_df: pd.DataFrame, spot_curve: Dict[float, float],
                settle_date: str) -> Dict[str, float]:
        """
        Price bond via discounted cashflows (clean + simple, no face_amt_cr needed)
        """
        # Ensure days_to_payment and time_fraction exist
        settle = pd.to_datetime(settle_date)
        if 'days_to_payment' not in cashflows_df.columns:
            cashflows_df['days_to_payment'] = (
                pd.to_datetime(cashflows_df['payment_date']) - settle
            ).dt.days

        cashflows_df['time_fraction'] = cashflows_df['days_to_payment'] / self.day_count_fraction

        # Get discount factors for each cashflow
        dfactors = self.discount_factors(spot_curve, cashflows_df['time_fraction'].values)

        # Clean price = sum of discounted total cashflows
        clean_price = float(np.sum(cashflows_df['total_cf'].values * dfactors))

        # --- simple accrued interest approximation ---
        # assume regular coupon periods and use first/last payment dates
        payment_dates = pd.to_datetime(cashflows_df['payment_date'])
        last_coupon_date = payment_dates.iloc[0]
        next_coupon_date = payment_dates.iloc[1] if len(payment_dates) > 1 else payment_dates.iloc[0]

        days_between_coupons = (next_coupon_date - last_coupon_date).days
        days_since_last_coupon = (settle - last_coupon_date).days
        days_since_last_coupon = max(0, min(days_since_last_coupon, days_between_coupons))

        coupon_amount = float(cashflows_df['coupon_payment'].iloc[0])
        accrued = coupon_amount * days_since_last_coupon / days_between_coupons

        dirty_price = clean_price + accrued

        return {
            "clean_price": round(clean_price, 4),
            "dirty_price": round(dirty_price, 4),
            "accrued_interest": round(accrued, 4),
            "ytm": 0.0725  # placeholder
        }
    



class RiskEngine:
    """Advanced risk calculator with scenarios"""
    
    def __init__(self, pricer: BondPricer):
        self.pricer = pricer
        
    def full_risk_profile(self, bond: Bond, spot_curve: Dict[float, float], 
                        settle_date: str = "2025-12-26") -> Dict[str, float]:
        """Complete risk profile for one bond"""
        cfs = self.pricer.bond_cashflows(bond, settle_date)
        base_price = self.pricer.price_bond(cfs, spot_curve, settle_date)['clean_price']
        
        # Duration via finite difference (1bp bumps)
        up_curve = {k: v + 0.0001 for k, v in spot_curve.items()}
        down_curve = {k: v - 0.0001 for k, v in spot_curve.items()}
        
        up_price = self.pricer.price_bond(cfs, up_curve, settle_date)['clean_price']
        down_price = self.pricer.price_bond(cfs, down_curve, settle_date)['clean_price']
        
        mod_duration = -(up_price - down_price) / (2 * base_price * 0.0001)
        dv01_cr = abs(mod_duration * base_price * 0.0001)  # Per 1bp
        
        return {
            'clean_price': base_price,
            'mod_duration': round(mod_duration, 4),
            'dv01_cr': round(dv01_cr, 4),
            'convexity': 0.0  # TODO: Add later
        }

def load_portfolio() -> List[Bond]:
    """Production portfolio loader"""
    portfolio_data = [
        Bond('GSEC_7.75_2027', 250, 0.0775, 1.5),
        Bond('GSEC_7.18_2033', 180, 0.0718, 7.5),
        Bond('IND_8.15_2028', 120, 0.0815, 2.8),
        Bond('IND_8.15_2028', 150, 0.0820, 2.9),
        Bond('IND_8.15_2028', 180, 0.0825, 3.0),
        Bond('IND_8.15_2028', 100, 0.075, 5)
        # ... add rest
    ]
    return portfolio_data

def main():
    """Production main with proper error handling"""
    print("🚀 Rates Risk Toolkit v2.0 - Production Grade")
    
    # Ensure outputs folder
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize
    pricer = BondPricer()
    risk_engine = RiskEngine(pricer)
    
    # Market data (RBI G-Sec curve proxy)
    base_curve = {0.25: 0.0625, 0.5: 0.0650, 1: 0.0675, 2: 0.0700, 5: 0.0725, 10: 0.0730}
    
    # Portfolio
    portfolio = load_portfolio()  # Test with 3 bonds first
    
    # Risk report
    results = []
    for bond in portfolio:
        risk_profile = risk_engine.full_risk_profile(bond, base_curve)
        risk_profile['bond_id'] = bond.bond_id
        risk_profile['face_amt_cr'] = bond.face_amt_cr
        results.append(risk_profile)
    
    risk_df = pd.DataFrame(results)
    total_dv01 = risk_df['dv01_cr'].sum()
    
    print(f"\n✅ Portfolio DV01: -{total_dv01:.2f}Cr per 1bp")
    print("\n📊 Risk Report:")
    print(risk_df.round(4))
    
    # --- NEW: convert portfolio list → DataFrame before saving ---
    portfolio_df = pd.DataFrame(
        [{
            "bond_id": b.bond_id,
            "face_amt_cr": b.face_amt_cr,
            "coupon_rate": b.coupon_rate,
            "maturity_years": b.maturity_years,
            "freq": b.freq
        } for b in portfolio]
    )

    # Export

    risk_df.to_csv("outputs/risk_raw.csv", index=False)
    portfolio_df.to_csv("outputs/portfolio_raw.csv", index=False)

    print("💾 Saved: outputs/risk_raw.csv and outputs/portfolio_raw.csv")
    return risk_df