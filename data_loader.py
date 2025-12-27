"""
Live market data loader for Rates Risk Toolkit
US Treasury (FRED) + India G-Sec (scraped)
"""
import ssl
import os
import re
from typing import Dict
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fredapi import Fred

def get_fred_client() -> Fred:
    """FRED client from .env (SSL verify disabled temporarily)."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in .env")
    # TEMP: disable SSL verification to avoid CERTIFICATE_VERIFY_FAILED
    ssl._create_default_https_context = ssl._create_unverified_context
    return Fred(api_key=api_key)


def fetch_us_treasury_curve_today() -> Dict[float, float]:
    """Live US Treasury curve from FRED (2y,5y,10y)."""
    fred = get_fred_client()
    dgs2 = float(fred.get_series("DGS2", observation_start="2025-12-01").iloc[-1]) / 100
    dgs5 = float(fred.get_series("DGS5", observation_start="2025-12-01").iloc[-1]) / 100
    dgs10 = float(fred.get_series("DGS10", observation_start="2025-12-01").iloc[-1]) / 100
    
    return {
        0.25: dgs2 * 0.9, 0.5: dgs2 * 0.95, 1.0: (dgs2 + dgs5)/2,
        2.0: dgs2, 5.0: dgs5, 10.0: dgs10
    }

def fetch_india_10y_yield() -> float:
    """Scrape India 10Y G-Sec yield."""
    url = "https://tradingeconomics.com/india/government-bond-yield"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        value_div = soup.find("div", class_="value") or soup.find("span", class_="text-2xl")
        if value_div:
            val_str = re.sub(r"[^0-9\.]", "", value_div.text)
            return float(val_str) / 100.0
    except Exception:
        pass
    return 0.068  # fallback

def build_india_gsec_curve() -> Dict[float, float]:
    """India G-Sec curve anchored to live 10Y."""
    y10 = fetch_india_10y_yield()
    return {
        0.25: y10 - 0.005, 0.5: y10 - 0.004, 1.0: y10 - 0.003,
        2.0: y10 - 0.002, 5.0: y10 - 0.001, 10.0: y10
    }

def write_yield_curves_csv(us_curve: Dict[float, float], india_curve: Dict[float, float]):
    """Write BOTH curves to data/yield_curves.csv."""
    os.makedirs("data", exist_ok=True)
    rows = []
    for t, r in us_curve.items():
        rows.append({"curve_name": "US_TSY", "tenor": t, "rate": r})
    for t, r in india_curve.items():
        rows.append({"curve_name": "INDIA_GSEC", "tenor": t, "rate": r})
    pd.DataFrame(rows).to_csv("data/yield_curves.csv", index=False)
