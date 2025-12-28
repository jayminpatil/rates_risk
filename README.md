# Rates Risk Toolkit 
**Live Fixed Income Portfolio Risk Analytics**  
*US Treasuries + India G-Secs | FRED API + Excel Dashboard*

[![Dashboard](https://via.placeholder.com/800x400/007ACC/FFFFFF?text=Live+Rates+Risk+Dashboard)](https://github.com/jayminpatil/rates_risk)

**Production-grade toolkit** for FI portfolio risk management. Fetches live curves, prices 10-bond portfolio (US+India), computes DV01/KRDs/scenarios, exports to Excel dashboard.

## ✨ **Live Features**
- ✅ **Live US Treasury curve** (FRED API: 2Y/5Y/10Y)
- ✅ **Live India 10Y G-Sec** (scraped TradingEconomics)
- ✅ **10-bond portfolio** (5 US + 5 India G-Secs)
- ✅ **Full risk metrics**: DV01, Duration, Convexity, KRDs (2Y/5Y/10Y)
- ✅ **Scenario analysis**: ±50bp parallel, bear steepener, bull flattener
- ✅ **Excel dashboard** auto-refreshes (Power Query + conditional formatting)

## 🚀 **Quick Start** (2 minutes)
Clone & install
git clone https://github.com/jayminpatil/rates_risk.git
cd rates_risk
pip install -r requirements.txt

Get FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
echo "FRED_API_KEY=your_key_here" > .env

Run live analysis
python main.py

**Excel dashboard**: Open `rates_dashboard.xlsx` → `Data → Refresh All`

## 📊 **Outputs Generated**

outputs/
├── risk_raw.csv (10 bonds × risk metrics)
├── scenario_raw.csv (4 scenarios × portfolio P&L)
├── portfolio_summary.csv (portfolio totals for Excel)
└── portfolio_raw.csv (bond definitions)

**Excel Summary Sheet** shows:
PORTFOLIO METRICS | PER-BOND RISK | SCENARIO P&L + Chart
Total MV: 158.23 Cr | Bond ID... | +50bp: -3.93 Cr
Duration: 5.23y | DV01 Cr... | Bear steep: -4.59 Cr
DV01: -1.62 Cr/bp | DV01%... | Chart with data labels

## 🏗️ **Architecture**

main.py → entrypoint
rates_risk.py → BondPricer + RiskEngine
data_loader.py → FRED API + India scrape
data/ → bonds_portfolio.csv (edit your bonds)

**Pricing**: Discounted cashflows on interpolated spot curve  
**Risks**: Finite difference (1bp bumps)  
**Scenarios**: Standard FI term structure views

## 📈 **Sample Output** (28-Dec-2025)

✅ US 10Y: 4.2% | India 10Y: 6.8%
✅ Portfolio MV: 158.23 Cr
✅ Portfolio DV01: -1.6234 Cr per 1bp
✅ Portfolio Duration: 5.2341 years
✅ Worst scenario: Bear steepener (-4.59 Cr)
✅ US 10Y: 4.2% | India 10Y: 6.8%
✅ Portfolio MV: 158.23 Cr
✅ Portfolio DV01: -1.6234 Cr per 1bp
✅ Portfolio Duration: 5.2341 years
✅ Worst scenario: Bear steepener (-4.59 Cr)

## 🔄 **Production Workflow**

Cron every 5min (add to crontab -e)
*/5 * * * * cd /path/to/rates_risk && /path/to/python main.py

Excel: Data → Refresh All (updates dashboard instantly)

## 🛠️ **Customization**

**Edit portfolio**: `data/bonds_portfolio.csv`

bond_id,face_amt_cr,coupon_rate,maturity_years,freq
UST2Y,10.0,0.042,2.0,2
IND10Y,20.0,0.070,10.0,2


**Add scenarios**: Edit `build_scenarios()` in `rates_risk.py`

## 📋 **Tech Stack**

Python 3.10+ | pandas | scipy | FRED API | Power Query
8 deps only (requirements.txt)

## 👨‍💼 **Built By**
**Jaymin Patil** | CFA L2 Candidate| [LinkedIn](https://linkedin.com/in/jayminpatil) | Dec 2025

---


