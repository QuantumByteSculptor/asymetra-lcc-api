# collect_universe.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def make_universe(seed: int = 42) -> List[Dict[str, str]]:
    # Convention:
    # - market: US / EU / UK / JP / CA / AU / CN / HK / IN / BR / MX / KR / SA / ZA / GLOBAL / G10 / EM
    # - asset_type: equity / etf / fx / commodity / crypto / rate
    #
    # NOTE: Yahoo tickers are messy for some exchanges. This list focuses on high hit-rate tickers.

    # -----------------------------
    # EQUITIES — US (large & liquid)
    # -----------------------------
    us_equity = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V","MA","UNH","XOM","AVGO","LLY",
        "PG","KO","PEP","COST","WMT","HD","MRK","ABBV","CVX","BAC","ORCL","CRM","ADBE","NFLX","AMD","INTC",
        "CSCO","TMO","ACN","MCD","NKE","DIS","LIN","WFC","PM","QCOM","INTU","TXN","AMAT","IBM","GE","CAT",
        "BA","NOW","SPGI","GS","MS","BLK","C","LOW","SBUX","DE","MDT","ISRG","ELV","VRTX","GILD","PFE",
        "UPS","HON","LMT","RTX","SCHW","ZTS","BKNG","ABNB","SNOW","PLTR","UBER","SHOP","SQ","PYPL","FDX",
    ]

    # -----------------------------
    # EQUITIES — EU (mix Germany/France/Netherlands/Spain/Italy)
    # Use exchange suffixes: .DE .PA .AS .MC .MI
    # -----------------------------
    eu_equity = [
        "SAP.DE","SIE.DE","DTE.DE","ALV.DE","BAS.DE","BAYN.DE","ADS.DE","BMW.DE","MBG.DE","VOW3.DE","RWE.DE",
        "AIR.PA","MC.PA","OR.PA","TTE.PA","SAN.PA","BNP.PA","DG.PA","CAP.PA","ENGI.PA","CS.PA","RI.PA",
        "ASML.AS","UNA.AS","INGA.AS","ADYEN.AS","PHIA.AS","DSM.AS",
        "IBE.MC","SAN.MC","ITX.MC","BBVA.MC",
        "ENEL.MI","ENI.MI","ISP.MI","STLAM.MI",
    ]

    # -----------------------------
    # EQUITIES — UK (.L)
    # -----------------------------
    uk_equity = [
        "AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GLEN.L","GSK.L","BATS.L","DGE.L","SHEL.L","VOD.L","BARC.L",
        "LSEG.L","PRU.L","NG.L","REL.L","AAL.L","BA.L",
    ]

    # -----------------------------
    # EQUITIES — Japan (.T)
    # -----------------------------
    jp_equity = [
        "7203.T","6758.T","9984.T","8306.T","9432.T","6861.T","8316.T","7974.T","8035.T","7267.T","4568.T",
        "6501.T","8058.T","8766.T","8411.T",
    ]

    # -----------------------------
    # EQUITIES — Canada (.TO)
    # -----------------------------
    ca_equity = [
        "RY.TO","TD.TO","BNS.TO","BMO.TO","ENB.TO","CNQ.TO","SU.TO","TRP.TO","CNR.TO","CP.TO",
    ]

    # -----------------------------
    # EQUITIES — Australia (.AX)
    # -----------------------------
    au_equity = [
        "BHP.AX","CBA.AX","RIO.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WOW.AX",
    ]

    # -----------------------------
    # EQUITIES — Emerging Markets (high hit-rate ADRs + a few local)
    # -----------------------------
    em_equity = [
        # China ADRs
        "BABA","JD","PDD","BIDU","NIO","LI","XPEV",
        # India (ADRs / US-listed)
        "INFY","WIT","HDB","IBN",
        # Brazil
        "VALE","PBR","ITUB","NU",
        # Mexico
        "FMX","BIMBOA.MX",  # local might fail, but ok to try
        # Korea ADRs
        "TSM",  # Taiwan (not KR, but Asia EM proxy)
    ]

    # -----------------------------
    # ETFs — Indices / Regions / Factors
    # -----------------------------
    etfs = [
        # US broad / style
        "SPY","IVV","VOO","QQQ","IWM","DIA","VTI","SCHB","RSP",
        # Volatility / defensive
        "USMV","SPLV","VTV","VUG","MTUM","QUAL",
        # Rates / bonds
        "IEF","TLT","SHY","LQD","HYG","TIP",
        # Regions
        "VEA","VWO","EWJ","EWU","FEZ","EEM","EFA","FXI","INDA","EWZ","EWW",
        # Sector SPDRs
        "XLF","XLK","XLE","XLY","XLP","XLV","XLI","XLB","XLU","XLC",
        # Gold / commodities proxies
        "GLD","SLV","USO","UNG","DBA","DBC",
    ]

    # -----------------------------
    # FX — G10 + a few EM
    # Yahoo format: "EURUSD=X"
    # -----------------------------
    fx = [
        "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","USDCAD=X","USDCHF=X","EURJPY=X","EURGBP=X",
        "USDMXN=X","USDZAR=X","USDBRL=X","USDINR=X",
    ]

    # -----------------------------
    # Commodities — futures & proxies (Yahoo coverage varies)
    # Use ETFs proxies as reliable baseline; futures tickers sometimes work:
    # Gold: GC=F, Oil: CL=F, NatGas: NG=F, Copper: HG=F, Corn: ZC=F, Wheat: ZW=F
    # -----------------------------
    commodities = [
        # Futures (often OK)
        "GC=F","SI=F","CL=F","NG=F","HG=F","ZC=F","ZW=F","ZS=F",
        # Proxies ETFs (more stable)
        "GLD","SLV","USO","UNG","DBA","DBC",
    ]

    # -----------------------------
    # Crypto — Yahoo tickers
    # -----------------------------
    crypto = [
        "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD",
    ]

    # -----------------------------
    # Rates / Macro proxies (Yahoo series)
    # ^TNX = US 10Y yield index, ^IRX = 13-week, ^FVX = 5Y, ^TYX = 30Y
    # -----------------------------
    rates = ["^TNX","^IRX","^FVX","^TYX"]

    # ---- Build dicts
    out: List[Dict[str, str]] = []
    def add_many(tickers: List[str], asset_type: str, market: str):
        for t in tickers:
            out.append({"ticker": t, "asset_type": asset_type, "market": market})

    add_many(us_equity, "equity", "US")
    add_many(eu_equity, "equity", "EU")
    add_many(uk_equity, "equity", "UK")
    add_many(jp_equity, "equity", "JP")
    add_many(ca_equity, "equity", "CA")
    add_many(au_equity, "equity", "AU")
    add_many(em_equity, "equity", "EM")

    add_many(etfs, "etf", "GLOBAL")
    add_many(fx, "fx", "GLOBAL")
    add_many(commodities, "commodity", "GLOBAL")
    add_many(crypto, "crypto", "GLOBAL")
    add_many(rates, "rate", "US")

    # De-dup by ticker (keep first occurrence)
    seen = set()
    dedup = []
    for r in out:
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            dedup.append(r)

    # Shuffle (stable) so later sampling is fair
    random.Random(seed).shuffle(dedup)
    return dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/universe.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max", type=int, default=0, help="Optional cap for sampling (0 = no cap)")
    args = ap.parse_args()

    uni = make_universe(seed=args.seed)

    if args.max and args.max > 0 and args.max < len(uni):
        uni = uni[: args.max]

    Path("data").mkdir(exist_ok=True)
    out = Path(args.out)
    out.write_text(json.dumps(uni, indent=2), encoding="utf-8")
    print(f"✅ wrote {out} ({len(uni)} tickers)")


if __name__ == "__main__":
    main()