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
        # US mid-cap / growth (S&P 400 / Russell 2000 proxies, high data availability)
        "NET","CRWD","DDOG","ZS","OKTA","FTNT","PANW","CYBR",
        "MDB","ESTC","ELASTIC","DSGX","GTLB","BILL","ZI",
        "ROKU","RBLX","HOOD","COIN","AFRM","SOFI","UPST","LC",
        "TTD","APPS","MGNI","PUBM","CRTO",
        "LYFT","GRAB","SE","MELI","NU","STNE",
        "CELH","HIMS","TMDX","IRTC","AXNX","INSP","SWAV","PRCT",
        "EXAS","NTRA","GH","VCYT","RXRX",
        "AFRM","OPEN","OPENDOOR","OFFERPAD",
        "SMAR","FROG","DOCN","BIGC","CLOU","WCLD",
        "WING","LULU","DECK","CROX","ONON","SKX",
        "HCA","MOH","CNC","CI","HUM",
        "AMT","CCI","SBAC","DLR","EQIX",
        "AVB","EQR","MAA","NLY","AGNC",
        "LEN","PHM","DHI","TOL","NVR",
        "CLF","NUE","STLD","RS","CMC",
        "MPC","VLO","PSX","HES","DVN","FANG",
        "CEG","VST","NRG","AES","PCG",
        "WDAY","VEEV","HUBS","BRZE","SPRK",
        "GLBE","RAMP","KRTX","TGTX","BCAB",
        "WOLF","ON","MCHP","MPWR","SWKS","QRVO",
        "ACLS","ENTG","MKSI","FORM","COHU","ACMR",
        "NVST","RVTY","A","KEYS","TRMB",
        "PODD","DXCM","ABMD","LMAT","NUVA",
        "MTCH","PINS","SNAP","RDDT","BMBL",
        "ZG","Z","RDFN","EXPI","OPEN",
        "ETSY","CHWY","W","PRTS","OLO",
        "HLT","MAR","H","WYNDM","STAY",
        "DAL","UAL","AAL","LUV","ALK","HA",
        "CCL","RCL","NCLH",
        "CAR","HTZ","AVIS",
        # More US growth / tech
        "TWLO","ZM","DOCU","BOX","DROPBOX","DBX","PCTY","PAYC","SAIC","LDOS",
        "CACI","MAXN","NOVA","ARRY","ENPH","SEDG","FSLR","SPWR",
        "PLUG","FCEL","BLOOM","BLDP","CLNE",
        "AMRC","HASI","BEPC","AY","CWEN","NEP",
        "IRM","CUBE","EXR","NSA","REXR","TRNO",
        "IIPR","SAFE","GMRE","PLYM",
        "GLPI","VICI","MGM","WYNN","LVS","CZR","PENN",
        "DKNG","FLUT","RSI","BALY",
        "CPRI","TPR","RL","PVH","VFC","HBI","SKX",
        "EL","COTY","REYN","CHD","CLX","HRL","K","GIS","CPB","CAG","SJM",
        "MKC","FLO","THRM","INGR","WH","TAST","CMG","DPZ","BLMN",
        "TXRH","JACK","EAT","DRI","CAKE","SHAK","RRGB",
        "VLKAF","EADSF","RLLCF","RYCEY","BAESF",
        # More biotech/pharma
        "MRNA","BNTX","NVAX","REGN","BIIB","ALNY","BMRN","RARE","UTHR",
        "HALO","ACAD","SAGE","SGEN","IMVT","KYMR","ARQT","DAWN",
        "CRSP","BEAM","EDIT","NTLA","BLUE","SGMO","TALS",
        # More semiconductors
        "LRCX","KLAC","ASML","SNPS","CDNS","MRVL","SLAB","SITM","CRUS",
        "DIOD","RMBS","AMBA","SMTC","AOSL","POWI",
        # Industrial / Materials
        "EMR","ETN","ROK","IR","GGG","FTV","GNRC","WTSLA",
        "APD","ECL","PPG","SHW","VMC","MLM","FAST","GWW","MSC",
        "ODFL","SAIA","XPO","CHRW","EXPD","JBHT","WERN",
        # Financial
        "KKR","APO","BX","CG","ARES","BLUE","OWL",
        "MSTR","ICE","CME","CBOE","NDAQ","LPLA","RJF","EVR","LAZ","HLI",
        "ALLY","SLM","CACC","OMF","EFC","NAVI",
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
        # Extra EU
        "ABI.BR","UCB.BR","SOLB.BR","ACKB.BR",
        "NOVN.SW","NESN.SW","ROG.SW","ZURN.SW","CFR.SW",
        "NOKIA.HE","STERV.HE",
        "VOLV-B.ST","ERICB.ST","SEB-A.ST","HM-B.ST","SAND.ST",
        "NDA-FI.HE","NESTE.HE",
        "OMV.VI","VIG.VI",
        "CRH.L","KGF.L","SVT.L","SSE.L","IHG.L",
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
        "BABA","JD","PDD","BIDU","NIO","LI","XPEV","DIDI","LABD",
        # India (ADRs / US-listed)
        "INFY","WIT","HDB","IBN","VEDL","SIFY","REDFY",
        # Brazil
        "VALE","PBR","ITUB","NU","BRFS","GGBR","EMBR","BBAS",
        # Mexico
        "FMX","BIMBOA.MX",
        # Korea / Taiwan ADRs
        "TSM","ASX","UMC","HIMX",
        # South Africa
        "MTN","NPN","SHP",  # local tickers — ok to try
        # Southeast Asia
        "SEA","GRAB","GCT",
        # Middle East (US-listed proxies)
        "SAUDIF","ARAMCO",
        # South Korea local
        "005930.KS","000660.KS","035420.KS","051910.KS",
        # Hong Kong
        "0700.HK","9988.HK","3690.HK","0941.HK","2318.HK","0005.HK",
    ]

    # -----------------------------
    # ETFs — Indices / Regions / Factors
    # -----------------------------
    etfs = [
        # US broad / style
        "SPY","IVV","VOO","QQQ","IWM","DIA","VTI","SCHB","RSP",
        "MDY","IJH","IJR","VB","VO",
        # Factor / style
        "USMV","SPLV","VTV","VUG","MTUM","QUAL","VLUE","SIZE",
        "MOAT","COWZ","CALF","DIVO","JEPI","JEPQ",
        # Rates / bonds (full spectrum)
        "IEF","TLT","SHY","LQD","HYG","TIP","BND","AGG","VCIT","VCSH",
        "BSV","BIV","BLV","BKLN","FLOT","NEAR","SHV",
        "EMB","PCY","VWOB","IAGG",
        # Regions
        "VEA","VWO","EWJ","EWU","FEZ","EEM","EFA","FXI","INDA","EWZ","EWW",
        "EWG","EWQ","EWI","EWP","EWN","EWL","EWD","EWK","NORW",
        "EWY","EWT","EIDO","THD","EWS","EWM","EWA","EWC","EWH",
        # Sector SPDRs + alternatives
        "XLF","XLK","XLE","XLY","XLP","XLV","XLI","XLB","XLU","XLC",
        "XLRE","XME","XOP","XBI","IBB","ARKG","ARKK","ARKF","ARKQ","ARKW",
        "SOXX","SMH","HACK","BUG","CIBR",
        "ICLN","QCLN","TAN","FAN","ACES",
        "BOTZ","ROBO","IRBO","AIQ",
        "HERO","ESPO","NERD",
        # Gold / commodities proxies
        "GLD","SLV","USO","UNG","DBA","DBC","PDBC",
        "GLDM","AAAU","IAU","SGOL",
        # Multi-asset / alternatives
        "VNQ","VNQI","RWX","REM","MORT",
        "BITO","GBTC","ETHE",
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