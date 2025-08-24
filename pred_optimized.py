import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import warnings, logging, os
from tabulate import tabulate

# -----------------------
# Suppress warnings/logs
# -----------------------
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
os.environ['LIGHTGBM_VERBOSITY'] = 'FATAL'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

days_array = [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
lookbacks = [30, 60, 90]

def prepare_dataset(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)-1):
        X.append(data[i-lookback:i])
        y.append(1 if data[i+1] > data[i] else 0)
    X = np.array(X).reshape(len(X), lookback)
    y = np.array(y)
    return X, y

def evaluate_best_model(symbol):
    try:
        df = yf.download(symbol, period='10y', interval='1d')
        if df.empty:
            print(f"⚠️ No data for {symbol}, skipping.")
            return None
        data_full = df["Close"].values
        dates_full = df.index
    except Exception as e:
        print(f"❌ Failed download for {symbol}: {e}")
        return None

    best_result = None
    for days in days_array:
        data = data_full[-days:]
        date_subset = dates_full[-days:]
        for lb in lookbacks:
            if lb >= len(data)-1:
                continue
            X, y = prepare_dataset(data, lb)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LGBMClassifier(verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            if (best_result is None) or (acc > best_result['accuracy']):
                best_result = {
                    'symbol': symbol,
                    'days': days,
                    'lookback': lb,
                    'accuracy': acc,
                    'model': model,
                    'X_last': X[-1],
                    'last_day_date': date_subset[-1],
                    'actual_last_dir': "UP" if data[-1] > data[-2] else "DOWN"
                }

    if best_result is None:
        print(f"⚠️ No valid model for {symbol}")
        return None

    # Next day prediction
    last_lb_days = data_full[-best_result['lookback']:]
    next_pred = best_result['model'].predict(last_lb_days.reshape(1, -1))[0]
    best_result['next_day_pred'] = "UP" if next_pred == 1 else "DOWN"

    # Backtest prediction
    backtest_pred = best_result['model'].predict(best_result['X_last'].reshape(1,-1))[0]
    best_result['backtest_pred'] = "UP" if backtest_pred == 1 else "DOWN"
    best_result['backtest_date'] = best_result['last_day_date'].strftime('%Y-%m-%d')

    return best_result

if __name__ == "__main__":
    # Only the symbols that previously failed
    symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'SBIN.NS', 'INFY.NS', 'HINDUNILVR.NS', 'LICI.NS', 'ITC.NS', 'SUNPHARMA.NS', 'AXISBANK.NS', 'NTPC.NS', 'ETERNAL.NS', 'ONGC.NS', 'ADANIPORTS.NS', 'BEL.NS', 'POWERGRID.NS', 'COALINDIA.NS', 'ADANIPOWER.NS', 'TATASTEEL.NS', 'HYUNDAI.NS', 'GRASIM.NS', 'DLF.NS', 'SBILIFE.NS', 'VBL.NS', 'VEDL.NS', 'HDFCLIFE.NS', 'HINDALCO.NS', 'ADANIGREEN.NS', 'AMBUJACEM.NS', 'BPCL.NS', 'PFC.NS', 'CIPLA.NS', 'GODREJCP.NS', 'LODHA.NS', 'BANKBARODA.NS', 'MAXHEALTH.NS', 'GAIL.NS', 'SHRIRAMFIN.NS', 'INDHOTEL.NS', 'MAZDOCK.NS', 'MUTHOOTFIN.NS', 'MANKIND.NS', 'TATACONSUM.NS', 'SWIGGY.NS', 'CGPOWER.NS', 'JINDALSTEL.NS', 'MOTHERSON.NS', 'CANBK.NS', 'RECLTD.NS', 'HAVELLS.NS', 'UNITDSPR.NS', 'ADANIENSOL.NS', 'ICICIGI.NS', 'MARICO.NS', 'GMRAIRPORT.NS', 'ICICIPRULI.NS', 'JSWENERGY.NS', 'INDIANB.NS', 'NAUKRI.NS', 'HINDPETRO.NS', 'SUZLON.NS', 'ASHOKLEY.NS', 'ABCAPITAL.NS', 'UNOMINDA.NS', 'COROMANDEL.NS', 'GVT&D.NS', 'LLOYDSME.NS', 'FORTIS.NS', 'PRESTIGE.NS', 'VMM.NS', 'OIL.NS', 'PATANJALI.NS', 'JSWINFRA.NS', 'JSL.NS', 'GODREJPROP.NS', 'AUROPHARMA.NS', 'COFORGE.NS', 'AIRTELPP.NS', 'MFSL.NS', 'PHOENIXLTD.NS', 'MOTILALOFS.NS', 'BDL.NS', 'LTF.NS', 'NAM-INDIA.NS', 'GLENMARK.NS', 'MPHASIS.NS', 'UPL.NS', 'KALYANKJIL.NS', 'ITCHOTELS.NS', 'HEXT.NS', 'FEDERALBNK.NS', 'BIOCON.NS', 'APLAPOLLO.NS', 'TATACOMM.NS', '360ONE.NS', 'ENDURANCE.NS', 'RADICO.NS', 'ASTRAL.NS', 'MEDANTA.NS', 'NH.NS', 'IPCALAB.NS', 'DELHIVERY.NS', 'CHOLAHLDNG.NS', 'GODIGIT.NS', 'COHANCE.NS', 'METROBRAND.NS', 'ASTERDM.NS', 'SUMICHEM.NS', 'APOLLOTYRE.NS', 'KIMS.NS', 'IGL.NS', 'EMCURE.NS', 'SONACOMS.NS', 'MSUMI.NS', 'AFFLE.NS', 'JBCHEPHARM.NS', 'SYNGENE.NS', 'EMAMILTD.NS', 'IRB.NS', 'SHYAMMETL.NS', 'AEGISLOG.NS', 'FSL.NS', 'PPLPHARMA.NS', 'INOXWIND.NS', 'ERIS.NS', 'BRIGADE.NS', 'WELCORP.NS', 'ANGELONE.NS', 'AADHARHFC.NS', 'CREDITACC.NS', 'CHALET.NS', 'KPIL.NS', 'CESC.NS', 'KEC.NS', 'SAGILITY.NS', 'ONESOURCE.NS', 'PNBHOUSING.NS', 'KARURVYSYA.NS', 'DEVYANI.NS', 'CROMPTON.NS', 'ABREL.NS', 'FIRSTCRY.NS', 'KFINTECH.NS', 'SAILIFE.NS', 'BIKAJI.NS', 'IIFL.NS', 'ANANTRAJ.NS', 'CGCL.NS', 'TECHNOE.NS', 'ACMESOLAR.NS', 'VENTIVE.NS', 'VINATIORGA.NS', 'FIVESTAR.NS', 'APTUS.NS', 'JUBLPHARMA.NS', 'TRITURBINE.NS', 'RATNAMANI.NS', 'RAINBOW.NS', 'PGEL.NS', 'AFCONS.NS', 'ELGIEQUIP.NS', 'SOBHA.NS', 'NIVABUPA.NS', 'SIGNATURE.NS', 'CUB.NS', 'TBOTEK.NS', 'ATHERENERG.NS', 'IGIL.NS', 'CIEINDIA.NS', 'DOMS.NS', 'TARIL.NS', 'ABDL.NS', 'DATAPATTNS.NS', 'PCBL.NS', 'GODREJAGRO.NS', 'THELEELA.NS', 'KIRLOSENG.NS', 'RRKABEL.NS', 'NCC.NS', 'HOMEFIRST.NS', 'BLUEJET.NS', 'INTELLECT.NS', 'LEMONTREE.NS', 'SYRMA.NS', 'FINPIPE.NS', 'MGL.NS', 'FINCABLES.NS', 'JINDALSAW.NS', 'GRAVITA.NS', 'ZYDUSWELL.NS', 'JYOTHYLAB.NS', 'SBFC.NS', 'RITES.NS', 'GRINFRA.NS', 'MINDACORP.NS', 'CELLO.NS', 'ALIVUS.NS', 'BALRAMCHIN.NS', 'CCL.NS', 'ACUTAAS.NS', 'TITAGARH.NS', 'BLACKBUCK.NS', 'JUBLINGREA.NS', 'STARCEMENT.NS', 'WELSPUNLIV.NS', 'METROPOLIS.NS', 'SUDARSCHEM.NS', 'GRANULES.NS', 'PVRINOX.NS', 'IXIGO.NS', 'ENGINERSIN.NS', 'EUREKAFORB.NS', 'SAPPHIRE.NS', 'SONATSOFTW.NS', 'AZAD.NS', 'CANFINHOME.NS', 'AETHER.NS', 'BIRLACORPN.NS', 'WABAG.NS', 'SAFARI.NS', 'MEDPLUS.NS', 'INDIASHLTR.NS', 'ASTRAMICRO.NS', 'HCG.NS', 'MAPMYINDIA.NS', 'JLHL.NS', 'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'JKTYRE.NS', 'TCI.NS', 'SHARDACROP.NS', 'UJJIVANSFB.NS', 'BECTORFOOD.NS', 'KIRLPNU.NS', 'VARROC.NS', 'RUSTOMJEE.NS', 'CAMPUS.NS', 'DODLA.NS', 'SANSERA.NS', 'PNCINFRA.NS', 'GALAXYSURF.NS', 'AJAXENGG.NS', 'MASTEK.NS', 'ARVIND.NS', 'EPL.NS', 'AKUMS.NS', 'TRIVENI.NS', 'MAXESTATES.NS', 'MAHLIFE.NS', 'CMSINFO.NS', 'DHANUKA.NS', 'JUSTDIAL.NS', 'CSBBANK.NS', 'ARVINDFASN.NS', 'CHEMPLASTS.NS', 'BAJAJELEC.NS', 'MOIL.NS', 'JUNIPER.NS', 'SYMPHONY.NS', 'THANGAMAYL.NS', 'KTKBANK.NS', 'HGINFRA.NS', 'AVL.NS', 'AHLUCONT.NS', 'SUPRAJIT.NS', 'SENCO.NS', 'GULFOILLUB.NS', 'IFBIND.NS', 'VMART.NS', 'GREENLAM.NS', 'RATEGAIN.NS', 'BANSALWIRE.NS', 'SUNTECK.NS', 'MASFIN.NS', 'GOKEX.NS', 'FEDFINA.NS', 'SIS.NS', 'EMIL.NS', 'BALAMINES.NS', 'VRLLOG.NS', 'JKIL.NS', 'RBA.NS', 'SAMHI.NS', 'ORIENTELEC.NS', 'AWFIS.NS', 'WONDERLA.NS', 'QUESS.NS', 'MEDIASSIST.NS', 'ORCHPHARMA.NS', 'GREENPLY.NS', 'DCBBANK.NS', 'NEOGEN.NS', 'GOCOLORS.NS', 'KOLTEPATIL.NS', 'EPACK.NS', 'CYIENTDLM.NS', 'ROSSARI.NS', 'GATEWAY.NS', 'FLAIR.NS', 'SAGCEM.NS', 'AJANTPHARM.NS', 'BAJAJCON.NS', 'TEAMLEASE.NS', 'JTLIND.NS', 'ARVSMART.NS', 'STYLAMIND.NS', 'VENUSPIPES.NS', 'PSPPROJECT.NS', 'MOLDTKPAC.NS', 'KRSNAA.NS', 'YATRA.NS', 'STOVEKRAFT.NS', 'SOMANYCERA.NS', 'LAXMIDENTL.NS', 'SPANDANA.NS', 'UDS.NS', 'SSFLPP.NS']
    results = []

    for sym in symbols:
        print(f"\nProcessing {sym} ...")
        best = evaluate_best_model(sym)
        if best:
            results.append(best)

    if results:
        # Sort by accuracy descending
        results.sort(key=lambda x: x['accuracy'], reverse=True)

        # Prepare table
        table = []
        for r in results:
            table.append([
                r['symbol'],
                r['days'],
                r['lookback'],
                f"{r['accuracy']:.4f}",
                f"{r['backtest_date']} ({r['backtest_pred']} / actual: {r['actual_last_dir']})",
                r['next_day_pred']
            ])

        headers = ["Symbol", "Days", "Lookback", "Accuracy", "Backtest (Pred / Actual)", "Next day pred"]
        print("\n" + tabulate(table, headers=headers, tablefmt="pretty"))
    else:
        print("⚠️ No symbols could be processed.")
