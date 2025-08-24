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

days_array = [700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
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
    data_full = yf.download(symbol, period='10y', interval='1d')["Close"].values
    dates_full = yf.download(symbol, period='10y', interval='1d').index
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

    # Next day prediction
    last_lb_days = data_full[-best_result['lookback']:]
    next_pred = best_result['model'].predict(last_lb_days.reshape(1, -1))[0]
    best_result['next_day_pred'] = "UP" if next_pred == 1 else "DOWN"

    # Backtest prediction (last day in dataset)
    backtest_pred = best_result['model'].predict(best_result['X_last'].reshape(1,-1))[0]
    best_result['backtest_pred'] = "UP" if backtest_pred == 1 else "DOWN"
    best_result['backtest_date'] = best_result['last_day_date'].strftime('%Y-%m-%d')

    return best_result

if __name__ == "__main__":
    symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'ITC.NS', 'ETERNAL.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 'DLF.NS', 'SBILIFE.NS', 'VBL.NS', 'PFC.NS', 'RECLTD.NS', 'ADANIENSOL.NS', 'INDIANB.NS', 'SUZLON.NS', 'ABCAPITAL.NS', 'LLOYDSME.NS', 'FORTIS.NS', 'PRESTIGE.NS', 'GVT&D.NS', 'PATANJALI.NS', 'MFSL.NS', 'MOTILALOFS.NS', 'KALYANKJIL.NS', 'ITCHOTELS.NS', 'HEXT.NS', '360ONE.NS', 'COHANCE.NS', 'ASTERDM.NS', 'KIMS.NS', 'PPLPHARMA.NS', 'BRIGADE.NS', 'WELCORP.NS', 'TIMKEN.NS', 'CHALET.NS', 'AADHARHFC.NS', 'SAGILITY.NS', 'ONESOURCE.NS', 'PNBHOUSING.NS', 'KARURVYSYA.NS', 'CROMPTON.NS', 'ABREL.NS', 'BIKAJI.NS', 'SAILIFE.NS', 'ANANTRAJ.NS', 'CGCL.NS', 'TECHNOE.NS', 'VENTIVE.NS', 'RATNAMANI.NS', 'TRITURBINE.NS', 'RAINBOW.NS', 'SIGNATURE.NS', 'TBOTEK.NS', 'AFCONS.NS', 'ATHERENERG.NS', 'DOMS.NS', 'TARIL.NS', 'ABDL.NS', 'THELEELA.NS', 'KIRLOSENG.NS', 'RRKABEL.NS', 'LEMONTREE.NS', 'JINDALSAW.NS', 'GRAVITA.NS', 'CELLO.NS', 'ALIVUS.NS', 'BALRAMCHIN.NS', 'TITAGARH.NS', 'GRANULES.NS', 'AZAD.NS', 'AETHER.NS', 'BIRLACORPN.NS', 'WABAG.NS', 'MEDPLUS.NS', 'INDIASHLTR.NS', 'ASTRAMICRO.NS', 'JLHL.NS', 'TCI.NS', 'BECTORFOOD.NS', 'KIRLPNU.NS', 'RUSTOMJEE.NS', 'DODLA.NS', 'ARVIND.NS', 'EPL.NS', 'AKUMS.NS', 'MAXESTATES.NS', 'CSBBANK.NS', 'CHEMPLASTS.NS', 'ARVINDFASN.NS', 'MOIL.NS', 'JUNIPER.NS', 'AVL.NS', 'THANGAMAYL.NS', 'KTKBANK.NS', 'GULFOILLUB.NS', 'BANSALWIRE.NS', 'SUNTECK.NS', 'MASFIN.NS', 'SIS.NS', 'FEDFINA.NS', 'EMIL.NS', 'JKIL.NS', 'BALAMINES.NS', 'VRLLOG.NS', 'SAMHI.NS', 'AWFIS.NS', 'WONDERLA.NS', 'ORCHPHARMA.NS', 'MEDIASSIST.NS', 'DCBBANK.NS', 'NEOGEN.NS', 'GOCOLORS.NS', 'KOLTEPATIL.NS', 'EPACK.NS', 'ROSSARI.NS', 'GATEWAY.NS', 'FLAIR.NS', 'BAJAJCON.NS', 'JTLIND.NS', 'ARVSMART.NS', 'VENUSPIPES.NS', 'MOLDTKPAC.NS', 'KRSNAA.NS', 'YATRA.NS', 'STOVEKRAFT.NS', 'LAXMIDENTL.NS', 'UDS.NS'] # <-- Add your list here
    results = []

    for sym in symbols:
        print(f"\nProcessing {sym} ...")
        best = evaluate_best_model(sym)
        results.append(best)

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
