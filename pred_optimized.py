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
        try:
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
        except Exception as e:
            print(f"⚠️ Error during training for {symbol} (days={days}): {e}")
            continue

    if best_result is None:
        print(f"⚠️ No valid model for {symbol}")
        return None

    try:
        # Next day prediction
        last_lb_days = data_full[-best_result['lookback']:]
        next_pred = best_result['model'].predict(last_lb_days.reshape(1, -1))[0]
        best_result['next_day_pred'] = "UP" if next_pred == 1 else "DOWN"

        # Backtest prediction
        backtest_pred = best_result['model'].predict(best_result['X_last'].reshape(1,-1))[0]
        best_result['backtest_pred'] = "UP" if backtest_pred == 1 else "DOWN"
        best_result['backtest_date'] = best_result['last_day_date'].strftime('%Y-%m-%d')
    except Exception as e:
        print(f"⚠️ Prediction step failed for {symbol}: {e}")
        return None

    return best_result

if __name__ == "__main__":
    # Only a few symbols for now
    symbols = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS']
    results = []

    for sym in symbols:
        try:
            print(f"\nProcessing {sym} ...")
            best = evaluate_best_model(sym)
            if best:
                results.append(best)
        except Exception as e:
            print(f"❌ Error processing {sym}: {e}")
            continue

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
