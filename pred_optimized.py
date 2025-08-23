import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import optuna
import warnings
import sys
warnings.filterwarnings('ignore')

# Suppress logs
import logging
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
import os
os.environ['LIGHTGBM_VERBOSITY'] = 'FATAL'

# ---------------------------
# Configuration
# ---------------------------
if len(sys.argv) > 1:
    stock_symbol = sys.argv[1]
else:
    stock_symbol = 'SBIN.NS'
nifty_symbol = '^NSEI'

# ---------------------------
# Advanced Feature Engineering (Enhanced)
# ---------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast).mean()
    exp2 = series.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_atr(high, low, close, window=14):
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def add_advanced_features(df, close_col, open_col, high_col, low_col, volume_col):
    df = df.copy()
    df['close'] = df[close_col]
    df['open'] = df[open_col] if open_col else df[close_col]
    df['high'] = df[high_col] if high_col else df[close_col]
    df['low'] = df[low_col] if low_col else df[close_col]
    df['volume'] = df[volume_col] if volume_col else 1

    # Returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility-scaled returns
    df['volatility_10'] = df['return'].rolling(10).std()
    df['volatility_30'] = df['return'].rolling(30).std()
    df['volatility_ratio'] = df['volatility_10'] / (df['volatility_30'] + 1e-6)

    # Moving Averages
    df['SMA5'] = df['close'].rolling(5).mean()
    df['SMA10'] = df['close'].rolling(10).mean()
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    df['EMA12'] = df['close'].ewm(span=12).mean()
    df['EMA26'] = df['close'].ewm(span=26).mean()

    # Golden/Death Cross
    df['MA_crossover'] = (df['SMA50'] > df['SMA200']).astype(int).diff()

    # RSI
    df['RSI'] = compute_rsi(df['close'], 14)
    df['RSI_regime'] = pd.cut(df['RSI'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought']).astype('category').cat.codes

    # MACD
    macd, signal, hist = compute_macd(df['close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    df['MACD_cross'] = (df['MACD'] > df['MACD_signal']).astype(int).diff()

    # ATR
    df['ATR'] = compute_atr(df['high'], df['low'], df['close'])
    df['ATR_ratio'] = df['ATR'] / df['close']

    # Bollinger Bands
    rolling_std = df['return'].rolling(20).std()
    upper = df['SMA20'] + 2 * rolling_std * df['close']
    lower = df['SMA20'] - 2 * rolling_std * df['close']
    df['BB_upper'] = upper
    df['BB_lower'] = lower
    df['BB_width'] = (upper - lower) / (df['SMA20'] + 1e-6)
    df['BB_position'] = (df['close'] - lower) / (upper - lower + 1e-6)
    df['BB_squeeze'] = (df['BB_width'] < df['BB_width'].rolling(20).mean()).astype(int)

    # Stochastic Oscillator
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-6)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    df['Stoch_signal'] = (df['Stoch_K'] > df['Stoch_D']).astype(int).diff()

    # Volume
    df['Volume_SMA'] = df['volume'].rolling(10).mean()
    df['Volume_ratio'] = df['volume'] / (df['Volume_SMA'] + 1e-6)
    df['Volume_volatility'] = df['volume'].pct_change().rolling(10).std()

    # Candlestick features
    df['candle_body'] = (df['close'] - df['open']).abs()
    df['candle_range'] = df['high'] - df['low']
    df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-6)
    df['candle_upper_shadow'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['candle_range'] + 1e-6)
    df['candle_lower_shadow'] = (df[['open','close']].min(axis=1) - df['low']) / (df['candle_range'] + 1e-6)

    # Lag features
    for lag in [1, 2, 3]:
        df[f'return_lag{lag}'] = df['return'].shift(lag)
        df[f'volume_lag{lag}'] = df['volume'].shift(lag)
        df[f'RSI_lag{lag}'] = df['RSI'].shift(lag)
        df[f'volatility_lag{lag}'] = df['volatility_10'].shift(lag)

    # Momentum & Acceleration
    df['ROC_5'] = df['close'].pct_change(5)
    df['ROC_10'] = df['close'].pct_change(10)
    df['acceleration'] = df['ROC_5'].diff()

    # Trend Strength
    df['trend_20'] = (df['close'] - df['SMA20']) / (df['ATR'] + 1e-6)
    df['trend_50'] = (df['close'] - df['SMA50']) / (df['ATR'] + 1e-6)
    df['trend_strength'] = (df['close'] - df['SMA200']) / df['SMA200']

    # Regime Detection
    df['vol_regime'] = df['volatility_30'].rolling(60).rank(pct=True)  # 0 to 1
    df['regime_vol_high'] = (df['vol_regime'] > 0.7).astype(int)
    df['regime_vol_low'] = (df['vol_regime'] < 0.3).astype(int)

    # Day of week and seasonal
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    # Detrended momentum
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['detrended_momentum'] = df['momentum_10'] - df['momentum_10'].rolling(100).mean()

    # Price position in recent window
    df['price_position_30'] = (df['close'] - df['close'].rolling(30).min()) / \
                              (df['close'].rolling(30).max() - df['close'].rolling(30).min() + 1e-6)

    return df

# ---------------------------
# NEW: Simple Up/Down Target
# ---------------------------
def create_target(close):
    """Predict next day's direction: 1 if up, 0 if down or flat"""
    return (close.pct_change(1).shift(-1) > 0).astype(int)

# ---------------------------
# Smooth Target Encoder
# ---------------------------
def smoothed_target_encode(series, target, smoothing=10):
    global_mean = target.mean()
    counts = series.value_counts()
    means = target.groupby(series).mean()
    smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return series.map(smooth)

# ---------------------------
# Optuna Objective Function (Enhanced)
# ---------------------------
def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 600, 1600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 15.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 15.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'random_state': 42,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'class_weight': 'balanced'
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Leakage-safe scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LGBMClassifier(**params)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)]
        )

        pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, pred_proba)
        balanced_acc = balanced_accuracy_score(y_val, (pred_proba > 0.5).astype(int))
        composite_score = 0.6 * auc + 0.4 * balanced_acc
        scores.append(composite_score)

    return np.mean(scores)

# ---------------------------
# Process Single Stock
# ---------------------------
def process_stock(symbol):
    try:
        print(f"\n🚀 Processing {symbol}...")
        stock_data = yf.download(symbol, period='9y', interval='1d')
        if stock_data.empty:
            print(f"❌ No data for {symbol}")
            return None
        stock_data.dropna(inplace=True)

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]

        close_col = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
        if close_col not in stock_data.columns:
            print(f"❌ No Close column found in {symbol} data")
            return None

        df = add_advanced_features(stock_data, close_col, 'Open', 'High', 'Low', 'Volume')

        # Nifty Market Factor
        df['nifty_return'] = 0.0
        nifty_data = yf.download(nifty_symbol, period='9y', interval='1d')
        if not nifty_data.empty:
            if isinstance(nifty_data.columns, pd.MultiIndex):
                nifty_data.columns = [col[0] for col in nifty_data.columns]
            nifty_close_col = 'Adj Close' if 'Adj Close' in nifty_data.columns else 'Close'
            if nifty_close_col in nifty_data.columns:
                nifty_rets = nifty_data[nifty_close_col].pct_change().reindex(df.index).fillna(0)
                df['nifty_return'] = nifty_rets

        # 🆕 Simple Up/Down Target
        df['target'] = create_target(df['close'])
        df.dropna(inplace=True)

        # Check for class imbalance (too few of one class)
        if df['target'].value_counts().min() < 10:
            print("❌ Not enough samples in one class")
            return None

        # Smooth encode categorical features
        df['dow_encoded'] = smoothed_target_encode(df['day_of_week'], df['target'])
        df['month_encoded'] = smoothed_target_encode(df['month'], df['target'])

        # Feature columns
        feature_columns = [
            'return', 'log_return', 'volatility_10', 'volatility_30', 'volatility_ratio',
            'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA200', 'EMA12', 'EMA26',
            'RSI', 'RSI_regime', 'MACD', 'MACD_signal', 'MACD_hist', 'MACD_cross',
            'ATR_ratio', 'BB_width', 'BB_position', 'BB_squeeze',
            'Stoch_K', 'Stoch_D', 'Stoch_signal',
            'Volume_ratio', 'Volume_volatility',
            'candle_body_ratio', 'candle_upper_shadow', 'candle_lower_shadow',
            'return_lag1', 'return_lag2', 'RSI_lag1', 'RSI_lag2',
            'ROC_5', 'ROC_10', 'acceleration',
            'trend_20', 'trend_50', 'trend_strength',
            'vol_regime', 'regime_vol_high', 'regime_vol_low',
            'detrended_momentum', 'price_position_30',
            'nifty_return', 'dow_encoded', 'month_encoded'
        ]
        feature_columns = [col for col in feature_columns if col in df.columns]

        if len(feature_columns) < 15:
            print("❌ Insufficient features")
            return None

        X = df[feature_columns]
        y = df['target']

        if y.nunique() < 2:
            print("❌ No class variance in target")
            return None

        # Optuna Optimization
        print("🔍 Running Bayesian Optimization...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        try:
            study.optimize(lambda trial: objective(trial, X[-1800:], y[-1800:]), n_trials=40, timeout=240)
        except Exception as e:
            print("Optuna failed:", e)
            return None

        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced'
        })

        # Final Ensemble with Calibrated Classifiers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        lgbm = LGBMClassifier(**best_params)
        rf = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=42)
        lr = LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000, random_state=42, solver='liblinear')

        # Calibrate probabilities
        lgbm_cal = CalibratedClassifierCV(lgbm, method='isotonic', cv=3)
        rf_cal = CalibratedClassifierCV(rf, method='isotonic', cv=3)

        ensemble = VotingClassifier(
            [('lgbm', lgbm_cal), ('rf', rf_cal), ('lr', lr)],
            voting='soft'
        )

        # Walk-forward evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        val_results_list = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            ensemble.fit(X_train, y_train)
            pred = ensemble.predict(X_val)
            prob = ensemble.predict_proba(X_val)[:, 1]

            fold_results = pd.DataFrame({
                'Date': y_val.index,
                'Actual': y_val.values,
                'Predicted': pred,
                'Prob_UP': prob
            })
            val_results_list.append(fold_results)

        val_results = pd.concat(val_results_list, axis=0)
        final_accuracy = accuracy_score(val_results['Actual'], val_results['Predicted'])  # ← Now: % correct direction

        # Predict next day
        latest_features = scaler.transform(X[feature_columns].iloc[-1:])
        prob_up = ensemble.predict_proba(latest_features)[0, 1]
        confidence_raw = max(prob_up, 1 - prob_up)
        direction = "UP" if prob_up > 0.5 else "DOWN"

        # Confidence filter
        if abs(prob_up - 0.5) < 0.1:
            direction = "NEUTRAL"
            confidence = 0.5
        else:
            confidence = confidence_raw

        last_pred = val_results['Predicted'].iloc[-1]

        print(f"✅ {symbol}: {direction} | Confidence: {confidence:.2f} | "
              f"CV Accuracy: {final_accuracy:.4f} | Last Pred: {last_pred}")

        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'accuracy': final_accuracy,
            'last_pred': int(last_pred),
            'prob_up': prob_up
        }

    except Exception as e:
        print(f"❌ Error processing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------------
# Run Single Stock
# ---------------------------
result = process_stock(stock_symbol)
