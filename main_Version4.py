import importlib
import subprocess
import sys

def install_if_missing(package_name, pip_name=None):
    try:
        importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package_name])

install_if_missing('yfinance')
install_if_missing('pandas')
install_if_missing('numpy')
install_if_missing('sklearn', 'scikit-learn')
install_if_missing('imbalanced_learn', 'imbalanced-learn')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import time
import os

# === Parameter ===
SYMBOL = 'BTC-USD'
INTERVAL = '5m'
START_CAPITAL = 10000
TRADE_FEE_RATE = 0.001
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.008
TRAIL_STOP_PCT = 0.02
FUTURE_WINDOW = 24
BUY_PROB_THRESHOLD = 0.2
LIVE_MODE = True
TRADE_LOG_FILE = 'live_trades.csv'

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs)).fillna(50)

def extract_features(df):
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['Close'])
    return df.dropna()

def train_model_from_df(df):
    df['target'] = (df['Close'].shift(-FUTURE_WINDOW) > df['Close'] * (1 + TAKE_PROFIT_PCT)).astype(int)
    df.dropna(inplace=True)
    X = df[['returns', 'volatility', 'rsi']]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_res, y_res = RandomOverSampler().fit_resample(X_scaled, y)
    model = HistGradientBoostingClassifier().fit(X_res, y_res)
    print(f"📊 Modell trainiert mit Accuracy: {model.score(X_scaled, y):.4f}")
    return model, scaler, ['returns', 'volatility', 'rsi']

def append_trade_to_csv(data):
    header = not os.path.exists(TRADE_LOG_FILE)
    pd.DataFrame([data]).to_csv(TRADE_LOG_FILE, mode='a', index=False, header=header)

def wait_for_next_candle(interval_minutes=5):
    now = datetime.utcnow()
    next_minute = (now.minute // interval_minutes + 1) * interval_minutes
    next_candle_time = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(minutes=next_minute)
    wait_seconds = (next_candle_time - now).total_seconds()
    if wait_seconds > 0:
        print(f"⏳ Warte {int(wait_seconds)} Sekunden bis nächste Candle um {next_candle_time.time()}")
        time.sleep(wait_seconds)
    else:
        print("⚠️ Nächste Candle-Zeit schon erreicht, prüfe sofort")

def wait_for_new_open(last_open, symbol, interval, poll_interval=10):
    wait_for_next_candle(5)
    print("⏳ Starte Polling auf neues Open...")

    while True:
        df = yf.download(symbol, period='1d', interval=interval, progress=False)
        if df.empty:
            print("❌ Keine Daten erhalten, versuche in 10s erneut")
            time.sleep(poll_interval)
            continue
        current_open = df.index[-1]
        if current_open > last_open:
            print(f"✅ Neue Candle verfügbar: {current_open}")
            return df, current_open
        print(f"⏳ Noch keine neue Candle, prüfe in {poll_interval}s erneut", end='\r')
        time.sleep(poll_interval)

def live_predict(model, scaler, features):
    capital = START_CAPITAL
    btc_amount = 0
    position = 0
    entry_price = 0
    total_fees = 0
    trades_since_retrain = 0
    highest_price = 0
    trade_id = 1

    df = yf.download(SYMBOL, period='1d', interval=INTERVAL, progress=False)
    df = extract_features(df)
    if df.empty:
        print("❌ Startdaten leer, beende")
        return
    last_open = df.index[-1]

    while True:
        df, last_open_new = wait_for_new_open(last_open, SYMBOL, INTERVAL)
        if df is None:
            continue
        last_open = last_open_new

        df = extract_features(df)
        if df.empty:
            print("⚠️ Features leer, nächste Runde")
            continue

        candle = df.iloc[-2]
        X = scaler.transform([candle[features].values])
        prob = model.predict_proba(X)[0, 1]
        close = float(candle['Close'])
        rsi = float(candle['rsi'])
        vol = float(candle['volatility'])
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if position == 0 and prob > BUY_PROB_THRESHOLD:
            fee = capital * TRADE_FEE_RATE
            btc_amount = (capital - fee) / close
            entry_price = close
            capital -= fee
            total_fees += fee
            highest_price = close
            position = 1
            print(f"🟢 TRADE #{trade_id} BUY {now} | Preis: {close:.2f} | RSI: {rsi:.2f} | Vol: {vol:.5f} | Gebühr: {fee:.2f}")

        elif position == 1:
            highest_price = max(highest_price, close)
            stop = entry_price * (1 - STOP_LOSS_PCT)
            take = entry_price * (1 + TAKE_PROFIT_PCT)
            trail = highest_price * (1 - TRAIL_STOP_PCT)

            exit_reason = None
            if close <= stop:
                exit_reason = 'Stop Loss'
            elif close >= take:
                exit_reason = 'Take Profit'
            elif close <= trail:
                exit_reason = 'Trail Stop'

            if exit_reason:
                fee = btc_amount * close * TRADE_FEE_RATE
                capital = btc_amount * close - fee
                total_fees += fee
                pnl = capital + total_fees - START_CAPITAL
                print(f"🔻 TRADE EXIT {now} | Grund: {exit_reason}")
                print(f"    Entry: {entry_price:.2f} | Exit: {close:.2f} | Δ: {close - entry_price:.2f}")
                print(f"    Kapital: {capital:.2f} | Gebühren: {total_fees:.2f} | PnL: {pnl:.2f}")
                append_trade_to_csv({
                    'zeit': now,
                    'preis': close,
                    'rsi': rsi,
                    'vol': vol,
                    'pnl': pnl,
                    'entry': entry_price,
                    'exit_reason': exit_reason
                })
                position = 0
                trades_since_retrain += 1

        if trades_since_retrain >= 10 and os.path.exists(TRADE_LOG_FILE):
            df_log = pd.read_csv(TRADE_LOG_FILE)
            df_log = extract_features(df_log)
            if not df_log.empty:
                model, scaler, features = train_model_from_df(df_log)
                trades_since_retrain = 0
                print(f"🔁 Modell nach {trades_since_retrain} Trades nachtrainiert")

        trade_id += 1

# Initialmodell trainieren
hist = yf.download(SYMBOL, period='30d', interval='5m', progress=False)
hist = extract_features(hist)
model, scaler, features = train_model_from_df(hist)

if LIVE_MODE:
    live_predict(model, scaler, features)