
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
install_if_missing('flask')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import time
import os
import threading
from flask import Flask, render_template, jsonify, request

# === Parameters ===
SYMBOL = 'BTC-USD'
INTERVAL = '5m'
START_CAPITAL = 10000
TRADE_FEE_RATE = 0.001
STOP_LOSS_PCT = 0.01
TAKE_PROFIT_PCT = 0.008
TRAIL_STOP_PCT = 0.02
FUTURE_WINDOW = 24
BUY_PROB_THRESHOLD = 0.2
TRADE_LOG_FILE = 'live_trades.csv'

# Global variables for the web app
app = Flask(__name__)
bot_running = False
bot_thread = None
current_status = {
    'capital': START_CAPITAL,
    'position': 0,
    'current_price': 0,
    'last_trade': None,
    'total_trades': 0,
    'pnl': 0
}

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
    print(f"📊 Model trained with Accuracy: {model.score(X_scaled, y):.4f}")
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
        print(f"⏳ Waiting {int(wait_seconds)} seconds until next candle at {next_candle_time.time()}")
        time.sleep(min(wait_seconds, 10))  # Limit sleep to 10 seconds for responsiveness
    else:
        print("⚠️ Next candle time already reached, checking immediately")

def wait_for_new_open(last_open, symbol, interval, poll_interval=10):
    wait_for_next_candle(5)
    print("⏳ Starting polling for new open...")

    retries = 0
    while bot_running and retries < 6:  # Max 60 seconds of retries
        df = yf.download(symbol, period='1d', interval=interval, progress=False)
        if df.empty:
            print("❌ No data received, trying again in 10s")
            time.sleep(poll_interval)
            retries += 1
            continue
        current_open = df.index[-1]
        if current_open > last_open:
            print(f"✅ New candle available: {current_open}")
            return df, current_open
        print(f"⏳ No new candle yet, checking again in {poll_interval}s")
        time.sleep(poll_interval)
        retries += 1
    
    return None, last_open

def live_predict(model, scaler, features):
    global current_status, bot_running
    
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
        print("❌ Start data empty, ending")
        return
    last_open = df.index[-1]

    while bot_running:
        df, last_open_new = wait_for_new_open(last_open, SYMBOL, INTERVAL)
        if df is None:
            continue
        last_open = last_open_new

        df = extract_features(df)
        if df.empty:
            print("⚠️ Features empty, next round")
            continue

        candle = df.iloc[-2]
        X = scaler.transform([candle[features].values])
        prob = model.predict_proba(X)[0, 1]
        close = float(candle['Close'])
        rsi = float(candle['rsi'])
        vol = float(candle['volatility'])
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update current status
        current_status['current_price'] = close
        current_status['capital'] = capital
        current_status['position'] = position

        if position == 0 and prob > BUY_PROB_THRESHOLD:
            fee = capital * TRADE_FEE_RATE
            btc_amount = (capital - fee) / close
            entry_price = close
            capital -= fee
            total_fees += fee
            highest_price = close
            position = 1
            current_status['position'] = position
            current_status['last_trade'] = f"BUY at {close:.2f}"
            print(f"🟢 TRADE #{trade_id} BUY {now} | Price: {close:.2f} | RSI: {rsi:.2f} | Vol: {vol:.5f} | Fee: {fee:.2f}")

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
                current_status['pnl'] = pnl
                current_status['total_trades'] = trade_id
                current_status['last_trade'] = f"SELL at {close:.2f} ({exit_reason})"
                print(f"🔻 TRADE EXIT {now} | Reason: {exit_reason}")
                print(f"    Entry: {entry_price:.2f} | Exit: {close:.2f} | Δ: {close - entry_price:.2f}")
                print(f"    Capital: {capital:.2f} | Fees: {total_fees:.2f} | PnL: {pnl:.2f}")
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
                print(f"🔁 Model retrained after {trades_since_retrain} trades")

        trade_id += 1

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/status')
def get_status():
    return jsonify(current_status)

@app.route('/trades')
def get_trades():
    if os.path.exists(TRADE_LOG_FILE):
        df = pd.read_csv(TRADE_LOG_FILE)
        return jsonify(df.tail(10).to_dict('records'))
    return jsonify([])

@app.route('/start_bot', methods=['POST'])
def start_bot():
    global bot_running, bot_thread
    if not bot_running:
        bot_running = True
        # Train initial model
        hist = yf.download(SYMBOL, period='30d', interval='5m', progress=False)
        hist = extract_features(hist)
        model, scaler, features = train_model_from_df(hist)
        
        bot_thread = threading.Thread(target=live_predict, args=(model, scaler, features))
        bot_thread.start()
        return jsonify({'status': 'Bot started'})
    return jsonify({'status': 'Bot already running'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    global bot_running
    bot_running = False
    return jsonify({'status': 'Bot stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
