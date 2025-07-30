# BTC-USD ML Trading Bot

Dieses Repository enthält einen Trading-Bot für BTC-USD mit Machine Learning, ausgeführt im Intervall von 5 Minuten via GitHub Actions.

## Features

- ML-basierte Trading-Entscheidungen für BTC-USD
- Automatisches Nachtrainieren des Modells nach 10 Trades
- Logging der Trades in `live_trades.csv`
- Ausführung alle 5 Minuten via GitHub Actions

## Lokal ausführen

```bash
pip install -r requirements.txt
python main.py
```

## Automatisch (GitHub Actions)

Die Action in `.github/workflows/bot.yml` startet das Skript alle 5 Minuten.

## Hinweise

Das Skript simuliert Trading und ist nicht für echten Handel geeignet!