import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import requests

# -----------------------------
# Initialize Session State Variables
# -----------------------------
if "edit_asset" not in st.session_state:
    st.session_state.edit_asset = None

# -----------------------------
# Live Exchange Rates Functions
# -----------------------------
def get_exchange_rates():
    url = "https://api.exchangerate-api.com/v4/latest/EUR"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["rates"]
        else:
            return None
    except Exception:
        return None

def get_conversion_factor(currency, rates):
    if currency.upper() in ["EURO", "EUR"]:
        return 1.0
    try:
        rate = rates.get(currency.upper())
        if rate is not None:
            return 1.0 / rate
        else:
            return 1.0
    except Exception:
        return 1.0

current_rates = get_exchange_rates()
if current_rates is None:
    st.warning("Failed to load live exchange rates. Using default values.")
    current_rates = {"USD": 1/1.08, "GBP": 1/0.85, "EUR": 1.0}

# -----------------------------
# Helper Functions for Persistence
# -----------------------------
def load_portfolio():
    if os.path.exists("portfolio.json"):
        with open("portfolio.json", "r") as f:
            return json.load(f)
    return []

def save_portfolio(portfolio):
    with open("portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=4)

def load_history():
    if os.path.exists("history.json"):
        with open("history.json", "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)

def load_cash():
    if os.path.exists("cash.json"):
        with open("cash.json", "r") as f:
            return json.load(f)
    return {"amount": 0.0, "currency": "EUR"}

def save_cash(cash):
    with open("cash.json", "w") as f:
        json.dump(cash, f, indent=4)

# -----------------------------
# Helper Function for Ticker Format
# -----------------------------
def get_final_ticker(ticker, asset_type, market):
    ticker = ticker.strip().upper()
    if asset_type == "Stock/ETF":
        if "." in ticker:
            return ticker
        else:
            if market == "LSE":
                return ticker + ".L"
            elif market == "EU":
                return ticker + ".PA"
            else:
                return ticker
    else:
        return ticker

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("Portfolio Tracker")

# -----------------------------
# Sidebar: Add New Asset and Cash Input
# -----------------------------
st.sidebar.header("Add a New Asset")
with st.sidebar.form("asset_form"):
    asset_type = st.selectbox("Asset Type", ["Stock/ETF", "Crypto"])
    
    if asset_type == "Crypto":
        top_crypto_list = [
            "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
            "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "UNI-USD",
            "LTC-USD", "LINK-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
            "XMR-USD", "ALGO-USD", "ICP-USD", "FTT-USD", "TRX-USD"
        ]
        ticker_input = st.selectbox("Crypto Ticker", top_crypto_list)
        market = None
    else:
        ticker_input = st.text_input("Ticker (e.g., AAPL, BATS, JNJ, JNJ.DE)")
        market = st.selectbox("Market/Exchange", ["US", "LSE", "EU"])
    
    currency = st.selectbox("Asset Currency", ["USD", "GBP", "EUR"])
    buy_price = st.number_input("Buy Price (optional)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
    quantity = st.number_input("Quantity", min_value=0.0, step=0.01, format="%.2f")
    submit = st.form_submit_button("Add Asset")
    
    if submit:
        if ticker_input.strip() == "" or quantity <= 0:
            st.sidebar.error("Please enter a valid ticker and a quantity > 0.")
        else:
            ticker_upper = ticker_input.strip().upper()
            final_ticker = get_final_ticker(ticker_upper, asset_type, market)
            try:
                data = yf.Ticker(final_ticker).history(period="1d")
                if data.empty:
                    raise ValueError("No data found.")
            except Exception:
                st.sidebar.error(f"Ticker '{final_ticker}' does not exist.")
            else:
                portfolio = load_portfolio()
                portfolio.append({
                    "ticker": ticker_upper,
                    "asset_type": asset_type,
                    "market": market,
                    "currency": currency,
                    "buy_price": buy_price,
                    "quantity": quantity,
                    "added_date": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                save_portfolio(portfolio)
                st.sidebar.success(f"Asset '{ticker_upper}' added!")

st.sidebar.header("Manage Cash")
with st.sidebar.form("cash_form"):
    cash_amount = st.number_input("Cash Amount", min_value=0.0, step=0.01, format="%.2f", value=load_cash().get("amount", 0.0))
    cash_currency = st.selectbox("Cash Currency", ["EUR", "USD", "GBP"],
                                  index=["EUR", "USD", "GBP"].index(load_cash().get("currency", "EUR")))
    cash_submit = st.form_submit_button("Update Cash")
    if cash_submit:
        cash = {"amount": cash_amount, "currency": cash_currency}
        save_cash(cash)
        st.sidebar.success("Cash amount updated!")

# -----------------------------
# Load Portfolio and Cash Data
# -----------------------------
portfolio = load_portfolio()
cash = load_cash()

# -----------------------------
# Create Tabs for Portfolio and History
# -----------------------------
tabs = st.tabs(["Portfolio", "History"])

# -----------------------------
# Tab 1: Portfolio
# -----------------------------
with tabs[0]:
    st.header("Portfolio Overview")
    
    # --- Stocks/ETFs Section ---
    st.subheader("Stocks/ETFs")
    stock_data = []
    total_stock_value_eur = 0
    total_stock_gain_eur = 0
    for asset in portfolio:
        if asset.get("asset_type") == "Stock/ETF":
            asset_currency = asset.get("currency", "USD")
            quantity = asset.get("quantity", 0)
            buy_price = asset.get("buy_price", 0)
            ticker = get_final_ticker(asset.get("ticker"), "Stock/ETF", asset.get("market"))
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if hist.empty:
                    continue
                info = stock.info
                current_price = info.get("regularMarketPrice") or hist["Close"].iloc[-1]
                conv_rate = get_conversion_factor(asset_currency, current_rates)
                price_eur = current_price * conv_rate
                buy_price_eur = buy_price * conv_rate
                value = current_price * quantity
                value_eur = price_eur * quantity
                invested_eur = buy_price_eur * quantity
                gain_eur = value_eur - invested_eur
                raw_sector = info.get("sector", "")
                long_name = info.get("longName", "").lower() if info.get("longName") else ""
                if any(word in long_name for word in ["etf", "index", "s&p", "fund"]) or raw_sector == "":
                    sector = "Index Fund"
                else:
                    sector = raw_sector or "Unknown"
                stock_data.append({
                    "Ticker": ticker,
                    "Sector": sector,
                    "Quantity": quantity,
                    "Buy Price": buy_price,
                    "Current Price": round(current_price, 2),
                    "Price (EUR)": round(price_eur, 2),
                    "Value": round(value, 2),
                    "Value (EUR)": round(value_eur, 2),
                    "Gain/Loss (EUR)": round(gain_eur, 2),
                    "Currency": asset_currency
                })
                total_stock_value_eur += value_eur
                total_stock_gain_eur += gain_eur
            except Exception:
                continue
    if stock_data:
        df_stocks = pd.DataFrame(stock_data)
        st.dataframe(df_stocks)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Stocks Gain/Loss (EUR)", value=f"€{total_stock_gain_eur:.2f}")
        with col2:
            st.metric(label="Total Stocks Value (EUR)", value=f"€{total_stock_value_eur:.2f}")
    else:
        st.info("No Stocks/ETFs to display.")
    
    st.markdown("---")
    
    # --- Crypto Section ---
    st.subheader("Crypto")
    crypto_data = []
    total_crypto_value_eur = 0
    total_crypto_gain_eur = 0
    for asset in portfolio:
        if asset.get("asset_type") == "Crypto":
            asset_currency = asset.get("currency", "USD")
            quantity = asset.get("quantity", 0)
            buy_price = asset.get("buy_price", 0)
            ticker = asset.get("ticker")
            try:
                crypto = yf.Ticker(ticker)
                hist = crypto.history(period="1d")
                if hist.empty:
                    continue
                info = crypto.info
                current_price = info.get("regularMarketPrice") or hist["Close"].iloc[-1]
                conv_rate = get_conversion_factor(asset_currency, current_rates)
                price_eur = current_price * conv_rate
                buy_price_eur = buy_price * conv_rate
                value = current_price * quantity
                value_eur = price_eur * quantity
                invested_eur = buy_price_eur * quantity
                gain_eur = value_eur - invested_eur
                crypto_data.append({
                    "Ticker": ticker,
                    "Quantity": quantity,
                    "Buy Price": buy_price,
                    "Current Price": round(current_price, 2),
                    "Price (EUR)": round(price_eur, 2),
                    "Value": round(value, 2),
                    "Value (EUR)": round(value_eur, 2),
                    "Gain/Loss (EUR)": round(gain_eur, 2),
                    "Currency": asset_currency
                })
                total_crypto_value_eur += value_eur
                total_crypto_gain_eur += gain_eur
            except Exception:
                continue
    if crypto_data:
        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Crypto Gain/Loss (EUR)", value=f"€{total_crypto_gain_eur:.2f}")
        with col2:
            st.metric(label="Total Crypto Value (EUR)", value=f"€{total_crypto_value_eur:.2f}")
    else:
        st.info("No Crypto to display.")
    
    # --- Cash Section ---
    st.markdown("---")
    st.subheader("Cash")
    cash_amount = cash.get("amount", 0.0)
    cash_currency = cash.get("currency", "EUR")
    conv_rate = get_conversion_factor(cash_currency, current_rates)
    cash_value_eur = cash_amount * conv_rate
    st.metric(label="Cash", value=f"€{cash_value_eur:.2f}")
    
    # --- Total Portfolio Value Counter ---
    total_portfolio_value = total_stock_value_eur + total_crypto_value_eur + cash_value_eur
    st.markdown("---")
    st.markdown(f"## Total Portfolio Value (EUR): €{total_portfolio_value:.2f}")
    
    # --- Portfolio Value Chart Expander ---
    with st.expander("Show Portfolio Value Chart"):
        history = load_history()
        if history:
            df_history = pd.DataFrame(history)
            df_history["date"] = pd.to_datetime(df_history["date"], format="%Y-%m-%d %H:%M", errors="coerce")
            df_history.sort_values(by="date", inplace=True)
            if "total_value_eur" in df_history.columns:
                default_y_min = float(df_history["total_value_eur"].min())
                default_y_max = float(df_history["total_value_eur"].max())
                y_min = st.number_input("Y-axis Min (EUR)", value=default_y_min)
                y_max = st.number_input("Y-axis Max (EUR)", value=default_y_max)
                default_x_min = df_history["date"].min().date()
                default_x_max = df_history["date"].max().date()
                x_min = st.date_input("X-axis Start", value=default_x_min)
                x_max = st.date_input("X-axis End", value=default_x_max)
                fig = px.line(
                    df_history, x="date", y="total_value_eur",
                    title="Portfolio Value History"
                )
                fig.update_layout(
                    xaxis_range=[x_min.strftime("%Y-%m-%d"), x_max.strftime("%Y-%m-%d")],
                    yaxis_range=[y_min, y_max]
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            else:
                st.write("No total portfolio value data found in history.")
        else:
            st.write("No history data available for chart.")
    
    # --- Manage Assets Section (Horizontal Layout) ---
    with st.expander("Manage Assets (Stocks/ETFs/Crypto)"):
        st.write("Each asset is displayed as **Ticker – Edit – Delete**. If there are more than 4, they wrap onto a new row.")
        assets = list(enumerate(portfolio))
        items_per_row = 4
        for row_start in range(0, len(assets), items_per_row):
            row_slice = assets[row_start : row_start + items_per_row]
            # Each asset occupies 3 columns: Ticker, Edit, Delete.
            row_cols = st.columns(len(row_slice) * 3)
            for j, (idx, asset) in enumerate(row_slice):
                col_ticker = row_cols[3*j]
                col_edit = row_cols[3*j + 1]
                col_delete = row_cols[3*j + 2]
                col_ticker.write(f"**{asset['ticker']}**")
                if col_edit.button("✏️", key=f"edit_{idx}"):
                    st.session_state.edit_asset = idx
                if col_delete.button("❌", key=f"delete_{idx}"):
                    del portfolio[idx]
                    save_portfolio(portfolio)
                    st.success("Asset deleted!")
                    st.experimental_rerun()
                if st.session_state.get("edit_asset") == idx:
                    with st.form(key=f"edit_form_{idx}"):
                        new_buy_price = st.number_input(
                            "New Buy Price",
                            min_value=0.0,
                            step=0.01,
                            format="%.2f",
                            value=asset.get("buy_price", 0.0)
                        )
                        if asset.get("asset_type") == "Crypto":
                            new_quantity = st.number_input(
                                "New Quantity",
                                min_value=0.0,
                                step=0.00000001,
                                format="%.8f",
                                value=asset.get("quantity", 0.0)
                            )
                        else:
                            new_quantity = st.number_input(
                                "New Quantity",
                                min_value=0.0,
                                step=0.01,
                                format="%.2f",
                                value=asset.get("quantity", 0.0)
                            )
                        submit_edit = st.form_submit_button("Save")
                        if submit_edit:
                            portfolio[idx]["buy_price"] = new_buy_price
                            portfolio[idx]["quantity"] = new_quantity
                            portfolio[idx]["added_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            save_portfolio(portfolio)
                            st.success("Asset updated!")
                            st.session_state.edit_asset = None
                            st.experimental_rerun()

# -----------------------------
# Tab 2: History
# -----------------------------
with tabs[1]:
    st.header("Performance History")
    
    if st.button("Save Performance Snapshot"):
        total_invested_stocks = 0
        total_value_stocks = 0
        total_invested_crypto = 0
        total_value_crypto = 0
        for asset in portfolio:
            asset_type = asset.get("asset_type")
            asset_currency = asset.get("currency", "USD")
            quantity = asset.get("quantity", 0)
            buy_price = asset.get("buy_price", 0)
            if asset_type == "Stock/ETF":
                ticker = get_final_ticker(asset.get("ticker"), "Stock/ETF", asset.get("market"))
            else:
                ticker = asset.get("ticker")
            try:
                tkr = yf.Ticker(ticker)
                hist = tkr.history(period="1d")
                if hist.empty:
                    continue
                info = tkr.info
                current_price = info.get("regularMarketPrice") or hist["Close"].iloc[-1]
                conv_rate = get_conversion_factor(asset_currency, current_rates)
                price_eur = current_price * conv_rate
                buy_price_eur = buy_price * conv_rate
                value = current_price * quantity
                value_eur = price_eur * quantity
                invested_eur = buy_price_eur * quantity
                if asset_type == "Stock/ETF":
                    total_invested_stocks += invested_eur
                    total_value_stocks += value_eur
                else:
                    total_invested_crypto += invested_eur
                    total_value_crypto += value_eur
            except Exception:
                continue
        
        cash_amount = cash.get("amount", 0.0)
        cash_currency = cash.get("currency", "EUR")
        cash_conv_rate = get_conversion_factor(cash_currency, current_rates)
        cash_value_eur = cash_amount * cash_conv_rate
        
        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stocks_invested_eur": total_invested_stocks,
            "stocks_value_eur": total_value_stocks,
            "stocks_gain_eur": round(total_value_stocks - total_invested_stocks, 2),
            "crypto_invested_eur": total_invested_crypto,
            "crypto_value_eur": total_value_crypto,
            "crypto_gain_eur": round(total_value_crypto - total_invested_crypto, 2),
            "cash_value_eur": round(cash_value_eur, 2),
            "total_invested_eur": total_invested_stocks + total_invested_crypto,
            "total_value_eur": total_value_stocks + total_value_crypto + cash_value_eur,
            "total_gain_eur": round((total_value_stocks + total_value_crypto) - (total_invested_stocks + total_invested_crypto), 2)
        }
        
        history = load_history()
        history.append(snapshot)
        save_history(history)
        st.success("Snapshot saved!")
    
    history = load_history()
    if history:
        df_history = pd.DataFrame(history)
        df_history["date"] = pd.to_datetime(df_history["date"], format="%Y-%m-%d %H:%M", errors="coerce")
        df_history.sort_values(by="date", ascending=False, inplace=True)
        
        df_history["date_table"] = df_history["date"].dt.strftime("%Y-%m-%d")
        df_display = df_history.drop(columns=["date"]).copy()
        df_display.set_index("date_table", inplace=True)
        
        numeric_cols = df_display.select_dtypes(include=["float64", "int64"]).columns
        df_display[numeric_cols] = df_display[numeric_cols].round(2)
        
        st.dataframe(df_display)
    else:
        st.info("No history snapshots yet.")
    
    # --- Manage History Section (Deletion Only) ---
    with st.expander("Manage History"):
        st.write("Delete snapshots:")
        history = load_history()
        if history:
            for idx, snapshot in enumerate(history):
                st.write(f"Snapshot Date: {snapshot.get('date')}, Total Value: €{snapshot.get('total_value_eur', 0.0):.2f}")
                if st.button("Delete Snapshot", key=f"del_snapshot_{idx}"):
                    del history[idx]
                    save_history(history)
                    st.success("Snapshot deleted.")
                    st.experimental_rerun()
        else:
            st.info("No snapshots available.")
