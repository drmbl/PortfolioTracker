import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import requests
from github import Github

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
    except Exception:
        pass
    return None

def get_conversion_factor(currency, rates):
    if currency.upper() in ["EURO", "EUR"]:
        return 1.0
    try:
        rate = rates.get(currency.upper())
        return 1.0 / rate if rate else 1.0
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
    return ticker

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("Portfolio Tracker")

# -----------------------------
# Sidebar: Add New Asset and Cash Input
# -----------------------------
# -----------------------------
# Sidebar: Add New Asset and Cash
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
    currency    = st.selectbox("Asset Currency", ["USD", "GBP", "EUR"])
    buy_price   = st.number_input("Buy Price (optional)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
    quantity    = st.number_input("Quantity",             min_value=0.0, step=0.01, format="%.2f")
    submit      = st.form_submit_button("Add Asset")

    if submit:
        if ticker_input.strip() == "" or quantity <= 0:
            st.sidebar.error("Please enter a valid ticker and a quantity > 0.")
        else:
            # 1) Save locally
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
                    "ticker":      ticker_upper,
                    "asset_type":  asset_type,
                    "market":      market,
                    "currency":    currency,
                    "buy_price":   buy_price,
                    "quantity":    quantity,
                    "added_date":  datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                save_portfolio(portfolio)
                st.sidebar.success(f"Asset '{ticker_upper}' added!")

                # 2) PUSH portfolio.json TO GITHUB
                try:
                    with open("portfolio.json", "r") as f:
                        new_portfolio = f.read()
                    gh       = Github(st.secrets["GITHUB_TOKEN"])
                    repo     = gh.get_repo("drmbl/PortfolioTracker")
                    contents = repo.get_contents("portfolio.json", ref="main")
                    repo.update_file(
                        path    = contents.path,
                        message = f"Auto‑update portfolio.json after adding {ticker_upper}",
                        content = new_portfolio,
                        sha     = contents.sha,
                        branch  = "main",
                    )
                    st.sidebar.success("portfolio.json pushed to GitHub ✅")
                except Exception as e:
                    st.sidebar.error(f"Failed to push portfolio.json: {e}")

# <-- the 'with st.sidebar.form("asset_form")' block ends here -->

# Now re‑open the sidebar for cash management at top level:
st.sidebar.header("Manage Cash")
with st.sidebar.form("cash_form"):
    cash_amount   = st.number_input(
        "Cash Amount",
        min_value=0.0,
        step=0.01,
        format="%.2f",
        value=load_cash().get("amount", 0.0)
    )
    cash_currency = st.selectbox(
        "Cash Currency",
        ["EUR", "USD", "GBP"],
        index=["EUR", "USD", "GBP"].index(load_cash().get("currency", "EUR"))
    )
    cash_submit = st.form_submit_button("Update Cash")
    if cash_submit:
        save_cash({"amount": cash_amount, "currency": cash_currency})
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
    total_stock_invested_eur = 0.0
    total_stock_value_eur    = 0.0
    total_stock_gain_eur     = 0.0

    # for 24h change calculation
    prev_stock_value_eur     = 0.0

    for asset in portfolio:
        if asset.get("asset_type") == "Stock/ETF":
            asset_currency = asset.get("currency", "USD")
            quantity       = asset.get("quantity", 0)
            buy_price      = asset.get("buy_price", 0)
            ticker         = get_final_ticker(asset.get("ticker", ""), "Stock/ETF", asset.get("market"))
            try:
                stock = yf.Ticker(ticker)
                hist  = stock.history(period="2d")
                if hist.shape[0] < 2:
                    continue
                close_prices      = hist["Close"].values
                prev_close        = close_prices[-2]
                current_price     = close_prices[-1]
                info              = stock.info
                # fallback to info price if needed
                if not current_price or current_price == 0:
                    current_price = info.get("regularMarketPrice") or current_price
                conv_rate         = get_conversion_factor(asset_currency, current_rates)
                invested_eur      = buy_price * conv_rate * quantity
                today_value_eur   = current_price * conv_rate * quantity
                yesterday_value_eur = prev_close * conv_rate * quantity
                gain_eur          = today_value_eur - invested_eur

                raw_sector = info.get("sector", "")
                long_name  = (info.get("longName") or "").lower()
                if any(w in long_name for w in ["etf", "index", "s&p", "fund"]) or raw_sector == "":
                    sector = "Index Fund"
                else:
                    sector = raw_sector or "Unknown"

                stock_data.append({
                    "Ticker": ticker,
                    "Sector": sector,
                    "Quantity": quantity,
                    "Buy Price": buy_price,
                    "Current Price": round(current_price, 2),
                    "Price (EUR)": round(current_price * conv_rate, 2),
                    "Value": round(current_price * quantity, 2),
                    "Value (EUR)": round(today_value_eur, 2),
                    "Gain/Loss (EUR)": round(gain_eur, 2),
                    "Currency": asset_currency
                })

                total_stock_invested_eur += invested_eur
                total_stock_value_eur    += today_value_eur
                total_stock_gain_eur     += gain_eur
                prev_stock_value_eur     += yesterday_value_eur

            except Exception:
                continue

    if stock_data:
        df_stocks = pd.DataFrame(stock_data)
        st.dataframe(df_stocks)
        # compute 24h percent change
        if prev_stock_value_eur > 0:
            pct_stock = 100 * (total_stock_value_eur - prev_stock_value_eur) / prev_stock_value_eur
        else:
            pct_stock = 0.0
        # determine color
        if pct_stock > 0:
            color_stock = "green"
            sign_stock = "+"
        elif pct_stock < 0:
            color_stock = "red"
            sign_stock = ""
        else:
            color_stock = "orange"
            sign_stock = ""
        col1, col2 = st.columns(2)
        col1.metric("Total Stocks Gain/Loss (EUR)", f"€{total_stock_gain_eur:.2f}")
        col2.markdown(
            f"Total Stocks Value (EUR): €{total_stock_value_eur:.2f} "
            f"<span style='color:{color_stock}'>({sign_stock}{pct_stock:.2f}% 24h)</span>",
            unsafe_allow_html=True
        )
    else:
        st.info("No Stocks/ETFs to display.")

    st.markdown("---")

    # --- Crypto Section ---
    st.subheader("Crypto")
    crypto_data = []
    total_crypto_invested_eur = 0.0
    total_crypto_value_eur    = 0.0
    total_crypto_gain_eur     = 0.0

    prev_crypto_value_eur     = 0.0

    for asset in portfolio:
        if asset.get("asset_type") == "Crypto":
            asset_currency = asset.get("currency", "USD")
            quantity       = asset.get("quantity", 0)
            buy_price      = asset.get("buy_price", 0)
            ticker         = asset.get("ticker")
            try:
                crypto = yf.Ticker(ticker)
                hist   = crypto.history(period="2d")
                if hist.shape[0] < 2:
                    continue
                close_prices        = hist["Close"].values
                prev_close_crypto   = close_prices[-2]
                current_price       = close_prices[-1]
                info                = crypto.info
                if not current_price or current_price == 0:
                    current_price = info.get("regularMarketPrice") or current_price
                conv_rate           = get_conversion_factor(asset_currency, current_rates)
                invested_eur        = buy_price * conv_rate * quantity
                today_value_eur     = current_price * conv_rate * quantity
                yesterday_value_eur_crypto = prev_close_crypto * conv_rate * quantity
                gain_eur            = today_value_eur - invested_eur

                crypto_data.append({
                    "Ticker": ticker,
                    "Quantity": quantity,
                    "Buy Price": buy_price,
                    "Current Price": round(current_price, 2),
                    "Price (EUR)": round(current_price * conv_rate, 2),
                    "Value": round(current_price * quantity, 2),
                    "Value (EUR)": round(today_value_eur, 2),
                    "Gain/Loss (EUR)": round(gain_eur, 2),
                    "Currency": asset_currency
                })

                total_crypto_invested_eur += invested_eur
                total_crypto_value_eur    += today_value_eur
                total_crypto_gain_eur     += gain_eur
                prev_crypto_value_eur     += yesterday_value_eur_crypto

            except Exception:
                continue

    if crypto_data:
        df_crypto = pd.DataFrame(crypto_data)
        st.dataframe(df_crypto)
        if prev_crypto_value_eur > 0:
            pct_crypto = 100 * (total_crypto_value_eur - prev_crypto_value_eur) / prev_crypto_value_eur
        else:
            pct_crypto = 0.0
        if pct_crypto > 0:
            color_crypto = "green"
            sign_crypto = "+"
        elif pct_crypto < 0:
            color_crypto = "red"
            sign_crypto = ""
        else:
            color_crypto = "orange"
            sign_crypto = ""
        col1, col2 = st.columns(2)
        col1.metric("Total Crypto Gain/Loss (EUR)", f"€{total_crypto_gain_eur:.2f}")
        col2.markdown(
            f"**Total Crypto Value (EUR): €{total_crypto_value_eur:.2f} "
            f"<span style='color:{color_crypto}'>({sign_crypto}{pct_crypto:.2f}% 24h)</span>**",
            unsafe_allow_html=True
        )
    else:
        st.info("No Crypto to display.")

    # --- Cash Section ---
    st.markdown("---")
    st.subheader("Cash")
    cash_amount    = cash.get("amount", 0.0)
    cash_currency  = cash.get("currency", "EUR")
    conv_rate      = get_conversion_factor(cash_currency, current_rates)
    cash_value_eur = cash_amount * conv_rate
    st.metric("Cash", f"€{cash_value_eur:.2f}")

    # --- Total Portfolio Section ---
    total_portfolio_value = total_stock_value_eur + total_crypto_value_eur + cash_value_eur
    # total gain excludes cash, as before
    total_portfolio_gain  = (total_stock_value_eur - total_stock_invested_eur) + \
                            (total_crypto_value_eur - total_crypto_invested_eur)
    # 24h change for portfolio excludes cash
    prev_portfolio_asset_value = prev_stock_value_eur + prev_crypto_value_eur
    if prev_portfolio_asset_value > 0:
        pct_total = 100 * ((total_stock_value_eur + total_crypto_value_eur) - prev_portfolio_asset_value) / prev_portfolio_asset_value
    else:
        pct_total = 0.0
    if pct_total > 0:
        color_total = "green"
        sign_total = "+"
    elif pct_total < 0:
        color_total = "red"
        sign_total = ""
    else:
        color_total = "orange"
        sign_total = ""

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Total Portfolio Gain/Loss (EUR)", f"€{total_portfolio_gain:.2f}")
    col2.markdown(
        f"**Total Portfolio Value (EUR): €{total_portfolio_value:.2f} "
        f"<span style='color:{color_total}'>({sign_total}{pct_total:.2f}% 24h)</span>**",
        unsafe_allow_html=True
    )

    # --- Store breakdowns for exact history snapshot ---
    st.session_state["latest_stocks_invested_eur"] = round(total_stock_invested_eur, 2)
    st.session_state["latest_stocks_value_eur"]    = round(total_stock_value_eur,    2)
    st.session_state["latest_stocks_gain_eur"]     = round(total_stock_gain_eur,     2)
    st.session_state["latest_crypto_invested_eur"] = round(total_crypto_invested_eur, 2)
    st.session_state["latest_crypto_value_eur"]    = round(total_crypto_value_eur,    2)
    st.session_state["latest_crypto_gain_eur"]     = round(total_crypto_gain_eur,     2)
    st.session_state["latest_cash_value_eur"]      = round(cash_value_eur,             2)
    st.session_state["latest_total_value_eur"]     = round(total_portfolio_value,     2)
    st.session_state["latest_total_gain_eur"]      = round(total_portfolio_gain,      2)

    # --- Portfolio Value Chart Expander ---
    with st.expander("Show Portfolio Value Chart"):
        history = load_history()
        if history:
            df_history = pd.DataFrame(history)
            df_history["date"] = pd.to_datetime(
                df_history["date"], format="%Y-%m-%d %H:%M", errors="coerce"
            )
            df_history.sort_values(by="date", inplace=True)
            fig = px.line(df_history, x="date", y="total_value_eur", title="Portfolio Value History")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No history data available for chart.")

    # --- Manage Assets Section ---
    with st.expander("Manage Assets (Stocks/ETFs/Crypto)"):
        st.write("Each asset is displayed as **Ticker – Edit – Delete**.")
        assets = list(enumerate(portfolio))
        items_per_row = 4
        for row_start in range(0, len(assets), items_per_row):
            row_slice = assets[row_start: row_start + items_per_row]
            row_cols = st.columns(len(row_slice) * 3)
            for j, (idx, asset) in enumerate(row_slice):
                col_ticker = row_cols[3*j]
                col_edit   = row_cols[3*j+1]
                col_delete = row_cols[3*j+2]
                col_ticker.write(f"**{asset['ticker']}**")
                if col_edit.button("✏️", key=f"edit_{idx}"):
                    st.session_state.edit_asset = idx
                if col_delete.button("❌", key=f"delete_{idx}"):
    # 1) Remove locally & save
                    del portfolio[idx]
                    save_portfolio(portfolio)
                    st.success("Asset deleted locally!")

    # 2) Push portfolio.json to GitHub
    try:
        with open("portfolio.json", "r") as f:
            new_portfolio = f.read()
        gh   = Github(st.secrets["GITHUB_TOKEN"])
        repo = gh.get_repo("drmbl/PortfolioTracker")
        contents = repo.get_contents("portfolio.json", ref="main")
        repo.update_file(
            path    = contents.path,
            message = f"Auto‑update portfolio.json after deletion {idx}",
            content = new_portfolio,
            sha     = contents.sha,
            branch  = "main",
        )
        st.success("portfolio.json pushed to GitHub ✅")
    except Exception as e:
        st.error(f"Failed to push portfolio.json: {e}")

    # 3) Rerun so UI updates
    if st.session_state.get("edit_asset") == idx:
                    with st.form(key=f"edit_form_{idx}"):
                        new_buy_price = st.number_input(
                            "New Buy Price", min_value=0.0, step=0.01, format="%.2f",
                            value=asset.get("buy_price", 0.0)
                        )
                        if asset.get("asset_type") == "Crypto":
                            new_quantity = st.number_input(
                                "New Quantity", min_value=0.0, step=0.00000001, format="%.8f",
                                value=asset.get("quantity", 0.0)
                            )
                        else:
                            new_quantity = st.number_input(
                                "New Quantity", min_value=0.0, step=0.01, format="%.2f",
                                value=asset.get("quantity", 0.0)
                            )
                        if st.form_submit_button("Save"):
                            asset["buy_price"] = new_buy_price
                            asset["quantity"]  = new_quantity
                            asset["added_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
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
        # pull every breakdown back out of session state:
        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stocks_invested_eur": st.session_state.get("latest_stocks_invested_eur"),
            "stocks_value_eur":    st.session_state.get("latest_stocks_value_eur"),
            "stocks_gain_eur":     st.session_state.get("latest_stocks_gain_eur"),
            "crypto_invested_eur": st.session_state.get("latest_crypto_invested_eur"),
            "crypto_value_eur":    st.session_state.get("latest_crypto_value_eur"),
            "crypto_gain_eur":     st.session_state.get("latest_crypto_gain_eur"),
            "cash_value_eur":      st.session_state.get("latest_cash_value_eur"),
            "total_invested_eur":  round(
                st.session_state.get("latest_stocks_invested_eur", 0.0)
              + st.session_state.get("latest_crypto_invested_eur", 0.0), 2
            ),
            "total_value_eur":     st.session_state.get("latest_total_value_eur"),
            "total_gain_eur":      st.session_state.get("latest_total_gain_eur"),
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
        st.dataframe(df_display)

        # --- Download Button for history.json ---
        st.download_button(
            label="Download history.json",
            data=json.dumps(history, indent=4),
            file_name="history.json",
            mime="application/json"
        )
    else:
        st.info("No history snapshots yet.")

# --- Manage History Section (Deletion Only) ---
with st.expander("Manage History"):
    st.write("Delete snapshots:")
    if history:
        for idx, snapshot in enumerate(history):
            # two columns: text on left, delete button on right
            col_text, col_button = st.columns([9, 1])
            col_text.write(f"{snapshot['date']}: €{snapshot['total_value_eur']:.2f}")

            # this button click is the *only* place we call experimental_rerun()
            if col_button.button("❌", key=f"del_snapshot_{idx}"):
                # 1) Remove locally and save
                del history[idx]
                save_history(history)
                st.success("Snapshot deleted locally!")

                # 2) DEBUG: report token presence
                has_token = "GITHUB_TOKEN" in st.secrets and bool(st.secrets["GITHUB_TOKEN"])
                st.write("• GITHUB_TOKEN present?", has_token)
                if not has_token:
                    st.error("❌ No GITHUB_TOKEN found in Streamlit Secrets!")
                else:
                    # 3) Push the updated history.json to GitHub
                    try:
                        # Read updated file
                        with open("history.json", "r") as f:
                            new_content = f.read()
                        st.write(f"• Read history.json ({len(new_content)} bytes)")

                        # Auth & repo
                        gh   = Github(st.secrets["GITHUB_TOKEN"])
                        user = gh.get_user().login
                        st.write(f"• Authenticated as GitHub user: {user}")

                        repo     = gh.get_repo("drmbl/PortfolioTracker")
                        st.write("• Repo found:", repo.full_name)

                        contents = repo.get_contents("history.json", ref="main")
                        st.write("• Current history.json SHA:", contents.sha)

                        # Commit change
                        commit = repo.update_file(
                            path    = contents.path,
                            message = f"Auto‑delete history entry {snapshot['date']}",
                            content = new_content,
                            sha     = contents.sha,
                            branch  = "main",
                        )
                        st.success("✅ history.json deletion pushed to GitHub!")
                        st.write("• New commit SHA:", commit["commit"].sha)
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Failed to push deletion to GitHub: {e}")
                        st.text(traceback.format_exc())