# ============================================================
# LIVE NIFTY OPTIONS PRICING & GREEKS ENGINE
# Streamlit Version
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime
import pytz

from scipy.stats import norm
from pnsea import NSE


# ============================================================
# TIMEZONE
# ============================================================

IST = pytz.timezone("Asia/Kolkata")


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NIFTY Options Pricing & Greeks Engine",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Live NIFTY Options Pricing & Greeks Engine")

st.caption("Black-Scholes Pricing | Greeks | IV-HV | Market vs BS Price")


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Settings")

r = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.25
)
r = r / 100

NIFTY_LOT_SIZE = st.sidebar.number_input(
    "NIFTY Lot Size",
    min_value=1,
    max_value=1000,
    value=65,
    step=1
)

refresh_button = st.sidebar.button("🔄 Refresh Live Data")


# ============================================================
# CONNECT TO NSE
# ============================================================

@st.cache_resource
def initialize_nse():
    return NSE()


# IMPORTANT: create NSE object here, before get_live_nifty_data()
nse = initialize_nse()


# ============================================================
# HISTORICAL VOLATILITY
# ============================================================

@st.cache_data(ttl=300)
def get_historical_volatility():
    nifty_history = yf.download(
        "^NSEI",
        period="6mo",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if nifty_history.empty:
        return np.nan

    if isinstance(nifty_history.columns, pd.MultiIndex):
        close_prices = nifty_history["Close"].iloc[:, 0]
    else:
        close_prices = nifty_history["Close"]

    close_prices = close_prices.dropna()

    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    daily_volatility = log_returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)

    return annual_volatility * 100


# ============================================================
# FETCH LIVE NSE OPTION CHAIN
# ============================================================

@st.cache_data(ttl=25)
def get_live_nifty_data():
    try:
        nifty = nse.options.option_chain("NIFTY")

        option_chain_data = nifty[0].copy()
        expiry_dates = nifty[1]
        nifty_spot = nifty[2]

        return option_chain_data, expiry_dates, nifty_spot

    except Exception as e:
        st.error("Unable to fetch NSE option chain data")
        st.exception(e)
        # Re-raise so the caller's try/except handles it and stops
        # instead of silently returning None and causing an
        # "cannot unpack NoneType" error downstream.
        raise


# ============================================================
# FIND NEAREST EXPIRY
# ============================================================

def get_nearest_expiry(expiry_dates):
    now = datetime.now(IST)
    valid_expiries = []

    for expiry in expiry_dates:
        try:
            expiry_date = datetime.strptime(expiry, "%d-%b-%Y").date()
            if expiry_date >= now.date():
                valid_expiries.append(expiry_date)
        except Exception:
            continue

    if len(valid_expiries) == 0:
        return None

    return min(valid_expiries)


# ============================================================
# CALCULATE TIME TO EXPIRY
# ============================================================

def calculate_time_to_expiry(nearest_expiry):
    now = datetime.now(IST)

    expiry_datetime = IST.localize(
        datetime.combine(nearest_expiry, datetime.min.time()).replace(
            hour=15, minute=30, second=0
        )
    )

    seconds_to_expiry = (expiry_datetime - now).total_seconds()

    if seconds_to_expiry <= 0:
        return 0, 0

    T = seconds_to_expiry / (365 * 24 * 60 * 60)
    days_to_expiry = seconds_to_expiry / (24 * 60 * 60)

    return T, days_to_expiry


# ============================================================
# BLACK-SCHOLES CALL PRICE
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    d1 = (
        np.log(S / K) + (r + (sigma ** 2) / 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return price, d1, d2


# ============================================================
# BLACK-SCHOLES PUT PRICE
# ============================================================

def black_scholes_put(S, K, T, r, sigma):
    d1 = (
        np.log(S / K) + (r + (sigma ** 2) / 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price, d1, d2


# ============================================================
# CALL GREEKS
# ============================================================

def call_greeks(S, K, T, r, sigma, d1, d2):
    delta = norm.cdf(d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    vega = S * norm.pdf(d1) * np.sqrt(T)

    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, theta, vega, rho


# ============================================================
# PUT GREEKS
# ============================================================

def put_greeks(S, K, T, r, sigma, d1, d2):
    delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )

    vega = S * norm.pdf(d1) * np.sqrt(T)

    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return delta, gamma, theta, vega, rho


# ============================================================
# MAIN ENGINE
# ============================================================

try:
    with st.spinner("Fetching live NIFTY option-chain data..."):
        option_chain_data, expiry_dates, nifty_spot = get_live_nifty_data()

    # ------------------------------------------------------
    # NIFTY SPOT
    # ------------------------------------------------------
    S = float(nifty_spot)

    # ------------------------------------------------------
    # CURRENT IST TIME
    # ------------------------------------------------------
    now = datetime.now(IST)

    # ------------------------------------------------------
    # FIND NEAREST EXPIRY
    # ------------------------------------------------------
    nearest_expiry = get_nearest_expiry(expiry_dates)

    if nearest_expiry is None:
        st.error("No valid expiry found.")
        st.stop()

    # ------------------------------------------------------
    # TIME TO EXPIRY
    # ------------------------------------------------------
    T, days_to_expiry = calculate_time_to_expiry(nearest_expiry)

    if T <= 0:
        st.error("Invalid time to expiry.")
        st.stop()

    # ------------------------------------------------------
    # ATM STRIKE SELECTION
    # ------------------------------------------------------
    option_chain_data["distance_from_spot"] = abs(
        option_chain_data["strikePrice"] - S
    )

    atm_row = option_chain_data.loc[
        option_chain_data["distance_from_spot"].idxmin()
    ]

    K = float(atm_row["strikePrice"])

    # ------------------------------------------------------
    # MARKET OPTION PRICE
    # ------------------------------------------------------
    call_price = float(atm_row["CE_lastPrice"])
    put_price = float(atm_row["PE_lastPrice"])

    # ------------------------------------------------------
    # IMPLIED VOLATILITY
    # ------------------------------------------------------
    call_iv = float(atm_row["CE_impliedVolatility"])
    put_iv = float(atm_row["PE_impliedVolatility"])

    # Convert IV % into decimal
    call_sigma = call_iv / 100
    put_sigma = put_iv / 100

    # ------------------------------------------------------
    # HISTORICAL VOLATILITY
    # ------------------------------------------------------
    hv_percent = get_historical_volatility()

    # ------------------------------------------------------
    # BLACK-SCHOLES CALCULATION
    # ------------------------------------------------------
    call_bs_price, d1_call, d2_call = black_scholes_call(
        S, K, T, r, call_sigma
    )

    put_bs_price, d1_put, d2_put = black_scholes_put(
        S, K, T, r, put_sigma
    )

    # ------------------------------------------------------
    # GREEKS
    # ------------------------------------------------------
    call_delta, call_gamma, call_theta, call_vega, call_rho = call_greeks(
        S, K, T, r, call_sigma, d1_call, d2_call
    )

    put_delta, put_gamma, put_theta, put_vega, put_rho = put_greeks(
        S, K, T, r, put_sigma, d1_put, d2_put
    )

    # ------------------------------------------------------
    # THETA PER DAY
    # ------------------------------------------------------
    call_theta_daily = call_theta / 365
    put_theta_daily = put_theta / 365

    # ------------------------------------------------------
    # THETA PER LOT PER DAY
    # ------------------------------------------------------
    call_theta_lot_daily = call_theta_daily * NIFTY_LOT_SIZE
    put_theta_lot_daily = put_theta_daily * NIFTY_LOT_SIZE

    # ------------------------------------------------------
    # VEGA PER 1% IV CHANGE
    # ------------------------------------------------------
    call_vega_1pct = call_vega / 100
    put_vega_1pct = put_vega / 100

    # ------------------------------------------------------
    # VEGA PER LOT
    # ------------------------------------------------------
    call_vega_lot = call_vega_1pct * NIFTY_LOT_SIZE
    put_vega_lot = put_vega_1pct * NIFTY_LOT_SIZE

    # ------------------------------------------------------
    # RHO PER 1% RATE CHANGE
    # ------------------------------------------------------
    call_rho_1pct = call_rho / 100
    put_rho_1pct = put_rho / 100

    # ------------------------------------------------------
    # IV - HISTORICAL VOLATILITY
    # ------------------------------------------------------
    call_iv_hv = call_iv - hv_percent
    put_iv_hv = put_iv - hv_percent

    # ------------------------------------------------------
    # MARKET PRICE VS BS PRICE
    # ------------------------------------------------------
    call_difference = call_price - call_bs_price
    put_difference = put_price - put_bs_price

    # ------------------------------------------------------
    # PUT-CALL PARITY
    # ------------------------------------------------------
    parity_call = call_price - put_price
    parity_theoretical = S - K * np.exp(-r * T)
    parity_difference = parity_call - parity_theoretical

    # ------------------------------------------------------
    # LAST UPDATE TIME
    # ------------------------------------------------------
    st.success(f"Last Updated (IST): {now.strftime('%d-%b-%Y %H:%M:%S')}")

    # ------------------------------------------------------
    # MARKET INFORMATION
    # ------------------------------------------------------
    st.subheader("📊 Market Information")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("NIFTY Spot", f"{S:,.2f}")

    with col2:
        st.metric("ATM Strike", f"{K:,.0f}")

    with col3:
        st.metric("Nearest Expiry", nearest_expiry.strftime("%d-%b-%Y"))

    with col4:
        st.metric("Days To Expiry", f"{days_to_expiry:.2f}")

    with col5:
        st.metric("Historical Volatility", f"{hv_percent:.2f}%")

    # ------------------------------------------------------
    # OPTION DATA
    # ------------------------------------------------------
    st.subheader("💰 Option Data")

    option_data = pd.DataFrame({
        "Metric": [
            "Market Price",
            "IV (%)",
            "Black-Scholes Price",
            "Market - BS Difference"
        ],
        "CALL": [call_price, call_iv, call_bs_price, call_difference],
        "PUT": [put_price, put_iv, put_bs_price, put_difference]
    })

    st.dataframe(option_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # GREEKS TABLE
    # ------------------------------------------------------
    st.subheader("📐 Greeks")

    greeks_data = pd.DataFrame({
        "Greek": [
            "Delta",
            "Gamma",
            "Theta / Day",
            "Theta / Lot / Day",
            "Vega / 1% IV",
            "Vega / Lot",
            "Rho / 1% Rate"
        ],
        "CALL": [
            call_delta, call_gamma, call_theta_daily,
            call_theta_lot_daily, call_vega_1pct, call_vega_lot,
            call_rho_1pct
        ],
        "PUT": [
            put_delta, put_gamma, put_theta_daily,
            put_theta_lot_daily, put_vega_1pct, put_vega_lot,
            put_rho_1pct
        ]
    })

    st.dataframe(greeks_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # VOLATILITY ANALYSIS
    # ------------------------------------------------------
    st.subheader("📈 Volatility Analysis")

    volatility_data = pd.DataFrame({
        "Metric": [
            "Historical Volatility",
            "Call IV",
            "Put IV",
            "Call IV - HV",
            "Put IV - HV"
        ],
        "Value": [hv_percent, call_iv, put_iv, call_iv_hv, put_iv_hv]
    })

    st.dataframe(volatility_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # BLACK-SCHOLES PARAMETERS
    # ------------------------------------------------------
    st.subheader("🧮 Black-Scholes Parameters")

    bs_data = pd.DataFrame({
        "Parameter": [
            "Spot Price (S)",
            "Strike Price (K)",
            "Time To Expiry (T)",
            "Risk Free Rate",
            "Call IV",
            "Put IV",
            "Call d1",
            "Call d2",
            "Put d1",
            "Put d2"
        ],
        "Value": [
            S, K, T, r * 100, call_iv, put_iv,
            d1_call, d2_call, d1_put, d2_put
        ]
    })

    st.dataframe(bs_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # PUT-CALL PARITY
    # ------------------------------------------------------
    st.subheader("⚖️ Put-Call Parity")

    parity_data = pd.DataFrame({
        "Metric": ["C - P", "S - K × exp(-rT)", "Difference"],
        "Value": [parity_call, parity_theoretical, parity_difference]
    })

    st.dataframe(parity_data, use_container_width=True, hide_index=True)

    # ------------------------------------------------------
    # OPTION CHAIN
    # ------------------------------------------------------
    st.subheader("📋 NIFTY Option Chain")

    display_columns = [
        "strikePrice",
        "CE_lastPrice",
        "CE_impliedVolatility",
        "CE_openInterest",
        "CE_totalTradedVolume",
        "PE_lastPrice",
        "PE_impliedVolatility",
        "PE_openInterest",
        "PE_totalTradedVolume"
    ]

    available_columns = [
        col for col in display_columns
        if col in option_chain_data.columns
    ]

    option_chain_display = option_chain_data[available_columns]

    st.dataframe(
        option_chain_display, use_container_width=True, hide_index=True
    )

    # ------------------------------------------------------
    # DOWNLOAD CSV
    # ------------------------------------------------------
    csv_data = option_chain_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Option Chain CSV",
        data=csv_data,
        file_name="NIFTY_option_chain.csv",
        mime="text/csv"
    )

    # ------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------
    st.divider()

    st.caption("NIFTY Options Pricing & Greeks Engine | Black-Scholes Model")
    st.caption("Market Data Source: pnsea NSE Option Chain")


# ============================================================
# ERROR HANDLING
# ============================================================

except Exception as e:
    st.error("Error while fetching or calculating data.")
    st.exception(e)
    st.stop()
