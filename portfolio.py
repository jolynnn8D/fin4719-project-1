import streamlit as st
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
import numpy as np

# Parameters
themes = {
    "ESG and Green Energy": ["ICLN", "PBW", "TAN"],
    "Tech": ["BOTZ", "ARKW", "KWEB"],
    "Value": ["AAPL", "MSFT", "AMZN"],
}

startdate = "2020-01-01"
enddate = "2021-12-31"

def compute_theme(theme):
    ticker_list = themes[theme]
    returns = get_returns(ticker_list, startdate, enddate)
    weights = calculate_mvp(returns)

    weight_dict = {}
    for (index, ticker) in enumerate(ticker_list):
        weight_dict[ticker] = weights[index][0]

    st.session_state.weights = weight_dict


def display():

    # Initialize values
    if "theme" not in st.session_state:
        st.session_state.theme = "ESG and Green Energy"
    if "weights" not in st.session_state:
        compute_theme(st.session_state.theme)
    if "risk" not in st.session_state:
        st.session_state.risk = "Low"

    st.header("Portfolio selection")
    
    portfolio_theme = st.selectbox("Pick a portfolio theme", themes.keys(), key="theme")  
    risk_slider = st.select_slider("What is your risk appetite?", options=["Low", "Medium", "High"], key="risk")
    st.button("Compute", on_click=compute_theme, args=(st.session_state.theme,))

    st.subheader("Portfolio weights")
    st.markdown("The portfolio allocation below is the **minimum variance portfolio** based on the portfolio theme that you chose. Feel free to change the weights.")
    

    for (ticker, weight) in st.session_state.weights.items():
        st.markdown(f"**{ticker}**")
        st.number_input("Weight", value=weight, key=f"{ticker}_weight")

def get_returns(tickers, startdate, enddate):
    data = yf.download(tickers, start=startdate, end=enddate)
    returns_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    returns_data = returns_data.dropna()
    returns_data = returns_data.reset_index()
    return returns_data

def calculate_mvp(returns):
    varcov = returns.cov()
    num_stocks = len(returns.columns) - 1 # -1 if date column is present, else remove
    ones_arr = np.array([[1]] * num_stocks)
    varcov_inv = np.linalg.pinv(varcov.values)
    weights = np.matmul(varcov_inv, ones_arr) / np.matmul(np.matrix.transpose(ones_arr), np.matmul(varcov_inv, ones_arr))

    return weights
