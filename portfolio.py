import streamlit as st
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
import numpy as np

# Parameters
themes = {
    "ESG and Green Energy": ["ICLN", "PBW", "TAN"],
    "Tech": ["BOTZ", "ARKW", "KWEB"],
    "Value": ["AAPL", "MSFT", "AMZN", "NFLX"]
}

startdate = "2020-01-01"
enddate = "2021-12-31"

def compute_theme(theme, risk):
    ticker_list = themes[theme]
    returns = get_returns(ticker_list, startdate, enddate)
    if risk == "Low":
        results = calculate_mvp(returns)
        st.session_state.portfolio_type = "minimum variance portfolio"
    elif risk == "High":
        results = maximize_sharpe_ratio(returns)
        st.session_state.portfolio_type = "maximum sharpe ratio portfolio"
    weights = results[0]
    expected_return = results[1]
    expected_variance = results[2]
    sharpe = expected_return / np.sqrt(expected_variance)
    
    weight_dict = {}
    for (index, ticker) in enumerate(ticker_list):
        weight_dict[ticker] = weights[index]

    st.session_state.weights = weight_dict
    st.session_state.expreturn = expected_return
    st.session_state.expvar = expected_variance
    

def display():

    # Initialize values
    if "theme" not in st.session_state:
        st.session_state.theme = "ESG and Green Energy"
    if "risk" not in st.session_state:
        st.session_state.risk = "Low"
    if "weights" not in st.session_state or \
        "expreturn" not in st.session_state or \
        "expvar" not in st.session_state or \
        "portfolio_type" not in st.session_state:
        compute_theme(st.session_state.theme, st.session_state.risk)
    

    st.header("Portfolio selection")
    
    portfolio_theme = st.sidebar.selectbox("Pick a portfolio theme", themes.keys(), key="theme")  
    risk_slider = st.sidebar.select_slider("What is your risk appetite?", options=["Low", "High"], key="risk")
    st.sidebar.button("Compute", on_click=compute_theme, args=(st.session_state.theme, st.session_state.risk))

    st.subheader("Portfolio weights")
    st.markdown(f"The portfolio allocation below is the **{st.session_state.portfolio_type}** based on the portfolio theme that you chose. Feel free to change the weights.")
    st.markdown(f"##### **Expected Return: {round(st.session_state.expreturn*100, 2)}%**")
    st.markdown(f"##### **Expected Variance: {round(st.session_state.expvar*100, 2)}%**")
    for (ticker, weight) in st.session_state.weights.items():
        st.markdown(f"**{ticker}**")
        st.number_input("Weight", value=weight, key=f"{ticker}_weight")

def get_returns(tickers, startdate, enddate):
    data = yf.download(tickers, start=startdate, end=enddate)
    returns_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    returns_data = returns_data.dropna()
    returns_data = returns_data.reset_index()
    returns_data = returns_data.drop(columns=["Date"])
    return returns_data

def calculate_mvp(returns):
    varcov = returns.cov()
    num_stocks = len(returns.columns) # -1 if date column is present, else remove
    ones_arr = np.array([[1]] * num_stocks)
    varcov_inv = np.linalg.pinv(varcov.values)
    weights = np.matmul(varcov_inv, ones_arr) / np.matmul(np.matrix.transpose(ones_arr), np.matmul(varcov_inv, ones_arr))
    flattened_weights = [weight for weight_list in weights for weight in weight_list]
    expected_returns = np.matmul(returns.mean(), flattened_weights)
    expected_variance = np.matmul(flattened_weights, np.matmul(varcov.values, weights))[0]
    return (flattened_weights, expected_returns*252, expected_variance*252)

def maximize_sharpe_ratio(returns):
    varcov = returns.cov()
    n = 500
    num_stocks = len(returns.columns)
    best_weights = []
    expected_return = 0
    expected_variance = 0
    max_sharpe_ratio = 0

    for i in range(n):
        weight = np.random.random(num_stocks)
        weight /= weight.sum()

        curr_return = np.sum(returns.mean() * weight) * 252
        curr_variance = np.matmul(np.matrix.transpose(weight), np.matmul(varcov*252, weight))
        curr_sharpe_ratio = curr_return / np.sqrt(curr_variance)
        
        if curr_sharpe_ratio > max_sharpe_ratio:
            expected_return = curr_return
            expected_variance = curr_variance
            max_sharpe_ratio = curr_sharpe_ratio
            best_weights = weight


    return (best_weights, expected_return, expected_variance, max_sharpe_ratio)
