import streamlit as st
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats 
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import datetime as dt

# Parameters
themes = {
    "ESG and Green Energy": ["ICLN", "PBW", "TAN"],
    "Tech": ["BOTZ", "ARKW", "KWEB"],
    "Value": ["AAPL", "MSFT", "AMZN", "NFLX"]
}

startdate = "2015-01-01"
enddate = "2021-12-31"
benchmark = "^GSPC"
value_key = f"{benchmark} Value"

def display():

    # Initialize values
    if "theme" not in st.session_state:
        st.session_state.theme = "ESG and Green Energy"
    if "risk" not in st.session_state:
        st.session_state.risk = "Low"
    if "short_sell" not in st.session_state:
        st.session_state.short_sell = True
    if "index_position" not in st.session_state:
        st.session_state.index_position = get_index_position(benchmark)
    if "weights" not in st.session_state or \
        "expreturn" not in st.session_state or \
        "expvar" not in st.session_state or \
        "portfolio_type" not in st.session_state or \
        "positions" not in st.session_state:
        compute_theme(st.session_state.theme, st.session_state.risk)
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Portfolio Selection")
        
        st.sidebar.selectbox("Pick a portfolio theme", themes.keys(), key="theme")  
        st.sidebar.select_slider("What is your risk appetite?", options=["Low", "High"], key="risk")
        if st.session_state.risk == "Low":
            st.sidebar.checkbox("Short Selling", key="short_sell")
        st.sidebar.button("Compute", on_click=compute_theme, args=(st.session_state.theme, st.session_state.risk))

        
        st.markdown(f"##### **Expected Return: {round(st.session_state.expreturn*100, 2)}%**")
        st.markdown(f"##### **Expected Variance: {round(st.session_state.expvar*100, 2)}%**")
        st.line_chart(st.session_state.positions)
        st.markdown(f"The portfolio allocation below is the **{st.session_state.portfolio_type}** based on the portfolio theme and risk that you chose.")
        for (ticker, weight) in st.session_state.weights.items():
            st.markdown(f"**{ticker}**")
            st.text_input("Weight", value=round(weight,2), key=f"{ticker}_weight", disabled=True)

        st.subheader("")
    
    with col2:
        st.header("Analytics")
        df = st.session_state.positions.reset_index()
        st.plotly_chart(rolling_beta(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]))
        st.plotly_chart(rolling_sharpe(df.iloc[:,0], df.iloc[:,1]))
        
        df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')
        st.plotly_chart(return_heatmap(df.iloc[:,0], df.iloc[:,1]))
        st.plotly_chart(return_barchart(df.iloc[:,0], df.iloc[:,1]))

# Main computation functions
def compute_theme(theme, risk):
    ticker_list = themes[theme]
    returns = get_returns(ticker_list, startdate, enddate)
    returns_without_date = returns.drop(columns=["Date"])
    if risk == "Low":
        results = calculate_mvp(returns_without_date, st.session_state.short_sell)
        st.session_state.portfolio_type = "minimum variance portfolio"
    elif risk == "High":
        results = monte_carlo(returns_without_date, "sharpe")
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
    st.session_state.positions = get_positions(returns, st.session_state.weights)
    

def calculate_mvp(returns, short_sell=True):
    varcov = returns.cov()
    num_stocks = len(returns.columns) # -1 if date column is present, else remove
    ones_arr = np.array([[1]] * num_stocks)
    varcov_inv = np.linalg.pinv(varcov.values)
    weights = np.matmul(varcov_inv, ones_arr) / np.matmul(np.matrix.transpose(ones_arr), np.matmul(varcov_inv, ones_arr))
    flattened_weights = [weight for weight_list in weights for weight in weight_list]
    expected_returns = np.matmul(returns.mean(), flattened_weights)
    expected_variance = np.matmul(flattened_weights, np.matmul(varcov.values, weights))[0]
    
    short_sell_required = any(w < 0 for w in flattened_weights)
    if short_sell_required and not short_sell:
        return monte_carlo(returns, "mvp")
    else:
        return (flattened_weights, expected_returns*252, expected_variance*252)

def monte_carlo(returns, type="sharpe", n=1000):
    varcov = returns.cov()
    num_stocks = len(returns.columns)
    best_weights = []
    expected_return = 0
    expected_variance = float('inf')
    max_sharpe_ratio = 0

    for i in range(n):
        weight = np.random.random(num_stocks)
        weight /= weight.sum()

        curr_return = np.sum(returns.mean() * weight) * 252
        curr_variance = np.matmul(np.matrix.transpose(weight), np.matmul(varcov*252, weight))

        if type == "sharpe":
            curr_sharpe_ratio = curr_return / np.sqrt(curr_variance)
            
            if curr_sharpe_ratio > max_sharpe_ratio:
                expected_return = curr_return
                expected_variance = curr_variance
                max_sharpe_ratio = curr_sharpe_ratio
                best_weights = weight

        elif type == "mvp":
            if curr_variance < expected_variance:
                expected_return = curr_return
                expected_variance = curr_variance
                best_weights = weight

    return (best_weights, expected_return, expected_variance)

 
 
### Utils and Organizing Data ###
def get_returns(tickers, startdate, enddate):
    data = yf.download(tickers, start=startdate, end=enddate)
    returns_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    returns_data = returns_data.dropna()
    returns_data = returns_data.reset_index()
    if len(tickers) < 2: 
        returns_data = returns_data.rename(columns={"Adj Close":tickers[0]})
    return returns_data

def get_positions(returns, weights):
    returns["NormReturns"] = 0
    for ticker, weight in weights.items():
        returns['NormReturns'] += (returns[ticker] + 1) * weight
    
    returns["Portfolio Value"] = 0
    returns.loc[0, "Portfolio Value"] = 100 * returns.loc[0, "NormReturns"]
    for i in range(1, len(returns)):
        returns.loc[i, 'Portfolio Value'] = returns.loc[i-1, 'Portfolio Value'] * returns.loc[i, 'NormReturns']

 
    positions = pd.concat([returns, st.session_state.index_position], axis=1)
    positions = positions.set_index("Date")
    return positions[["Portfolio Value", value_key]]

def get_index_position(benchmark):
    returns = get_returns([benchmark], startdate, enddate)
    returns[benchmark] += 1
    returns[value_key] = 0
    returns.loc[0, value_key] = 100 * returns.loc[0, benchmark]
    for i in range(1, len(returns)):
        returns.loc[i, value_key] = returns.loc[i-1, value_key] * returns.loc[i, benchmark]
    return returns[value_key]

### Analytics ###

def rolling_beta(dates, asset_pr, benchmark_pr, window = 180):
    # find returns 
    benchmark_rt = benchmark_pr.pct_change().dropna()
    asset_rt = asset_pr.pct_change().dropna()
    
    # Initialise results list
    rolling_beta = []

    #get df size -1 for dropna 
    length = len(dates)
    
    # beta of a window
    def beta(benchmark_rt, asset_rt):

        # x is the benchmark, y is the asset linregress(x,y) -- CAPM: R_i = a + beta(R_m - R_f) + e
        ols = stats.linregress(benchmark_rt, asset_rt)
        beta = ols.slope
        return beta

    # for loop for 1-day step
    for i in range(length-window):
        rolling_beta.append(beta(benchmark_rt[i:i+window], asset_rt[i:i+window]))
        
    plt_data = pd.DataFrame({'Date': dates[window:], 'Rolling Beta': rolling_beta})
    
    # plot
    fig = px.line(plt_data, x = 'Date', y = 'Rolling Beta', title = str('Rolling Beta' + ' (' + str(window) + '-' + 'days' + ')'))
    
    return fig 

def rolling_sharpe(dates, asset_pr, window = 180):
    asset_rt = asset_pr.pct_change().dropna()
    
    # Initialise results list
    rolling_sharpe = []

    #get df size -1 for dropna 
    length = len(dates)
    
    # beta of a window
    def sharpe(asset_rt):
        sharpe = np.mean(asset_rt)/np.std(asset_rt)
        return sharpe

    # for loop for 1-day step
    for i in range(length-window):
        rolling_sharpe.append(sharpe(asset_rt[i:i+window]))
        
    plt_data = pd.DataFrame({'Date': dates[window:], 'Rolling sharpe': rolling_sharpe})
    
    # plot
    fig = px.line(plt_data, x = 'Date', y = 'Rolling sharpe', title = str('Rolling Sharpe' + ' (' + str(window) + '-' + 'days' + ')'))
    
    return fig 

def return_heatmap(dates, asset_pr):

    # convert dtype to dates and resample to month end price
    data = pd.DataFrame([dates, asset_pr]).T
    data.columns = ['Date', 'asset_pr']
    data.Date = pd.to_datetime(dates)
    data.set_index('Date', inplace = True)    
    data = data.resample('M').last()
    data = data.reset_index()
    
    # find asset monthly retrun
    data['asset_rt'] = data.asset_pr.pct_change()*100
    data['asset_rt'] = data['asset_rt'].round(2)
    data.dropna()
    
    # get year and month 
    asset_rt = data.asset_rt
    year = data.Date.dt.strftime("%Y")
    month = data.Date.dt.strftime("%m")
    
    #reshape dataframe
    df = pd.DataFrame([year, month, asset_rt]).T
    df.columns = ['Year', 'Month', 'Return']
    df = df.pivot_table(index='Year', columns='Month', values='Return')
    
    fig = px.imshow(df, labels=dict(x="Month", y="Year", color="Return"), title="Monthly Return Heatmap",
                    color_continuous_scale ='RdYlGn' , color_continuous_midpoint = 0,
                    text_auto=True)
    
    return fig

def return_barchart(dates, asset_pr):

    # convert dtype to dates and resample to month end price
    data = pd.DataFrame([dates, asset_pr]).T
    data.columns = ['Date', 'asset_pr']
    data.Date = pd.to_datetime(dates)
    data.set_index('Date', inplace = True)    
    data = data.resample('Y').last()
    data = data.reset_index()
    
    # find asset monthly retrun
    data['asset_rt'] = data.asset_pr.pct_change()*100
    data['asset_rt'] = data['asset_rt'].round(2)
    data.dropna()
    
    # get year and month 
    asset_rt = data.asset_rt
    year = data.Date.dt.strftime("%Y")
    
    #reshape dataframe
    df = pd.DataFrame([year[1:], asset_rt]).T
    df.columns = ['Year', 'Return']
    
    fig = px.bar(df, x='Return', y='Year', orientation='h', title="Monthly Return Barchart", text_auto = True)
    
    return fig

