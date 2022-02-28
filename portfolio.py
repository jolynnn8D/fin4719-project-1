import warnings
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px 
import json
from datetime import datetime as dt

import analytics 
import configuration

with open('./chart_desc.json','rb') as f:
    chart_desc = json.load(f)

startdate = "2015-01-01"
enddate = "2021-12-31"
base_date = "2019-01-01"
benchmark = configuration.config["benchmark"]
startdate = configuration.config["startdate"]
enddate = configuration.config["enddate"]
themes = configuration.config["funds"]
value_key = f"{benchmark} Value"

def display():
    """
    This is the main UI function. All streamlit frontend functions go here. 
    """
    # Initialize values
    if "theme" not in st.session_state:
        st.session_state.theme = "ESG and Green Energy"
    if "risk" not in st.session_state:
        st.session_state.risk = "Low"
    # if "short_sell" not in st.session_state:
    #     st.session_state.short_sell = True
    if "index_position" not in st.session_state:
        st.session_state.index_position = get_index_position(benchmark)
    if "weights" not in st.session_state or \
        "expreturn" not in st.session_state or \
        "expvar" not in st.session_state or \
        "portfolio_type" not in st.session_state or \
        "positions" not in st.session_state:
        compute_theme(st.session_state.theme, st.session_state.risk)
    
    st.sidebar.selectbox("Pick a fund theme", themes.keys(), key="theme")  
    st.sidebar.select_slider("What is your risk appetite?", options=["Low", "High"], key="risk")
    st.sidebar.button("Refresh", on_click=compute_theme, args=(st.session_state.theme, st.session_state.risk))
    
    header_col1, header_col2 = st.columns(2)
    with header_col1: st.header("Portfolio Composition and Returns")
    with header_col2: st.header("Analytics")
    
    
    metric_col1, metric_col2, extra_col = st.columns([1,1,2])
    with metric_col1: st.metric('Historical Return (%)', round(st.session_state.expreturn*100, 2))
    with metric_col2: st.metric('Historical Volatility (%)', round(np.sqrt(st.session_state.expvar)*100, 2))
        
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown('##### Growth of $10,000 USD Since Inception')
        st.markdown(f"*{chart_desc['historical_perf']}*")
        st.subheader("")
        # st.line_chart(st.session_state.positions)
        st.plotly_chart(
            plot_perf_comparison_price(
                st.session_state.positions.iloc[:, 0], st.session_state.positions.iloc[:, 1], base_date=startdate, rebase=10000)
        )
        
    with row1_col2:
        df_charts = st.session_state.positions.reset_index()
        st.markdown('##### \t Rolling Beta (6-month)')
        st.markdown(f"*{chart_desc['rolling_beta']}*")
        st.plotly_chart(analytics.rolling_beta(df_charts.iloc[:,0], df_charts.iloc[:,1], df_charts.iloc[:,2]))
        
    
    row2_col1, row2_col2 = st.columns(2) 
    with row2_col1:
        st.markdown('##### Composition')
        st.markdown(f"The portfolio allocation below is the **{st.session_state.portfolio_type}** based on the portfolio theme and risk that you chose.")
        st.plotly_chart(plot_weight_pie_charts(st.session_state.weights), use_container_width=True)
    with row2_col2:
        st.markdown('##### \t Rolling Sharpe Ratio (6-month)')
        st.markdown(f"*{chart_desc['rolling_sharpe']}*")
        st.plotly_chart(analytics.rolling_sharpe(df_charts.iloc[:,0], df_charts.iloc[:,1]))
        
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown('##### Risk-Return Performance')
        for field, df in st.session_state.combined_analytics.items():
            st.markdown(f"**{field}**")
            if field !="Sharpe Ratio":
                df = df.multiply(100)
                df = df.applymap("{0:.2f}%".format)
            else:    
                df = df.applymap("{0:.3f}".format)
            st.dataframe(df.style)
        
    with row3_col2:
        df_charts["Date"] = df_charts["Date"].dt.strftime('%Y-%m-%d')
        st.markdown('##### \t Monthly Returns Heatmap')
        st.markdown(f"*{chart_desc['heatmap']}*")
        st.plotly_chart(analytics.return_heatmap(df_charts.iloc[:,0], df_charts.iloc[:,1]))
    
    row4_col1, row4_col2 = st.columns(2)    
    with row4_col1: pass
    with row4_col2:
        st.markdown('##### \t Calendar Year Returns')
        st.markdown(f"*{chart_desc['calendar_year']}*")
        st.plotly_chart(analytics.return_barchart(df_charts.iloc[:,0], df_charts.iloc[:,1]))


# Main computation functions
def compute_theme(theme, risk):
    """
    Compute the MVP and max sharpe ratio portfolio for a chosen theme, and saves the result to session state.

    Args:
        theme (string): Name of the theme fund chosen.
        risk (string): Risk profile of investor, either High or Low.
    """
    ticker_list = themes[theme]["tickers"]
    returns = get_returns(ticker_list, startdate, enddate)
    returns_without_date = returns.drop(columns=["Date"])
    if risk == "Low":
        results = calculate_mvp(returns_without_date)
        st.session_state.portfolio_type = "minimum variance portfolio"
    elif risk == "High":
        results = monte_carlo(returns_without_date, "sharpe")
        st.session_state.portfolio_type = "maximum sharpe ratio portfolio"
    weights = results[0]
    expected_return = results[1]
    expected_variance = results[2]
    
    weight_dict = {}
    for (index, ticker) in enumerate(ticker_list):
        weight_dict[ticker] = weights[index]

    st.session_state.weights = weight_dict
    st.session_state.expreturn = expected_return
    st.session_state.expvar = expected_variance
    st.session_state.positions = get_positions(returns, st.session_state.weights)

    benchmark_ret = st.session_state.index_position.iloc[:, :2].set_index('Date')    
    portfolio_ret = returns.set_index('Date').loc[:, 'NormReturns']
    portfolio_analytics = calc_portfolio_analytics(portfolio_ret, df_label=st.session_state.theme)
    benchmark_analytics= calc_portfolio_analytics(benchmark_ret, df_label=benchmark)
    
    st.session_state.combined_analytics = compile_analytics(portfolio_analytics, benchmark_analytics)

        
    
def calculate_mvp(returns, short_sell=False):
    """
    Calculate MVP given the returns of several stocks.

    Args:
        returns (pd.DataFrame): DataFrame containing daily returns of different stock tickers.
        short_sell (boolean, optional): Set to True to allow short selling in the MVP. Defaults to False.

    Returns:
        tuple (list, float, float): Tuple of weights, annualized returns and annualized variance.
    """
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
    """
    Performs the Monte Carlo simulation.

    Args:
        returns (pd.DataFrame): DataFrame containing daily returns of different stock tickers.
        type (string, optional): Either sharpe or mvp for the type of simulation.
        n (int, optional): Number of iterations.

    Returns:
        tuple (list, float, float): Tuple of weights, annualized returns and annualized variance.
    """
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


def annualize_ret(ret, n_days=None):
    """Calculate annualized returns for a given return series with daily interval.

    Args:
        ret (pd.Series): Simple return series (1 + r) ...
        n_days (int, optional): Define window to annualize returns over. If None, annaulize returns since inception. Defaults to None.

    Returns:
        float: Annualized returns over n years
    """
    ret = ret.values
    n_trading_days = int(252*n_days)
    if n_days is None:
        obs = ret
    else:
        obs = ret[-n_trading_days:]
    
    cumret = ret[-n_trading_days:].prod()
    ann_ret =  cumret ** (252 / n_trading_days) - 1
    
    return ann_ret


def annualize_std(ret, n_days=None):
    """Annualize standard deviation for a given return series with daily interval.

    Args:
        ret (pd.Series): Simple return series (1 + r) ...
        n_days (int, optional): Define window to annualize returns over. If None, annaulize returns since inception. Defaults to None.

    Returns:
        float: Annualized standard deviation over n years
    """
    ret = ret.values
    n_trading_days = int(252*n_days)
    
    if n_days is None:
        obs = ret
        
    else:
        obs = ret[-n_trading_days:]
        
    std = obs.std()
    ann_std = std * np.sqrt(252 / n_trading_days) 
    
    return ann_std

def annualize_sharpe_ratio(ann_ret, ann_std, rf=0):
    return (ann_ret-rf) / ann_std
    
 
def calc_portfolio_analytics(ret, df_label='Benchmark'):
    """Calculate annualized portfolio analytics for a given return series with daily interval.

    Args:
        ret (pd.Series): Simple return series (1 + r) ...
        n_days (int, optional): Define window to annualize returns over. If None, annaulize returns since inception. Defaults to None.

    Returns:
        pd.DataFrame: Annualized portfolio analytics
    """
    
    ann_ret_dict = {
        "3 Month" : annualize_ret(ret, n_days=252/4),
        "6 Month" : annualize_ret(ret, n_days=252/2),
        "1 Year" : annualize_ret(ret, n_days=252),
        "3 Year" : annualize_ret(ret, n_days=252*3),
        "5 Year" : annualize_ret(ret, n_days=252*5)
    }
    
    ann_std_dict = {
        "3 Month" : annualize_std(ret, n_days=252/4),
        "6 Month" : annualize_std(ret, n_days=252/2),
        "1 Year" : annualize_std(ret, n_days=252),
        "3 Year" : annualize_std(ret, n_days=252*3),
        "5 Year" : annualize_std(ret, n_days=252*5)
    }
    
    ann_sr_dict = {
        "3 Month" : annualize_sharpe_ratio(ann_ret_dict["3 Month"], ann_std_dict["3 Month"]),
        "6 Month" : annualize_sharpe_ratio(ann_ret_dict["6 Month"], ann_std_dict["6 Month"]),
        "1 Year" : annualize_sharpe_ratio(ann_ret_dict["1 Year"], ann_std_dict["1 Year"]),
        "3 Year" : annualize_sharpe_ratio(ann_ret_dict["3 Year"], ann_std_dict["3 Year"]),
        "5 Year" : annualize_sharpe_ratio(ann_ret_dict["5 Year"], ann_std_dict["5 Year"])
    }
    
    fields = ['Returns','Volatility','Sharpe Ratio']
    analytics = pd.DataFrame([ann_ret_dict, ann_std_dict, ann_sr_dict],
                              index = fields)

    analytics.name = df_label
    
    return analytics
    

def compile_analytics(pf_analytics1, pf_analytics2):
    """Compile analytics using dataframe generated from `calc_portfolio_analytics()`"""
    
    label1 = pf_analytics1.name
    label2 = pf_analytics2.name
    
    analytics_dict = {}
    for field in ['Returns','Volatility','Sharpe Ratio']:
        tbl = pd.concat([pf_analytics1.loc[[field]],pf_analytics2.loc[[field]]])
        tbl.index = [label1,label2]
        tbl.columns.name = field
        analytics_dict[field] = tbl
    
    return analytics_dict

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

 
    positions = pd.concat([returns, st.session_state.index_position[value_key]], axis=1)
    positions = positions.set_index("Date")
    return positions[["Portfolio Value", value_key]]

def get_index_position(benchmark):
    returns = get_returns([benchmark], startdate, enddate)
    returns[benchmark] += 1
    returns[value_key] = 0
    returns.loc[0, value_key] = 100 * returns.loc[0, benchmark]
    for i in range(1, len(returns)):
        returns.loc[i, value_key] = returns.loc[i-1, value_key] * returns.loc[i, benchmark]
    return returns


# input price series with date set as index
def calc_il_price(price, base_date=None, end_date=None, rebase=1000):
    """Calculate index level

    Args:
        price (pd.Series): Single price series with date set as index
        base_date (str): Start date of the reference period for calculating index levels. If None, take the earliest available date in price sample. Defaults to None.
        end_date (str, optional): Ending date of reference period for calculating index levels. If None, take the last available date in price sample. Defaults to None.
        rebase (int, optional): Starting index level. Defaults to 1000.

    Returns:
        pd.Series: Timeseries of index levels
    """
    not_in_daterange = (pd.to_datetime(base_date) < price.index.min())
    
    # check if base_date is within sample date range in price series
    if not_in_daterange:
        warnings.warn(f"Error! base_date is out of sample date range... Defaulting to earliest available date in sample: {price.index.min()}. Otherwise, please use input date range between {price.index.min()} and {price.index.max()}")
        base_date = price.index.min()
        
    # subset price data
    if end_date is not None:
        price = price[base_date:end_date] 
    
    else:
        price = price[base_date:]   
    
    # cumulative returns calculation
    lret = np.log(price / price.shift(1))
    cumlret = np.cumsum(lret)
    cumret = np.exp(cumlret)
    cumret.iloc[0] = 1 # set 
    
    il = cumret*rebase
    
    return il

# input price series with date set as index
def calc_il(ret, base_date=None, end_date=None, rebase=1000):
    """Calculate index level

    Args:
        returns (pd.Series): Single simple returns series with date set as index
        base_date (str): Start date of the reference period for calculating index levels. If None, take the earliest available date in price sample. Defaults to None.
        end_date (str, optional): Ending date of reference period for calculating index levels. If None, take the last available date in price sample. Defaults to None.
        rebase (int, optional): Starting index level. Defaults to 1000.

    Returns:
        pd.Series: Timeseries of index levels
    """
    not_in_daterange = (pd.to_datetime(base_date) < ret.index.min())
    
    # check if base_date is within sample date range in price series
    if not_in_daterange:
        warnings.warn(f"Error! base_date is out of sample date range... Defaulting to earliest available date in sample: {ret.index.min()}. Otherwise, please use input date range between {ret.index.min()} and {ret.index.max()}")
        base_date = ret.index.min()
        
    # subset price data
    if end_date is not None:
        ret = ret[base_date:end_date] 
    
    else:
        ret = ret[base_date:]   
       
    
    # cumulative returns calculation
    cumret = np.cumprod(ret)
    cumret.iloc[0] = 1 # set 
    
    il = cumret*rebase
    
    return il


def compare_perf(portfolio, benchmark, base_date, end_date=None, rebase=1000):
    portfolio_il = calc_il(portfolio, base_date, end_date, rebase)
    benchmark_il = calc_il(benchmark, base_date, end_date, rebase)
    
    compare_il = pd.merge(portfolio_il.to_frame(),
                          benchmark_il, 
                          how='left',
                          on='Date')
    
    return compare_il

def compare_perf_price(portfolio, benchmark, base_date, end_date=None, rebase=1000):
    portfolio_il = calc_il_price(portfolio, base_date, end_date, rebase)
    benchmark_il = calc_il_price(benchmark, base_date, end_date, rebase)
    
    compare_il = pd.merge(portfolio_il.to_frame(),
                          benchmark_il, 
                          how='left',
                          on='Date')
    
    return compare_il

def plot_perf_comparison(portfolio, benchmark, base_date, end_date=None, rebase=1000):
    """Generate plotly fig object that compares performance between benchmark and portfolio

    Args:
        portfolio (pd.Series): Timeseries of portfolio returns (returns)
        benchmark (pd.Series): Timeseries of benchmark returns (returns)
        base_date (str): Starting date of reference period
        end_date (str, optional): Ending date of reference period. If None, default to last available date in sample data. Defaults to None.
        rebase (int, optional): Reference level. Defaults to 1000.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object 
    """
    
    compare_il = compare_perf(portfolio, benchmark, base_date, end_date, rebase)
    
    # plot covid data
    fig = px.line(compare_il)
    fig.add_hline(y=rebase, line_dash="dash", line_color="black", line_width=1) # rebase
    fig.update_layout(yaxis_title='Index Level',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            title=''
                            )
    )
    
    return fig

def plot_perf_comparison_price(portfolio, benchmark, base_date, end_date=None, rebase=1000):
    """Generate plotly fig object that compares performance between benchmark and portfolio

    Args:
        portfolio (pd.Series): Timeseries of portfolio values (price)
        benchmark (pd.Series): Timeseries of benchmark value (price)
        base_date (str): Starting date of reference period
        end_date (str, optional): Ending date of reference period. If None, default to last available date in sample data. Defaults to None.
        rebase (int, optional): Reference level. Defaults to 1000.

    Returns:
        plotly.graph_objects.Figure: Plotly figure object 
    """
    
    compare_il = compare_perf_price(portfolio, benchmark, base_date, end_date, rebase)
    
    # plot covid data
    fig = px.line(compare_il)
    fig.add_hline(y=rebase, line_dash="dash", line_color="black", line_width=1) # rebase
    fig.update_layout(yaxis_title='Index Level',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            title=''
                            )
    )
    
    return fig


def plot_weight_pie_charts(weight_dict):
    
    plt_data = pd.DataFrame(
        [(ticker, weight) for ticker,weight in weight_dict.items()],
        columns=['Ticker','Weight']
    )
    fig = px.pie(plt_data, values='Weight', names='Ticker', hole=0.4)
    fig.update_traces(textinfo='label+percent')
    fig.update_layout(showlegend=False)
    
    return fig
