
import streamlit
import numpy as np
import datetime as dt
import yahoo_fin.stock_info as si
import yfinance as yf
from sklearn import linear_model
import scipy.stats as st
import pandas as pd
from dateutil.relativedelta import relativedelta
import plotly.express as px 
import warnings
warnings.filterwarnings("ignore")



ticker_choices = ['ANDE','DOW','GM','HWKN','JPM','NWN','SCVL','SRCE','TSLA', 'WMT']    
benchmark = ['^GSPC']

def display():
    st.subheader("Active Investing Strategy")


    # buttons i'll add in later, ps went to sleep
    # date picker - from and to, up to 5 years from current date - done
    # stock picker - choose from ticker_choices list - done
    
    # outputs to add in later
    # historical backtest chart -- done
    # expected returns annualized -- done
    # strategy suggestion (e.g. buy on x day and sell on x day)

    # initialize values
    if "ticker" not in st.session_state:
        st.session_state.ticker = "ANDE"
    if "st_start_date" not in st.session_state:
        st.session_state.st_start_date = dt.date.today() - relativedelta(years = 5)
    if "st_end_date" not in st.session_state:
        st.session_state.st_end_date = dt.date.today()
    if "fig" not in st.session_state or \
        "expreturn" not in st.session_state:
        Compute(st.session_state.ticker, benchmark, st.session_state.st_start_date - relativedelta(years = 1), st.session_state.st_end_date)
    
    
    # input widgets    
    st.sidebar.selectbox("Select a stock", ticker_choices, key='ticker')          #stock picker
    st.sidebar.date_input("Start Date", value = dt.date.today() - relativedelta(years = 5), 
                          min_value = dt.date.today() - relativedelta(years = 5), max_value = dt.date.today(), key="st_start_date")
    st.sidebar.date_input("End Date", value = dt.date.today(), 
                          min_value = st.session_state.st_start_date, max_value = dt.date.today(), key="st_end_date")
    
    st.sidebar.button("Compute", on_click=produce_portfolio, args=(st.session_state.ticker, benchmark, 
                                                                   st.session_state.st_start_date - relativedelta(years = 1), st.session_state.st_end_date))

    # display
    st.markdown(f"#### **Ticker: {st.session_state.ticker}**")
    st.markdown(f"##### **Expected Return (Annualized): {round(st.session_state.expreturn, 2)}%**")
    st.plotly_chart(st.session_state.fig)


winar_dict = {
    'winar5': [+1, +1],
    'winar6': [+2, +5],
    'winar7': [+2, +10],
    'winar8': [+2, +20],
    }

def Compute(ticker, benchmark, start_date, end_date):
    ticker = [ticker]
    st.session_state.fig, st.session_state.expreturn = produce_portfolio(ticker, benchmark, start_date, end_date)




# get returns data
def get_returns(ticker_list, benchmark, start_date, end_date):
    
    data = yf.download(ticker_list + benchmark, start=start_date, end=end_date)

    main_data = data["Adj Close"] / data["Adj Close"].shift(1) - 1
    main_data = main_data.dropna()
    main_data = main_data.reset_index()
    data_ret = main_data.copy()

    return data_ret

def do_event_study(
    data_ret,
    eventdate,
    ticker,
    estimation_period=252,
    before_event=20,
    event_window_start=-20,
    event_window_end=20,
    benchmark="^GSPC",
    ):

    # Generate post-event indicator
    data_ret["post_event"] = (data_ret["Date"] >= eventdate).astype(
        int
    )  # 1 if after event, 0 otherwise
    data_ret = (
        data_ret.reset_index()
    )  # pushes out the current index column and create a new one

    # Identify the index for the event date
    event_date_index = data_ret.groupby(["post_event"])["index"].transform("min").max()
    data_ret["event_date_index"] = event_date_index

    # Create the variable day relative to event
    data_ret["rel_day"] = data_ret["index"] - data_ret["event_date_index"]

    # Identify estimation period
    estimation = data_ret[
        (data_ret["rel_day"] < -before_event)
        & (data_ret["rel_day"] >= -estimation_period - before_event)
    ]

    # Identify event period
    event = data_ret[
        (data_ret["rel_day"] <= event_window_end)
        & (data_ret["rel_day"] >= event_window_start)
    ]

    # Calculate expected returns with the market model
    x_df = estimation[benchmark].values.reshape(-1, 1)

    # Create an empty list to store betas
    betas = []

    # Calculate betas for the market model
    for y in [benchmark, ticker]:
        y_df = estimation[y].values.reshape(-1, 1)
        reg = linear_model.LinearRegression()
        betas.append(reg.fit(x_df, y_df).coef_)

    # Convert the list to a Numpy Array
    beta_np = np.array(betas)
    beta_np

    # Expected Returns via Beta
    # Need Numpy Array to do Calculations!
    sp500array = event[benchmark].values
    expected_returns = np.outer(sp500array, beta_np)
    expected_returns = pd.DataFrame(expected_returns, index=event.index)
    expected_returns.columns = [benchmark, ticker]
    expected_returns = expected_returns.rename(columns={ticker: "expected_return"})
    del expected_returns[benchmark]

    # Abnormal Returns
    event = pd.concat([event, expected_returns], axis=1, ignore_index=False)

    event["abnormal_return"] = event[ticker] - event["expected_return"]

    # Event CAR
    winar5 = event[(event["rel_day"] <= 1) & (event["rel_day"] >= 1)][
        "abnormal_return"
    ].sum()  # Event Day 1

    # Post Event CAR
    winar6 = event[(event["rel_day"] <= 5) & (event["rel_day"] >= 2)][
        "abnormal_return"
    ].sum()  # CAR[2,5]
    winar7 = event[(event["rel_day"] <= 10) & (event["rel_day"] >= 2)][
        "abnormal_return"
    ].sum()  # CAR[2,10]
    winar8 = event[(event["rel_day"] <= 20) & (event["rel_day"] >= 2)][
        "abnormal_return"
    ].sum()  # CAR[2,20]

    return (
        winar8,
        winar7,
        winar6,
        winar5,
    )

# get earnings surprise from yahoo finance
def get_earnings_surprise(ticker_list, start_date, end_date):

    esp_events = pd.DataFrame

    for ticker in ticker_list:
        earnings_dict = si.get_earnings_history(ticker)
        temp_earnings_df = pd.DataFrame.from_dict(earnings_dict)
        # print (temp_earnings_df.head())
        temp_earnings_df['startdatetime'] = pd.to_datetime(temp_earnings_df['startdatetime']).dt.date
        temp_earnings_df = temp_earnings_df[(temp_earnings_df['startdatetime'] > start_date) & (temp_earnings_df['startdatetime'] < end_date)]
        temp_earnings_df = temp_earnings_df[['ticker','startdatetime','epsestimate','epsactual']]
        
        if esp_events.empty:
            esp_events = temp_earnings_df
        else:
            esp_events = pd.concat([esp_events,temp_earnings_df], ignore_index = True)

    esp_events.columns = ['Ticker','Date','estimate','actual']
    esp_events['Type'] =  np.where(esp_events['actual'] >= esp_events['estimate'], 1, -1)
    esp_events['Date'] = pd.to_datetime(esp_events['Date'])


    #remove this if not doing for positive events only
    pos_events = esp_events[esp_events["Type"] == 1].set_index("Ticker")
    return pos_events

def calculate_results(returns_df, surprise_df, benchmark):

    cars = []

    # for ticker, eventdate in pos_events.items():
    benchmark = benchmark[0]
    
    for index, row in surprise_df.iterrows():
        data_ret = returns_df[["Date", index, benchmark]].copy()
        cars.append(do_event_study(data_ret, ticker=index, eventdate=row["Date"]))

    cars = pd.DataFrame(cars)
    cars.columns = [
        "winar8",
        "winar7",
        "winar6",
        "winar5",
    ]

    # Calculate the Mean and Standard Deviation of the AAR
    mean_AAR = cars.mean()
    std_AAR = cars.sem()

    # Put everything in Dataframes
    stats = pd.DataFrame(mean_AAR, columns=['Mean AAR'])
    stats['STD AAR'] = std_AAR
    stats['T-Test'] = mean_AAR / std_AAR

    # Note method sf (survival function) from scipy.stats.t (or st.t) calculates P-values from T-stats
    # The method sf takes two arguments: T-statistic and degree of freedom, i.e., sf(absolute value of t-statistic, degree of freedom)
    # For one-tail test multiply the function output by 1, for two-tail test multiply it by 2
    stats['P-Value']  = st.t.sf(np.abs(stats['T-Test']), len(cars)-1)*2

    return stats

# conduct an event study to find which 
def event_study(ticker_list, benchmark = ['^GSPC'], start_date = 'default end_date - 6 years', end_date = dt.date.today()):

    earnings_start_date = start_date + relativedelta(years = 1)

    returns_df = get_returns(ticker_list, benchmark, start_date, end_date)
    earnings_df = get_earnings_surprise(ticker_list, earnings_start_date, end_date)
    results_df = calculate_results(returns_df, earnings_df, benchmark)
    results_df = results_df[results_df['P-Value'] < 0.05]

    return returns_df, earnings_df, results_df

# get the strategy to use and the respective post-event timeframes
def get_strat(results, winar_dict = winar_dict):
    strategy_df = pd.DataFrame

    for index, row in results.iterrows():

        if not strategy_df.empty:
            if any(item in ['winar7', 'winar8'] for item in strategy_df.index) == True:
                if index == 'winar6':
                    if any(item in ['winar7', 'winar8'] for item in strategy_df.index) == True:
                        continue
                elif index == 'winar7':
                    if 'winar8' in strategy_df.index:
                        continue

        startAR = winar_dict[index][0]
        endAR = winar_dict[index][1]
        if row['Mean AAR'] >= 0:
            strat = 'long'
        else:
            strat = 'short'

        temp_strat = pd.DataFrame([[startAR, endAR, strat, row['Mean AAR']]], columns = ['start','end','strategy','mean AAR'])
        temp_strat.index = [index]
        
        if strategy_df.empty:
            strategy_df = temp_strat
        else:
            strategy_df = pd.concat([strategy_df, temp_strat])
            
    return strategy_df

# get the dates to hold the stock
def get_strat_dates(earnings, strategy_df):
    strat_dates = pd.DataFrame

    for index, row in earnings.iterrows():
        for strat_index, strat_row in strategy_df.iterrows():

            temp_dates = np.arange(strat_row['start'],strat_row['end']+1).astype('timedelta64[D]')
            temp_dates = pd.DataFrame(row['Date'] + temp_dates, columns = ['Date'])
            temp_dates['ticker'] = index
            if strat_row['strategy'] == 'long':
                temp_dates['strategy'] = 1
            else:
                temp_dates['strategy'] = -1

            if strat_dates.empty:
                strat_dates = temp_dates
            else:
                strat_dates = pd.concat([strat_dates, temp_dates], ignore_index=True)

    return strat_dates


# generate the portfolio results in a graph and get the annualized returns
def get_portfolio(returns, strat_dates):
    portfolio_df = returns.merge(strat_dates, on=['Date'])
    ticker = portfolio_df['ticker'][0]
    portfolio_df.drop(['^GSPC','ticker'], axis=1, inplace=True)
    portfolio_df.columns = ['Date','ret','strategy']
    # portfolio_df['returns'] = portfolio_df
    portfolio_df['returns'] = (portfolio_df['ret'] * portfolio_df['strategy']) + 1
    portfolio_df['value'] = portfolio_df['returns'].cumprod()
    first_date = pd.DataFrame([[portfolio_df['Date'][0] - relativedelta(days=1), 0, 0, 1, 1]], columns = ['Date','ret','strategy','returns','value'])

    portfolio_value = pd.concat([first_date, portfolio_df], ignore_index = True)
    portfolio_value.set_index('Date', inplace=True)
    portfolio_value.drop(['ret', 'strategy', 'returns'], axis = 1, inplace=True)
    portfolio_value.rename(columns={'value':ticker}, inplace=True)

    fig = px.line(portfolio_value, width=800, height=400)
    fig.update_layout(title='Earnings Strategy Portfolio Value Relative to Start Date',
                    yaxis_title='Value',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        title=''
                        )
    )

    portfolio_summary = pd.concat([portfolio_value.head(1), portfolio_value.tail(1)])
    portfolio_returns = portfolio_summary[ticker].max()
    date_diff = int((portfolio_summary.index.max() - portfolio_summary.index.min()) / np.timedelta64(1,'D'))
    total_years = date_diff / 365
    annualized_return = ((portfolio_returns**(1/total_years)) - 1) * 100

    return fig, annualized_return


# full run function for everything
def produce_portfolio(ticker_list, benchmark = ['^GSPC'], start_date = 'default end_date - 6 years', end_date = dt.date.today()):

    if start_date == 'default end_date - 6 years':
        start_date = end_date - relativedelta(years = 6)
        
    returns, earnings, results = event_study(ticker_list, benchmark, start_date, end_date)
    
    if results.empty:
        return 'No suggested strategy for ' + ticker_list[0], 0
    else:
        strategy_df = get_strat(results, winar_dict)
        strat_dates = get_strat_dates(earnings, strategy_df)
        fig, annualized_return = get_portfolio(returns, strat_dates)
        return fig, annualized_return

    


    
