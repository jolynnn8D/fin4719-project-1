import pandas as pd
import numpy as np
import scipy.stats as stats 
import plotly.express as px 

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
    fig = px.line(plt_data, x = 'Date', y = 'Rolling Beta')
    
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
    fig = px.line(plt_data, x = 'Date', y = 'Rolling sharpe')
    
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
    year = data.Date.dt.year.astype(int)
    month = data.Date.dt.month.astype(int)
    
    #reshape dataframe
    df = pd.DataFrame([year, month, asset_rt]).T
    df.columns = ['Year', 'Month', 'Return']
    df = df.pivot_table(index='Year', columns='Month', values='Return')
    
    fig = px.imshow(df, labels=dict(x="Month", y="Year", color="Return"),
                    color_continuous_scale =["red","white","green"] , color_continuous_midpoint = 0,
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
    data.dropna(inplace=True)
       
    # get year and month 
    asset_rt = data.asset_rt
    # year = data.Date.dt.strftime("%Y")
    year = data.Date.dt.year.astype(int)
    
    #reshape dataframe
    df = pd.DataFrame([year[1:], asset_rt]).T
    df.columns = ['Year', 'Return']

    # create negative ret flag
    df['is_negative']  = np.where(df['Return'] < 0, True , False)
    
    color_mapping = {
        True : "red",
        False : "green"
    }
    
    fig = px.bar(
        df, 
        x='Return', 
        y='Year', 
        color='is_negative',
        color_discrete_map=color_mapping,
        orientation='h', 
        text_auto=True)
    fig.update_layout(showlegend=False)
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    
    return fig