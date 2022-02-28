import streamlit as st
import numpy as np
import pandas as pd
import json
import portfolio 

themes = {
    "ESG and Green Energy": ["ICLN", "PBW", "TAN"],
    "Tech": ["BOTZ", "ARKW", "KWEB"],
    "Value": ["AAPL", "MSFT", "AMZN", "NFLX"]
}

with open('investment_obj.json', 'rb') as f:
     investment_obj = json.load(f)


startdate = "2015-01-01"
enddate = "2021-12-31"

risk_options = ['Low','High']

def display():
    
    if "index_position" not in st.session_state:
        st.session_state.index_position = portfolio.get_index_position(portfolio.benchmark)
    
    st.header("Welcome to Sleep Wealth.")
    st.selectbox("Select risk level:", options=risk_options, key="risk")
    st.markdown("---")
    st.header("Welcome to Sleep Wealth.")

    if "returns_data" not in st.session_state:
        st.session_state.returns_data = {}
        for theme, tickers in themes.items():
            returns = portfolio.get_returns(tickers, startdate, enddate)
            # returns_without_date = returns.drop(columns=["Date"])
            st.session_state.returns_data[theme] = returns.set_index('Date')

    low_risk, high_risk = generate_summary()
    
    # calculate portfolio returns: weighted average of constituent returns 
    portfolio_risk_ret_dict = {}
    portfolio_ret_dict = {}
    for risk_level in risk_options:
        if risk_level == 'Low':
            risk_result = low_risk
        elif risk_level == 'High':
            risk_result = high_risk
            
        for theme, tickers in themes.items():
            constituent_ret = st.session_state.returns_data[theme]
            w = risk_result[theme][0] # extract weights from results dictionary
            theme_portfolio_ret = (constituent_ret[tickers] * w).sum(axis=1) # calculate portfolio returns
            portfolio_ret_dict[theme] = theme_portfolio_ret
        
        portfolio_risk_ret_dict[risk_level] = portfolio_ret_dict # store portfolio returns at each risk level
    
    portfolios = st.columns(3)
    column = 0
    
    # risk condition
    if st.session_state.risk == "Low":
        risk_result = low_risk
    if st.session_state.risk == "High":
        risk_result = high_risk
    
    
    for theme in themes.keys():
        with portfolios[column]:
            st.subheader(theme)
            # st.markdown("##### **Low Risk**")
            st.markdown(f"*{investment_obj[theme]}*")
            st.markdown(f"**Historical Mean Return: {round(risk_result[theme][1]*100, 2)}%**")
            st.markdown(f"**Historical Variance: {round(risk_result[theme][2]*100, 2)}%**")
            st.number_input("Your stake: ", min_value=0.0, max_value=1.0, value=1/3, key=theme +"_weights")
            # st.markdown("##### **High Risk**")
            # st.markdown(f"**Expected Return: {round(risk_result[theme][1]*100, 2)}%**")
            # st.markdown(f"**Expected Variance: {round(risk_result[theme][2]*100, 2)}%**")
            # st.number_input("Your stake: ", min_value=0.0, max_value=1.0, value=1/6, key=theme +"_high")

        column += 1
    
    # store thematic weights
    theme_w = [st.session_state[theme+"_weights"] for theme in themes.keys()]
    
    
    # sanity check total weights
    if sum(theme_w) != 1:
        st.markdown("<span style='color:red'> Warning! Total weight allocated exceed 1. Please re-allocate portfolio weights.</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    # create blended portfolio
    selected_risk = st.session_state.risk
    blended_ret = pd.DataFrame(portfolio_risk_ret_dict[selected_risk])
    blended_ret = blended_ret.dot(np.array(theme_w)) + 1 # returns are in terms of (1 + returns)
    blended_ret.name = "Blended"
    benchmark_ret = st.session_state.index_position.iloc[:, :2].set_index('Date')
    combined_ret = pd.merge(blended_ret.to_frame(), benchmark_ret, on='Date').dropna()
    
    # portfolio analytics
    blended_analytics = portfolio.calc_portfolio_analytics(combined_ret.iloc[:, 0], df_label="Blended")
    benchmark_analytics= portfolio.calc_portfolio_analytics(combined_ret.iloc[:, 1], df_label="Benchmark")
    st.header("Blended Portfolio Overview")
    
    combined_analytics = portfolio.compile_analytics(blended_analytics,benchmark_analytics)
    
    # index level
    combined_il = portfolio.compare_perf(combined_ret.iloc[:, 0], combined_ret.iloc[:, 1], base_date=startdate)
    st.plotly_chart(portfolio.plot_perf_comparison(combined_ret.iloc[:, 0], combined_ret.iloc[:, 1], base_date=startdate), 
                    use_container_width=True)
    
    analytics_col, weights_col = st.columns(2)
    with analytics_col:
        for field, df in combined_analytics.items():
            st.markdown(f"##### {field}")
            st.dataframe(df, height = 1600)

    
    with weights_col:
        # plot weight chart
        theme_w_dict = {theme: st.session_state[theme+"_weights"] for theme in themes.keys()}
        st.markdown(f"##### Composition")
        st.plotly_chart(portfolio.plot_weight_pie_charts(theme_w_dict), title="", use_container_width=True)




def generate_summary():
    low_risk = {}
    high_risk = {}
    for theme, returns in st.session_state.returns_data.items():
        returns_without_date = returns.reset_index().drop(columns=["Date"])
        low_risk_result = portfolio.calculate_mvp(returns_without_date)
        high_risk_result = portfolio.monte_carlo(returns_without_date, "sharpe")
        low_risk[theme] = low_risk_result
        high_risk[theme] = high_risk_result
    return (low_risk, high_risk)
