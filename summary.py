import streamlit as st

import portfolio 

themes = {
    "ESG and Green Energy": ["ICLN", "PBW", "TAN"],
    "Tech": ["BOTZ", "ARKW", "KWEB"],
    "Value": ["AAPL", "MSFT", "AMZN", "NFLX"]
}

startdate = "2015-01-01"
enddate = "2021-12-31"


def display():
    st.header("PaGrowth")
    
    if "returns_data" not in st.session_state:
        st.session_state.returns_data = {}
        for theme, tickers in themes.items():
            returns = portfolio.get_returns(tickers, startdate, enddate)
            returns_without_date = returns.drop(columns=["Date"])
            st.session_state.returns_data[theme] = returns_without_date

    portfolios = st.columns(3)
    column = 0
    low_risk, high_risk = generate_summary()
    for theme in themes.keys():
        with portfolios[column]:
            st.subheader(theme)
            st.markdown("##### **Low Risk**")
            st.markdown(f"**Expected Return: {round(low_risk[theme][1]*100, 2)}%**")
            st.markdown(f"**Expected Variance: {round(low_risk[theme][2]*100, 2)}%**")
            st.number_input("Your stake: ", min_value=0.0, max_value=1.0, value=1/6, key=theme +"_low")
            st.markdown("##### **High Risk**")
            st.markdown(f"**Expected Return: {round(high_risk[theme][1]*100, 2)}%**")
            st.markdown(f"**Expected Variance: {round(high_risk[theme][2]*100, 2)}%**")
            st.number_input("Your stake: ", min_value=0.0, max_value=1.0, value=1/6, key=theme +"_high")

        column += 1


def generate_summary():
    low_risk = {}
    high_risk = {}
    for theme, returns_without_date in st.session_state.returns_data.items():
        low_risk_result = portfolio.calculate_mvp(returns_without_date)
        high_risk_result = portfolio.monte_carlo(returns_without_date, "sharpe")
        low_risk[theme] = low_risk_result
        high_risk[theme] = high_risk_result
    return (low_risk, high_risk)
