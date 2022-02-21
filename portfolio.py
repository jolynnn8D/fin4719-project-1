import streamlit as st

def display():
    st.subheader("Portfolio selection")
    risk_slider = st.slider("What is your risk appetite?", 1, 7)
    portfolio_theme = st.selectbox("Pick a portfolio theme", ["ESG", "Tech", "Value"])      