import streamlit as st

import portfolio
import earnings_strategy
import summary

# Page Config
project_title = "Sleep Wellth"
st.set_page_config(page_title=project_title, layout="wide")

# Sidebar

st.sidebar.title("Features")

pages = {
        "Summary": summary, 
        "PaGrowth": portfolio,
        "QuantiFi": earnings_strategy,
    }
page = st.sidebar.radio("", tuple(pages.keys()))
pages[page].display()
