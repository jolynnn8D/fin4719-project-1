import streamlit as st

import portfolio
import events

# Page Config
project_title = "FIN4719 Project 1"
st.set_page_config(page_title=project_title, layout="wide")

# Sidebar

st.sidebar.title("Features")

pages = {
        "Portfolio Selection": portfolio,
        "Event Studies": earnigns_strategy,
    }
page = st.sidebar.radio("", tuple(pages.keys()))
pages[page].display()
