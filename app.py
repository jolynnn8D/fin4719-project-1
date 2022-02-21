import streamlit as st

import portfolio
import events

# Page Config
project_title = "FIN4719 Project"

st.set_page_config(page_title=project_title)
st.title(project_title)

# Sidebar

st.sidebar.title("Features")

pages = {
        "Portfolio Selection": portfolio,
        "Event Studies": events,
    }
page = st.sidebar.radio("", tuple(pages.keys()))
pages[page].display()