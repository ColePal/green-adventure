import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path

import pandas as pd
import shap
from streamlit_shap import st_shap

pages = {
    "Home": [
        st.Page("about_our_project.py", title="About"),
    #    st.Page("trial.py", title="Try it out"),
    ],
    "Text Classification Models": [
        st.Page("relevance_detection.py", title="Relevance Model"),
        st.Page("stance_detection.py", title="Stance Model"),
        st.Page("stance_direction.py", title="Polarity Model"),
        #st.Page("about_our_project.py", title="About"),
    ],
    
}
pg = st.navigation(pages)
pg.run()


                
