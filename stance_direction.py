import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path

import pandas as pd
import shap
from streamlit_shap import st_shap
from helpers import model_layout
from helpers import ROOTPATH

sample_dict = {"Chronic Lyme disease leaves patients with mild to severe impediments...":"Chronic Lyme disease leaves patients with mild to severe impediments, impacting their daily lives and overall well-being. This can range from physical limitations, such as joint pain and fatigue, to cognitive challenges like memory issues and brain fog.",
               "Salmonella, like other bacteria such as E. coli, C. difficile and Borrelia...":"Salmonella, like other bacteria such as E. coli, C. difficile, and Borrelia burgdorferi, can cause a range of health issues, including severe gastrointestinal distress, systemic infections, and long-term complications if not properly treated.",
               "Evidence suggests that nausea, fatigue and hearing loss are more likely...":"Evidence suggests that nausea, fatigue, and hearing loss are more likely the result of other underlying conditions such as immune system dysfunction, chronic viral infections, or side effects from long-term antibiotic use rather than chronic Lyme disease itself."}

model_name = "stance_dir_abstract_feature_final_final"
model = joblib.load(ROOTPATH / f"models/{model_name}/{model_name}.pkl")
#model = joblib.load(ROOTPATH / "models/stance_dir_abstract_feature_final/stance_dir_abstract_feature_final_final.pkl")

st.title("Stance Direction Model")
st.write("The stance direction model determines which side an author takes in the PTLDS debate")
top_container = st.container()
output_area = st.container()
chart_area = st.container()

scores = {"f1":0.73,
          "accuracy": 0.58}

declarations = {
    "true": "Your message supports the existance of PTLDS",
    "false": "Your message refutes the existance of PTLDS"
}

parameters = {
    "sample_dict":sample_dict,
    "model":model,
    "output_area":output_area,
    "chart_area":chart_area,
    "scores": scores,
    "top_container":top_container,
    "declaration":declarations
}

model_layout(parameters)
