import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path
import torch
import pandas as pd
import shap
from streamlit_shap import st_shap
from helpers import model_layout
from helpers import ROOTPATH

sample_dict = {"Many researchers believe that PTLDS symptoms actually have nothing to do...":"Many researchers believe that PTLDS symptoms actually have nothing to do with Lyme disease, attributing these symptoms instead to other underlying conditions such as chronic fatigue syndrome, fibromyalgia, or autoimmune disorders.",
               "PLTDS is caused by clusters of Lyme bacteria that resist traditional methods of...":"PTLDS is caused by clusters of Lyme bacteria that resist traditional methods of antibiotic treatment, leading to persistent symptoms due to ongoing infection and inflammation.",
               "The hypothesis of PTLDS has been thouroughly refuted, with evidence...":"The hypothesis of PTLDS has been thoroughly refuted, with evidence suggesting that the symptoms often attributed to this condition are more likely caused by alternative diagnoses such as immune dysfunction, chronic fatigue syndrome, or psychological factors."}
#st.set_page_config(layout="wide")
model_name = "stance_det_abstract_feature_final_final_final"
model = joblib.load(ROOTPATH / f"models/{model_name}/{model_name}.pkl")
#model = joblib.load(ROOTPATH / "models/stance_det_abstract_feature_final_final/stance_det_abstract_feature_final_final_final.pkl")

st.title("Stance Detection Model")
st.write("The stance detection model determines whether or not an author takes a side in the PTLDS debate")
top_container = st.container()
output_area = st.container()
chart_area = st.container()

scores = {"f1":0.86,
          "accuracy": 0.77}

declarations = {
    "true": "Your message takes a stance on the topic of PTLDS",
    "false": "Your message does not take a side in the PTLDS debate"
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