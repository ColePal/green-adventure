import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path

import pandas as pd
import shap
from streamlit_shap import st_shap
#st.set_page_config(layout="wide")
from helpers import model_layout
from helpers import ROOTPATH

sample_dict = {"Lyme disease is a common occurence in dairy animals and other..." : "Lyme disease is a common occurence in dairy animals and other livestock, caused by the bacterium Borrelia burgdorferi, transmitted through the bite of infected ticks. It can lead to severe health issues if not promptly treated, including arthritis, neurological problems, and heart abnormalities.",
               "Chronic Lyme disease can have a life-long impact on a patients...": "Chronic Lyme disease can have a life-long impact on a patients health, leading to persistent symptoms such as fatigue, joint pain, cognitive difficulties, and neurological issues. These lasting effects can significantly diminish quality of life and require ongoing medical management and support.",
               "The Chronic Lyme disease is a popular myth within the medical community...": "The Chronic Lyme disease is a popular myth within the medical community, often mistakenly attributed to various non-specific symptoms that do not have a clear cause. However, for many sufferers, the persistent symptoms they experience are very real, leading to ongoing debate and the need for more nuanced understanding and treatment approaches."}

#st.set_page_config(layout="wide")
model_name = "relevance_class_abstract_feature_final_final"
model = joblib.load(ROOTPATH / f"models/{model_name}/{model_name}.pkl")

st.title("Relevance Classification Model")
st.write("We have created a text classification model using simple tools. Does it work? Lets find out!")
top_container = st.container()
output_area = st.container()
chart_area = st.container()

scores = {"f1":0.63,
          "accuracy": 0.82}
declarations = {
    "true": "Your message is relevant to the debate around Lyme disease",
    "false": "Your message is not relevant to the debate around Lyme disease"
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
