import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path
import torch
import pandas as pd
import shap
from streamlit_shap import st_shap

st.title("The Great Debate: PTLDS")
st.subheader("Also known as post-treatment Lyme disease syndrome or chronic Lyme disease")
st.write("""PTLDS is a condition where symptoms persist even after normal Lyme disease treatment.
         Symptoms include fatigue, joint pain, brain fog, heart palpatations, hearing loss, sleep
         difficulties and others.""")
debate_columns = st.columns(2)
with debate_columns[0]:
    st.subheader("PTLDS is real and is caused by ongoing infection by Lyme bacteria")
    st.write("""Surviving Bacteria: One theory suggests that clusters of Lyme bacteria survive 
             the antibiotic treatment, potentially reactivating later and causing inflammation.""")
    st.write("""Autoimmune Response: Another popular theory is that Lyme disease triggers an autoimmune
            disorder. This means the immune system mistakenly attacks the body, causing symptoms even
            without active infection.""")
    st.write("""Bacterial Debris: There's also a hypothesis that leftover bacterial debris triggers ongoing
            inflammation.""")
    st.write("""(Gurarie, 2023)""")
with debate_columns[1]:
    st.subheader("PTLDS is not real and is caused by a variety of other sources.")
    st.write("""Lack of Scientific Evidence: Some scientists and medical professionals argue that there
              is little scientific evidence to support the existence of chronic Lyme disease. They believe
              that the symptoms attributed to PTLDS are not caused by ongoing infection but by other factors.""")
    st.write("""Alternative Causes: They suggest that symptoms like fatigue, pain, and cognitive issues could be
              due to immune dysfunction, chronic fatigue syndrome, or depression""")
    st.write("""Risks of Prolonged Antibiotic Use: They caution against prolonged or repeated antibiotic
              treatment, stating that it can be harmful and is not supported by evidence""")
    st.write("""(Susnjak, 2024)""")
st.caption("References")
st.caption("""Gurarie, M. (2023, December 18). What you need to know about post-treatment Lyme disease syndrome:
            Sometimes called chronic Lyme disease. Verywell Health. Medically reviewed by A. C. 
           Chandrasekaran, MD. Retrieved from https://www.verywellhealth.com/chronic-lyme-disease-5181468""")
st.caption("""Susnjak, T. (2024). Applying BERT and ChatGPT for sentiment analysis of Lyme disease in 
           scientific literature. In L. Gilbert (Ed.), Borrelia burgdorferi: Methods and protocols 
           (Vol. 2742, pp. [Chapter pages]). Springer Science+Business Media. https://doi.org/10.1007/978-1-0716-3561-2_14""")
    
