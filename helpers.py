import streamlit as st
import time
import joblib
from rel_model import relclf
from pathlib import Path
import torch
import pandas as pd
import shap
from streamlit_shap import st_shap

ROOTPATH = Path(__file__).parent.resolve()

#def predict(model, phrase, output_area, chart_area, scores):
def predict(params, phrase):
    st.session_state.text_to_classify = phrase
    with params["output_area"]:
        with st.spinner("Classifying your message..."):
            prediction = params["model"].predict(pd.DataFrame({"abstract":phrase}, index=range(0,1)))
        if prediction > 0.5:
            st.header(params["declaration"]["true"])
        else:
            st.header(params["declaration"]["false"])
        
        #st.header(f'Your message is {"relevant" if prediction > 0.5 else "not relevant"} to the debate around lyme disease!')
        st.caption(f"This model achieves an f1 score of {params["scores"]["f1"]} and an accuracy score of {params["scores"]["accuracy"]} under test conditions")
    with params["chart_area"]:
        with st.spinner("Trying to figure out why..."):
            st_shap(params["model"].explain_abstract(pd.DataFrame({"abstract":phrase}, index=[0])), height=200)
        with st.spinner("Stalling for time..."):
            time.sleep(5)  
            st_shap(params["model"].waterfall(pd.DataFrame({"abstract":phrase}, index=range(0,1))))
        with st.spinner("Spinning up a circular graph..."):
            time.sleep(5)
            st.pyplot(params["model"].feature_importance())

#def model_layout(model, sample_dict, output_area, chart_area, scores):
def model_layout(params):
    with params["top_container"]:
        with open(ROOTPATH / "rel_model.py", 'r') as file:
            model_code = file.read()
            file.close()

        with st.expander("Model Code"):
            st.code(model_code)

        if 'text_to_classify' not in st.session_state:
            st.session_state.text_to_classify = ""
        with st.form(key="my_form"):
            #with text_input_div[0]:
            text_to_classify_area = st.text_area("Type your text here", value=st.session_state.text_to_classify)
            #with text_input_div[1]:
            #st.form_submit_button("GO!", on_click= predict, args=(model, text_to_classify_area, output_area, chart_area))
            st.form_submit_button("GO!", on_click= predict, args=(params, text_to_classify_area))

        st.write("Or try one of ours!")
        preformed_texts = st.columns(3)
        for i in range(len(preformed_texts)):
            sample_dict = params["sample_dict"]
            with preformed_texts[i]:
                #st.button(list(sample_dict.keys())[i],on_click=predict,args=(model, sample_dict.get(list(sample_dict.keys())[i]), output_area, chart_area))
                st.button(list(sample_dict.keys())[i],on_click=predict,args=(params, sample_dict.get(list(sample_dict.keys())[i])))