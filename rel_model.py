import re
import joblib
import shap
import pandas as pd
from textblob import TextBlob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn import metrics
from textblob import TextBlob
from xgboost import DMatrix
import xgboost as xgb
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from time import time

ROOTPATH = Path(__file__).parent.resolve()

def create_list(string, n):
    out = []
    for i in range(n):
        out.append(f"{string}_{i}")
    return out

class relclf:
    def acc_metric(self,predictions,dtrain):
        targets = dtrain.get_label()
        classifications = [1 if x > 0.5 else 0 for x in predictions]
        return 'Accuracy', metrics.accuracy_score(targets, classifications)
    
    def __init__(self,
                 text_columns=["abstract"],
                 category_columns=[],
                 numeric_columns=[],
                 bert_columns=[],
                 blob_columns=[],
                 name="default"):
        self.name=name
        self.text_columns = text_columns
        self.category_columns = category_columns
        self.numeric_columns = numeric_columns
        self.bert_columns = bert_columns
        self.blob_columns = blob_columns
        
            
    def transform(self, data):


        df = pd.DataFrame({"index":range(0,len(data))})
        
        for text in self.blob_columns:
            blobs = [TextBlob(str(x)) for x in data[text] ]
            df[f'{text}_neg_sentiment'] = [x.sentiment[0] for x in blobs]
            df[f'{text}_pos_sentiment'] = [x.sentiment[1] for x in blobs]
            df[f'{text}_polarity'] = [x.sentiment.polarity for x in blobs]
        
        for text in self.text_columns:
            vect = joblib.load(ROOTPATH / f'models/{self.name}/{text}_transformer.pkl')
            init_clf = joblib.load(ROOTPATH / f'models/{self.name}/{text}_initial_classifier.pkl')
            lda = joblib.load(ROOTPATH / f"models/{self.name}/{text}_lda.pkl")

            data_dtm = vect.transform(data[text])
            lda_features = lda.transform(data_dtm)
            
            df = pd.concat([df,pd.DataFrame(lda_features, columns=create_list(f"{text}_lda", self.n_components))],axis=1)
            data_matrix = DMatrix(data_dtm)

            init_prediction = init_clf.predict(data_matrix)
            df[f"{text}_prediction"] = init_prediction

        

        for category in self.category_columns:
            dummies = pd.get_dummies(data[category], prefix=category).astype(int)
            df = pd.concat([df, dummies.reset_index(drop=True)], axis=1)
            
        for numeric in self.numeric_columns:
            numeric_df = data[[numeric]].reset_index(drop=True)
            df = pd.merge(df, numeric_df, left_index=True, right_index=True)
            
        df.columns = df.columns.str.replace('[<,>]', '', regex=True).astype(str)
        df.drop(columns=df.columns[df.columns.str.contains("[\[\]<]")], inplace=True)
        df.drop(columns=["index"], inplace=True)
        
        return df[self.features]

    def tokenizer(self, s):
        pos = 0
        offset_ranges = []
        input_ids = []
        for word in s.split():
            start, end = pos, pos + len(word)
            offset_ranges.append((start, end))
            input_ids.append(word)
            pos = end + 1  # +1 for the space
        return {"input_ids": input_ids, "offset_mapping": offset_ranges}
    
    def fit(self, 
    X_train, 
    y_train, 
    vect_params={"stop_words":'english', "max_features":5000, "ngram_range":(1,3),"max_df":0.85,"lowercase":True,"min_df":0.0001},
    lda_params={"n_components":10,"random_state":0,"max_iter":100}):
        df = pd.DataFrame({"index":range(0,len(X_train))})
        self.n_components = lda_params["n_components"]
        self.transformers = {}
        self.init_clfs = {}
        
        vect = TfidfVectorizer(stop_words=vect_params["stop_words"], max_features=vect_params["max_features"], ngram_range=vect_params["ngram_range"],max_df=vect_params["max_df"],lowercase=vect_params["lowercase"],min_df=vect_params["min_df"])
        params = {}
        
        for text in self.blob_columns:
            blobs = [TextBlob(x) for x in X_train[text] ]
            df[f'{text}_neg_sentiment'] = [x.sentiment[0] for x in blobs]
            df[f'{text}_pos_sentiment'] = [x.sentiment[1] for x in blobs]
            df[f'{text}_polarity'] = [x.sentiment.polarity for x in blobs]
        
        print("Blobbing Complete!")
        
        for text in self.text_columns:
            train_dtm = vect.fit_transform(X_train[text])
            lda = LatentDirichletAllocation(n_components=lda_params["n_components"], random_state=lda_params["random_state"],max_iter=lda_params["max_iter"])
            lda_features = lda.fit_transform(train_dtm)
            
            try:
                joblib.dump(lda, f"models/{self.name}/{text}_lda.pkl")
            except:
                try:
                    os.makedirs(f"models/{self.name}/", exist_ok=False)
                    joblib.dump(lda, f"models/{self.name}/{text}_lda.pkl")
                except:
                    os.makedirs(f"models/", exist_ok=True)
                    os.makedirs(f"models/{self.name}/", exist_ok=False)
                    joblib.dump(lda, f"models/{self.name}/{text}_lda.pkl")

            df = pd.concat([df, pd.DataFrame(lda_features, columns=create_list(f"{text}_lda", self.n_components))],axis=1)

            joblib.dump(vect, f"models/{self.name}/{text}_transformer.pkl")
            train_matrix = DMatrix(train_dtm, label= y_train)
            init_clf = xgb.train({}, dtrain = train_matrix, num_boost_round=100)
            joblib.dump(init_clf, f"models/{self.name}/{text}_initial_classifier.pkl")
            y_pred_class = init_clf.predict(train_matrix)
            df[f"{text}_prediction"] = y_pred_class
            
        print("Vectorized and LDA'd")

            
        for category in self.category_columns:
            dummies = pd.get_dummies(X_train[category], prefix=category).astype(int)
            df = pd.concat([df, dummies.reset_index(drop=True)], axis=1)
        for numeric in self.numeric_columns:
            numeric_df = X_train[[numeric]].reset_index(drop=True)
            df = pd.merge(df, numeric_df, left_index=True, right_index=True)
        df.columns = df.columns.str.replace('[<,>]', '', regex=True).astype(str)
        df.drop(columns=df.columns[df.columns.str.contains("[\[\]<]")], inplace=True)
        
        
        features = list(df.drop(columns=["index"]).columns)
        #print(df[features])
        train_matrix = DMatrix(df[features], label=y_train, enable_categorical=False)
        
        print("Performing Initial Training")
        
        clf = xgb.train(params, dtrain= train_matrix,num_boost_round=100)
        


        print("Blind Pruning...")
        
        while (True):
            
            import_dict = clf.get_score(importance_type='weight')

            total_value = np.sum(list(import_dict.values()))
            sorted_dict = dict(sorted(import_dict.items(), key=lambda item: item[1], reverse=True))

            smallest_value = list(sorted_dict.values())[-1]

            print(f"smallest {smallest_value}")
            print("import")
            print(import_dict.keys())

            filtered_dict = {k: v for k, v in import_dict.items() if (v > smallest_value or smallest_value/total_value >= 0.05) }
            print("filtered")
            print(filtered_dict.keys())
            
            features = list(filtered_dict.keys())

            print("snip")
            
            if (import_dict == filtered_dict):
                self.features=features
                train_matrix = DMatrix(df[features], label=y_train, enable_categorical=False)
                clf = xgb.train(params, dtrain= train_matrix,num_boost_round=100)
                self.final_classifier = clf
                return

            train_matrix = DMatrix(df[features], label=y_train, enable_categorical=False)
            clf = xgb.train(params, dtrain= train_matrix,num_boost_round=100)

    def predict(self, data):
        
        print("Performing prediction...")
        
        df = self.transform(data)
        clf = self.final_classifier
        pred_matrix = DMatrix(df, enable_categorical=False)
        predictions = clf.predict(pred_matrix)
        return predictions
    
    
    def explain_features(self, data):
        df = self.transform(data)
        clf = self.final_classifier
        explainer = shap.Explainer(clf)
        shap_values = explainer(df)
        return shap.summary_plot(shap_values.values, df)

    def waterfall(self, data):
        df = self.transform(data)
        clf = self.final_classifier
        explainer = shap.Explainer(clf)
        shap_values = explainer(df)

        fig, ax = plt.subplots()
        plt.title("Feature Importance for classified text")
        
        return shap.plots.waterfall(shap_values[0],show=True)
        #return 

    def explain_abstract(self, data, index=0):
        class Tokenizer:
            def __init__(self):
                pass
            def __call__(self, text):
                pos = 0
                offset_ranges = []
                input_ids = []
                for word in text.split():
                    start, end = pos, pos + len(word)
                    offset_ranges.append((start, end))
                    input_ids.append(word)
                    pos = end + 1  # +1 for the space
                return {"input_ids": input_ids, "offset_mapping": offset_ranges}
            def decode(self, ids):
                return " ".join(ids)
        clf = self.final_classifier
        my_tokenizer = Tokenizer()
        def model_predict(texts):
            transformed_texts = [my_tokenizer(text)["input_ids"] for text in texts]
            df2 = pd.DataFrame({"index": range(len(transformed_texts)) })
            df2["abstract"] = ""
            for row in range(len(transformed_texts)):
                transformed_text = " ".join(transformed_texts[row])
                df2.loc[row,"abstract"] = transformed_text
            my_df = self.transform(df2.drop(columns=["index"]))
            dmatrix = xgb.DMatrix(my_df, enable_categorical=True)
            return clf.predict(dmatrix)
        masker = shap.maskers.Text(my_tokenizer)
        explainer = shap.Explainer(model_predict, masker)
        row_to_explain = data['abstract'][index]
        shap_values = explainer([row_to_explain])
        return shap.plots.text(shap_values[0])
        
    def feature_importance(self):
        clf = self.final_classifier
        import_dict = clf.get_score(importance_type='weight')
        print(import_dict)
        importance = pd.DataFrame(import_dict, columns=import_dict.keys(), index=range(0,1))
        
        fig, ax = plt.subplots()
        ax.pie(importance.iloc[0].to_numpy(),
               labels=importance.columns,
               autopct='%1.1f%%',
               startangle=90)
        plt.title("WARNING: PIE CHART FOR ENTERTAINMENT PURPOSES ONLY")
        return plt

    def save(self):
        joblib.dump(self, f"models/{self.name}/{self.name}.pkl")