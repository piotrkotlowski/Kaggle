from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection  import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import CategoricalNB







# df = pd.read_csv("merged_dataset.csv", engine="python", sep=",")
# X = df.drop(columns=["Is.Fraudulent"]).copy()
# y = df["Is.Fraudulent"].copy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ValidationData = pd.concat([X_val, y_val], axis=1)  
# ValidationData.to_csv("ValidationData.csv", index=False)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# y_test = pd.DataFrame(y_test, columns=["Is.Fraudulent"])
# y_train = pd.DataFrame(y_train, columns=["Is.Fraudulent"])


# TestData = pd.concat([X_test, y_test], axis=1)  
# TestData.to_csv("TestData.csv", index=False)

# TrainData = pd.concat([X_train, y_train], axis=1)  
# TrainData.to_csv("TrainData.csv", index=False)




class TimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, is_catboost=False, is_NB=False):
        self.is_catboost = is_catboost
        self.is_NB = is_NB
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        df = X.copy()  
        if "Transaction.Date" not in df.columns: 
            raise ValueError("What are you doing man?")
        df["Transaction.Date"] = pd.to_datetime(df["Transaction.Date"], format = "ISO8601")

        df["day"] = df["Transaction.Date"].dt.day.astype(float)

        df["month"] = df["Transaction.Date"].dt.month.astype(float)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["month_angle"]=np.arctan2(df["month_sin"],df["month_cos"]).astype(float) 

        if "Transaction.Hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["Transaction.Hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["Transaction.Hour"] / 24)
            df["hour_angle"] = np.arctan2(df["hour_sin"], df["hour_cos"]).astype(float)
        else:
            raise ValueError("What are you doing man? Time")

        df["Transaction.Weekday"] = df["Transaction.Date"].dt.weekday + 1
        df["Transaction.Weekday"] = df["Transaction.Weekday"].astype(int)
        df["FirstPartMonth"]=df["day"].apply(lambda x: 1 if x<=12 else 0).astype(int) 

        weekDaysEncoded=pd.get_dummies(df["Transaction.Weekday"], dtype=int) 
        if self.is_catboost:
            result=df[["month_angle", "hour_angle","FirstPartMonth","Transaction.Weekday"]]
        elif self.is_NB: 
            result=pd.concat([df[["month", "day","FirstPartMonth"]], weekDaysEncoded], axis=1)
        else:
           result=pd.concat([df[["month_angle", "hour_angle","FirstPartMonth"]], weekDaysEncoded], axis=1)
    
        return result.to_numpy()
    def get_feature_names_out(self,input_features=None):
        if self.is_catboost:
            return np.array(["month_angle", "hour_angle", "FirstPartMonth", "Transaction.Weekday"])
        elif self.is_NB:
            dummy_columns = [f"Transaction.Weekday_{i}" for i in range(1, 8)]
            return np.array(["month", "day", "FirstPartMonth"] + dummy_columns)
        else:
            dummy_columns = [f"Transaction.Weekday_{i}" for i in range(1, 8)]
            return np.array(["month_angle", "hour_angle", "FirstPartMonth"] + dummy_columns)





class AgeTransfomer(BaseEstimator, TransformerMixin): 

    def __init__(self): 
        pass   
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        df=X.copy() 
        if "Customer.Age" not in df.columns: 
            raise ValueError("What are you doing man? Age")
        df["Is.Minor"]=df["Customer.Age"].apply(lambda x : 1 if x<18 else 0) 
        df["Is.Senior"]=df["Customer.Age"].apply(lambda x : 1 if x>60 else 0) 

        return pd.concat([df[["Is.Minor"]], df[["Is.Senior"]]], axis = 1).to_numpy().astype(int)
    def get_feature_names_out(self,input_features=None):
        return np.array(["Is.Minor", "Is.Senior"])





class SexTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self): 
         pass    
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        df=X.copy() 
        if "sex" not in df.columns: 
            raise ValueError("What are you doing man? Sex")
        df["male"]=df["sex"].apply(lambda x : 1 if x=="M" else 0)

        return df[["male"]].to_numpy().astype(int)
    def get_feature_names_out(self,input_features=None):
        return np.array(["Sex"])





class BinaryPassthroughTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self): 
        pass   
    def fit(self, X, y=None):
        return self  
    def transform(self, X):  
        df=X.copy()
        return df[["Address.Match"]].to_numpy().astype(int)
    def get_feature_names_out(self,input_features=None):
        return np.array(["Address.Match"])



class HighAmountTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self): 
        pass   
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        df = X.copy()
        if "Transaction.Amount" not in df.columns: 
            raise ValueError("What are you doing man? Amount")
        HighAmountInt=df["Transaction.Amount"].quantile(0.95)
        df["Is.HighAmount"]=df["Transaction.Amount"].apply(lambda x : 1 if x>=HighAmountInt else 0) 
        return df[["Is.HighAmount"]].to_numpy().astype(int)
    def get_feature_names_out(self,input_features=None):
        return np.array(["Is.HighAmount"])




def KCrossData():
    df =pd.read_csv("../../data/TrainData.csv")
    X = df.drop(columns=["Is.Fraudulent"]).copy()
    y = df["Is.Fraudulent"].copy()
    return X,y

def getTrainingData():
    df =pd.read_csv("../../data/TrainData.csv")
    X = df.drop(columns=["Is.Fraudulent"]).copy()
    y = df["Is.Fraudulent"].copy()
    return X,y
def getTestData():
    df=pd.read_csv("../../data/TestData.csv")
    X = df.drop(columns=["Is.Fraudulent"]).copy()
    y = df["Is.Fraudulent"].copy()
    return X,y
    
def getValidationData():
    df=pd.read_csv("../../data/ValidationData.csv")
    X = df.drop(columns=["Is.Fraudulent"]).copy()
    y = df["Is.Fraudulent"].copy()
    return X,y



def GetScalePosWeight(y): 
    return y.sum()/y.shape[0]



def PipelineModel(model,Numerical=['Transaction.Amount', 'Customer.Age','Account.Age.Days','Quantity'],
                    CatBasic=["Payment.Method",'browser','Product.Category','Device.Used','source'],n=30):
    NB = isinstance(model, CategoricalNB)
    column_transformer=PipeLineColumnTransformer(Numerical,CatBasic, NB = NB)
    selector=CreateFeatureSelector(n)
    classifier_pipeline=Pipeline([
    ('preprocessor', column_transformer),  
    #('smote', SMOTE(sampling_strategy=0.1,random_state=42))
    ('featureselection',selector),
    ('model',model)
     ])
    return classifier_pipeline

def CatBoostTransformer(Numerical=['Transaction.Amount', 'Customer.Age','Account.Age.Days','Quantity']): 
    time_transformer=TimeTransformer(is_catboost=True)
    column_transformer = ColumnTransformer([
        ('time_features', time_transformer,["Transaction.Date","Transaction.Hour"]), 
        ("high_amount",HighAmountTransformer(),["Transaction.Amount"]),
        ("numerical",StandardScaler(),Numerical), 
        ("age",AgeTransfomer(),["Customer.Age"]), 
        ("dropColumns",'drop',["Transaction.Date","Transaction.Hour"])],remainder="passthrough") 
    return column_transformer

def CreateFeatureSelector(n=30): 
    log_clf = LogisticRegression(C=0.1, class_weight="balanced",  penalty='l1', 
    solver='liblinear', random_state=42)
    selector = RFE(estimator=log_clf, n_features_to_select=n, step=1)
    return selector

def PipeLineGradient(Numerical=['Transaction.Amount', 'Customer.Age','Account.Age.Days','Quantity'],
                    CatBasic=["Payment.Method",'browser','Product.Category','Device.Used','source']): 
    column_transformer=PipeLineColumnTransformer(Numerical,CatBasic)
    selector=CreateFeatureSelector()
    LR_pipeline=Pipeline([
    ('preprocessor', column_transformer),
    #('smote', SMOTE(sampling_strategy=0.1,random_state=42)),
    ('featureselection',selector)]) 
    return LR_pipeline

def PipeLineColumnTransformer(Numerical=['Transaction.Amount', 'Customer.Age','Account.Age.Days','Quantity'],
                    CatBasic=["Payment.Method",'browser','Product.Category','Device.Used','source'],CatBoost=False,NB=False):

    time_transformer = TimeTransformer(is_catboost=CatBoost,is_NB=NB)
    Scaler = StandardScaler() if not NB else MinMaxScaler()
    column_transformer = ColumnTransformer([
        ('time_features', time_transformer,["Transaction.Date","Transaction.Hour"]), 
        ("high_amount",HighAmountTransformer(),["Transaction.Amount"]),
        ("numerical",Scaler,Numerical), 
        ("age",AgeTransfomer(),["Customer.Age"]),
        ("sex",SexTransformer(),["sex"]),
        ("AddressMatch",BinaryPassthroughTransformer(),["Address.Match"]),
        ("catBasic", OneHotEncoder(drop='if_binary' , handle_unknown='ignore'),CatBasic)], remainder='drop')
    return column_transformer



def PredictionQualityInfo(y_pred,y_test):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    print("Classification Report:\n", classification_report(y_test, y_pred))


def FitPredictResult(model,X_train,X_test,y_train,y_test):
    classifier=PipelineModel(model)
    classifier.fit(X_train,y_train) 
    y_pred=classifier.predict(X_test) 
    PredictionQualityInfo(y_pred,y_test) 





