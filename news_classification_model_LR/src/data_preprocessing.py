import pandas as pd
import re
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, label_column, text_column

def clean_text(text):
    text= text.lower()
    text= re.sub(r"[^a-zA-Z\s]","",text)
    return text

def preprocessing_data():
    df= pd.read_csv(RAW_DATA_PATH, sep='\t', engine='python')

    print("Original Data:")
    print(df.head())

    print("Dataset info:")
    print(df.info())

    print("Columns:")
    print(df.columns )

    print("Categories:")
    print(df["category"].unique())

    print("Performing Data Cleaning...")

    df.dropna(subset=["category","title","content"], inplace=True) #dropping 'filename' and null values
    df["text"]= df["title"]+" "+df["content"]

    df["text"]= df["text"].apply(clean_text)

    df=df[["text","category"]]
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed Dataset:")
    print(df.head())

if __name__=="__main__":
    preprocessing_data()

# this file is responsible for the preprocessing of data to make it ready for feature extraction
# in this, the null values and the 