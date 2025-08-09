import pandas as pd

def drop_columns(df,columns_to_drop): 
    df.drop(columns= columns_to_drop,inplace=True)

def convert_string_to_nums(df,list_columns_name):
    if df is not None:
        for col in list_columns_name:
            df[col] = df[col].astype("category").cat.codes
    else:
        print("data frame is none")

def scailing_data(df):
    if df is not None:
        for col in df.columns :
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0  # أو خليه NaN مثلاً          
    else:
        print("data frame is none")


def save_data_csv(df,file_path):
    df.to_csv(file_path, index=False)

def data_information(df):
    print(df.head(20))
    print(df.describe())
    df.info()
    print(df.shape)
