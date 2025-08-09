from preprocessing import *
from data_loader import load_data

df = load_data("src\\data\\gender_classification_v7.csv")

convert_string_to_nums(df,["gender"])

split_index = int(len(df) * 0.8)
cols = df.shape[1]

x_df = df.iloc[:,:cols - 1]
y_df = df.iloc[:,cols - 1]

x_np = x_df.to_numpy()
y_np = y_df.to_numpy()

y_np = y_np.reshape(-1,1)

x_training = x_np[:split_index]
y_training = y_np[:split_index]

x_testing = x_np[split_index:]
y_testing = y_np[split_index:]


