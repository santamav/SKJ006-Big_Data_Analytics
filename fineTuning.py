import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

reddit_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Reddit_Data.csv'
twitter_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Twitter_Data.csv'

# Load train and test dataset in a pandas dataframe
df_reddit = pd.read_csv(reddit_dataset)
df_twitter = pd.read_csv(twitter_dataset)

# Concatenate both datasets in order to get a bigger dataset
df = pd.concat([df_reddit, df_twitter], ignore_index=True)

print(df.head(), '\n')
print(df['category'].value_counts())
print()