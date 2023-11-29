import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

reddit_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Reddit_Data.csv'
twitter_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Twitter_Data.csv'
output_dir = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics'

# Load train and test dataset in a pandas dataframe
df_reddit = pd.read_csv(reddit_dataset)
df_twitter = pd.read_csv(twitter_dataset)

df_reddit = df_reddit.rename(columns={'clean_comment': 'text', 'category': 'label'})
df_twitter = df_twitter.rename(columns={'clean_text': 'text', 'category': 'label'})

# Concatenate both datasets in order to get a bigger dataset
df = pd.concat([df_reddit, df_twitter], ignore_index=True)
df = df.dropna()

print(df.head(), '\n')
print(df['label'].value_counts())
print()


# Contar el número de comentarios negativos
num_negative = df[df['label'] == -1].shape[0]

# Filtrar aleatoriamente el mismo número de comentarios positivos y neutros
positive_comments = df[df['label'] == 1].sample(n=num_negative, random_state=42)
neutral_comments = df[df['label'] == 0].sample(n=num_negative, random_state=42)

# Combinar los DataFrames resultantes
df_balanced = pd.concat([positive_comments, neutral_comments, df[df['label'] == -1]])

# Mezclar el DataFrame para que las muestras estén en orden aleatorio
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Convertir la columna 'label' a 'numpy.int64'
df_balanced['label'] = df_balanced['label'].astype(np.int64)

print(df_balanced.head(), '\n')
print(df_balanced['label'].value_counts())
print()

# Step 2: Split the data into train, validation, and test sets
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Step 3: Get the tokenizer and the classification model
#model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device))


# Tokenize the text data
train_tokenized = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, return_tensors="pt")
val_tokenized = tokenizer(val_data['text'].tolist(), truncation=True, padding=True, return_tensors="pt")
test_tokenized = tokenizer(test_data['text'].tolist(), truncation=True, padding=True, return_tensors="pt")

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_tokenized['input_ids']),
    torch.tensor(train_tokenized['attention_mask']),
    torch.tensor(train_data['label'].tolist())
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_tokenized['input_ids']),
    torch.tensor(val_tokenized['attention_mask']),
    torch.tensor(val_data['label'].tolist())
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_tokenized['input_ids']),
    torch.tensor(test_tokenized['attention_mask']),
    torch.tensor(test_data['label'].tolist())
)


# Step 4: Prepare the arguments of the trainer and set the trained object
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Custom data collator
class CustomDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[2] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Define training arguments
batch_size = 8
logging_steps = len(train_data) // batch_size
training_args = TrainingArguments(
    output_dir="sentiment_movies",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    #warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="steps",
    #logging_steps = len(emotions["train"]) // batch_size,
    #eval_steps = 50,
    #save_strategy="no",
    fp16 = True,
    #logging_dir="./logs",
    disable_tqdm=False
)

# Define Trainer with the tokenized datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics = compute_metrics,
    data_collator = CustomDataCollator()
)


# Step 5: Fine-tune the model
trainer.train()


# Step 6: Evaluate the model with the test dataset (English)
results = trainer.evaluate(test_dataset)
print(results)

preds_output = trainer.predict(test_dataset)
print(preds_output.metrics)


# Save model and weights
trainer.save_model(output_dir + "/model")
#model.save_pretrained(output_dir)

# Save tokenizer
tokenizer.save_pretrained(output_dir + "/tokenizer")