import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import Dataset, DatasetDict


reddit_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Reddit_Data.csv'
twitter_dataset = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\Dataset\Twitter_Data.csv'
model_directory = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\reddit_model'
tokenizer_directory = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\reddit_tokenizer'
model_name = "xlm-roberta-base"


def load_data(reddit_dataset, twitter_dataset):
    # Load train and test dataset in a pandas dataframe
    df_reddit = pd.read_csv(reddit_dataset)
    df_twitter = pd.read_csv(twitter_dataset)

    df_reddit = df_reddit.rename(columns={'clean_comment': 'text', 'category': 'label'})
    df_twitter = df_twitter.rename(columns={'clean_text': 'text', 'category': 'label'})
    
    # Concatenate both datasets in order to get a bigger dataset
    df = pd.concat([df_reddit, df_twitter], ignore_index=True)

    return df


def preprocesing_data(df):
    df = df.dropna()

    # Contar el número de comentarios negativos
    num_negative = df[df['label'] == -1].shape[0]

    # Filtrar aleatoriamente el mismo número de comentarios positivos y neutros
    positive_comments = df[df['label'] == 1].sample(n=num_negative, random_state=42)
    neutral_comments = df[df['label'] == 0].sample(n=num_negative, random_state=42)

    # Combinar los DataFrames resultantes
    df_balanced = pd.concat([positive_comments, neutral_comments, df[df['label'] == -1]])

    # Mezclar el DataFrame para que las muestras estén en orden aleatorio
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convertir la columna 'label' a 'numpy.int64'
    df['label'] = df['label'].astype(np.int64)

    # Reemplazar todos los valores -1 en la columna 'label' con 2
    df['label'] = df['label'].replace(-1, 2)

    return df


def split_data(df):
    # Dividir el DataFrame en conjuntos de entrenamiento y prueba
    train_df = df.sample(frac=0.8, random_state=42)
    temp_df = df.drop(train_df.index)
    test_df = temp_df.sample(frac=0.5, random_state=42)
    val_df = temp_df.drop(test_df.index)

    # Convertir los DataFrames a objetos Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Crear el objeto DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    })

    return dataset


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_directory)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    val_dataset = tokenized_datasets["val"]
    
    return train_dataset, test_dataset, val_dataset


def init_trainer(train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    batch_size = 16 #8
    training_args = TrainingArguments(
        output_dir="reddit_sentiment",
        num_train_epochs=2, #3
        learning_rate=2e-5, #5e-5
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16 = True,
        disable_tqdm=False
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def eval(trainer, test_dataset):
    results = trainer.evaluate(test_dataset)
    print(results)
    
    
def save_trained_model(trainer, output_dir):
    # Save model and weights
    trainer.save_model(output_dir)


def main():
    df = load_data(reddit_dataset, twitter_dataset)
    df = preprocesing_data(df)
    dataset = split_data(df)
    train_dataset, test_dataset, val_dataset = tokenize_dataset(dataset)
    trainer = init_trainer(train_dataset, val_dataset)    
    trainer.train()
    eval(trainer, test_dataset)
    save_trained_model(trainer, model_directory)
    
    
if __name__== "__main__":
    main()