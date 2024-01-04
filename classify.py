from transformers import pipeline
import ast
import pandas as pd


model_directory = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\reddit_model'
tokenizer_directory = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\reddit_tokenizer'
reddit_data_directory = r'C:\Users\gonza\OneDrive\Escritorio\master_2023\Big_Data_Analitycs\Sistemes-Intelligents_Big-Data-Analytics\subreddits.csv'

classifier = pipeline(
    "text-classification",
    model=model_directory,
    tokenizer=tokenizer_directory
    )

df = pd.read_csv(reddit_data_directory)

df['body'] = df['body'].fillna('empty')

df['comments'] = df['comments'].apply(ast.literal_eval)
subreddit_heads = df['head'].tolist()
subreddit_bodies = df['body'].tolist()

for i in range(len(subreddit_heads)):
    #if len(subreddit_heads[i].split()) > 165:
    #   subreddit_heads[i] = "empty"
    if len(subreddit_bodies[i].split()) > 165:
        subreddit_bodies[i] = "empty"

subreddit_heads_classification = classifier(subreddit_heads)
subreddit_bodies_classification = classifier(subreddit_bodies)

heads_classification_df = pd.DataFrame(subreddit_heads_classification)
bodies_classification_df = pd.DataFrame(subreddit_bodies_classification)

df = pd.merge(df, heads_classification_df, left_index=True, right_index=True)
df = pd.merge(df, bodies_classification_df, left_index=True, right_index=True)

df = df.rename(columns={'label_x': 'head_label', 'label_y': 'body_label'})
df = df[['head', 'head_label', 'body', 'body_label', 'timestamp', 'comments']]

label_map = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': -1}
df['head_label'] = df['head_label'].replace(label_map)
df['body_label'] = df['body_label'].replace(label_map)

df.loc[df['body'] == 'empty', 'body_label'] = 'empty'

#COMMENTS

df['body_comments'] = df['comments'].apply(lambda comments: [comment['body'] for comment in comments])
df['timestamp_comments'] = df['comments'].apply(lambda comments: [comment['timestamp'] for comment in comments])

subreddit_comments_classification_lst = []
for comments in df['body_comments']:
    for i in range(len(comments)):
        if len(comments[i].split()) > 150:
            comments[i] = "empty"
    subreddit_comments_classification = classifier(comments)
    subreddit_comments_classification_lst.append(subreddit_comments_classification)
  
new_lst = []
for lst in subreddit_comments_classification_lst:
    label_lst = []
    score_lst = []
    for item in lst:
        label = item['label']
        score = item['score']
        label_lst.append(label)
        score_lst.append(score)
    dic = {'label': label_lst, 'score': score_lst}
    new_lst.append(dic)
  
comments_classification_df = pd.DataFrame(new_lst)
df = pd.merge(df, comments_classification_df, left_index=True, right_index=True)

df = df.rename(columns={'label': 'label_comments'})
df = df[['head', 'head_label', 'body', 'body_label', 'timestamp', 'body_comments', 'label_comments', 'timestamp_comments']]

for i in range(len(df['body_comments'])):
    for j in range(len(df['body_comments'][i])): 
        if df['body_comments'][i][j] == "empty":
            df['label_comments'][i][j] = "empty"
        elif df['label_comments'][i][j] == "LABEL_0":
            df['label_comments'][i][j] = 0
        elif df['label_comments'][i][j] == "LABEL_1":
            df['label_comments'][i][j] = 1
        else:
            df['label_comments'][i][j] = -1
            
df['comments'] = df.apply(lambda row: {'body_comments': row['body_comments'], 
                                       'label_comments': row['label_comments'],
                                       'timestamp_comments': row['timestamp_comments']},
                          axis=1
                          )

df = df[['head', 'head_label', 'body', 'body_label', 'timestamp', 'comments']]

print(df.head())

df.to_csv('subreddits_classification.csv', index=False)
#df2 = pd.DataFrame(df['comments'][0])