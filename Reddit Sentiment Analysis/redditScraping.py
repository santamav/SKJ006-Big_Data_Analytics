from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import pandas as pd
import praw
from datetime import datetime, timezone
import pytz
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import re

# 
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.quit()

# Accessing Reddit API
reddit = praw.Reddit (
    client_id="qpyOMjZ8iMic4Ify8gJnqw",
    client_secret="yowC8cmlWzzAIlH-eON629_HyjO4qA",
    user_agent="zpopScrapper"
    #username="",
    #password=""
    )

# Set the path to the Tesseract executable (change this path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def convert_timestamp(timestamp_utc):
    fmt = "%d-%m-%Y %H:%M:%S"
    time_from_utc = datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)
    time_from = time_from_utc.astimezone(tz=pytz.timezone("US/Central"))
    timestamp = time_from.strftime(fmt)
    #print(timestamp)
    return timestamp

# Clean the extracted text
def clean_text(text):
    # Remove non-alphanumeric characters and extra whitespaces
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespaces and convert to lowercase
    cleaned_text = ' '.join(cleaned_text.split()).lower()
    return cleaned_text


def download_images(image_urls):
    image_lst = []
    image_text_lst = []
    for image_url in image_urls:
        if image_url:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_data = Image.open(BytesIO(response.content))
                #image = cv2.imread(image_data, 0)
                image_text = pytesseract.image_to_string(image_data)
                image_text_lst.append(clean_text(image_text))
                image_lst.append(image_data)
            else:
                image_lst.append(None)
                image_text_lst.append(None)
        else:
            image_lst.append(None)
            image_text_lst.append(None)
    return image_lst, image_text_lst


def get_subreddits():
    # Initialize empty lists to store data
    headlines = []
    body = []
    num_comments = []
    comments = []
    timestamp = []
    image_urls = []

    # Scraping a Subreddit
    for submission in reddit.subreddit('OpenAI').hot(limit=25):  #.top(time_filter="all")  .top(time_filter="day", limit=5)   .hot(limit=5)
        headlines.append(submission.title)
        body.append(submission.selftext)
        num_comments.append(submission.num_comments)
        timestamp.append(convert_timestamp(submission.created_utc))
        #print('HEADLINES: ', headlines)
        #print ('NUM_COMMENTS: ', num_comments)

        # Initialize empty list for comments of each submission
        submission_comments = []

        # Iterate through the top-level comments of the submission (limit to 2 comments)
        for comment in submission.comments[:2]:
            # Check if the comment is a top-level comment (not a reply)
            if isinstance(comment, praw.models.Comment):
                submission_comments.append({
                    'body': comment.body,
                    'timestamp': convert_timestamp(comment.created_utc)
                    })

        comments.append(submission_comments)
        #print ('COMMENTS: ', comments, '\n')
        
        # Check if the submission has a URL (to filter out text-only posts)
        url = str(submission.url)
        if url.endswith("jpg") or url.endswith("jpeg") or url.endswith("png"):
            image_urls.append(url)
        else:
            image_urls.append(None)
            
        images, images_text = download_images(image_urls)
        
    # To scrape different types of information use any of the following code lines: 
    """ 
    print(submission.selftext) 
    print(submission.title)
    print(submission.id)
    print(submission.author)
    print(submission.score)
    print(submission.upvote_ratio)
    print(submission.url)
    print(submission.num_comments)
    """

    data = {
        'head': headlines,
        'body': body,
        'timestamp': timestamp,
        'num_comments': num_comments,
        'comments': comments,
        'images': images,
        'text_images': images_text
    }

    df = pd.DataFrame(data)
    print(df)
    #print(df['text_images'])
    return df


df = get_subreddits()
# Display images using matplotlib
for index, row in df.iterrows():
    # Check if the 'image_data' column contains image data
    if row['images']:
        plt.figure(figsize=(6, 6))
        plt.imshow(row['images'])
        plt.title(row['head'])
        plt.show()
df.to_csv('subreddits.csv', index=False)