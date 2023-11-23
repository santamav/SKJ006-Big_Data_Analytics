from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import pandas as pd
import praw

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

# Initialize empty lists to store data
headlines = []
body = []
num_comments = []
comments = []

# Scraping a Subreddit
for submission in reddit.subreddit('OpenAI').hot(limit=5):  #.top(time_filter="all")  .top(time_filter="day", limit=5)   .hot(limit=5)
    headlines.append(submission.title)
    body.append(submission.selftext)
    num_comments.append(submission.num_comments)
    #print('HEADLINES: ', headlines)
    #print ('NUM_COMMENTS: ', num_comments)

    # Initialize empty list for comments of each submission
    submission_comments = []

    # Iterate through the top-level comments of the submission (limit to 2 comments)
    for comment in submission.comments[:2]:
        # Check if the comment is a top-level comment (not a reply)
        if isinstance(comment, praw.models.Comment):
            submission_comments.append(comment.body)

    comments.append(submission_comments)
    #print ('COMMENTS: ', comments, '\n')
    
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

# TODO: Preparar el formato del csv
data = {
    'headlines': headlines,
    'body': body,
    'num_comments': num_comments,
    'comments': comments
}

df = pd.DataFrame(data)
print(df)