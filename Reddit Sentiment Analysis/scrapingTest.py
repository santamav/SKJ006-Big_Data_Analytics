from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService


#driver = webdriver.Chrome(ChromeDriverManager().install())
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
#
#
#
driver.quit()

import pandas as pd
import praw

# Accessing Reddit API
user_agent = "zpopScrapper"
reddit = praw.Reddit (
client_id="qpyOMjZ8iMic4Ify8gJnqw",
client_secret="yowC8cmlWzzAIlH-eON629_HyjO4qA",
user_agent=user_agent,
#username="",
#password=""
)

# Scraping a Subreddit
headlines = set ( )
for submission in reddit.subreddit('OpenAI').hot(limit=5):
    headlines.add(submission.title)
    print(len(headlines))
    print (submission.title)
    # TODO: Preparar el formato del csv
# To scrape different types of information use any of the following code lines: 
"""  
print (submission.title)
print (submission.id)
print (submission.author)
print (submission.score)
print (submission.upvote_ratio)
print (submission.url)
"""

# Saving the scraped data
df=pd.DataFrame(headlines)