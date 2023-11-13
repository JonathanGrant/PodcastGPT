# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import requests
import datetime
import re
from bs4 import BeautifulSoup as BS
import jonlog
from ChatPodcastGPT import *

logger = jonlog.getLogger()


# -

class MoneyStuff:
    BASE_URL = 'https://newsletterhunt.com/'
    LIST_URL = BASE_URL + 'newsletters/money-stuff-by-matt-levine'
    
    def list(self):
        html_content = requests.get(self.LIST_URL).content
        soup = BS(html_content, 'html.parser')
        now = datetime.datetime.now()
        # Extracting articles with their titles, publish dates, and links
        articles = []
        for article in soup.find_all('article'):
            title = article.find('h2')
            date_text = article.find('time').get_text(strip=True) if article.find('time') else None
            link = article.find('a', href=True)
        
            if title and date_text and link and 'hours ago' in date_text.lower():
                articles.append({
                    'title': title.get_text(strip=True),
                    'link': link['href'],
                })
        return articles

    def get(self, article):
        html_content = requests.get(self.BASE_URL + article['link']).content
        soup = BS(html_content, 'html.parser')
        soup = BS(str(soup.find('iframe')['srcdoc']), 'html.parser')
        all_text = soup.find(class_='body-component__content').get_text()
        # Replace carriage returns and non-breaking spaces with normal spaces
        text = all_text.replace('\r', '\n').replace('\xa0', ' ')
        text = re.sub('\[\d\]', '', text)
        lines = [x for x in text.split('\n') if x]
        return lines


def run(ndays=1):
    articles = MoneyStuff().list()
    now = datetime.datetime.now()
    for article in articles:
        logger.info(f"Writing {article}")
        lines = MoneyStuff().get(article)
        ep = Episode(
            topic=article['title'],
            episode_type='pure_tts',
        )
        ep.step(msg=lines)
        ep.upload("(New OpenAI APIs v1) [MoneyStuff] " + article['title'][:200], f'Test MoneyStuff tts: {article["title"]}')




