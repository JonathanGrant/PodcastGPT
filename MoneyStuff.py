# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
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

def clean_text(txt):
    txt = txt.replace('\r', '\n').replace('\xa0', ' ')
    txt = re.sub('\[\d\]', '', txt)
    return txt


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
        
            if title and date_text and link and 'hours ago' in date_text:
                articles.append({
                    'title': title.get_text(strip=True),
                    'link': link['href']
                })
        return articles

    def get(self, article):
        html_content = requests.get(self.BASE_URL + article['link']).content
        soup = BS(html_content, 'html.parser')
        soup = BS(str(soup.find('iframe')['srcdoc']), 'html.parser')
        all_text = soup.find(class_='body-component__content').get_text()
        # Replace carriage returns and non-breaking spaces with normal spaces
        text = clean_text(all_text)
        lines = [x for x in text.split('\n') if x]
        return lines


# +
# class ZviBlog:
#     BASE_URL = "https://thezvi.wordpress.com/"
#     LIST_URL = BASE_URL

#     def list(self):
#         html_content = requests.get(self.LIST_URL).content
#         soup = BS(html_content, 'html.parser')
#         articles = soup.find_all('div', class_='post')
    
#         # Current time and 24 hours ago
#         now = datetime.datetime.now()
#         one_day_ago = now - datetime.timedelta(days=2)
    
#         articles_list = []
    
#         for article in articles:
#             # Extracting the title and URL - these selectors might need adjustment
#             title = clean_text(article.find(class_='entry-title').get_text(strip=True))
#             article_url = article.find('a', href=True)['href']
#             # Extracting and parsing the publish datetime - this format might need adjustment
#             article_date = article_url.split(self.BASE_URL)[1][:10]
#             article_time = article.find(title=True)['title']
#             datetime_str = f'{article_date} {article_time}'
#             article_datetime = datetime.datetime.strptime(datetime_str, '%Y/%m/%d %I:%M %p')
    
#             if article_datetime >= one_day_ago:
#                 articles_list.append({
#                     'datetime': article_datetime, 'title': title, 'link': article_url
#                 })

#         return articles_list

#     def get(self, article):
#         html_content = requests.get(article['link']).content
#         soup = BS(html_content, 'html.parser')
#         all_text = soup.find('div', class_='entry-content').get_text()
#         # Replace carriage returns and non-breaking spaces with normal spaces
#         text = clean_text(all_text)
#         lines = [x for x in text.split('\n') if x]
#         return lines
# -

def run(ndays=1):
    for src, prefix in [(MoneyStuff(), 'MoneyStuff')]:
        print(prefix)
        if prefix == 'MoneyStuff': continue
        articles = src.list()
        now = datetime.datetime.now()
        for article in articles:
            logger.info(f"Writing {article}")
            lines = src.get(article)
            ep = Episode(
                topic=article['title'],
                episode_type='pure_tts',
            )
            ep.step(msg=lines)
            ep.upload(f"[{prefix}] " + article['title'][:200], f'{prefix} tts: {article["title"]}')

# +
# run(1)

# +
# # %debug
# -


