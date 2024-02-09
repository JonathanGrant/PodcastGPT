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
import imaplib
import email
import bs4
import re
import functools
from email.header import decode_header
from email.utils import parsedate_to_datetime
import datetime
from lxml import etree as ET
import pytz
import html

from ChatPodcastGPT import *
# -

MAX_LENGTH = 50_000
TXT_MODEL = 'gpt-3.5-turbo-0125'


def clean_text(text):
    # HTML unescape
    text = html.unescape(text)

    # Replace non-breaking spaces and other similar whitespace characters
    text = text.replace(u'\xa0', ' ').replace(u'\u200c', '')

    # Optional: Remove other unwanted characters or sequences
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters

    # Correcting misplaced spaces before apostrophes
    text = re.sub(r"\s+'", "'", text)
    
    # Correcting a space + s pattern that should be 's
    text = re.sub(r"(\w+)\s+s\b", r"\1's", text)

    # Removing extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


class ZohoMail:
    def __init__(self, username, password):
        self._mail = imaplib.IMAP4_SSL('imap.zoho.com')
        self._username = username
        self._password = password
        self.login()

    def list_ids(self, *args, **kwargs):
        resp = self._mail.uid('search', 'ALL', *args, **kwargs)
        return resp[1][0].split()

    @functools.cache
    def get_msg(self, msg_id):
        _status, msg = self._mail.uid('fetch', msg_id, '(RFC822)')
        msg = email.message_from_bytes(msg[0][1])
        return msg
    
    def get_msg_parts(self, msg_id):
        msg = self.get_msg(msg_id)
        parts = msg.walk()
        return list(parts)

    def get_html_text(self, msg_id, raw=False):
        parts = self.get_msg_parts(msg_id)
        text_content = []
        for part in parts:
            if part.get_content_type() != 'text/html':
                continue
            if not raw:
                soup = bs4.BeautifulSoup(part.get_payload(decode=True).decode(), 'html.parser')
                text_content.append(clean_text(soup.get_text()))
            else:
                text_content.append(part.get_payload(decode=True).decode())
        return '\n'.join(text_content)

    def get_email_metadata(self, msg_id):
        msg = self.get_msg(msg_id)
        from_header = decode_header(msg.get("From"))[0]
        sender = from_header[0]
        if isinstance(sender, bytes):
            # if it's a bytes type, decode to str
            sender = sender.decode(from_header[1])
        # Decode email subject
        subject_header = decode_header(msg.get("Subject"))[0]
        subject = subject_header[0]
        if isinstance(subject, bytes):
            # if it's a bytes type, decode to str
            subject = subject.decode(subject_header[1])
        # Date
        date = parsedate_to_datetime(msg.get("Date"))
        return {'sender': sender, 'subject': subject, 'date': date}
    
    def login(self):
        self._mail.login('jonreads@zoho.com', 'pxg8myj6GMX!zqn@pfd')
        self._mail.select('INBOX')


# +
# mail = ZohoMail('jonreads@zoho.com', 'pxg8myj6GMX!zqn@pfd')
# mail.list_ids()
# # mail.get_email_metadata(b'')
# txt = mail.get_html_text(b'43')
# with open('/Users/jong/Documents/example_email.txt', 'w') as f:
#     f.write(txt)
# -

def rewrite_email(txt):
    chat = Chat(
        '''Clean and slightly re-write text into something that has a strong voice.
Keep the information the same.
Keep the length the same.
Separate every few sentances with a single newline character.''',
    )
    return chat.message(txt, model='gpt-4-0125-preview')


# +
# rewrite_email(txt)

# +
class PodcastXMLHandler:
    def __init__(self):
        self.root = ET.Element("channel")  # 'channel' is typically used in podcast RSS feeds
        self.tree = ET.ElementTree(self.root)

    def to_xml(self, filepath):
        self.tree.write(filepath, encoding='utf-8', xml_declaration=True, pretty_print=True)

    @classmethod
    def from_xml(cls, filepath):
        self = cls()
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        return self

    def contains_episode(self, episode_name):
        for episode in self.root.findall('./channel/item'):
            title = episode.find('title').text
            if title == episode_name:
                return True
        return False

"""
pd = PodcastXMLHandler.from_xml('/Users/jong/Downloads/podcast.xml')
pd.contains_episode('cs.IR: Recent Research Papers on Data Science and Cybersecurity.')
pd.remove_episodes_older_than(datetime.timedelta(days=30))
pd.to_xml('/Users/jong/Downloads/podcast2.xml')
"""
pass


# -

def run(timediff):
    mail = ZohoMail('jonreads@zoho.com', 'pxg8myj6GMX!zqn@pfd')

    podcast_args = {
        'org': 'JonReads',
        'repo': 'JonReads.github.io',
        'xml_path': 'podcast.xml',
        'clean_timedelta': datetime.timedelta(days=30),
    }
    pd = PodcastRSSFeed(**podcast_args)
    pd.remove_episodes_older_than(open(pd.download_podcast_xml()).read(), podcast_args['clean_timedelta'])
    pd = PodcastXMLHandler.from_xml(pd.download_podcast_xml())
    now = datetime.datetime.now(pytz.utc)

    for mid in mail.list_ids():
        mail_meta = mail.get_email_metadata(mid)
        if now - mail_meta['date'] > timediff:
            continue

        title_long = f'[{mail_meta["sender"].split(" <")[0]}] {re.sub(r"[^a-zA-Z0-9]", "_", mail_meta["subject"])}'
        title = title_long[:200]
        if pd.contains_episode(title):
            continue

        # Make and publish episode
        logger.info(f'Making {mail_meta=} {mid=} {now - mail_meta["date"]}')
        ep_text = mail.get_html_text(mid)
        if len(ep_text) > MAX_LENGTH:
            continue
        ep_text = rewrite_email(ep_text).split('\n')
        ep_lines = [''.join(ep_text[i:i+3]) for i in range(0, len(ep_text), 3)]  # Turn into speakings

        ep = Episode(
            topic=title,
            episode_type='pure_tts',
            podcast_args=podcast_args.values(),
            host_voices=get_random_voices(2),
        )
        ep.step(msg=ep_lines)
        ep.upload(title, f'{title_long}')

# +
# run(datetime.timedelta(days=1))
# -




