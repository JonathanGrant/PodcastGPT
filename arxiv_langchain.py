# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: '3.11'
#     language: python
#     name: '3.11'
# ---

# +
from ChatPodcastGPT import *
import collections
import concurrent.futures
import os
import feedparser
import logging
import re
import tempfile
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
import retrying
import random
from IPython.display import Audio
import datetime

MAX_TOKENS = 4096
JOIN_NUM_DEFAULT = 300


# -

class PDFEpisode(Episode):
    PDFPart = collections.namedtuple('PDFPart', 'title text')

    def __init__(self, title, model='gpt-3.5-turbo-16k', **kwargs):
        self.title = title
        self.model = model
        self.topic = kwargs.pop('topic', self.title) or self.title
        self._kwargs = kwargs
        self.join_num = JOIN_NUM_DEFAULT
        if 'podcast_args' in self._kwargs: self._kwargs.pop('podcast_args')
        super().__init__(topic=self.topic, **kwargs)

    def parse_pdf(self, file):
        """Parse a PDF and extract the text."""
        with open(file, "rb") as f:
            pdf = PdfReader(f)
            return ''.join(page.extract_text() for page in pdf.pages)

    def split_into_parts(self, text, max_tokens=MAX_TOKENS // 2):
        """Split the text into parts based on titles and tokens."""
        lines = text.split("\n")
        parts = []
        current_part = []
        current_title = 'Abstract'
        for line in lines:
            if re.match(r'\d+\s[A-Za-z]', line):
                if current_part:
                    parts.append(self.PDFPart(current_title, "\n".join(current_part)))
                current_title = line
                current_part = []
            else:
                current_part.append(line)

            while Chat.num_tokens_from_text('\n'.join(current_part)) > max_tokens:
                part_text = '\n'.join(current_part)
                shortened_part, current_part = part_text[:max_tokens * 2], [part_text[max_tokens * 2:]]
                parts.append(self.PDFPart(current_title, shortened_part))

        if current_part:
            parts.append(self.PDFPart(current_title, "\n".join(current_part)))
        return parts

    def process_pdf(self, pdf_path):
        text = self.parse_pdf(pdf_path)
        parts = self.split_into_parts(text)
        return parts
    
    def write_one_part(self, chat_msg):
        chat = PodcastChat(**{**self._kwargs, 'topic': self.title})
        msg = chat.step(msg=chat_msg, model=self.model, skip_aud=True)
        return msg

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def concat_podcast_parts(self, texts):
        """Given list of texts from chatGPT about the podcast."""
        chat = PodcastChat(max_length=12_000, **{**self._kwargs, 'topic': self.title})
        msg = "Rewrite the following podcast episode as one complete single episode.\n"
        msg += "\n".join(texts)
        msg, aud = chat.step(msg=msg, model='gpt-3.5-turbo-16k', ret_aud=True)
        if len(msg) < 500:
            raise ValueError(f"Returned msg too short. Suspecting an error. [{msg=}]")
        return msg, aud

    def step(self):
        include = f" Remember to respond with the hosts names like {self.chat._hosts[0]}: and {self.chat._hosts[1]}:"
        outline = self.data[0].text
        # logger.info(f"Outline: {outline}")
        intro_msg = f"Write the intro for a podcast about a paper: {self.title}. The abstract for the paper is {outline}. Only write the introduction.{include}"

        # Get parts
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as tpe:
            jobs = [tpe.submit(self.write_one_part, intro_msg)]
            jobs.extend([
                tpe.submit(self.write_one_part, f"Rewrite the text from the paper {self.title} part {part.title} into a podcast section. Explain everything other than the title as if the listener has no idea. Do not include any intro such as saying welcome back, just get right to it. The text in the paper is: {part.text}.{include}")
                for part in self.data
            ])
            job2idx = {j:i for i, j in enumerate(jobs)}
            self.texts  = [None] * len(jobs)
            for i, job in enumerate(concurrent.futures.as_completed(jobs)):
                logger.info(f"Part: {i} / {len(jobs)} = {100.0*i/len(jobs):,.5f}%")
                jobid = job2idx[job]
                text = job.result()
                self.texts[jobid] = text

            # Combine texts into one podcast and ask chatGPT to re-write it.
            jobs = [
                tpe.submit(self.concat_podcast_parts, self.texts[txt_i*self.join_num:(txt_i+1)*self.join_num])
                for txt_i in range(0, len(self.texts), self.join_num)
            ]
            job2idx = {j:i for i, j in enumerate(jobs)}
            self.sounds, self.summary_texts = [None] * len(jobs), [None] * len(jobs)
            for i, job in enumerate(concurrent.futures.as_completed(jobs)):
                logger.info(f"Join Part: {i} / {len(jobs)} = {100.0*i/len(jobs):,.5f}%")
                jobid = job2idx[job]
                text, aud = job.result()
                self.sounds[jobid], self.summary_texts[jobid] = aud, text

        return outline, '\n'.join(self.summary_texts)


class ArxivEpisode(PDFEpisode):
    ArxivPart = collections.namedtuple('ArxivPart', 'title text')

    def __init__(self, arxiv_id, model='gpt-3.5-turbo-16k', **kwargs):
        self.arxiv_id = arxiv_id
        self.model = model
        self.data = self.process_pdf(self.arxiv_id)
        self.title = self.arxiv_title = self.get_title(self.arxiv_id)
        self._kwargs = kwargs
        super().__init__(title=self.arxiv_title, topic=self.arxiv_title, **kwargs)

    def split_into_parts(self, text, max_tokens=8_000):
        """Split the text into parts based on tokens."""
        lines = text.split("\n")
        parts = []
        current_part = [text]
        current_title = 'Paper'

        while Chat.num_tokens_from_text('\n'.join(current_part)) > max_tokens:
            part_text = '\n'.join(current_part)
            shortened_part, current_part = part_text[:max_tokens * 2], [part_text[max_tokens * 2:]]
            parts.append(self.ArxivPart(current_title, shortened_part))

        if current_part:
            parts.append(self.ArxivPart(current_title, "\n".join(current_part)))
        return parts

    def process_pdf(self, arxiv_id):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "file.pdf")
            self.arxiv_download(arxiv_id, file)
            text = self.parse_pdf(file)
        parts = self.split_into_parts(text)
        return parts
    
    def arxiv_download(self, arxiv_id, out_file):
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url)
        with open(out_file, "wb") as f:
            f.write(response.content)
    
    def get_title(self, arxiv_id):
        url = f"https://arxiv.org/abs/{arxiv_id}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', {'class': 'title mathjax'}).text.strip().split('\n')[-1].strip()
        return title


class CommercialGenerator:
    def get_random_company(self):
        chat = Chat("Return simple plaintext responses only.")
        with open("nouns.txt") as f:
            nouns = f.read().splitlines()
        random_noun = random.choice(nouns)
        return chat.message(f"Write just 1 funny, weird, creative made up company that doesn't exist involving {random_noun}.", temperature=1)

    def generate(self, company=None):
        if company is None:
            company = self.get_random_company()
        chat = PodcastChat(f"Very short commercial for {company}", host_voices=[GttsTTS(GttsTTS.MAN), GttsTTS(GttsTTS.WOMAN)])
        chat._history[-1] = {"role": "user", "content": f"Generate a very funny, weird, and short commercial for {company}. Make sure to say phrases like 'This podcast brought to you by' or 'Sponsored by'"}
        return chat.step()


class ArxivRunner:
    def __init__(self, category, start=0, limit=5):
        self.category = category
        self.start = start
        self.limit = limit

    def get_top(self):
        """Retrieve top Arxiv entries based on category."""
        url = f'http://export.arxiv.org/api/query?search_query=cat:{self.category}&start={self.start}' \
              f'&max_results={self.limit}&sortBy=lastUpdatedDate&sortOrder=descending'
        data = feedparser.parse(url)
        return [entry['id'].split('/')[-1] for entry in data['entries']]


# +
JINGLE_FILE_PATH = 'jazzstep.mp3'
MODEL = 'gpt-3.5-turbo-16k'
HOST_VOICES = [GttsTTS(GttsTTS.MAN), GttsTTS(GttsTTS.WOMAN)]
PODCAST_ARGS = ("ArxivPodcastGPT", "ArxivPodcastGPT.github.io", "podcasts/ComputerScience/Consolidated/podcast.xml")

def create_large_episode(arxiv_category, limit=5):
    """Create a podcast episode with Arxiv papers."""
    with open(JINGLE_FILE_PATH, 'rb') as jingle_file:
        jingle_audio = jingle_file.read()
    jingle_audio = jingle_audio[:len(jingle_audio) // 4]  # Shorten to just 4 sec

    audios, texts = [jingle_audio], []
    
    for arxiv_id in ArxivRunner(arxiv_category, limit=limit).get_top():
        try:
            arxiv_episode = ArxivEpisode(arxiv_id, model=MODEL, podcast_args=PODCAST_ARGS, host_voices=HOST_VOICES)
            outline, txt = arxiv_episode.step()
        except Exception as e:
            logger.exception(f"Error processing arxiv_id {arxiv_id}: {e}")
            continue

        audios.append(b''.join(arxiv_episode.sounds))
        audios.append(jingle_audio)
        arxiv_title = re.sub('[^0-9a-zA-Z]+', ' ', arxiv_episode.arxiv_title)
        texts.append(f'ChatGPT generated podcast using model={MODEL} for https://arxiv.org/abs/{arxiv_id} {arxiv_title}')
        
        try:
            commercial_text, commercial_sound = CommercialGenerator().generate()
            audios.append(commercial_sound)
            audios.append(jingle_audio)
        except Exception as e:
            logger.error("Unable to generate commercial")
            logger.exception(e)
    
    return audios, texts


# -

def get_title(texts):
    chat = Chat("Return just simple plaintext.")
    return chat.message(
        "Given the following papers, write a clickbait title that captures all of them. " + 
        ", ".join(txt.split(' Title ')[-1] for txt in texts)
    )


class AudioCompletedEpisode(Episode):
    def __init__(self, sounds, podcast_args):
        self.sounds = sounds
        self.pod = PodcastRSSFeed(*podcast_args)


# +
arxiv_categories = ["AI", "CL", "CC", "CE", "CG", "GT", "CV", "CY", "CR", "DS", "DB", "DL", "DM", "DC", "ET", "FL", "GL", "GR", "AR", "HC", "IR", "IT", "LO", "LG", "MS", "MA", "MM", "NI", "NE", "NA", "OS", "OH", "PF", "PL", "RO", "SI", "SE", "SD", "SC", "SY"]

def run(arxiv_category):
    # TODO: Multi thread each part
    audios, texts = create_large_episode(arxiv_category)
    ep = AudioCompletedEpisode(audios, podcast_args=PODCAST_ARGS)
    ep.upload(f'{datetime.datetime.now():%Y-%m-%d} {arxiv_category}: {get_title(texts)}', '\n\n'.join(texts))


# -

"""TODO:

"""
pass

# +
# a, b = CommercialGenerator().generate()
# a
# -


