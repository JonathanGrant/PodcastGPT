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

MAX_TOKENS = 60_000 # GPT4-128k
JOIN_NUM_DEFAULT = 300
DEFAULT_TEXTGEN_MODEL = 'gpt-4-1106-preview'


# -

class PDFEpisode(Episode):
    PDFPart = collections.namedtuple('PDFPart', 'title text')

    def __init__(self, title, model=DEFAULT_TEXTGEN_MODEL, **kwargs):
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

    def split_into_parts(self, text, max_tokens=MAX_TOKENS):
        """Split the text into parts based on titles and tokens."""
        lines = text.split("\n")
        parts = []
        current_part = []
        current_title = 'Paper'
        for line in lines:
            current_part.append(line)

            while Chat.num_tokens_from_text('\n'.join(current_part)) > max_tokens:
                part_text = '\n'.join(current_part)
                shortened_part, current_part = part_text[:max_tokens * 2], [part_text[max_tokens * 2:]]
                logger.info("PartAdd1")
                parts.append(self.PDFPart(current_title, shortened_part))

        if current_part:
            logger.info("PartAdd2")
            parts.append(self.PDFPart(current_title, "\n".join(current_part)))
        return parts

    def process_pdf(self, pdf_path):
        text = self.parse_pdf(pdf_path)
        parts = self.split_into_parts(text)
        return parts

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def write_one_part(self, chat_msg):
        chat = PodcastChat(**{**self._kwargs, 'topic': self.title})
        msg, aud = chat.step(msg=chat_msg, model=self.model, ret_aud=True)
        return msg, aud

    def step(self):
        outline = self.data[0].text

        # Get parts
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as tpe:
            jobs = ([
                tpe.submit(self.write_one_part, f"""Explain the paper \"{self.title}\" in full verbose detail.
Assume the listener doesn't know anything.
Follow this guide:

Introduction (500+ words)
Contextual background: Why this paper is significant in its field.
Key Concepts and Background: Explanation of the main scientific concepts or theories addressed in the paper.
(Optional) Breakdown of complex vocabulary used.

Core (2500+ words)
Detailed discussion of the research paper’s objectives.
Methodology and techniques used.
Key findings and results.

Implications and Applications (500+ words)
Analysis of the potential impact of these findings on the field.

Conclusion (500+ words)
Recap of the main points discussed in the episode.
Personal reflections on the paper and its broader relevance.

Respond with the hosts names before each line like {self.chat._hosts[0]}: and {self.chat._hosts[1]}:
The text in the paper is:
{part.text}
""")
                for part in self.data
            ])
            job2idx = {j:i for i, j in enumerate(jobs)}
            self.sounds, self.summary_texts = [None] * len(jobs), [None] * len(jobs)
            for i, job in enumerate(concurrent.futures.as_completed(jobs)):
                logger.info(f"Part: {i} / {len(jobs)} = {100.0*i/len(jobs):,.5f}%")
                jobid = job2idx[job]
                text, aud = job.result()
                self.summary_texts[jobid] = text
                self.sounds[jobid] = aud

        return outline, '\n'.join(self.summary_texts)


class ArxivEpisode(PDFEpisode):
    ArxivPart = collections.namedtuple('ArxivPart', 'title text')

    def __init__(self, arxiv_id, model=DEFAULT_TEXTGEN_MODEL, **kwargs):
        self.arxiv_id = arxiv_id
        self.model = model
        self.data = self.process_pdf(self.arxiv_id)
        self.title = self.arxiv_title = self.get_title(self.arxiv_id)
        self._kwargs = kwargs
        super().__init__(title=self.arxiv_title, topic=self.arxiv_title, **kwargs)

    def split_into_parts(self, text, max_tokens=MAX_TOKENS):
        """Split the text into parts based on tokens."""
        lines = text.split("\n")
        parts = []
        current_part = [text]
        current_title = 'Paper'

        while Chat.num_tokens_from_text('\n'.join(current_part)) > max_tokens:
            part_text = '\n'.join(current_part)
            shortened_part, current_part = part_text[:max_tokens * 2], [part_text[max_tokens * 2:]]
            logger.info("PartAdd3")
            parts.append(self.ArxivPart(current_title, shortened_part))

        if current_part:
            logger.info(f"PartAdd4, {len(parts)}")
            parts.append(self.ArxivPart(current_title, "\n".join(current_part)))
        if len(parts) > 1:
            raise Exception("More than 1 part, giving up")
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
        chat = PodcastChat(f"Very short commercial for {company}", host_voices=[OpenAITTS(OpenAITTS.MAN), OpenAITTS(OpenAITTS.WOMAN)])
        chat._history[-1] = {"role": "user", "content": f"Generate a very funny, weird, and short commercial for {company}, who is sponsoring the podcast."}
        return chat.step(model=DEFAULT_TEXTGEN_MODEL)


class ArxivRunner:
    def __init__(self, category, start=0, limit=5):
        self.category = category
        self.start = start
        self.limit = limit

    def get_top(self):
        """Retrieve top Arxiv entries based on category."""
        # url = f'http://export.arxiv.org/api/query?search_query=cat:{self.category}&start={self.start}' \
        #       f'&max_results={self.limit}&sortBy=submittedDate&sortOrder=descending'
        url = f'https://arxiv.org/list/{self.category}/recent'
        print(url)
        html = requests.get(url).content
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        for item in soup.find_all('dt'):
            title = item.find_next_sibling('dd').find('div', class_='list-title').text.replace('Title:', '').strip()
            identifier = item.find('span', class_='list-identifier').a.text
            pdf_link = 'https://arxiv.org' + item.find('span', class_='list-identifier').find('a', title='Download PDF')['href']
        
            articles.append({
                'title': title,
                'ID': identifier,
                'pdf': pdf_link
            })
        return [a["pdf"].split('/')[-1] for a in articles]
        
        # data = feedparser.parse(url)
        # return [entry['id'].split('/')[-1] for entry in data['entries']]


# +
JINGLE_FILE_PATH = 'jazzstep.mp3'
MODEL = DEFAULT_TEXTGEN_MODEL
HOST_VOICES = [OpenAITTS(OpenAITTS.MAN), OpenAITTS(OpenAITTS.WOMAN)]
PODCAST_ARGS = ("ArxivPodcastGPT", "ArxivPodcastGPT.github.io", "podcasts/ComputerScience/Consolidated/podcast.xml")

def create_large_episode(arxiv_category, limit=5):
    """Create a podcast episode with Arxiv papers."""
    with open(JINGLE_FILE_PATH, 'rb') as jingle_file:
        jingle_audio = jingle_file.read()
    jingle_audio = jingle_audio[:len(jingle_audio) // 4]  # Shorten to just 4 sec

    audios, texts = [jingle_audio], []
    successes = 0
    
    for arxiv_id in ArxivRunner(arxiv_category, limit=limit).get_top():
        if successes >= limit:
            break

        logger.info(f"Trying arxiv ID {arxiv_id} in {arxiv_category} with {successes}/{limit}")
        try:
            arxiv_episode = ArxivEpisode(arxiv_id, model=MODEL, podcast_args=PODCAST_ARGS, host_voices=HOST_VOICES)
            outline, txt = arxiv_episode.step()
            logger.info(f"Got outline: {outline[:500]}")
        except Exception as e:
            logger.exception(f"Error processing arxiv_id {arxiv_id}: {e}")
            continue

        audios.append(b''.join(arxiv_episode.sounds))
        audios.append(jingle_audio)
        arxiv_title = re.sub('[^0-9a-zA-Z]+', ' ', arxiv_episode.arxiv_title)
        texts.append(f'ChatGPT generated podcast using model={MODEL} for https://arxiv.org/abs/{arxiv_id} {arxiv_title}')
        successes += 1
        
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

def run(arxiv_category, upload=True, limit=5):
    # TODO: Multi thread each part
    audios, texts = create_large_episode(arxiv_category, limit=limit)
    ep = AudioCompletedEpisode(audios, podcast_args=PODCAST_ARGS)
    if upload:
        ep.upload(f'{datetime.datetime.now():%Y-%m-%d} {arxiv_category}: {get_title(texts)}', '\n\n'.join(texts))
    return ep

# +
# ep = run("astro-ph", upload=False, limit=5)
# IPython.display.Audio(b''.join(ep.sounds))

# +
# ep.upload(f'{datetime.datetime.now():%Y-%m-%d} astro-ph', '\n\n'.join(texts))

# +
# IPython.display.Audio(b''.join(ep.sounds))
# -


