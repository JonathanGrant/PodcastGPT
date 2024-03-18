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
from urllib.parse import unquote

# MAX_TOKENS = 60_000 # GPT4-128k
MAX_TOKENS = 60_000
JOIN_NUM_DEFAULT = 300
# DEFAULT_TEXTGEN_MODEL = 'gpt-4-0125-preview'
# DEFAULT_TEXTGEN_MODEL = "AWS/" + AWSChat.MODELS["claude-3-sonnet"]
DEFAULT_TEXTGEN_MODEL = "ANTHROPIC/" + AnthropicChat.MODELS["claude-haiku"]
JINGLE_FILE_PATH = 'jazzstep.mp3'
with open(JINGLE_FILE_PATH, 'rb') as jingle_file:
    JINGLE_AUDIO = jingle_file.read()
JINGLE_AUDIO = JINGLE_AUDIO[:len(JINGLE_AUDIO) // 4]  # Shorten to just 4 sec
# -

METAPROMPT_SYSTEM = """You are an award-winning podcast with hosts Tom and Jen. Your podcast investigates research papers
with intrigue and depth, blending expert analysis with compelling storytelling to illuminate
cutting-edge discoveries.

Here is the title of the research paper you will be discussing in this episode:

<paper_title>
{PAPER_TITLE}
</paper_title>

And here is the full text of the paper:

<paper_text>
{ENTIRE_PAPER_TEXT}
</paper_text>

To create this podcast episode, follow these steps:

<intro>
Begin the episode with an engaging introduction that captures the listener's attention. Have Tom and
Jen introduce themselves and the name of the podcast. Then, introduce the paper you will be
discussing by stating its title and authors. Provide a brief, high-level overview of the paper's
main findings and conclusions to pique the listener's interest.
</intro>

<body>
Next, dive into the details of the paper. Explain the research question, hypothesis, methodology,
results, and implications in an engaging, story-like manner. Use dialogue between Tom and Jen to
make the explanation more conversational and accessible. Assume the listener has no prior knowledge
of the topic, so be sure to define any technical terms and provide clear explanations of complex
concepts.

As you discuss the paper, emphasize strong narrative storytelling. Bring the research to life by
painting a vivid picture of the experiments conducted, the challenges faced by the researchers, and
the excitement of their discoveries. Use analogies, examples, and anecdotes to make the science more
relatable and memorable.

Throughout the discussion, explore the broader relevance and potential impact of the research.
Consider how the findings might be applied in the real world, what new questions they raise, and how
they fit into the bigger picture of the field.
</body>

<reflections>
After thoroughly explaining the paper, have Tom and Jen offer their personal reflections and
opinions. What did they find most interesting or surprising about the research? What are the
strengths and limitations of the study? Do they agree with the authors' conclusions? Encourage a
thoughtful and nuanced discussion that considers multiple perspectives.
</reflections>

<conclusion>
Conclude the podcast episode with a summary of the key takeaways from the paper. Reiterate the most
important findings and their implications. End on a thought-provoking note that leaves the listener
with something to ponder.
</conclusion>

<format>
Format the entire podcast transcript with Tom: and Jen: before each line to indicate which host is
speaking. Use line breaks and paragraph spacing to clearly distinguish between different sections
and ideas.
</format>

<answer>
Write the complete podcast episode transcript here, following the structure outlined above. Tell the
story of the paper completely, in full verbose detail, as a single response. Make sure to teach
complex topics in an intuitive way. The episode should be informative, entertaining, and very
detailed - a systematic and narrative review of the research paper.
</answer>"""

SHORT_SYSTEM = """You are an award-winning podcast with hosts Tom and Jen.
Your podcast has {style} commercials relevant to the paper you just covered.

<format>
Format the entire podcast commercial transcript with Tom: and Jen: before each line to indicate which host is
speaking. Use line breaks and paragraph spacing to clearly distinguish between different sections
and ideas.
</format>"""
COMMERCIAL_STYLES = [
    "insane",
    "multi-dimensional",
    "star wars",
    "medieval fantasy",
    "AGI themed",
    "kitchen gadget",
    "get rich quick scheme",
    "bizarre sports training equipment",
    "retro-futurism",
    "underwater cities",
    "space colonization",
    "time travel adventures",
    "cyberpunk gadgets",
    "steampunk inventions",
    "utopian societies",
    "dystopian futures",
    "virtual reality experiences",
    "augmented reality tools",
    "eco-friendly innovations",
    "survival gear for the apocalypse",
    "alien technology",
    "superhero gadgets",
    "magical artifacts",
    "historical reenactments",
    "luxury lifestyle",
    "minimalist living",
    "smart home devices",
    "extreme sports equipment",
    "pet care innovations",
    "health and wellness gadgets",
    "beauty and personal care inventions",
    "fashion and style trends",
    "food and beverage innovations",
    "art and design tools",
    "music and entertainment technology",
    "travel and adventure gear",
    "educational tools and toys",
    "gaming and esports equipment",
    "automotive and transportation innovations",
    "agricultural and gardening innovations",
    "construction and DIY tools",
    "safety and security gadgets",
    "finance and investment tools",
    "social networking innovations",
    "news and media trends",
    "philanthropy and social impact",
    "spirituality and mindfulness",
    "cultural and ethnic heritage",
    "sci-fi and fantasy gadgets",
    "mythological creatures and worlds",
    "ancient civilizations and technologies",
    "parallel universes",
    "mystery and detective gear",
    "horror and thriller themes",
    "romantic and relationship aids",
    "comedy and satire products",
    "children's toys and games",
    "teen lifestyle and gadgets",
    "elderly care innovations",
    "healthcare and medical devices",
    "space exploration tools",
    "underground living",
    "floating cities",
    "digital nomad gadgets",
    "off-grid living essentials",
    "extreme weather gear",
    "wildlife and nature exploration",
    "ocean exploration technologies",
    "mountain climbing equipment",
    "desert survival gear",
    "polar exploration tools",
    "jungle survival equipment",
    "urban living innovations",
    "rural living essentials",
    "fantasy sports and leagues",
    "reality bending devices",
    "memory enhancement tools",
    "intelligence augmentation",
    "emotional wellbeing gadgets",
    "interdimensional travel devices",
    "quantum computing applications",
    "nano-technology gadgets",
    "biotechnology innovations",
    "genetic engineering kits",
    "robotics and automation",
    "artificial intelligence applications",
    "blockchain and cryptocurrency tools",
    "virtual worlds and metaverses",
    "cyberspace security",
    "ethical hacking tools",
    "spy and surveillance gadgets",
    "military and defense innovations",
    "peacekeeping and conflict resolution",
    "disaster relief and recovery",
    "sustainable living solutions",
    "renewable energy gadgets",
    "waste management innovations",
    "water purification technologies",
    "air quality improvement devices",
    "soil regeneration and protection",
    "wildlife conservation tools",
    "climate change mitigation",
    "space debris management",
    "asteroid mining technologies",
    "universal translation devices",
    "teleportation technologies",
    "anti-gravity devices",
    "time manipulation gadgets"
]


def clean_text(text):
    # Remove References Section
    text = re.sub(r'\bReferences?\b.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove Footnotes Section
    text = re.sub(r'\bFootnotes?\b.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove Figure/Table Insertions
    text = re.sub(r'- INSERT FIGURE \d AROUND HERE -', '', text)
    text = re.sub(r'- INSERT TABLE \d AROUND HERE -', '', text)
    # Remove extra whitespaces but keep line breaks
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text


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

    @classmethod
    def from_file(cls, filepath, *args, **kwargs):
        ep = cls(*args, **kwargs)
        ep.data = ep.process_pdf(filepath)
        return ep

    def parse_pdf(self, file):
        """Parse a PDF and extract the text."""
        try:
            with open(file, "rb") as f:
                pdf = PdfReader(f)    
                txt = clean_text('\n'.join(page.extract_text() for page in pdf.pages))
        except:
            with open(file, 'ab') as f:
                f.write(b'%%EOF')
            with open(file, "rb") as f:
                pdf = PdfReader(f)    
                txt = clean_text('\n'.join(page.extract_text() for page in pdf.pages))
        if len(txt) < 1000: raise Exception("Text too small, cleaner must have messed up.")
        return txt

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
    def write_one_part(self, chat_msg, with_commercial=False):
        extra_system = f"""This podcast investigates research papers with intrigue and depth, blending expert analysis with compelling storytelling to illuminate cutting-edge discoveries.
Similar to the style of the Darknet Diaries podcast.
Tell the story of the paper completely, in full verbose detail, as a single response.
Emphasize strong narrative storytelling.
Assume the listener doesn't know anything.
Afterwards, give your personal reflections on the paper and its broader relevance.
Respond with the hosts names before each line like {self.chat._hosts[0]}: and {self.chat._hosts[1]}:
"""
        # chat = PodcastChat(**{**self._kwargs, 'topic': self.title, 'extra_system': extra_system})
        chat = PodcastChat(**{**self._kwargs, 'system': METAPROMPT_SYSTEM.format(PAPER_TITLE=self.title, ENTIRE_PAPER_TEXT=chat_msg), 'topic': self.title})
        # msg, aud = chat.step(msg=chat_msg, model=self.model, ret_aud=True, min_length=200)
        msg, aud = chat.step(msg=None, model=self.model, ret_aud=True, min_length=200)
        # chat._history.pop(2)
        # chat._history[0]['content'] = chat._history[0]['content'][:len(chat._history[0]['content']) - len(extra_system)]
        system = chat._history[0]['content']
        chat._history[0]['content'] = SHORT_SYSTEM.format(style=random.choice(COMMERCIAL_STYLES))
        print(f"{len(system)=} {len(chat._history[0]['content'])=}")
        com_msg, com_aud = chat.step(msg="Generate a funny, weird, and concise commercial for a company that now exists as a result of this paper.", model=self.model, ret_aud=True)
        msg = '\n'.join([msg, com_msg])
        try:
            aud = merge_mp3s([aud, JINGLE_AUDIO, com_aud])
        except Exception as e:
            logger.exception(e)
            raise
        return msg, aud

    def step(self):
        outline = self.data[0].text

        # Get parts
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as tpe:
            jobs = ([
                # tpe.submit(self.write_one_part, f"Title: \"{self.title}\"\nText:\n{part.text}", with_commercial=True)
                tpe.submit(self.write_one_part, part.text, with_commercial=True)
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

    def __init__(self, arxiv_id, id_is_url=False, title=None, model=DEFAULT_TEXTGEN_MODEL, **kwargs):
        self.arxiv_id = arxiv_id
        self.id_is_url = id_is_url
        self.title = title
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
        if self.id_is_url:
            url = arxiv_id
        else:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url)
        with open(out_file, "wb") as f:
            f.write(response.content)

    def get_title(self, arxiv_id):
        if self.title is not None:
            return self.title
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
        if self.category == 'psyarxiv': return self.get_top_psyarxiv()
        if self.category == 'osf': return self.get_top_osf()
        if self.category == 'econpapers': return self.get_top_econpapers()
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

    def get_top_psyarxiv(self):
        url = 'https://share.osf.io/api/v3/index-card-search?cardSearchFilter%5BresourceType%5D=Preprint&cardSearchFilter%5Bpublisher%5D%5B%5D=https%3A%2F%2Fosf.io%2Fpreprints%2Fpsyarxiv&cardSearchFilter%5BaccessService%5D=https%3A%2F%2Fosf.io%2F&cardSearchText%5B*%2Ccreator.name%2CisContainedBy.creator.name%5D=&page%5Bcursor%5D=&page%5Bsize%5D=100&sort=-dateCreated'
        print(url)
        data = requests.get(url, headers={'Accept': 'application/vnd.api+json'}).json()
        data = [x for x in data['included'] if x["type"] == "index-card"]
        return [(f"{x['attributes']['resourceIdentifier'][0]}/download/", x['attributes']['resourceMetadata']['title'][0]['@value']) for x in data]

    def get_top_osf(self):
        url = 'https://share.osf.io/api/v3/index-card-search?cardSearchFilter%5BresourceType%5D=Preprint&cardSearchFilter%5BaccessService%5D=https%3A%2F%2Fosf.io%2F&cardSearchText%5B*%2Ccreator.name%2CisContainedBy.creator.name%5D=&page%5Bcursor%5D=&page%5Bsize%5D=100&sort=-dateCreated'
        print(url)
        data = requests.get(url, headers={'Accept': 'application/vnd.api+json'}).json()
        data = [x for x in data['included'] if x["type"] == "index-card"]
        return [(f"{x['attributes']['resourceIdentifier'][0]}/download/", x['attributes']['resourceMetadata']['title'][0]['@value']) for x in data]
    
    def get_top_econpapers(self):
        base_url = 'https://econpapers.repec.org/'
        url = base_url + "scripts/search.pf?ft=&adv=true&wp=on&pl=&auth=on&online=on&sort=rank&lgc=AND&aus=&ar=on&kw=&jel=&nep=&ni=7+day&nit=epdate"
        print(url)
        html = requests.get(url).content
        soup = BeautifulSoup(html, 'html.parser')
        paper_links = [x for x in soup.find_all('a') if x['href'].startswith('/paper/') and x['href'].endswith('.htm')]
        data = []
        for link in paper_links:
            title = link.text
            url = base_url + link['href']
            html = requests.get(url).content
            soup = BeautifulSoup(html, 'html.parser')
            x = soup.find('b', text='Downloads:')
            paper_url = x.parent.find('a').text
            data.append((paper_url, title))
            if len(data) >= self.limit * 2: break
        return data


# +
MODEL = DEFAULT_TEXTGEN_MODEL
# HOST_VOICES = [OpenAITTS(OpenAITTS.MAN), OpenAITTS(OpenAITTS.WOMAN)]
# HOST_VOICES = [GoogleTTS(GoogleTTS.MAN), GoogleTTS(GoogleTTS.WOMAN)]
HOST_VOICES = get_random_voices()
PODCAST_ARGS = ("ArxivPodcastGPT", "ArxivPodcastGPT.github.io", "podcasts/ComputerScience/Consolidated/podcast.xml")

def create_large_episode(arxiv_category, limit=5, add_commercials=False):
    """Create a podcast episode with Arxiv papers."""
    audios, texts = [JINGLE_AUDIO], []
    successes = 0
    
    for arxiv_id in ArxivRunner(arxiv_category, limit=limit).get_top():
        if successes >= limit:
            break

        arxiv_kwargs = {'id_is_url': False}
        if isinstance(arxiv_id, tuple):
            arxiv_id, arxiv_title = arxiv_id
            arxiv_kwargs['title'] = arxiv_title
            arxiv_kwargs['id_is_url'] = True
            
        logger.info(f"Trying arxiv ID {arxiv_id} in {arxiv_category} with {successes}/{limit}")
        try:
            arxiv_episode = ArxivEpisode(arxiv_id, model=MODEL, podcast_args=PODCAST_ARGS, host_voices=HOST_VOICES, **arxiv_kwargs)
            outline, txt = arxiv_episode.step()
            logger.info(f"Got outline: {outline[:500]}")
        except Exception as e:
            logger.exception(f"Error processing arxiv_id {arxiv_id}: {e}")
            continue

        audios.append(merge_mp3s(arxiv_episode.sounds))
        audios.append(JINGLE_AUDIO)
        arxiv_title = re.sub('[^0-9a-zA-Z]+', ' ', arxiv_episode.arxiv_title)
        texts.append(f'ChatGPT generated podcast using model={MODEL} for https://arxiv.org/abs/{arxiv_id} {arxiv_title}')
        successes += 1
        logger.info(texts[-1])

        if not add_commercials:
            continue
        try:
            commercial_text, commercial_sound = CommercialGenerator().generate()
            audios.append(commercial_sound)
            audios.append(JINGLE_AUDIO)
        except Exception as e:
            logger.error("Unable to generate commercial")
            logger.exception(e)
    
    return audios, texts


# -

def get_title(texts):
    chat = Chat("Return just simple plaintext.")
    return chat.message(
        "Given the following papers, write a clickbait title that captures all of them. " + 
        ", ".join(txt.split(' Title ')[-1] for txt in texts),
        model=DEFAULT_TEXTGEN_MODEL
    )


class AudioCompletedEpisode(Episode):
    def __init__(self, sounds, podcast_args):
        self.sounds = sounds
        self.pod = PodcastRSSFeed(*podcast_args)


# +
arxiv_categories = ["AI", "CL", "CC", "CE", "CG", "GT", "CV", "CY", "CR", "DS", "DB", "DL", "DM", "DC", "ET", "FL", "GL", "GR", "AR", "HC", "IR", "IT", "LO", "LG", "MS", "MA", "MM", "NI", "NE", "NA", "OS", "OH", "PF", "PL", "RO", "SI", "SE", "SD", "SC", "SY"]
other_categories = ['econpapers', 'psyarxiv']

def run(arxiv_category, upload=True, limit=5):
    audios, texts = create_large_episode(arxiv_category, limit=limit)
    ep = AudioCompletedEpisode(audios, podcast_args=PODCAST_ARGS)
    if upload:
        ep.upload(f'{datetime.datetime.now():%Y-%m-%d} {arxiv_category}: {get_title(texts)}', '\n\n'.join(texts))
    return ep


# -

# Drive podcast episode with custom list of PDFs
def episode_with_pdfs(dirname, upload=None):
    papers = os.listdir(dirname)
    audios, texts = [], []
    for i, paper in enumerate(papers):
        logger.info(f"{i=}/{len(papers)} Working on {paper=}")
        title = os.path.splitext(paper)[0]
        path = os.path.join(dirname, paper)

        try:
            ep = PDFEpisode.from_file(path, title, model=MODEL, podcast_args=PODCAST_ARGS, host_voices=HOST_VOICES)
            outline, txt = ep.step()
            logger.info(f"Got outline: {outline[:100]}")
        except Exception as e:
            logger.exception(f"Error processing paper {paper=}: {e=}")
            continue

        audios.append(merge_mp3s(ep.sounds))
        audios.append(JINGLE_AUDIO)
        arxiv_title = re.sub('[^0-9a-zA-Z]+', ' ', title)
        texts.append(f'ChatGPT generated podcast using model={MODEL} for {title}')
        logger.info(texts[-1])

    ep = AudioCompletedEpisode(audios, podcast_args=PODCAST_ARGS)
    if upload is not None:
        ep.upload(f'{datetime.datetime.now():%Y-%m-%d} {upload}: {get_title(texts)}', '\n\n'.join(texts))
    return ep

# +
# # %%time
# sub = 'osf'
# ep = run(sub, upload=True, limit=5)
# IPython.display.Audio(merge_mp3s(ep.sounds))

# # # d = '/Users/jong/Documents/PodPapers/Conciousness'
# # # ep = episode_with_pdfs(d)
# # # IPython.display.Audio(merge_mp3s(ep.sounds))
# -




