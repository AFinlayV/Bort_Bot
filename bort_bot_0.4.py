"""
BORT is a discord chatbot that uses embeddings and semantic search to simulate a long term memory

[X] - switch to pinecone for memory embedding/ search/ storage

"""

import asyncio
import datetime
import os
import openai
import json
from time import time, sleep
from uuid import uuid4
import pinecone
import tiktoken
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.agents import initialize_agent, load_tools
import disnake
from disnake.ext import commands


class Bort(commands.Cog):
    def __init__(self):
        self.intents = disnake.Intents.all()
        self.bot = commands.Bot(
            command_prefix='/',
            intents=self.intents
        )
        self.context_size = CONFIG['context_size']
        self.recent_message_count = CONFIG['recent_message_count']
        self.token_limit = CONFIG['token_limit']
        self.prompt_file = CONFIG['prompt_file']
        self.discord_token = os.environ.get(CONFIG["env_vars"]["sandbox_discord_token"])
        self.wolfram_id = os.environ.get(CONFIG["env_vars"]["wolfram_id"])
        self.google_id = os.environ.get(CONFIG["env_vars"]["google_id"])
        self.google_key = os.environ.get(CONFIG["env_vars"]["google_key"])
        self.pinecone_key = os.environ.get(CONFIG["env_vars"]["pinecone_key_env"])
        self.gpt_settings = CONFIG['gpt_settings']
        self.intents = disnake.Intents.all()
        self.chat_llm = OpenAI(
            model_name=CONFIG['gpt_chat_model'],
            temperature=gpt_settings['chat_temp'],
            top_p=gpt_settings['top_p']
        )
        self.util_llm = OpenAI(
            model_name=CONFIG['gpt_util_model'],
            temperature=gpt_settings['util_temp'],
            top_p=gpt_settings['top_p']
        )
        self.tools = load_tools(
            [
                'llm-math',
                'google-search',
                'wolfram-alpha'
            ],
            llm=self.util_llm
        )
        self.agent = initialize_agent(
            llm=self.chat_llm,
            tools=self.tools,
            prompt_template=str(),
            token_limit=self.token_limit,
            context_size=self.context_size,
            recent_message_count=self.recent_message_count
        )


class User:
    def __init__(self, ctx):
        self.user_id = ctx.author.id
        self.channel_id = ctx.channel.id
        self.all_messages = []
        self.most_recent_message = str()
        self.custom_rules = []
        self.summary = str()


class Conversation:
    def __init__(self, ctx, user):
        self.user_id = ctx.author.id
        self.channel_id = ctx.channel.id
        self.messages = []
        self.system_message = str()
        self.user = user
        self.user_summary = user.summary
        self.user_custom_rules = user.custom_rules



# load config vars
with open('config.json', 'r', encoding='utf-8') as infile:
    CONFIG = json.load(infile)
context_size = CONFIG['context_size']
recent_message_count = CONFIG['recent_message_count']
token_limit = CONFIG['token_limit']
prompt_file = CONFIG['prompt_file']
discord_token = os.environ.get(CONFIG["env_vars"]["sandbox_discord_token"])
wolfram_id = os.environ.get(CONFIG["env_vars"]["wolfram_id"])
google_id = os.environ.get(CONFIG["env_vars"]["google_id"])
google_key = os.environ.get(CONFIG["env_vars"]["google_key"])
pinecone_key = os.environ.get(CONFIG["env_vars"]["pinecone_key_env"])

# Initialize pinecone, langchain, and discord
intents = disnake.Intents.all()
bort = commands.Bot(
    intents=intents,
    command_prefix='/'
)
pinecone.init(
    api_key=pinecone_key,
    environment=CONFIG["pinecone_environment"]
)
vdb = pinecone.Index(CONFIG["pinecone_index"])
gpt_settings = CONFIG['gpt_settings']
llm = OpenAI(
    model_name='text-davinci-003',
    temperature=gpt_settings['temp'],
    top_p=gpt_settings['top_p']
)
util_llm = OpenAI(
    model_name=gpt_settings['engine'],
    temperature=0.0,
    top_p=gpt_settings['top_p']
)
tools = load_tools(
    [
        'llm-math',
        'google-search',
        'wolfram-alpha'
    ],
    llm=util_llm)
bort_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description")


def vprint(*args, **kwargs):
    if CONFIG['verbose']:
        print(*args, **kwargs)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def count_tokens(text):
    print('counting tokens')
    encoding = tiktoken.get_encoding("gpt2")
    num_tokens = len(encoding.encode(text))
    vprint('num_tokens: %s' % num_tokens)
    return num_tokens


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def get_response(prompt, agent):
    # get a response from the agent
    response = agent.run(prompt)
    return response


def save_memory(message, vdb):
    # save a memory
    # vectorize the message
    vector = gpt3_embedding(message)
    unique_id = str(uuid4())
    save_vector_to_pinecone(vector, unique_id, vdb)
    save_message_to_json(message, unique_id)
    pass


def save_message_to_json(message, unique_id):
    # save a message to the json file
    pass


def get_similar_memories(query):
    # get similar memories from the vdb
    pass


def get_recent_memories():
    # get recent memories from the vdb
    pass


def save_vector_to_pinecone(vector, memory, vdb):
    # save vector to vdb
    pass


def trim_prompt(prompt):
    print('trimming prompt')
    limit = CONFIG['token_limit']
    tokens = count_tokens(prompt)
    if tokens > limit:
        print('trimming prompt')
        lines = prompt.splitlines()
        while tokens > limit:
            lines = lines[1:]
            prompt = '  '.join(lines)
            tokens = count_tokens(prompt)
    return prompt


def process_message(discord_text):
    with open(CONFIG['langchain_file'], 'r', encoding='utf-8') as infile:
        format_template = str(infile.read())
    prompt = PromptTemplate(
        template=format_template,
        input_variables=['memories', 'recent', 'message']
    )
    memories = get_similar_memories(discord_text)
    recent = get_recent_memories()
    message = discord_text
    prompt = prompt.format(memories=memories, recent=recent, message=message)
    prompt = trim_prompt(prompt)
    response = get_response(prompt)
    return response


async def send_response(ctx, discord_text, response):
    chunk_size = 1500
    user = ctx.author.name
    try:
        if not ctx.author.bot:
            if len(response) > chunk_size:
                chunks = []
                current_chunk = ""
                for word in response.split():
                    if len(current_chunk + word) + 1 > chunk_size:
                        chunks.append(current_chunk + "...")
                        current_chunk = ""
                    current_chunk += word + " "
                chunks.append(current_chunk)
                for chunk in chunks:
                    chunk = chunk.replace(':', ':\n')
                    chunk = chunk.replace('*', '\n*')
                    print(chunk)
                    await ctx.send(f"```{chunk}```")
                    sleep(3)
            else:
                response = response.replace(':', ':\n')
                response = response.replace('**', '\n**')
                print(response)
                await ctx.send(f"```{response}```")
    except Exception as oops:
        await ctx.channel.send(f'Error: {oops} \n {user}: {discord_text[:20]}...')
        print(f'Error: {oops} \n {user}: {discord_text[:20]}...')


@bort.command()
async def b(ctx, *args):
    discord_text = " ".join(args)
    response = await asyncio.get_event_loop().run_in_executor(None, process_message, discord_text)
    await send_response(ctx, discord_text, response)


def main():
    try:
        bort.run(os.environ['SANDBOX_DISCORD_TOKEN'])
    except Exception as oops:
        print("ERROR IN MAIN", oops)


if __name__ == "__main__":
    main()
