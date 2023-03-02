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

"""
OLD CODE

def init_discord_client():
    intents = discord.Intents.all()
    client = discord.Client(intents=intents,
                            shard_id=0,
                            shard_count=1,
                            reconnect=True)
    return client
    
    
def get_last_messages(limit, server_id):
    try:
        # get {limit} most recent json files from nexus/ folder and return
        # a concatenated string of all the ['message'] fields
        print('getting most recent messages')
        files = os.listdir('nexus')
        # make a dict of {filename: timestamp}
        file_dict = {}
        for file in files:
            file_dict[file] = os.path.getmtime('nexus/' + file)
        # sort the dict by timestamp
        sorted_files = sorted(file_dict.items(), key=lambda x: x[1], reverse=True)
        # get the first {limit} files
        sorted_files = sorted_files[:limit]
        # get the messages from the files
        messages = []
        for file in sorted_files:
            message = load_json('nexus/' + file[0])
            if message['server'] == server_id:
                messages.append(message['message'])
        message = [message for message in messages if message != '']
        # concatenate the messages
        output = ' '.join(messages)
        return output
    except Exception as oops:
        print('Error getting last messages:', oops)
        return 'Error getting last messages: %s' % oops


def gpt3_completion(prompt):
    print('generating completion')
    max_retry = CONFIG['max_retry']
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=CONFIG["gpt_settings"]["engine"],
                prompt=prompt,
                temperature=CONFIG["gpt_settings"]["temp"],
                max_tokens=CONFIG["gpt_settings"]["tokens"],
                top_p=CONFIG["gpt_settings"]["top_p"],
                frequency_penalty=CONFIG["gpt_settings"]["freq_pen"],
                presence_penalty=CONFIG["gpt_settings"]["pres_pen"])
            text = response['choices'][0]['text'].strip()
            # text = re.sub('[\r\n]+', '\n', text)
            # text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            vprint('saved gpt3 log')
            vprint('gpt response: %s' % text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def load_conversation(results, server_id):
    try:
        print('loading relevant messages from past conversations')
        result = list()
        for m in results['matches']:
            info = load_json('nexus/%s.json' % m['id'])
            if info['server'] == server_id:
                result.append(info)
        vprint('result: %s' % result)
        ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
        messages = [i['message'] for i in ordered]
        print('loaded %s messages' % len(messages))
        vprint('messages: %s' % messages)
        return '\n'.join(messages).strip()
    except Exception as oops:
        print('Error loading conversation:', oops)
        return ''




def process_message(discord_message):
    context_size = CONFIG['context_size']
    recent_message_count = CONFIG['recent_message_count']
    token_limit = CONFIG['token_limit']
    prompt_file = CONFIG['prompt_file']
    try:
        print('processing message')
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()
        message = discord_message.content
        user = discord_message.author.name
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        # message = '%s: %s - %s' % (user, timestring, a)
        message = f'{user}: {message}'
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        server_id = discord_message.guild.id
        metadata = {'speaker': user, 'time': timestamp, 'message': message, 'timestring': timestring,
                    'uuid': unique_id, 'server': server_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=context_size)
        print('results: %s' % results)
        conversation = load_conversation(
            results, server_id)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        recent = get_last_messages(recent_message_count, server_id)
        print('conversation: %s' % conversation)
        prompt = open_file(prompt_file)\
            .replace('<<CONVERSATION>>', conversation)\
            .replace('<<RECENT>>', recent)\
            .replace('<<MESSAGE>>', message)
        num_tokens = count_tokens(prompt)
        while num_tokens > token_limit:
            print('prompt too long, trimming')
            prompt = trim_prompt(prompt)
            num_tokens = count_tokens(prompt)
            print(f'prompt reduced to {num_tokens} tokens')
        print('prompt: %s' % prompt)
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        # message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output
        vector = gpt3_embedding(message)
        unique_id = str(uuid4())
        metadata = {'speaker': 'BORT', 'time': timestamp, 'message': message, 'timestring': timestring,
                    'uuid': unique_id, 'server': server_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        print('\n\nBORT: %s' % output)
        return {'output': output, 'user': discord_message.author.name}
    except Exception as oops:
        print('Error processing message:', oops)
        return {'output': 'Error processing message: %s' % oops, 'user': discord_message.author.name}

   @bort.event
    async def on_message(message):
        print('message received from discord from %s' % message.author.name)
        vprint('message: %s' % message.content)
        limit = CONFIG['discord_chunk_size']
        if message.author == bort.user:
            vprint('message from self, ignoring')
            return
        elif message.author.bot:
            vprint('message from bot, ignoring')
            return
        elif message.content.startswith('!') or message.content == "":
            vprint('message is humans whispering, ignoring')
            return
        elif message.content.startswith('/dream'):
            print('dreaming')
            output = dream.get_dream()
            vprint('dream: %s' % output)
            # split output into chunks of 1500 characters, separated at the end of a word, and ending with '...'
            # if there is a split happening and send them as separate messages with a 1 sec delay
            message_parts = [output[i:i + limit] for i in range(0, len(output), limit)]
            for part in message_parts:
                if len(part) == limit:
                    vprint('sending chunk to discord')
                    part = part + '...'
                await message.channel.send(part)
            print("sent response to discord")
            vprint('message: %s' % output)
        else:
            print('sending message to process')
            # use asyncio to run the process_message function in the background
            output = await asyncio.get_event_loop().run_in_executor(None, process_message, message)
            # Split the output into chunks of 1500 characters and send them as separate messages with a 1 sec delay
            message_parts = [output['output'][i:i + limit] for i in range(0, len(output['output']), limit)]
            for part in message_parts:
                if len(part) == limit:
                    vprint('sending chunk to discord')
                    part = part + '...'
                await message.channel.send(part)
            print("sent response to discord")
            vprint('message: %s' % output['output'])

"""
