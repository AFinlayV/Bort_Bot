"""
BORT is a discord chatbot that uses embeddings and semantic search to simulate a long term memory

[X] - switch to pinecone for memory embedding/ search/ storage
"""

import asyncio
import datetime
import discord
import os
import openai
import json
from time import time, sleep
from uuid import uuid4
import dream
import pinecone
import tiktoken

with open('config.json', 'r', encoding='utf-8') as infile:
    CONFIG = json.load(infile)


def vprint(*args, **kwargs):
    if CONFIG['verbose']:
        print(*args, **kwargs)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def init_discord_client():
    intents = discord.Intents.all()
    client = discord.Client(intents=intents,
                          shard_id=0,
                          shard_count=1,
                          reconnect=True)
    return client


# Functions from RAVEN to load and save memories, rewrite these using langchain and async_openai
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


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


def get_last_messages(limit):
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
        message = load_json('nexus/' + file[0])['message']
        messages.append(message)
    # concatenate the messages
    output = ' '.join(messages)
    return output


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


def load_conversation(results):
    print('loading relevant messages from past conversations')
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    vprint('result: %s' % result)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    print('loaded %s messages' % len(messages))
    vprint('messages: %s' % messages)
    return '\n'.join(messages).strip()


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
        metadata = {'speaker': user, 'time': timestamp, 'message': message, 'timestring': timestring,
                    'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=context_size)
        print('results: %s' % results)
        conversation = load_conversation(
            results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id'
        recent = get_last_messages(recent_message_count)
        print('conversation: %s' % conversation)
        prompt = open_file(prompt_file)\
            .replace('<<CONVERSATION>>', conversation)\
            .replace('<<RECENT>>', recent)\
            .replace('<<MESSAGE>>', message)
        num_tokens = count_tokens(prompt)
        while num_tokens > token_limit:
            print('prompt too long, trimming')
            context_size = int(context_size - 1)
            results = vdb.query(vector=vector, top_k=context_size)
            conversation = load_conversation(results)
            prompt = open_file(prompt_file) \
                .replace('<<CONVERSATION>>', conversation) \
                .replace('<<RECENT>>', recent) \
                .replace('<<MESSAGE>>', message)
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
                    'uuid': unique_id}
        save_json('nexus/%s.json' % unique_id, metadata)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        print('\n\nBORT: %s' % output)
        return {'output': output, 'user': discord_message.author.name}
    except Exception as oops:
        print('Error processing message:', oops)
        return {'output': 'Error processing message: %s' % oops, 'user': discord_message.author.name}


if __name__ == "__main__":
    # init_keys()
    bort = init_discord_client()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=CONFIG["pinecone_environment"])
    vdb = pinecone.Index(CONFIG["pinecone_index"])


    @bort.event
    async def on_ready():
        channel = bort.get_channel(int(os.environ['BORT_DISCORD_CHAN_ID']))
        if channel is not None:
            await channel.send(f"{bort.user} has connected to Discord!")
            print(f"{bort.user} has connected to Discord!")
        else:
            print("Channel not found")


    @bort.event
    async def on_message(message):
        print('message received from discord from %s' % message.author.name)
        vprint('message: %s' % message.content)
        limit = CONFIG['discord_chunk_size']
        if message.author == bort.user:
            vprint('message from self, ignoring')
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


    bort.run(os.environ['BORT_DISCORD_TOKEN'])
