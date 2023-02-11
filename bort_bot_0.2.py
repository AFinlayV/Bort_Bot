"""
BORT is a discord chatbot that uses embeddings and semantic search to simulate a long term memory
"""

import asyncio
import datetime
import discord
import os
import openai
import json
from time import time, sleep
from uuid import uuid4
import numpy as np
from numpy.linalg import norm


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def init_discord_client():
    intents = discord.Intents.all()
    bort = discord.Client(intents=intents,
                          shard_id=0,
                          shard_count=1,
                          reconnect=True)
    return bort


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


# def init_keys():
#     # load openai api key
#     if not os.environ.get('OPENAI_API_KEY'):
#         with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
#             key = f.read().strip()
#             openai.api_key = key
#             os.environ["OPENAI_API_KEY"] = key
#
#     # load discord auth token
#     if not os.environ.get('BORT_DISCORD_TOKEN'):
#         auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
#         os.environ['BORT_DISCORD_TOKEN'] = auth['token'].strip()
#         os.environ['BORT_DISCORD_CHAN_ID'] = auth['chan_id'].strip()


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    #   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector,
            'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('internal_notes/%s' % filename, info)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:-1]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.5, top_p=1.0, tokens=1024, freq_pen=0.0, pres_pen=0.0):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen)
            text = response['choices'][0]['text'].strip()
            # text = re.sub('[\r\n]+', '\n', text)
            # text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def save_message(discord_message, vector):
    text = discord_message.content
    user = discord_message.author.name
    timestamp = time()
    timestring = timestamp_to_datetime(timestamp)
    message = '%s: %s' % ([{user}], text)
    info = {'speaker': user,
            'time': timestamp,
            'vector': vector,
            'message': message,
            'uuid': str(uuid4()),
            'timestring': timestring}
    filename = f'log_{timestamp}_{user}.json'
    save_json('nexus/%s' % filename, info)


def load_memories(vector):
    conversation = load_convo()
    memories = fetch_memories(vector, conversation, 10)
    notes = summarize_memories(memories)
    recent = get_last_messages(conversation, 4)
    return notes, recent


def generate_prompt(notes, recent, a):
    prompt = open_file('BORT_Prompt.txt') \
        .replace('<<NOTES>>', notes) \
        .replace('<<CONVERSATION>>', recent) \
        .replace('<<MESSAGE>>', a)
    return prompt


def save_response(response):
    timestamp = time()
    vector = gpt3_embedding(response)
    timestring = timestamp_to_datetime(timestamp)
    message = '%s: %s' % ('Bort', response)
    info = {
        'speaker': 'Bort',
        'time': timestamp,
        'vector': vector,
        'message': message,
        'uuid': str(uuid4()),
        'timestring': timestring
    }
    filename = f'log_{timestamp}_Bort.json'
    save_json('nexus/%s' % filename, info)


def process_message(discord_message):
    try:
        message = f'{discord_message.author.name}: {discord_message.content}'
        message_vector = gpt3_embedding(message)
        save_message(discord_message, message_vector)
        notes, recent = load_memories(message_vector)
        prompt = generate_prompt(notes, recent, message)
        response = gpt3_completion(prompt)
        save_response(response)
        return {'output': response, 'user': discord_message.author.name}
    except Exception as oops:
        print('Error processing message:', oops)
        return {'output': 'Error processing message: %s' % oops, 'user': discord_message.author.name}


if __name__ == "__main__":
    #init_keys()
    bort = init_discord_client()


    @bort.event
    async def on_ready():
        channel = bort.get_channel(int(os.environ.get['BORT_DISCORD_CHAN_ID']))
        if channel is not None:
            await channel.send(f"{bort.user} has connected to Discord!")
        else:
            print("Channel not found")


    @bort.event
    async def on_message(message):
        if message.author == bort.user:
            return
        elif message.content.startswith('!') or message.content == "":
            return
        else:
            # use asyncio to run the process_message function in the background
            output = await asyncio.get_event_loop().run_in_executor(None, process_message, message)
            await message.channel.send(output['output'])


    bort.run(os.environ.get['BORT_DISCORD_TOKEN'])
