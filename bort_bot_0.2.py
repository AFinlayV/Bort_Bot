"""
BORT is a discord chatbot that uses embeddings and semantic search to simulate a long term memory

Ideas for new features:
X Dream - Get random memories of chats and generate a dream. Have gpt take details from the memories and generate a dreamlike narative and some image prompts so users can "see the dream"
- Research - Use langchain search tools to allow bort to do research on topics that have come up a lot in conversation
- Ruminate - Use langchain to have bort "think about" a topic on it's own. This could be used to generate a random
topic for the dream feature.
- Silo Memories - Create separate folders of memories for each discord server and each user. This would allow for more privacy and more specific memories.
- Switch from 'discord' library to 'disnake' library to allow for slash commands
- Switch from 'openai' to langchain
- Switch from the .Client to the .Bot class
- Add slash commands.
- Add a way to generate "rules" when a user requests a specific way for the bot to behave. These rules could be siloed by user and server.
- Maybe have users somehow add a secure api key to their account so that they can use their own gpt3 api key. This will be necessary if I want to grow the bot beyond my personal server, as I am currently paying for all the api requests it does.


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
import dream

VERBOSE = False


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


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


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    print('fetching memories')
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    vprint('found %s memories' % len(ordered))
    vprint('top memory score: %s' % ordered[0]['score'])
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        vprint('memories fetched: %s' % len(ordered))
        return ordered
    except:
        return ordered


def load_convo():
    try:
        print('loading convo')
        files = os.listdir('nexus')
        files = [i for i in files if '.json' in i]  # filter out any non-JSON files
        result = list()
        for file in files:
            data = load_json('nexus/%s' % file)
            result.append(data)
        ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
        vprint('loaded %s conversations' % len(ordered))
        return ordered
    except:
        return "No Conversation Found"


def summarize_memories(memories):  # summarize a block of memories into one payload
    try:
        print('summarizing memories')
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
        vprint('saved notes')
        vprint('notes: %s' % notes)
        return notes
    except:
        return 'No memories found'


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


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=1024, freq_pen=0.0, pres_pen=0.0):
    print('generating completion')
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
            vprint('saved gpt3 log')
            vprint('gpt response: %s' % text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def save_message(discord_message, vector):
    print('saving message')
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
    vprint('saved message')


def load_memories(vector):
    try:
        print('loading memories')
        conversation = load_convo()
        memories = fetch_memories(vector, conversation, 10)
        notes = summarize_memories(memories)
        recent = get_last_messages(conversation, 4)
        vprint('loaded memories')
        vprint('notes: %s' % notes)
        vprint('recent: %s' % recent)
        return notes, recent
    except Exception as oops:
        print('Error loading memories:', oops)
        return '', ''


def generate_prompt(notes, recent, a):
    print('generating prompt')
    prompt = open_file('BORT_Prompt.txt')
    prompt_len = len(prompt) + len(a) + len(notes) + len(recent)
    # This is a hack to keep the prompt under the token limit
    # This can be done more precisely by using the token count of the prompt,
    # rather than the estimate from the string length.
    if prompt_len < 25000:
        prompt = prompt.replace('<<NOTES>>', notes).replace('<<CONVERSATION>>', recent).replace('<<MESSAGE>>', a)
    else:
        prompt = prompt.replace('<<NOTES>>', '').replace('<<CONVERSATION>>', recent).replace('<<MESSAGE>>', a)
    vprint('generated prompt')
    vprint('prompt length: %s' % len(prompt))
    vprint('prompt: %s' % prompt)
    return prompt


def save_response(response):
    print('saving response')
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
    vprint('saved response')
    vprint('filename: %s' % filename)
    vprint('response: %s' % response)



def process_message(discord_message):
    try:
        print('processing message')
        message = f'{discord_message.author.name}: {discord_message.content}'
        message_vector = gpt3_embedding(message)
        save_message(discord_message, message_vector)
        notes, recent = load_memories(message_vector)
        prompt = generate_prompt(notes, recent, message)
        response = gpt3_completion(prompt)
        save_response(response)
        vprint('processed message')
        vprint('user: %s' % discord_message.author.name)
        vprint('response: %s' % response)
        return {'output': response, 'user': discord_message.author.name}
    except Exception as oops:
        print('Error processing message:', oops)
        return {'output': 'Error processing message: %s' % oops, 'user': discord_message.author.name}


if __name__ == "__main__":
    # init_keys()
    bort = init_discord_client()


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
        limit = 1500
        if message.author == bort.user:
            vprint('message from self, ignoring')
            return
        elif message.content.startswith('!') or message.content == "":
            vprint('message is humans whispering, ignoring')
            return
        elif message.content.startswith('/dream'):
            print('dreaming')
            output = dream.main()
            vprint('dream: %s' % output)
            # split output into chunks of 1500 characters, separated at the end of a word, and ending with '...'
            # if there is a split happening and send them as separate messages with a 1 sec delay
            message_parts = [output[i:i + 1500] for i in range(0, len(output), 1500)]
            for part in message_parts:
                if len(part) == 1500:
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
            message_parts = [output['output'][i:i + 1500] for i in range(0, len(output['output']), 1500)]
            for part in message_parts:
                if len(part) == 1500:
                    vprint('sending chunk to discord')
                    part = part + '...'
                await message.channel.send(part)
            print("sent response to discord")
            vprint('message: %s' % output['output'])

    bort.run(os.environ['BORT_DISCORD_TOKEN'])
