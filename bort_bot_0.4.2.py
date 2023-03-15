"""
ARHHHHHGHGHGHGhhh I'll just have gpt4 do it...

Here goes nothing...
"""
import datetime
import openai
import pinecone
import os
import json
from uuid import uuid4
from time import time
import heapq
import discord
import asyncio

with open('config.json') as f:
    CONFIG = json.load(f)


def vprint(*args, **kwargs):
    if CONFIG['verbose']:
        print(*args, **kwargs)


def init_discord_client():
    intents = discord.Intents.all()
    client = discord.Client(intents=intents,
                            shard_id=0,
                            shard_count=1,
                            reconnect=True)
    return client


def retrieve_related_memories(message):
    # Retrieve related memories based on the given message
    vector = gpt3_embedding(message.content)
    relevant_memories = vdb.query(vector=vector, top_k=10)

    recent_memories = heapq.nlargest(10, os.listdir("nexus"), key=lambda f: os.path.getmtime(os.path.join("nexus", f)))
    recent_memories = [json.load(open(os.path.join("nexus", memory_file))) for memory_file in recent_memories]

    memories = [json.load(open(os.path.join("nexus", key))) for key in relevant_memories.keys()]

    return memories + recent_memories


async def process_message(message):
    retrieved_memories = retrieve_related_memories(message)
    system_message = get_system_message_from_memories(retrieved_memories)
    user_message = {"role": "user", "content": message.content}

    assistant_messages = [{"role": "assistant", "content": memory["message"]} for memory in retrieved_memories]

    messages_list = [system_message] + assistant_messages + [user_message]
    prompt = get_prompt(message.author.id, messages_list)

    response = openai.ChatCompletion.create(**prompt)
    output = response.choices[0]['message']['content'].strip()

    return {"output": output, "message_id": message.id}


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def summarize_memories(memories):
    summary = ""
    for memory in memories:
        speaker = memory["speaker"]
        content = memory["message"]
        summary += f"{speaker}: {content}\n"
    return summary.strip()


def get_system_message_from_memories(memories):
    summary = summarize_memories(memories)
    return {
        "role": "system",
        "content": f"Here's a summary of the previous related memories:\n{summary}"
    }


def save_memory(message, speaker, server_id):
    timestamp = time()
    timestring = timestamp_to_datetime(timestamp)
    vector = gpt3_embedding(message)
    unique_id = str(uuid4())
    metadata = {'speaker': speaker, 'time': timestamp, 'message': message, 'timestring': timestring,
                'uuid': unique_id, 'server': server_id}
    save_json(os.path.join('nexus', f'{unique_id}.json'), metadata)
    vdb.upsert([(unique_id, vector)])


def get_prompt(user, messages_list):
    prompt = {
        "model": "gpt-3.5-turbo",
        "messages": messages_list,
        "temperature": CONFIG["gpt_chat_settings"]["temperature"],
        "top_p": CONFIG["gpt_chat_settings"]["top_p"],
        "max_tokens": CONFIG["gpt_chat_settings"]["max_tokens"],
        "frequency_penalty": CONFIG["gpt_chat_settings"]["frequency_penalty"],
        "presence_penalty": CONFIG["gpt_chat_settings"]["presence_penalty"],
        "user": user,
    }
    return prompt


if __name__ == "__main__":
    bort = init_discord_client()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=CONFIG["pinecone_environment"])
    vdb = pinecone.Index(CONFIG["pinecone_index"])


    @bort.event
    async def on_ready():
        channel = bort.get_channel(int(os.environ['SANDBOX_DISCORD_CHAN_ID']))
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

        if message.content.startswith('/bort') or \
                message.content.startswith('/Bort') or \
                message.channel.id == int(os.environ['SANDBOX_DISCORD_CHAN_ID']):
            print('sending message to process')
            output = await process_message(message)
            message_parts = [output[i:i + limit] for i in range(0, len(output), limit)]

            for part in message_parts:
                if len(part) == limit:
                    vprint('sending chunk to discord')
                    part = part + '...'
                await message.channel.send(part)
            print("sent response to discord")
            vprint('message: %s' % output)


    bort.run(os.environ['SANDBOX_DISCORD_TOKEN'])
