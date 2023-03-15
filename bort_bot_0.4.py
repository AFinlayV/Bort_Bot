"""
Bort 0.4

Starting again from scratch. There is a new version of the gpt api. gpt-3.5-turbo. It is a lot faster, and 10x cheaper.
I also want to rewrite use the langchain library from the ground up rather than building functions that are openai calls.
The end goal is to create a chatbot that can exist on multiple discord servers simultaneously.
It will generate responses to users with a call to a langchain agent, which make the api call to openai.
The langchain agent will have a long-term memory of the conversations it has had with a user.
The long-term memory will be stored as a vector database, on pinecone.io with a uuid and a vector of the text.

Ihe text of the memory is stored locally in a json file with the following fields:
-message
-speaker
-time
-timestring
-uuid
-server
the filename is the uuid of the memory.

The bot will build a profile of each user locally as a json file with the following fields:
-username (the discord username)
-user_id (the discord user id)
-name (the name of the user, as they have told the bot)
-location (the location of the user, as they have told the bot)
-servers (a list of all the servers the user is on)
-list of uuids (for all memories associated with the user)
-description (a summary of all the information that the llm can derive from the text of the user's memories)
-rulebook (a list of any rules that user has added to the bot)
-tone (a description of the tone the user typically uses, so that the bot can generate responses in the same tone)
filename is the user_id

The bot will build a profile of each server locally as a json file with the following fields:
-servername (the discord server name)
-server_id (the discord server id)
-users (a list of all the users on the server)
-list of uuids (for all memories associated with the server)
-description (a summary of all the information that the llm can derive from the text of the server's memories)
filename is server_id

The bot will use langchain tools to access information from various sources on the internet.
- google search api
- wolfram alpha api
- docsearch over the contents of the discord server
- user memory recall (maybe, or maybe use langchain's built in memory functions?)
- llm-math
- weather/news headlines
- wikipedia
- summary of any link posted in chat

The bot will operate asynchronously, so that it can respond to multiple users simultaneously.

The bot will generate its response using gpt3.5, but the system message that is sent to the bot will be
composed dynamically by text-davinci-003.
davinci is 10x more expensive so the dynamic composition of the prompt should only happen occasionally, not every message


for the call to gp3.5, the prompt will be composed of the following fields:
The {system} message will be composed of the following fields, and processed through the langchain pipeline:
- main BORT Prompt
- username
- user description
- server description
- recent user memories
- recent server memories
- relevant server memories
- relevant user memories
- user tone
- user rules
- server rules
the {messages} filed will be composed of the most recent messages in the channel.
The {user} field will be the user's discord ID.

The llm will evaluate its own responses to make sure they are appropriate, and safe.
"""

import os
import sys
import time
import json
import random
import asyncio
import logging
import datetime
import openai
import disnake
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.llms import (
    OpenAIChat,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import pinecone

from disnake.ext import commands

CONFIG = json.load(open('config.json'))
CHANNEL_ENV_NAME = "SANDBOX_DISCORD_CHAN_ID"
TOKEN_ENV_NAME = "SANDBOX_DISCORD_TOKEN"


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)


def save_message(message):
    pass


def get_moderation_results(message):
    text = message.content
    mod_res = openai.Moderation.create()
    return mod_res


def get_recent_messages(message, num=10):
    recent_messages = []
    # load the nexus folder for every json file in the folder
    # for each json file, load the json file to a dict, append the dict to the recent_messages list
    filelist = os.listdir('nexus')
    for file in filelist:
        with open('nexus/' + file) as f:
            recent_messages.append(json.load(f))
    # filter out only messages with the same channel_id as the message
    recent_messages = [message for message in recent_messages if message['channel_id'] == message.channel.id]
    # sort the list by time
    recent_messages = sorted(recent_messages, key=lambda k: k['time'])
    # return the last 10 messages
    recent_messages = recent_messages[-num:]
    # filter out all fields except "message" and "speaker" into a dict with key as speaker and value as message
    recent_messages = {message['speaker']: message['message'] for message in recent_messages}
    return recent_messages


def get_summary(text, max_length=100):
    # use the openai summarization api to summarize the text
    # return the summary
    system_prompt = """
You are an assistant to a chatbot who is having a conversation with a user. The bot has asked you to help it summarize messages
and try to help it pick up on any details or themes present in the chats. you will me given a set of messages, and you will give a detailed verbose
response that clearly tells the bot what past memories are relevant to the current conversation, and what the user has said in the past.
    """
    summary = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                           max_length=max_length,
                                           temperature=0,

                                           messages=[
                                               {"role": "system",
                                                "content": system_prompt},
                                               {"role": "assistant",
                                                "content": "What is the text you would like for me to summarize"},
                                               {"role": "user",
                                                "content": text}
                                           ]
                                           )

    return summary["choices"][0]["message"]["content"]


def get_similar_memories(message):
    # get a list of 10 memories that are most similar to the message from pinecone return a list of the uuid keys
    # load the pinecone vector database
    # get the vector for the message
    # query the vector database for the 10 most similar vectors
    # return the uuids of the 10 most similar vectors
    server_id = message.guild.id
    vdb = pinecone.Index(CONFIG["pinecone_index"])
    query_vector = embed_text(message.content)
    results = vdb.query(vector=query_vector, top_k=CONFIG["context_size"])
    result = []
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['uuid'])
        if info['server'] == server_id:
            result.append(info)
    summary = get_summary("\n".join([m['message'] for m in result]))
    #get similar memories to the summary
    query_vector = embed_text(summary)
    results = vdb.query(vector=query_vector, top_k=CONFIG["context_size"])
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['uuid'])
        result.append(info)
        # return a list of the messages, in order by time
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)
    messages = [i['message'] for i in ordered]
    print('loaded %s messages' % len(messages))
    return messages


def get_user_rules(message):
    pass


def get_server_rules(message):
    pass


def get_user_profile(message):
    pass


def get_constitution(message):
    pass


def build_system_prompt(parsed_message):
    pass


def parse_message(message):
    text = message.content
    user_id = message.author.id
    user_name = message.author.name
    server = message.guild.name
    server_id = message.guild.id
    channel = message.channel.name
    channel_id = message.channel.id
    moderation_results = get_moderation_results(message)
    recent_messages = get_recent_messages(message)
    similar_memories = get_similar_memories(message)
    # user_rules = get_user_rules(message)
    # server_rules = get_server_rules(message)
    # user_profile = get_user_profile(message)
    # constitution = get_constitution(message)
    vector = embed_text(text)
    parsed_message = {
        'text': text,
        'user_id': user_id,
        'user_name': user_name,
        'server': server,
        'server_id': server_id,
        'channel': channel,
        'channel_id': channel_id,
        'moderation_results': moderation_results,
        'recent_messages': recent_messages,
        'similar_memories': similar_memories,
        # 'user_rules': user_rules,
        # 'server_rules': server_rules,
        # 'user_profile': user_profile,
        # 'constitution': constitution,
        'vector': vector
    }
    return parsed_message


def build_prompt(parsed_message):
    pass


async def generate_response(prompt, llm):
    response = await llm.agenerate(prompt)
    return response


async def send_message(message, response):
    await message.channel.send(response)


def embed_text(text):
    content = text.encode(encoding='ASCII', errors='ignore').decode()
    vector = OpenAIEmbeddings.embed_query(content)
    return vector


def main():
    bot = commands.Bot(command_prefix="/")
    bot.remove_command('help')
    llm = OpenAIChat(
        model_name=CONFIG["gpt_settings"]["engine"],
        temperature=CONFIG["gpt_settings"]["temp"],
        max_tokens=CONFIG["gpt_settings"]["max_tokens"],
        top_p=CONFIG["gpt_settings"]["top_p"],
        frequency_penalty=CONFIG["gpt_settings"]["freq_pen"],
        presence_penalty=CONFIG["gpt_settings"]["pres_pen"],
    )
    pinecone.init(
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment=CONFIG["pinecone_environment"]
    )

    @bot.event
    async def on_ready():
        print(f'{bot.user} has connected to Discord!')



    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        if message.content.startswith(CONFIG['prefix']) or message.channel.id == os.environ[CHANNEL_ENV_NAME]:
            parsed_message = await parse_message(message)
            await save_message(parsed_message)
            system_prompt = await build_system_prompt(parsed_message)
            prompt = await build_prompt(parsed_message)
            response = await generate_response(prompt, llm)
            await save_message(response)
            await send_message(response)
        if message.content.startswith('!'):
            return

    bot.run(os.environ[TOKEN_ENV_NAME])



if __name__ == '__main__':
    main()

"""
"""
