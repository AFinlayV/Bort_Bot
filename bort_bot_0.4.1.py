import os
import json
import asyncio
import discord
import pinecone
import dream
import openai
from time import sleep
import glob

def init_discord_bot():
    intents = discord.Intents.all()
    bot = discord.Client(intents=intents)
    return bot


def vprint(message):
    if CONFIG['verbose']:
        print(message)


def get_response(prompt):
    print('generating completion')
    vprint('prompt: %s' % prompt)
    max_retry = CONFIG['max_retry']
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(prompt)
            text = response['choices'][0]["messages"]["content"].strip()
            vprint('gpt response: %s' % text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def get_last_messages(limit, server_id):
    try:
        print('getting most recent messages')

        # Use glob to get all json files in the nexus folder
        files = glob.glob('nexus/*.json')

        # Sort files based on their modification time
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)

        # Load messages from files and filter by server_id
        messages = []
        for file in sorted_files[:limit]:
            message = load_json(file)
            if message['server'] == server_id and message['message'] != '':
                messages.append({"role": message['speaker'], "content": message['message']})

        return messages
    except Exception as oops:
        print('Error getting last messages:', oops)
        return [{'role': 'system', 'content': f'Error getting last messages: {oops}'}]

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


async def get_messages_list(message):
    messages_list = []
    # add the system prompt
    with open('BORT_Prompt_3.5.txt') as infile:
        system_prompt = infile.read()
    messages_list.append({"role": "system", "content": system_prompt})
    # get the last 10 messages from the channel
    for msg in message.channel.history(limit=20):
        if msg.author == bort.user:
            messages_list.append({"role": "assistant", "content": msg.content})
        else:
            messages_list.append({"role": "user", "content": msg.content})
    return messages_list


def process_message(message):
    user = message.author.name
    messages_list = message.channel.history(limit=20)
    dialog_list = []
    for msg in messages_list:
        if msg.author == bort.user:
            dialog_list.append({"role": "assistant", "content": msg.content})
        else:
            messages_list.append({"role": "user", "content": msg.content})
    prompt = get_prompt(user, messages_list)
    response = get_response(prompt)
    return response


if __name__ == "__main__":
    # init_keys()
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    with open('config.json') as f:
        CONFIG = json.load(f)
    bort = init_discord_bot()
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
        elif message.content.startswith('!'):
            vprint('message is humans whispering, ignoring')
            return
        elif message.content.startswith('/bort') or \
                message.content.startswith('/Bort') or \
                message.channel.id == int(os.environ['SANDBOX_DISCORD_CHAN_ID']):
            print('sending message to process')
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


    bort.run(os.environ['SANDBOX_DISCORD_TOKEN'])
