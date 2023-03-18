import discord
from discord.ext import commands
import openai
import tiktoken
import json
import os
import uuid
from datetime import datetime
from time import sleep
from glob import glob
import logging
import asyncio


# Your GPT4Chat class

class GPT4Chat:
    def __init__(self):
        self.VERBOSE = True
        self.config = self.load_config()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.conversation_memory = [{"role": "system", "content": self.config["system_prompt"]}]
        os.makedirs("log", exist_ok=True)
        self.memory_limit = 30
        self.respond_to_all_channels = self.config["respond_to_all_channels"]

    def load_config(self):
        self.vprint("Loading config...")
        with open("config4.json", "r") as config_file:
            return json.load(config_file)

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)

    def load_recent_memories(self):
        self.vprint("Loading recent memories...")
        log_files = glob("log/*.json")
        log_files.sort(key=os.path.getmtime)

        recent_messages = []
        for log_file in log_files[-self.memory_limit:]:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                recent_messages.append({"role": log_data["speaker"], "content": log_data["message"]})

        return recent_messages

    def num_tokens_from_messages(self, messages, model="gpt-4"):
        self.vprint("Calculating number of tokens...")
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model.startswith("gpt-4"):
            num_tokens = 0
            for message in messages:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += -1
            num_tokens += 2
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    def ensure_token_limit(self):
        self.vprint("Ensuring token limit...")
        while True:
            total_tokens = self.num_tokens_from_messages(self.conversation_memory)
            if total_tokens < 7000:
                break
            else:
                self.conversation_memory.pop(1)
                self.conversation_memory.pop(1)

    def save_message_to_log(self, message, speaker):
        self.vprint("Saving message to log...")
        log_entry = {
            "message": message,
            "speaker": speaker,
            "time": datetime.utcnow().timestamp(),
            "timestring": datetime.utcnow().strftime("%A, %B %d, %Y at %I:%M%p "),
            "uuid": str(uuid.uuid4()),
        }

        log_filename = os.path.join("log", f"{log_entry['uuid']}.json")

        with open(log_filename, "w") as log_file:
            json.dump(log_entry, log_file, indent=4)

    def get_gpt_response(self, user_message):
        self.vprint("Getting GPT response...")
        self.update_conversation_memory("user", user_message)
        self.save_message_to_log(user_message, "user")
        self.ensure_token_limit()

        prompt = {
            "model": "gpt-4",
            "messages": self.conversation_memory,
            "temperature": self.config["gpt_chat_settings"]["temperature"],
            "top_p": self.config["gpt_chat_settings"]["top_p"],
            "max_tokens": self.config["gpt_chat_settings"]["max_tokens"],
            "frequency_penalty": self.config["gpt_chat_settings"]["frequency_penalty"],
            "presence_penalty": self.config["gpt_chat_settings"]["presence_penalty"],
        }
        self.vprint(prompt)
        response = openai.ChatCompletion.create(**prompt)
        gpt_response = response["choices"][0]["message"]["content"]
        self.update_conversation_memory("assistant", gpt_response)
        self.save_message_to_log(gpt_response, "assistant")

        return gpt_response

    def update_conversation_memory(self, role, content):
        self.vprint("Updating conversation memory...")
        self.conversation_memory.append({"role": role, "content": content})
        for memory in self.conversation_memory:
            self.vprint(memory["role"] + ":", memory["content"])

    def generate_response(self, user_message):
        self.vprint("Generating response...")
        return self.get_gpt_response(user_message)

    def main(self):
        self.vprint("Loading recent memories...")
        self.conversation_memory.extend(self.load_recent_memories())

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            for memory in self.conversation_memory:
                self.vprint(memory["role"] + ":", memory["content"])
            gpt_response = self.get_gpt_response(user_input)
            print("GPT-4:", gpt_response)


# Create a bot instance with the command prefix you'd like to use
intents = discord.Intents.all()
bot = commands.Bot(intents=intents, command_prefix="/")

# Load the GPT-4 chat model (modify this according to your model loading method)
gpt4_chat = GPT4Chat()

# Set up the channel ID
BORT_DISCORD_CHANNEL_ID = 1071975175802851461


def split_response(text, max_length):
    split_texts = []
    while len(text) > max_length:
        index = text.rfind(" ", 0, max_length)
        if index == -1:
            index = max_length
        split_texts.append(text[:index])
        text = text[index:].strip()
    split_texts.append(text)
    return split_texts


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


async def generate_and_send_response(channel, question):
    print('generating response...')
    if question.startswith("!"):
        return  # Ignore messages that start with !

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, gpt4_chat.generate_response, question.strip())
    print('response generated')

    max_length = 2000  # Set your desired max_length
    split_texts = split_response(response, max_length)
    for split_text in split_texts:
        await channel.send(split_text)
        sleep(1)



@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!'):
        await bot.process_commands(message)  # Pass the message object instead of ctx
    else:
        question = message.content.strip()
        await generate_and_send_response(message.channel, question)  # Pass the channel instead of ctx


@bot.command()
async def bort(ctx, *, question):
    if not ctx.author.bot:
        await generate_and_send_response(ctx, ctx.message.content)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gpt4chat.log", mode="a")
    ]
)

bot_token = os.environ.get("BORT_DISCORD_TOKEN")
if bot_token is None:
    print("Error: BORT_DISCORD_TOKEN environment variable not found.")
    exit(1)

# Run the bot with your token
bot.run(bot_token)
