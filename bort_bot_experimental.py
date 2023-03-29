import discord
from discord.ext import commands
import openai
import tiktoken
import json
import os
import uuid
from datetime import datetime
from time import sleep
import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gpt4chat.log", mode="a")
    ]
)
logging.info("Starting GPT4Chat...")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_config():
    logging.info("Loading config...")
    with open("config4.json", "r") as config_file:
        config = json.load(config_file)
        for key, value in config.items():
            logging.info(f"{key}: {value}")
        return config


def num_tokens_from_messages(messages):
    logging.info("Getting number of tokens from messages...")
    return sum([num_tokens_from_string(str(message)) for message in messages])


def save_message_to_log(message, speaker):
    logging.info("Saving message to log...")
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


class GPT4Chat:

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.config = load_config()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.conversation_memory = [{"role": "system", "content": self.config["system_prompt"]}]
        os.makedirs("log", exist_ok=True)
        self.memory_limit = 30
        self.respond_to_all_channels = self.config["respond_to_all_channels"]
        self.model = "gpt-4"
        self.token_count = num_tokens_from_messages([self.conversation_memory[-1]])
        self.conversation_memory.extend(self.load_recent_memories())

    def get_gpt_response(self, user_message, tries=0):
        max_tries = 5
        try:
            logging.info("Getting GPT response...")
            self.update_conversation_memory("user", user_message)
            save_message_to_log(user_message, "user")

            prompt = {
                "model": self.model,
                "messages": self.conversation_memory,
                "temperature": self.config["gpt_chat_settings"]["temperature"],
                "top_p": self.config["gpt_chat_settings"]["top_p"],
                "max_tokens": self.config["gpt_chat_settings"]["max_tokens"],
                "frequency_penalty": self.config["gpt_chat_settings"]["frequency_penalty"],
                "presence_penalty": self.config["gpt_chat_settings"]["presence_penalty"],
            }
            logging.info("Prompt:")
            for key, value in prompt.items():
                logging.info(f"{key}: {value}")
            for message in prompt["messages"]:
                logging.info(f"{message['role']}: {message['content']}")
            token_count = num_tokens_from_string(str(prompt))
            if token_count >= self.config["prompt_token_limit"]:
                logging.info("Too many tokens, reducing conversation memory by removing oldest messages...")
                removed_message = self.conversation_memory.pop(1)
                logging.info(f"Removed message: {removed_message}")
                return self.get_gpt_response(user_message)
            response = openai.ChatCompletion.create(**prompt)
            gpt_response = response["choices"][0]["message"]["content"]
            self.update_conversation_memory("assistant", gpt_response)
            save_message_to_log(gpt_response, "assistant")

            return gpt_response
        except Exception as e:
            logging.error(e)
            if "maximum context length" in str(e):
                logging.info("Too many tokens, reducing conversation memory by removing oldest messages...")
                self.conversation_memory.append({"role": "system", "content": self.config["system_prompt"]})
                self.token_count = num_tokens_from_messages([self.conversation_memory[-1]])
                return self.get_gpt_response(user_message)
            elif tries < max_tries:
                logging.info("Retrying...")
                sleep(1)
                return self.get_gpt_response(user_message, tries + 1)
            else:
                logging.info("Failed to get GPT response.")
                return f"I'm sorry, The server returned an error. Please try again. Error: {e}"

    def load_recent_memories(self):
        logging.info("Loading recent memories...")

        def memory_from_log_file(filename):
            with open(os.path.join("log", filename), "r") as log_file:
                log_data = json.load(log_file)
                return {
                    "role": log_data["speaker"],
                    "content": log_data["message"],
                    "time": log_data["time"]
                }

        log_files = os.listdir("log")
        memories = [memory_from_log_file(filename) for filename in log_files]

        # Sort memories by time in descending order
        memories.sort(key=lambda x: x['time'], reverse=True)

        # Get the most recent memories
        most_recent_memories = memories[:self.memory_limit]

        # Reverse the most recent memories to have the oldest at the beginning
        most_recent_memories.reverse()

        # Remove the time field from each memory
        for memory in most_recent_memories:
            del memory['time']
        return most_recent_memories

    def update_conversation_memory(self, role, content):
        logging.info("Updating conversation memory...")
        self.conversation_memory.append({"role": role, "content": content})


# Create a bot instance with the command prefix you'd like to use
intents = discord.Intents.all()
bot = commands.Bot(intents=intents, command_prefix="/")

# Load the GPT-4 chat model (modify this according to your model loading method)
gpt4_chat = GPT4Chat()

# Set up the channel ID
if gpt4_chat.config["experimental"]:
    CHAN_ID = gpt4_chat.config["experimental_channel"]
    logging.info(f"Using experimental mode. {CHAN_ID}")
else:
    CHAN_ID = gpt4_chat.config["main_channel"]
    logging.info(f"Using production mode. {CHAN_ID}")


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
    if question.startswith("!"):
        return  # Ignore messages that start with !
    print('generating response...')
    logging.info(f"Message: {question.strip()}")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, gpt4_chat.get_gpt_response, question.strip())
    print('response generated')
    max_length = 2000  # Set your desired max_length
    split_texts = split_response(response, max_length)
    for split_text in split_texts:
        await channel.send(split_text)
        sleep(1)


@bot.event
async def on_message(message):
    if message.author == bot.user or message.author.bot:
        return
    elif message.content.startswith('!'):
        await bot.process_commands(message)  # Pass the message object instead of ctx
    elif message.channel.id != CHAN_ID:
        return
    else:
        question = message.content.strip()
        await generate_and_send_response(message.channel, question)  # Pass the channel instead of ctx


@bot.command()
async def bort(ctx, *, question):
    if not ctx.author.bot:
        await generate_and_send_response(ctx, ctx.message.content)


if gpt4_chat.config["experimental"]:
    bot_token = os.environ.get("SANDBOX_DISCORD_TOKEN")
    logging.warning(
        f"Experimental mode is enabled. This is not recommended for production use. Token:{bot_token[:5]}...{bot_token[-5:]}")
else:
    bot_token = os.environ.get("BORT_DISCORD_TOKEN")
if bot_token is None:
    print("Error: BORT_DISCORD_TOKEN environment variable not found.")
    exit(1)

# Run the bot with your token
bot.run(bot_token)
