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


class GPT4Chat:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.config = self.load_config()
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.conversation_memory = [{"role": "system", "content": self.config["system_prompt"]}]
        os.makedirs("log", exist_ok=True)
        self.memory_limit = 30
        self.respond_to_all_channels = self.config["respond_to_all_channels"]
        self.model = "gpt-4"
        # Load recent memories on startup
        self.conversation_memory.extend(self.load_recent_memories())
        if self.config["experimental"]:
            self.model = "gpt-3.5-turbo"
        self.token_count = self.num_tokens_from_messages([self.conversation_memory[-1]])

    def load_config(self):
        logging.info("Loading config...")
        with open("config4.json", "r") as config_file:
            config = json.load(config_file)
            for key, value in config.items():
                logging.info(f"{key}: {value}")
            return config

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
        most_recent_memories = []

        for memory in memories:
            self.ensure_token_count(memory)
            if len(most_recent_memories) < self.memory_limit:
                most_recent_memories.append(memory)
            else:
                break

        # Reverse the most recent memories to have the oldest at the beginning
        most_recent_memories.reverse()

        # Remove the time field from each memory
        for memory in most_recent_memories:
            del memory['time']
        return most_recent_memories

    def num_tokens_from_messages(self, messages):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        try:
            num_tokens = 2  # Start with 2 tokens for the initial token count
            for message in messages:
                num_tokens += 4  # Add 4 tokens for each message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
            return num_tokens
        except Exception as e:
            logging.error(e)
            return 0

    def ensure_token_count(self, message):
        logging.info("Ensuring token count...")

        # Calculate the token count for the new message
        new_message_tokens = self.num_tokens_from_messages([message])

        # Recalculate the total token count based on the current conversation memory
        self.token_count = self.num_tokens_from_messages(self.conversation_memory)

        # Check if the new token count would exceed the limit
        if self.token_count + new_message_tokens > self.config["prompt_token_limit"]:
            logging.info("Token count exceeded, reducing conversation memory by removing oldest non-system messages...")

            # Remove the oldest non-system messages until the token count is below the limit
            safety_counter = 0  # Add a safety counter to prevent infinite loops
            max_retries = len(
                self.conversation_memory) - 1  # Set the maximum number of retries to the number of messages excluding the system prompt

            while self.token_count + new_message_tokens > self.config[
                "prompt_token_limit"] and safety_counter < max_retries:
                # Skip the first item (system prompt) when removing messages
                removed_message = self.conversation_memory.pop(1)
                removed_tokens = self.num_tokens_from_messages([removed_message])
                self.token_count -= removed_tokens

                # Increment the safety counter
                safety_counter += 1

        # Add the new message tokens to the total token count
        self.token_count += new_message_tokens

    def save_message_to_log(self, message, speaker):
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

    def get_gpt_response(self, user_message, tries=0):
        max_tries = 5
        try:
            logging.info("Getting GPT response...")
            self.update_conversation_memory("user", user_message)
            self.save_message_to_log(user_message, "user")

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
            response = openai.ChatCompletion.create(**prompt)
            gpt_response = response["choices"][0]["message"]["content"]
            self.update_conversation_memory("assistant", gpt_response)
            self.save_message_to_log(gpt_response, "assistant")

            return gpt_response
        except Exception as e:
            logging.error(e)
            if "maximum context length" in str(e):
                logging.info("Too many tokens, reducing conversation memory by removing oldest messages...")
                self.conversation_memory.append({"role": "system", "content": self.config["system_prompt"]})
                self.token_count = self.num_tokens_from_messages([self.conversation_memory[-1]])
                return self.get_gpt_response(user_message)
            elif tries < max_tries:
                logging.info("Retrying...")
                sleep(1)
                return self.get_gpt_response(user_message, tries + 1)
            else:
                logging.info("Failed to get GPT response.")
                return f"I'm sorry, The server returned an error. Please try again. Error: {e}"

    def update_conversation_memory(self, role, content):
        logging.info("Updating conversation memory...")
        self.conversation_memory.append({"role": role, "content": content})
        self.token_count += 4


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
    logging.info(f"System Prompt: {gpt4_chat.conversation_memory[0]['content']}")
    logging.info(f"Response: {response}")
    logging.info(f"Response length: {len(response)}")
    logging.info(f"Prompt token count: {gpt4_chat.num_tokens_from_messages(gpt4_chat.conversation_memory)}")
    logging.info(
        f"Response token count: {gpt4_chat.num_tokens_from_messages([{'role': 'assistant', 'content': response}])}")
    logging.info(
        f"Total token count: {gpt4_chat.num_tokens_from_messages(gpt4_chat.conversation_memory) + gpt4_chat.num_tokens_from_messages([{'role': 'assistant', 'content': response}])}")

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
