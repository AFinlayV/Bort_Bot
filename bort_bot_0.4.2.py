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
        self.token_count = self.num_tokens_from_messages([self.conversation_memory[-1]])
        self.model = "gpt-4"
        self.token_count = self.num_tokens_from_messages([self.conversation_memory[-1]])

        # Load recent memories on startup
        self.conversation_memory.extend(self.load_recent_memories())
        if self.config["experimental"]:
            self.model = "gpt-3.5-turbo"

    def load_config(self):
        logging.info("Loading config...")
        with open("config4.json", "r") as config_file:
            config = json.load(config_file)
            for key, value in config.items():
                logging.info(f"{key}: {value}")
            return config


    def ensure_token_count(self, message):
        logging.info("Ensuring token count...")
        if self.token_count > self.config["prompt_token_limit"]:
            logging.info("Token count exceeded, reducing conversation memory by removing oldest messages...")
            self.conversation_memory.append({"role": "system", "content": self.config["system_prompt"]})
            self.token_count = self.num_tokens_from_messages([self.conversation_memory[-1]])
        self.token_count += self.num_tokens_from_messages([message])

    def load_recent_memories(self):
        logging.info("Loading recent memories...")
        memories = []
        for filename in os.listdir("log"):
            with open(os.path.join("log", filename), "r") as log_file:
                memories.append(json.load(log_file))
        memories.sort(key=lambda x: x["time"])
        # Format the memories to match the format of the conversation memory
        for memory in memories:
            memory["role"] = memory["speaker"]
            memory["content"] = memory["message"]
            del memory["speaker"]
            del memory["message"]
            del memory["time"]
            del memory["timestring"]
            del memory["uuid"]
        return memories[:self.memory_limit]

    def num_tokens_from_messages(self, messages, model="gpt-4"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-4":
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
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

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
    logging.warning(f"Experimental mode is enabled. This is not recommended for production use. Token:{bot_token[:5]}...{bot_token[-5:]}")
else:
    bot_token = os.environ.get("BORT_DISCORD_TOKEN")
if bot_token is None:
    print("Error: BORT_DISCORD_TOKEN environment variable not found.")
    exit(1)

# Run the bot with your token
bot.run(bot_token)
