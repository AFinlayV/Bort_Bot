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


class GPT4Chat(commands.Bot):
    def __init__(self, memory_limit=10, config_file="config4.json", log_dir="log", command_prefix="/", intents=None):
        super().__init__(intents=intents, command_prefix=command_prefix)
        self.memory_limit = memory_limit
        self.config = self.load_config(config_file)
        self.recent_memories = self.load_recent_memories(log_dir)
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

        memories.sort(key=lambda x: x['time'])  # If you want the most recent memories last
        # remove the time field from each memory
        for memory in memories:
            del memory['time']
        return memories[:self.memory_limit]

    def process_message(self, message, speaker, system_prompt=None):
        def num_tokens_from_messages(messages):
            encoding = tiktoken.encoding_for_model(
                self.model) if self.model in tiktoken.MODELS else tiktoken.get_encoding("cl100k_base")
            num_tokens = 2 + sum(4 + sum(len(encoding.encode(value)) for key, value in msg.items()) for msg in messages)
            return num_tokens

        # Ensure token count
        logging.info("Ensuring token count...")
        new_message_tokens = num_tokens_from_messages([{"role": speaker, "content": message}])
        self.token_count = num_tokens_from_messages(self.conversation_memory)

        if self.token_count + new_message_tokens > self.config["prompt_token_limit"]:
            logging.info("Token count exceeded, reducing conversation memory by removing oldest non-system messages...")
            while self.token_count + new_message_tokens > self.config["prompt_token_limit"]:
                removed_message = self.conversation_memory.pop(1)
                self.token_count -= num_tokens_from_messages([removed_message])

        self.token_count += new_message_tokens

        # Save message to log
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

        # Update conversation memory
        logging.info("Updating conversation memory...")
        if speaker != "system" and system_prompt is not None and (
                not self.conversation_memory or self.conversation_memory[0]["role"] != "system"):
            self.conversation_memory.insert(0, {"role": "system", "content": system_prompt})
            self.token_count += 4
        self.conversation_memory.append({"role": speaker, "content": message})
        self.token_count += 4

    def update_system_prompt(self, messages_list):
        pass
        # logging.info("Updating system prompt...")
        # messages = "\n".join(messages_list)
        # prompt = f"""
        # Given the following conversation history: {messages}
        # Come up with a new system prompt that matches the tone and subject matter of the conversation history.
        #
        # NEW SYSTEM PROMPT:
        # """
        # response = openai.Completion.create(
        #     engine="davinci",
        #     prompt=prompt,
        #     temperature=0.0
        # )
        # self.config["system_prompt"] = response["choices"][0]["text"]
        # self.conversation_memory[0]["content"] = self.config["system_prompt"]

    def get_gpt_response(self, user_message, tries=0):
        max_tries = 5
        try:
            logging.info("Getting GPT response...")
            self.process_message(user_message, "user", system_prompt=self.config["system_prompt"])

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

    def split_response(self, text, max_length):
        split_texts = []
        while len(text) > max_length:
            index = text.rfind(" ", 0, max_length)
            if index == -1:
                index = max_length
            split_texts.append(text[:index])
            text = text[index:].strip()
        split_texts.append(text)
        return split_texts

    async def on_ready(self):
        print(f"We have logged in as {self.user}")

    async def generate_and_send_response(self, channel, question):
        if question.startswith("!"):
            return  # Ignore messages that start with !
        print('generating response...')
        logging.info(f"Message: {question.strip()}")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_gpt_response, question.strip())
        print('response generated')
        logging.info(f"System Prompt: {self.conversation_memory[0]['content']}")
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

    async def on_message(self, message, bot):
        if message.author == bot.user or message.author.bot:
            return
        elif message.content.startswith('!'):
            await bot.process_commands(message)  # Pass the message object instead of ctx
        elif message.channel.id != self.chan_id:
            return
        else:
            question = message.content.strip()
            await self.generate_and_send_response(message.channel, question)  # Pass the channel instead of ctx

    @commands.command()
    async def bort(self, ctx, *, question):
        if not ctx.author.bot:
            await self.generate_and_send_response(ctx, ctx.message.content)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("gpt4chat.log", mode="a")
        ]
    )
    logging.info("Starting GPT4Chat...")

    intents = discord.Intents.all()
    bot = GPT4Chat(intents=intents, command_prefix="/")
    bot.add_command(bot.bort)
    # Create a bot instance with the command prefix you'd like to use
    intents = discord.Intents.all()
    bot = commands.Bot(intents=intents, command_prefix="/")

    # Load the GPT-4 chat model (modify this according to your model loading method)
    gpt4_chat = GPT4Chat()

    # Set up the channel ID
    if gpt4_chat.config["experimental"]:
        CHAN_ID = gpt4_chat.config["experimental_channel"]
        logging.info(f"Using experimental mode. {CHAN_ID}")
        bot_token = os.environ.get("SANDBOX_DISCORD_TOKEN")
        logging.warning(
            f"Experimental mode is enabled. This is not recommended for production use. Token:{bot_token[:5]}...{bot_token[-5:]}")
    else:
        CHAN_ID = gpt4_chat.config["main_channel"]
        logging.info(f"Using production mode. {CHAN_ID}")
        bot_token = os.environ.get("BORT_DISCORD_TOKEN")
        logging.info("Production mode is enabled. Token: {bot_token[:5]}...{bot_token[-5:]}")

    if bot_token is None:
        print("Error: BORT_DISCORD_TOKEN environment variable not found.")
        exit(1)

    # Run the bot with your token
    bot.run(bot_token)


if __name__ == "__main__":
    main()
