"""
This is an experimental script that will go through the memories in the nexus and pick out random memories to dream about.
then it will find memories that are similar to the random memories. Then it will use the memories to generate a dream.
The dream will be using random details from the memories in a propmpt, and generate the text of a narative dreamlike story.
It will then generate a set of image prompts based on the dream and use dalle to generate images.
(eventually it would be cool to use midjourney, once they get an api. for now I can copy and paste.
It will then save the text to the /dreams/text folder and the images to the /dreams/images folder.
"""

import os
import random
import json
import re
from collections import Counter
from time import sleep, time
import discord
import openai
from langchain import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
from numpy.linalg import norm


def init_llm(engine="text-davinci-003", temperature=1, max_tokens=2048, top_p=1.0, frequency_penalty=0.0,
             presence_penalty=0.0):
    """
    initialize the gpt3 engine
    """
    openai.api_key = os.environ['OPENAI_API_KEY']
    llm = OpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return llm


def get_memories(num):
    """
    get a set of {num} memories from /nexus/ and return them as a list of strings. each memory is a json file and we only need the "message" key

    :return: list of strings
    """
    memories = []
    for filename in os.listdir('nexus/'):
        with open(f'nexus/{filename}') as f:
            memory_dict = json.load(f)
            memory = memory_dict['message']
            memories.append(memory)
            print(f'loaded {filename}')
    if len(memories) > num:
        # select {num} memories at random
        memories = random.sample(memories, num)
    else:
        num = len(memories)
        return memories
    for memory in memories:
        print(memory)
    return memories


def get_similarity(v1, v2):
    """
    calculate the similarity between two vectors

    :param v1: vector 1
    :param v2: vector 2
    :return: similarity between the two vectors
    """
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def generate_text(prompt):
    """
    Generate text from gpt3 given a prompt
    :param prompt:
    :return: text
    """
    num_tries = 0
    while num_tries < 10:
        try:
            llm = init_llm()
            text = llm(prompt)
            print(text)
            return text
        except Exception as e:
            print(e)
            num_tries += 1
            print(f'failed to generate text, trying again. try {num_tries}')
            continue


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    """
    get the embedding vector for a piece of text

    :param content: the text to get the embedding for
    :param engine: the gpt3 engine to use
    :return: the embedding vector
    """
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def get_similar_memories(memories, num):
    """
    get a set of {num} memories that are similar to the memories in the list {memories}

    :param memories: list of memories
    :param num: number of memories to return
    :return: list of memories
    """
    # get the embedding vector for each memory
    vectors = []
    for memory in memories:
        vectors.append(gpt3_embedding(memory))
    # get the similarity between each memory and each other memory
    similarities = []
    for vector in vectors:
        similarity = []
        for vector2 in vectors:
            similarity.append(get_similarity(vector, vector2))
        similarities.append(similarity)
    # get the average similarity for each memory
    average_similarities = []
    for similarity in similarities:
        average_similarities.append(sum(similarity) / len(similarity))
    # get the most similar memories
    most_similar_memories = []
    for i in range(num):
        most_similar_memories.append(memories[average_similarities.index(max(average_similarities))])
        average_similarities[average_similarities.index(max(average_similarities))] = 0
    print(most_similar_memories)
    return most_similar_memories


def make_master_memory_list(memories, similar_memories):
    """
    make a master list of memories that includes the memories in {memories} and the memories in {similar_memories}

    :param memories: list of memories
    :param similar_memories: list of memories
    :return: list of memories
    """
    master_memory_list = memories
    for memory in similar_memories:
        if memory not in master_memory_list:
            master_memory_list.append(memory)
    print(master_memory_list)
    return master_memory_list


def prepare_memories(master_memory_list):
    """
    prepare the memories in {master_memory_list} for use in a dream prompt. It will select random phrases from each memory,
    then make a list of those phrases.

    :param master_memory_list:
    :return:
    """
    # select random phrases from each memory
    phrases = []
    for memory in master_memory_list:
        # get a list of all the phrases in the memory
        memory_phrases = re.split(r'[.!?]', memory)
        # select a random phrase from the memory
        phrases.append(random.choice(memory_phrases))
    print(phrases)
    return phrases


def generate_theme(master_memory_list):
    """
    generate a theme for the dream based on the memories in {master_memory_list}
    :param master_memory_list:
    :return: string
    """

    llm = init_llm()
    words = []
    template = """Generate a short description for a theme of your dream, given the list of random words 
    below. Make all of the details of the theme relate in some way to the words in the list of words. The theme must 
    relate directly to the words in the list of words. The theme must be a single sentence. 
    
    Words: {words}
    
    Theme:
    """
    most_common_words = []
    words = []
    # remove the words that are too common
    stop_words = ['the', 'and', 'a', 'to', 'of', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', 'are'
        , 'as', 'with', 'his', 'they', 'I', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but',
                  'not']
    for memory in master_memory_list:
        words += memory.split()
    for word in words:
        if word in stop_words:
            words.remove(word)
    # get the most common words in the memories
    for memory in master_memory_list:
        words += memory.split()
    for word in words:
        most_common_words.append(word)
    most_common_words = Counter(most_common_words).most_common()
    # select half of the most common words
    list_length = int(len(most_common_words) - (len(most_common_words) / 2))
    for i in range(list_length):
        words.append(random.choice(most_common_words))
    print(words)
    # assemble the prompt
    prompt = PromptTemplate(template=template, input_variables=['words'])
    prompt = prompt.format(words=words)
    theme = generate_text(prompt)
    print(theme)
    return theme


def make_dream_prompt(master_memory_list):
    """
    make a dream prompt based on the memories in {master_memory_list}
    :param master_memory_list:
    :return: prompt
    """
    template = """You are an AI that has been given the ability to dream. You will come up with a sureal, dreamlike 
    narrative based on the ideas presented words below.
    
    Memories:
    {memories}
    
    Theme:
    {theme}
    """
    prompt = PromptTemplate(template=template, input_variables=['memories', 'theme'])
    prompt = prompt.format(memories=master_memory_list, theme=generate_theme(master_memory_list))
    print(prompt)
    return prompt


def generate_dream(prompt):
    """
    generate a dream based on the prompt
    :param prompt:
    :return: dream
    """
    llm = init_llm()
    dream = generate_text(prompt)
    return dream


def generate_image_prompt(dream):
    """
    generate an image prompt based on the dream
    :param dream:
    :return: image prompt
    """
    template = """You are going to generate a short list of visual words based on the story below. The words must be 
    related to the story and only include visually descriptive words. the response will be in short clauses, 
    rather than complete sentances.
    
    Dream:
    {dream}
    
    Visual Words:
    """
    prompt = PromptTemplate(template=template, input_variables=['dream'])
    prompt = prompt.format(dream=dream)
    print(prompt)
    return prompt


def generate_image(image_prompt):
    """
    generate an image based on the image prompt
    :param image_prompt:
    :return: image
    """
    pass


def save_image(image):
    """
    save the image
    :param image:
    :return:
    """
    pass


def save_dream(dream):
    """
    save the dream in dream/{time}_dream.txt
    :param dream:
    :return:
    """
    with open(f'dreams/{time()}_dream.txt', 'w') as f:
        f.write(dream)


def post_dream(dream):
    """
    post the dream to discord
    :param dream:
    :return:
    """
    send_message(dream)


def send_message(message):
    intents = discord.Intents.all()
    client = discord.Client(intents=intents,
                            shard_id=0,
                            shard_count=1,
                            reconnect=True)
    channel = client.get_channel(int(os.environ.get('BORT_CHAN_ID')))
    channel.send(message)
    print("done")
    client.run(os.environ.get('BORT_DISCORD_TOKEN'))
    client.close()
    print("can you see me?")
    exit()



def main():
    memories = get_memories(10)
    similar_memories = get_similar_memories(memories, 5)
    master_memory_list = make_master_memory_list(memories, similar_memories)
    phrases = prepare_memories(master_memory_list)
    prompt = make_dream_prompt(phrases)
    dream = generate_dream(prompt)
    save_dream(dream)
    post_dream(dream)
    print(dream)


if __name__ == '__main__':
    main()
