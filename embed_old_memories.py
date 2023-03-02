"""
Go through the /old/ directory and find all the json files.
Embed the text in the message field, and put that on pinecone with the uuid as the key.
"""

import json
import os
import pinecone
import openai

with open ("config.json", "r") as f:
    CONFIG = json.load(f)

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def main():
    # Change this to the folder where the json files are
    folder = "nexus"
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=CONFIG["pinecone_environment"])
    vdb = pinecone.Index(CONFIG["pinecone_index"])
    payload = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):

            with open(os.path.join(folder, filename), "r") as f:
                data = json.load(f)
                vector = gpt3_embedding(data["message"])
                uuid = str(data["uuid"])
                payload.append((uuid, vector))
    #save payload to local file
    with open("payload.txt", "w") as f:
        f.write(str(payload))
    #save payload to pinecone
    vdb.upsert(payload)

