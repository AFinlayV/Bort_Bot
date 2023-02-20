"""
This script will take BORT's old memories and upload them to the new pinecone database.

The format of the old memories is :
log_1676065949.186774_AlexThe5th.json
{
  "message": "[{'AlexThe5th'}]: ok maybe now?",
  "speaker": "AlexThe5th",
  "time": 1676065892.817868,
  "timestring": "Friday, February 10, 2023 at 04:51PM ",
  "uuid": "0b26e205-3eb7-46fc-8ed6-1996889098e0",
  "vector": <<vector>>
}

the format of the new one is:
0b26e205-3eb7-46fc-8ed6-1996889098e0.json
{
  "message": "[{'AlexThe5th'}]: ok maybe now?",
  "speaker": "AlexThe5th",
  "time": 1676065892.817868,
  "timestring": "Friday, February 10, 2023 at 04:51PM ",
  "uuid": "0b26e205-3eb7-46fc-8ed6-1996889098e0",
}

It needs to send the vector to pinecone with the info from the old file as metadata

"""

import os
import json
import pinecone
from uuid import uuid4

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment='us-east1-gcp')
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def upload_old_memories():
    # load the old memories
    old_memories = os.listdir('nexus')
    old_memories = [x for x in old_memories if x.startswith('log_')]
    print(f'Found {len(old_memories)} old memories')

    # load the pinecone index
    index = pinecone.Index('bort')
    vector_list = []
    # loop through the old memories
    for old_memory in old_memories:
        # load the old memory
        old_memory = load_json(os.path.join('nexus', old_memory))

        # get the vector
        vector = old_memory['vector']
        uuid = old_memory['uuid']

        # get the metadata
        metadata = {k: v for k, v in old_memory.items() if k != 'vector'}
        save_json(os.path.join('nexus', f'{uuid}.json'), metadata)
        # upload to pinecone
        index.upsert([(uuid, vector)])

upload_old_memories()