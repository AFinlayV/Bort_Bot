"""
This script will go through a folder of json files and add the field "server" with the value of 1071975175165333544 in each file
"""

import json
import os

def main():
# Change this to the folder where the json files are
    folder = "nexus"
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as f:
                data = json.load(f)
                data["server"] = 1071975175165333544
            with open(os.path.join(folder, filename), "w") as f:
                json.dump(data, f)

if __name__ == "__main__":
    main()