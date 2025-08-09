import os
import json

def load_constants(constants_path):
    with open(constants_path, "r") as f:
        return json.load(f)
    
def set_envvars(constants_path='constants.json'):
    constants = load_constants(constants_path)
    
    # Load keys
    curr_path = os.path.dirname(os.path.abspath(__file__))
    keys_path = os.path.join(curr_path, constants["keys_path"])
    with open(keys_path, "r") as f:
        keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = keys[constants["openai_api_key_name"]]
