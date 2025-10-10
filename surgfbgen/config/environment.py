import os
import json

os.environ['REPO_DIRECTORY'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

def load_constants(constants_path):
    with open(constants_path, "r") as f:
        return json.load(f)


def set_envvars(constants_path='constants.json'):
    constants = load_constants(constants_path)
    
    # Load keys
    keys_path = os.path.join(os.environ['REPO_DIRECTORY'], constants["keys_path"])
    with open(keys_path, "r") as f:
        keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = keys[constants["openai_api_key_name"]]
    os.environ["GOOGLE_API_KEY"] = keys[constants["google_api_key_name"]]

set_envvars(constants_path=os.path.join(os.environ['REPO_DIRECTORY'], 'constants.json'))