import os
import json

os.environ['REPO_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

def set_envvars(env_path: str):
    with open(env_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value

set_envvars(env_path=os.path.join(os.environ['REPO_DIR'], '.env'))