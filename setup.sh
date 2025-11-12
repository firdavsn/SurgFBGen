if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
else
    SCRIPT_PATH=$(realpath "$0")
fi
REPO_DIR=$(realpath $(dirname $SCRIPT_PATH))

echo "REPO_DIR: $REPO_DIR"

setup_env() {
    cd $REPO_DIR
    deactivate
    rm -rf .venv
    uv sync
    source .venv/bin/activate
}

setup_surgvlp() {
    cd $REPO_DIR
    rm -rf SurgVLP
    source .venv/bin/activate
    uv pip install git+https://github.com/openai/CLIP.git
    uv pip install git+https://github.com/CAMMA-public/SurgVLP.git
    git clone https://github.com/CAMMA-public/SurgVLP.git
}

setup_cotracker() { 
    cd $REPO_DIR
    rm -rf co-tracker
    source .venv/bin/activate
    git clone https://github.com/facebookresearch/co-tracker
    cd co-tracker
    uv pip install -e .
    uv pip install matplotlib flow_vis tqdm tensorboard
    cd ..
    mkdir -p checkpoints
    cd checkpoints
    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
    mv scaled_offline.pth cotracker3.pth
    cd $REPO_DIR
    uv pip install imageio[ffmpeg]
}

main() {
    setup_env
    setup_surgvlp
    setup_cotracker
}

main
