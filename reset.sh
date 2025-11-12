if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
else
    SCRIPT_PATH=$(realpath "$0")
fi
REPO_DIR=$(realpath $(dirname $SCRIPT_PATH))

cd $REPO_DIR
deactivate
rm -rf .venv
rm -rf SurgVLP
rm -rf co-tracker