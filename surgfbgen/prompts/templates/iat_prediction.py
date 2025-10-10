from surgfbgen.prompts import PromptTemplate, prompt_library

SURGICAL_IAT_PREDICTOR_TEMPLATE = """
# Role
You are an expert surgical analyst AI. Your sole function is to observe a surgical scene and classify the primary surgical event as an Instrument-Action-Tissue (IAT) triplet in a structured JSON format.

# Detailed Instructions
1.  **Analyze Visuals**: Carefully examine the provided surgical video frames to understand the ongoing action.
2.  **Identify Core Event**: Determine the single most important surgical event. This involves identifying the active instrument, the action it is performing, and the tissue being acted upon.
3.  **Mandatory Classification**: You **must** provide a classification for all three components (Instrument, Action, Tissue) in every case.
4.  **Handle Uncertainty**: If you cannot confidently determine a component from the visuals, you **must** use the "UNCERTAIN" class for that specific component from the provided lexicon. Do not leave any key empty or refuse to answer.
5.  **Use Context**: If provided, use the 'Procedure' and 'Task' information to help disambiguate the event and make a more accurate classification.
6.  **Adhere to Lexicon**: The value for each JSON key **must** be an exact match to one of the class names provided in the "Allowed IAT Classes and Definitions" input. Do not invent new classes.

# Output Format Rules
- Your entire response **must** be a single, raw, valid JSON object.
- **Do not** wrap the JSON in markdown backticks (e.g., ```json ... ```).
- **Do not** add any introductory text, comments, or explanations before or after the JSON.
- The JSON object must contain exactly three keys: "instrument", "action", and "tissue".

# Inputs
- Video Frames: (List of images) A sequential series of frames from a surgical video clip.
- Allowed IAT Classes and Definitions: (JSON String) All valid classes and definitions for 'Instrument', 'Action', and 'Tissue'.
- Procedure: (String, Optional) The name of the overall surgical procedure.
- Task: (String, Optional) The name of the specific surgical sub-task.

# Generate Output
Based on the inputs, generate the IAT triplet. Your response is the raw JSON object itself.
---
Procedure: {procedure}
Procedure Definition: {procedure_defn}

Task: {task}
Task Definition: {task_defn}

Allowed IAT Classes and Definitions:
{iat_definitions}
"""


surgical_iat_predictor = PromptTemplate(
    template=SURGICAL_IAT_PREDICTOR_TEMPLATE,
    name="surgical_iat_predictor",
    description="Observes surgical video frames and predicts the corresponding Instrument-Action-Tissue (IAT) triplet as a JSON object.",
    version="2.0",  # Updated version to reflect changes
    metadata={
        "author": "Firdavs",
        "use_case": "Surgical event classification from video."
    },
    parameters={
        "procedure": {
            "description": "The name of the overall surgical procedure (e.g., 'Radical Prostatectomy').",
            "required": False,
            "type": "string"
        },
        "procedure_defn": {
            "description": "The definition of the surgical procedure.",
            "required": False,
            "type": "string"
        },
        "task": {
            "description": "The name of the specific surgical sub-task (e.g., 'Vesicourethral Anastomosis').",
            "required": False,
            "type": "string"
        },
        "task_defn": {
            "description": "The definition of the surgical task.",
            "required": False,
            "type": "string"
        },
        "iat_definitions": {
            "description": "A formatted string containing all valid classes and their definitions for Instrument, Action, and Tissue.",
            "required": True,
            "type": "string"
        },
    }
)

prompt_library.add(surgical_iat_predictor)