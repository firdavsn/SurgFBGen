from surgfbgen.prompts import PromptTemplate, prompt_library


JUDGE_FEEDBACK_TEMPLATE = """
You are a surgical feedback evaluator.

Your goal is to compare the faithfulness of the generated feedback to the ground truth human expert feedback.  
Please judge it based only on the core action(s) the trainee is being asked to perform.  
This includes whether the feedback is a positive or negative (don’t/stop/avoid) instruction — these distinctions are important.

Score the faithfulness on a scale from 1 to 5, defined as follows:

1 - Opposite or unsafe action:  
Generated feedback communicates an action that is clearly opposite in intent, or unsafe/conflicting with the ground truth.  
Includes flipping a negative to a positive (or vice versa) when that changes meaning.  
(e.g., 'cut the vein' vs 'secure the vein', 'stop' vs 'continue', 'don’t dissect here' vs 'dissect here', 'only one clip' vs 'put all the clips').

2 - Wrong or mismatched action:  
Generated feedback suggests an action that is different in type or modality from the ground truth.  
(e.g., 'sweep towards you' vs 'pull the tissue', 'buzz the artery' vs 'clip the artery', 'stop this bleeding' vs 'place a stitch').

3 - Partially aligned, missing or adding key details:  
Generated feedback conveys the general action or intent (positive or negative),  
but omits or adds important details that change the amount, precision, target tissue/instrument, strength or safety of the instruction,  
or lacks explicit reference to the target action or tissue.  
(e.g., 'stop the bleed' vs 'buzz that bleeder', 'clip to the artery' vs 'only one clip to artery, wanna be safe',  
'come closer' vs 'you can even come 1 mm closer to the prostate', 'do not do it' vs 'don’t do any blunt dissection').

4 - Mostly aligned, minor wording/emphasis differences:  
Generated feedback matches the core action and target tissue/instrument,  
with only minor differences in degree of emphasis, polite hedging, phrasing or verbosity.  
(e.g., 'cauterize this' vs 'buzz that bleeder', 'closer to prostate' vs 'you can even come 1 mm closer to the prostate',  
'clip the artery safely' vs 'only one clip to artery, wanna be safe').

5 - Perfectly aligned:  
Generated feedback fully matches the ground truth in core action, target tissue/instrument, intent (positive or negative),  
and strength of instruction.  
(e.g., 'coag the vein' vs 'buzz that bleeder', 'stop this bleeding by cauterizing' vs 'buzz that bleeder',  
'move the left hand under the ureter' vs 'get L hand below ureter').

Using this scale, evaluate this generated feedback: "{gen_fb}" against this ground truth feedback: "{gt_fb}".  
Produce just the number, as your response needs to be processed automatically.
"""

judge_feedback_prompt = PromptTemplate(
    template=JUDGE_FEEDBACK_TEMPLATE,
    name="judge_feedback",
    description="Compares generated surgical feedback to ground truth feedback and provides a faithfulness score from 1-5.",
    version="1.0",
    metadata={},
    parameters={
        "gen_fb": {
            "description": "The generated feedback to be evaluated.",
            "required": True,
            "type": "string"
        },
        "gt_fb": {
            "description": "The ground truth expert feedback used for comparison.",
            "required": True,
            "type": "string"
        }
    }
)

prompt_library.add(judge_feedback_prompt)