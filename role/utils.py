import json
import re

REVIEW_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "strengths": {
                "type": "ARRAY",
                "description": "List of key strengths of the paper.",
                "items": {"type": "STRING"}
            },
            "weaknesses": {
                "type": "ARRAY",
                "description": "List of key weaknesses of the paper.",
                "items": {"type": "STRING"}
            },
            "justification": {
                "type": "STRING",
                "description": "A concise paragraph summarizing the overall reasoning."
            },
            "score": {
                "type": "STRING",
                "description": "The numerical rating (0, 2, 4, 6, 8, or 10).",
                "enum": ["0", "2", "4", "6", "8", "10"]
            }
        },
        "required": ["strengths", "weaknesses", "justification", "score"]
    }

AC_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "final_justification": {"type": "STRING"},
        "final_decision": {
            "type": "STRING", 
            "enum": ["Accept", "Reject"]
        },
        "review_evaluation": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "justification": {"type": "STRING"},
                    "score": {
                        # FIX: Must be STRING to allow the 'enum' property
                        "type": "STRING", 
                        "description": "Review quality score (0, 2, 4, 6, 8, or 10).",
                        # FIX: Enum values must be string literals
                        "enum": ["0", "2", "4", "6", "8", "10"]
                    }
                },
                "required": ["id", "justification", "score"]
            }
        }
    },
    "required": ["final_justification", "final_decision", "review_evaluation"]
}

def clean_and_load_json(text_with_wrappers):
    """
    Strips markdown wrappers and cleans up non-standard characters from LLM output.
    """
    # 1. Strip the markdown code block wrappers (```json\n and \n```)
    # This pattern is robust: it matches optional leading/trailing whitespace, 
    # the code block tags, and captures the inner JSON content.
    match = re.search(r"```json\s*(\{.*\})\s*```", text_with_wrappers, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
    else:
        # If markdown wrappers aren't found, assume the text is mostly JSON
        json_content = text_with_wrappers.strip()

    # 2. Replace non-standard spaces (like \xa0, the non-breaking space) with standard space.
    # The \s pattern in Python includes common white space but sometimes misses specific non-ASCII ones.
    # Using .replace() is direct for the observed \xa0 in your prompt (which appears as ' ').
    json_content = json_content.replace('\xa0', ' ')
    
    # 3. Load the cleaned content
    try:
        data = json.loads(json_content)
        return data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON after cleaning: {e}")
        import pdb; pdb.set_trace()
        return None