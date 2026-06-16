import json
import re
from pathlib import Path
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    data_dir = Path(DATA_PATH)
    train_path = data_dir / TRAIN_FILE
    labels_path = data_dir / LABELS_FILE

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            # We ensure the returned dict contains at least the 5 required fields.
            # Extra fields from 'ep' are also preserved as per requirements.
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], title: str, description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    taxonomy = """
- interview: One host + one guest. The guest's knowledge or experience drives the episode.
- solo: A single host speaking directly to the audience — opinion, reflection, or explainer.
- panel: Multiple guests (usually 3+) in structured discussion. No single voice dominates.
- narrative: Reported or documentary storytelling. Events unfold across the episode.
"""

    prompt_parts = [
        "You are a podcast episode classifier. Your task is to classify a podcast episode into one of these four categories.",
        "If the provided description is nonsensical or does not fit any of these categories, you must return 'unknown' as the label.",
        taxonomy.strip(),
        "\nReturn your answer in the following format:",
        "Label: [label]",
        "Reasoning: [brief reasoning]",
        "\nThe label must be exactly one of: interview, solo, panel, narrative, or unknown.\n",
    ]

    if labeled_examples:
        prompt_parts.append("Here are some examples:")
        for ex in labeled_examples:
            prompt_parts.append(f"Title: {ex['title']}")
            prompt_parts.append(f"Description: {ex['description']}")
            prompt_parts.append(f"Label: {ex['label']}\n")

    prompt_parts.append("Now, classify this episode:")
    prompt_parts.append(f"Title: {title}")
    prompt_parts.append(f"Description: {description}")
    prompt_parts.append("Label: ?")

    return "\n".join(prompt_parts)


def classify_episode(title: str, description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    prompt = build_few_shot_prompt(labeled_examples, title, description)

    try:
        completion = _client.chat.completions.create(
            model=LLM_MODEL, 
            messages=[
                {
                    "role": "system", 
                    "content": "You are a classifier. Return 'unknown' if the content is not a podcast description."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300, 
            temperature=0,
        )
        response_text = completion.choices[0].message.content

        # Simple parsing for Label and Reasoning
        label_match = re.search(r"Label:\s*(\w+)", response_text, re.IGNORECASE)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

        label = label_match.group(1).lower().strip() if label_match else "unknown"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text.strip()

        if label not in VALID_LABELS:
            label = "unknown"

        return {
            "label": label,
            "reasoning": reasoning,
        }
    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"Error during classification: {str(e)}",
        }
