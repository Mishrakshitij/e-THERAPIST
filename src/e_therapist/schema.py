"""The PSYCON schema and shared classifier label/input conventions."""

from __future__ import annotations

from typing import Any, TypedDict


LABEL_VOCAB: dict[str, tuple[str, ...]] = {
    "gender_age": (
        "female_young",
        "female_adult",
        "female_elder",
        "male_young",
        "male_adult",
        "male_elder",
    ),
    "persona": (
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ),
    "sentiment": ("negative", "neutral", "positive"),
    "approach": ("directive", "non_directive", "eclectic"),
    "politeness": ("impolite", "moderately_polite", "polite"),
    "ipc": (
        "confrontational",
        "imposing",
        "dissatisfied",
        "uncertain",
        "compliant",
        "helpful",
        "understanding",
        "empathetic",
        "directing",
    ),
}
LABELS = LABEL_VOCAB
LABEL_TO_ID = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in LABEL_VOCAB.items()
}
ID_TO_LABEL = {task: dict(enumerate(labels)) for task, labels in LABEL_VOCAB.items()}
SPEAKERS = ("patient", "therapist")
GENDERS = ("female", "male")
AGES = ("young", "adult", "elder")
ISSUES = (
    "anxiety",
    "bipolar_disorder",
    "depression",
    "disruptive_behaviour_and_dissocial_disorders",
    "ptsd",
    "schizophrenia",
    "stress",
)
TURN_FIELDS = (
    "conversation_id",
    "turn_id",
    "speaker",
    "utterance",
    "issue",
    "gender",
    "age",
    "persona",
    "sentiment",
    "ipc",
    "politeness",
    "approach",
)


class Turn(TypedDict):
    conversation_id: str
    turn_id: int
    speaker: str
    utterance: str
    issue: str | None
    gender: str | None
    age: str | None
    persona: str | None
    sentiment: str | None
    ipc: str | None
    politeness: str | None
    approach: str | None


def normalize_label(value: Any) -> str | None:
    """Normalize casing and separators; an empty cell represents missing data."""
    if value is None or not str(value).strip():
        return None
    return "_".join(str(value).strip().casefold().replace("-", " ").split())


def build_classifier_input(
    task: str,
    response: str,
    user_text: str = "",
    sentiment: str | None = None,
) -> str:
    """Format exactly the same features for classifier training and rewards.

    Targets such as persona, approach and politeness are never input features.
    For the two sentiment-conditioned classifiers, ``sentiment`` must describe
    the preceding patient utterance, never the therapist response.
    """
    if task not in LABEL_VOCAB:
        raise ValueError(f"Unknown classifier task: {task!r}")
    if task == "sentiment":
        text = user_text or response
    elif task == "approach":
        if not user_text.strip():
            raise ValueError("The approach classifier requires a patient utterance")
        text = f"Therapist: {response}\nPatient: {user_text}"
    elif task in ("politeness", "ipc"):
        if sentiment not in LABEL_VOCAB["sentiment"]:
            raise ValueError(f"{task} requires a valid patient sentiment")
        text = f"Therapist: {response}\nPatient sentiment: {sentiment}"
    else:
        text = response
    if not (user_text if task == "sentiment" and user_text else response).strip():
        raise ValueError("Classifier text must not be blank")
    return text


def format_classifier_text(
    task: str,
    response: str,
    user_utterance: str = "",
    user_sentiment: str | None = None,
) -> str:
    """Keyword-compatible convenience wrapper for example dictionaries."""
    return build_classifier_input(task, response, user_utterance, user_sentiment)
