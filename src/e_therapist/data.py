"""Validated turns, conversation splits and leakage-free training examples.

The canonical CSV keeps all source turn identifiers, including empty utterances.
Empty strings in label columns are read as ``None``. Missing labels are excluded
only from their corresponding classifier task; they are never guessed here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    AGES,
    GENDERS,
    ISSUES,
    LABEL_TO_ID,
    LABEL_VOCAB,
    SPEAKERS,
    TURN_FIELDS,
    Turn,
    build_classifier_input,
)


def _id_key(identifier: str) -> tuple[int, int | str]:
    return (0, int(identifier)) if identifier.isdecimal() else (1, identifier)


def group_conversations(turns: Iterable[Turn]) -> dict[str, list[Turn]]:
    """Group and order by original identifiers without renumbering turns."""
    grouped: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        grouped[turn["conversation_id"]].append(turn)
    return {
        cid: sorted(grouped[cid], key=lambda turn: turn["turn_id"])
        for cid in sorted(grouped, key=_id_key)
    }


def conversation_profile(turns: Sequence[Turn]) -> dict[str, str | None]:
    """Fill dialogue attributes from known cells, rejecting conflicting values."""
    profile: dict[str, str | None] = {}
    for field in ("issue", "gender", "age", "persona", "approach"):
        values = {turn.get(field) for turn in turns if turn.get(field) is not None}
        if len(values) > 1:
            cid = turns[0]["conversation_id"] if turns else "<empty>"
            raise ValueError(f"Conversation {cid}: conflicting {field}: {sorted(values)}")
        profile[field] = next(iter(values), None)
    gender, age = profile["gender"], profile["age"]
    profile["gender_age"] = f"{gender}_{age}" if gender and age else None
    return profile


def validate_turns(turns: Sequence[Turn]) -> None:
    """Reject invalid labels, duplicate IDs and inconsistent dialogue metadata."""
    seen: set[tuple[str, int]] = set()
    vocab = {
        "speaker": SPEAKERS,
        "issue": ISSUES,
        "gender": GENDERS,
        "age": AGES,
        **{key: labels for key, labels in LABEL_VOCAB.items() if key != "gender_age"},
    }
    for turn in turns:
        missing = set(TURN_FIELDS) - turn.keys()
        if missing:
            raise ValueError(f"Missing turn columns: {sorted(missing)}")
        cid, tid = turn["conversation_id"], turn["turn_id"]
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("conversation_id must be a nonempty string")
        if not isinstance(tid, int) or isinstance(tid, bool) or tid < 1:
            raise ValueError(f"Conversation {cid}: turn_id must be a positive integer")
        if (cid, tid) in seen:
            raise ValueError(f"Duplicate turn ID: {cid}/{tid}")
        seen.add((cid, tid))
        if not isinstance(turn["utterance"], str):
            raise ValueError(f"{cid}/{tid}: utterance must be a string")
        for field, labels in vocab.items():
            value = turn[field]
            if value is None and field != "speaker":
                continue
            if value not in labels:
                raise ValueError(f"{cid}/{tid}: invalid {field} {value!r}")
    for dialogue in group_conversations(turns).values():
        conversation_profile(dialogue)


def load_turns(path: str | Path) -> list[Turn]:
    """Load canonical CSV or JSONL and validate its records.

    A directory resolves to its ``psycon.csv``. JSONL uses explicit JSON nulls;
    CSV uses empty cells for missing labels. Text is retained verbatim.
    """
    path = Path(path)
    if path.is_dir():
        path /= "psycon.csv"
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(TURN_FIELDS) - set(reader.fieldnames):
                raise ValueError(f"{path}: expected canonical columns {TURN_FIELDS}")
            records = list(reader)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("Turn data must be a .csv or .jsonl file")
    turns: list[Turn] = []
    for line, row in enumerate(records, start=2 if path.suffix == ".csv" else 1):
        try:
            turn = {field: row[field] for field in TURN_FIELDS}
            if turn["conversation_id"] is None:
                raise ValueError("conversation_id cannot be null")
            turn["conversation_id"] = str(turn["conversation_id"])
            if isinstance(turn["turn_id"], (bool, float)):
                raise ValueError("turn_id must be an integer")
            turn["turn_id"] = int(turn["turn_id"])
            for field in TURN_FIELDS[4:]:
                if turn[field] == "":
                    turn[field] = None
            turns.append(turn)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: malformed turn: {exc}") from exc
    validate_turns(turns)
    return [turn for dialogue in group_conversations(turns).values() for turn in dialogue]


def conversation_fingerprint(turns: Sequence[Turn]) -> str:
    """Identify identical transcripts despite whitespace, case or identifier changes."""
    transcript = [
        (turn["speaker"], " ".join(turn["utterance"].casefold().split()))
        for turn in sorted(turns, key=lambda turn: turn["turn_id"])
    ]
    return hashlib.sha256(json.dumps(transcript, ensure_ascii=False).encode()).hexdigest()


def make_splits(
    turns: Sequence[Turn],
    seed: int = 10,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    stratify_by: str | None = None,
) -> dict[str, list[str]]:
    """Deterministic conversation splits that keep duplicate transcripts together.

    Largest-remainder rounding determines dialogue targets. Duplicate groups are
    indivisible, so actual sizes can differ from targets for small datasets.
    Hash ordering makes the result independent of input order and Python's RNG.
    With ``stratify_by="approach"``, sufficiently represented known classes get
    at least one dialogue in every nonzero split. Validation and test allocations
    are rounded independently, with global totals reconciled in the largest class.
    """
    if len(ratios) != 3 or any(not math.isfinite(r) or r < 0 for r in ratios):
        raise ValueError("ratios must contain three finite nonnegative values")
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    if stratify_by not in (None, "issue", "gender", "age", "gender_age", "persona", "approach"):
        raise ValueError(f"Unknown dialogue stratification field: {stratify_by!r}")
    grouped = group_conversations(turns)
    duplicates: dict[str, list[str]] = defaultdict(list)
    for cid, dialogue in grouped.items():
        duplicates[conversation_fingerprint(dialogue)].append(cid)
    names = ("train", "validation", "test")
    exact = [len(grouped) * ratio for ratio in ratios]
    targets = [math.floor(value) for value in exact]
    remainder_order = sorted(range(3), key=lambda i: (-(exact[i] - targets[i]), i))
    for i in remainder_order[: len(grouped) - sum(targets)]:
        targets[i] += 1
    ordered = sorted(
        duplicates.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
        ),
    )
    result: dict[str, list[str]] = {name: [] for name in names}
    if stratify_by is None:
        strata = {None: ordered}
        stratum_targets = {None: targets}
    else:
        strata = defaultdict(list)
        for fingerprint, ids in ordered:
            labels = {conversation_profile(grouped[cid])[stratify_by] for cid in ids}
            if len(labels) != 1:
                raise ValueError(f"Duplicate transcript has conflicting {stratify_by}: {ids}")
            strata[next(iter(labels))].append((fingerprint, ids))
        sizes = {label: sum(len(ids) for _, ids in groups) for label, groups in strata.items()}
        minima = {}
        stratum_targets = {}
        active_splits = sum(ratio > 0 for ratio in ratios)
        for label, size in sizes.items():
            eligible = label is not None and len(strata[label]) >= active_splits
            minimum = [int(eligible and ratio > 0) for ratio in ratios]
            budgets = [0, *(math.floor(size * ratio + 0.5) for ratio in ratios[1:])]
            budgets[0] = size - sum(budgets)
            for i in range(3):
                while budgets[i] < minimum[i]:
                    donor = max(range(3), key=lambda j: (budgets[j] - minimum[j], -j))
                    if budgets[donor] <= minimum[donor]:
                        raise ValueError("Too few conversations to preserve stratified coverage")
                    budgets[donor] -= 1
                    budgets[i] += 1
            minima[label] = minimum
            stratum_targets[label] = budgets
        totals = [sum(budget[i] for budget in stratum_targets.values()) for i in range(3)]
        while totals != targets:
            source = max(range(3), key=lambda i: (totals[i] - targets[i], -i))
            destination = max(range(3), key=lambda i: (targets[i] - totals[i], -i))
            candidates = [
                label
                for label, budget in stratum_targets.items()
                if budget[source] > minima[label][source]
            ]
            if not candidates:
                raise ValueError("Split ratios cannot accommodate every stratification class")
            label = max(candidates, key=lambda item: (sizes[item], item or ""))
            stratum_targets[label][source] -= 1
            stratum_targets[label][destination] += 1
            totals[source] -= 1
            totals[destination] += 1
    for label, groups in strata.items():
        assigned = [0, 0, 0]
        for _, ids in groups:
            destination = max(range(3), key=lambda i: (stratum_targets[label][i] - assigned[i], -i))
            result[names[destination]].extend(ids)
            assigned[destination] += len(ids)
    result = {name: sorted(ids, key=_id_key) for name, ids in result.items()}
    validate_splits(turns, result)
    return result


def validate_splits(turns: Sequence[Turn], splits: Mapping[str, Sequence[str]]) -> None:
    """Require complete, disjoint ID coverage and no duplicate-transcript leakage."""
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Splits must contain train, validation and test")
    grouped = group_conversations(turns)
    owner: dict[str, str] = {}
    transcript_owner: dict[str, str] = {}
    for split, ids in splits.items():
        for cid in ids:
            if cid not in grouped:
                raise ValueError(f"Unknown conversation in {split}: {cid}")
            if cid in owner:
                raise ValueError(f"Conversation appears more than once in splits: {cid}")
            owner[cid] = split
            fingerprint = conversation_fingerprint(grouped[cid])
            if fingerprint in transcript_owner and transcript_owner[fingerprint] != split:
                raise ValueError(f"Duplicate transcript crosses splits: {cid}")
            transcript_owner[fingerprint] = split
    if set(owner) != set(grouped):
        raise ValueError(
            f"Unassigned conversations: {sorted(set(grouped) - set(owner), key=_id_key)}"
        )


def select_split(
    turns: Sequence[Turn], splits: Mapping[str, Sequence[str]], split: str
) -> list[Turn]:
    """Select whole dialogues from a validated split assignment."""
    validate_splits(turns, splits)
    if split not in splits:
        raise ValueError(f"Unknown split: {split!r}")
    ids = set(splits[split])
    return [turn for turn in turns if turn["conversation_id"] in ids]


def format_prompt(profile: Mapping[str, str | None], context: Sequence[Mapping[str, Any]]) -> str:
    """Format past utterances and available user profile; no target labels enter."""
    attributes = [
        f"{field}={profile[field]}"
        for field in ("gender", "age", "persona")
        if profile.get(field) is not None
    ]
    lines = ["Patient profile: " + ", ".join(attributes)] if attributes else []
    for turn in context:
        role = turn["speaker"].capitalize()
        sentiment = turn.get("sentiment") if turn["speaker"] == "patient" else None
        annotation = f" (sentiment={sentiment})" if sentiment is not None else ""
        lines.append(f"{role}{annotation}: {turn['utterance']}")
    lines.append("Therapist:")
    return "\n".join(lines)


def build_examples(turns: Sequence[Turn], context_window: int = 4) -> list[dict[str, Any]]:
    """Create therapist response examples from strictly preceding turns.

    Blank turns remain in the canonical dataset but are omitted from examples and
    context. Initial therapist greetings without a preceding patient are skipped.
    Gaps in source turn IDs reset context and patient conditioning, because the
    missing intervening text cannot be assumed to continue the same exchange.
    Dialogue profile information is supplied explicitly by the task; future
    utterances, response labels and approach labels never enter the prompt.
    """
    if (
        not isinstance(context_window, int)
        or isinstance(context_window, bool)
        or context_window < 1
    ):
        raise ValueError("context_window must be a positive integer")
    examples: list[dict[str, Any]] = []
    for dialogue in group_conversations(turns).values():
        profile = conversation_profile(dialogue)
        past: list[Turn] = []
        user: Turn | None = None
        previous_turn_id: int | None = None
        segment_id = dialogue[0]["turn_id"]
        for turn in dialogue:
            if previous_turn_id is not None and turn["turn_id"] != previous_turn_id + 1:
                past = []
                user = None
                segment_id = turn["turn_id"]
            previous_turn_id = turn["turn_id"]
            if not turn["utterance"].strip():
                continue
            if turn["speaker"] == "patient":
                user = turn
            elif user is not None:
                context = [
                    {
                        field: previous[field]
                        for field in ("turn_id", "speaker", "utterance", "sentiment")
                    }
                    for previous in past[-context_window:]
                ]
                # A series of therapist turns can displace the latest user from
                # the window. Keep its text as an explicit input in that case.
                if not any(item["turn_id"] == user["turn_id"] for item in context):
                    trailing = context[-(context_window - 1) :] if context_window > 1 else []
                    context = [
                        {
                            field: user[field]
                            for field in ("turn_id", "speaker", "utterance", "sentiment")
                        }
                    ] + trailing
                labels = {
                    "gender_age": profile["gender_age"],
                    "persona": profile["persona"],
                    "sentiment": user["sentiment"],
                    "approach": profile["approach"],
                    "politeness": turn["politeness"],
                    "ipc": turn["ipc"],
                }
                examples.append(
                    {
                        "conversation_id": turn["conversation_id"],
                        "turn_id": turn["turn_id"],
                        "segment_id": segment_id,
                        "speaker": "therapist",
                        "utterance": turn["utterance"],
                        "response": turn["utterance"],
                        "context": context,
                        "user_utterance": user["utterance"],
                        "user_turn_id": user["turn_id"],
                        "user_sentiment": user["sentiment"],
                        "profile": dict(profile),
                        "labels": labels,
                        "prompt": format_prompt(profile, context),
                    }
                )
            past.append(turn)
    return examples


def build_classifier_examples(turns: Sequence[Turn], task: str) -> list[dict[str, Any]]:
    """Return supervised records for one of the six classifiers.

    Sentiment uses only patient turns. Other classifiers use therapist turns.
    Missing task labels and missing required conditioning inputs are excluded.
    Gender/age and persona classifier inputs contain response text only.
    Patient conditioning never crosses a gap in source turn IDs.
    """
    if task not in LABEL_VOCAB:
        raise ValueError(f"Unknown classifier task: {task!r}")
    examples: list[dict[str, Any]] = []
    for dialogue in group_conversations(turns).values():
        profile = conversation_profile(dialogue)
        user: Turn | None = None
        previous_turn_id: int | None = None
        for turn in dialogue:
            if previous_turn_id is not None and turn["turn_id"] != previous_turn_id + 1:
                user = None
            previous_turn_id = turn["turn_id"]
            if not turn["utterance"].strip():
                continue
            if turn["speaker"] == "patient":
                user = turn
                if task != "sentiment":
                    continue
            elif task == "sentiment":
                continue
            label = profile[task] if task in ("gender_age", "persona", "approach") else turn[task]
            if label is None:
                continue
            if task == "approach" and user is None:
                continue
            if task in ("politeness", "ipc") and (user is None or user["sentiment"] is None):
                continue
            text = build_classifier_input(
                task,
                turn["utterance"],
                user["utterance"] if user else "",
                user["sentiment"] if user else None,
            )
            examples.append(
                {
                    "conversation_id": turn["conversation_id"],
                    "turn_id": turn["turn_id"],
                    "speaker": turn["speaker"],
                    "text": text,
                    "label": label,
                    "label_id": LABEL_TO_ID[task][label],
                    "task": task,
                }
            )
    return examples


classifier_examples = build_classifier_examples


def load_examples(
    path: str | Path,
    split: str = "train",
    context_window: int = 4,
) -> list[dict[str, Any]]:
    """Load response examples for one split using the adjacent ``splits.json``."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    turns = load_turns(path)
    splits = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    return build_examples(select_split(turns, splits, split), context_window)


def dataset_stats(
    turns: Sequence[Turn],
    splits: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Compute public counts from records, with label units made explicit."""
    grouped = group_conversations(turns)
    profiles = [conversation_profile(dialogue) for dialogue in grouped.values()]
    fingerprints = Counter(conversation_fingerprint(dialogue) for dialogue in grouped.values())
    stats: dict[str, Any] = {
        "conversations": len(grouped),
        "turns": len(turns),
        "nonempty_turns": sum(bool(turn["utterance"].strip()) for turn in turns),
        "empty_turns": sum(not turn["utterance"].strip() for turn in turns),
        "speakers": dict(sorted(Counter(turn["speaker"] for turn in turns).items())),
        "duplicate_transcript_groups": sum(count > 1 for count in fingerprints.values()),
        "dialogue_labels": {},
        "turn_labels": {},
        "missing_labels": {},
        "generation_examples": len(build_examples(turns)),
        "classifier_examples": {
            task: len(build_classifier_examples(turns, task)) for task in LABEL_VOCAB
        },
    }
    for field in ("issue", "gender", "age", "gender_age", "persona", "approach"):
        stats["dialogue_labels"][field] = dict(
            sorted(
                Counter(
                    profile[field] for profile in profiles if profile[field] is not None
                ).items()
            )
        )
        stats["missing_labels"][field + "_conversations"] = sum(
            profile[field] is None for profile in profiles
        )
    for field, speaker in (
        ("sentiment", "patient"),
        ("politeness", "therapist"),
        ("ipc", "therapist"),
    ):
        eligible = [
            turn for turn in turns if turn["speaker"] == speaker and turn["utterance"].strip()
        ]
        stats["turn_labels"][field] = dict(
            sorted(Counter(turn[field] for turn in eligible if turn[field] is not None).items())
        )
        stats["missing_labels"][field + "_nonempty_" + speaker + "_turns"] = sum(
            turn[field] is None for turn in eligible
        )
    if splits is not None:
        validate_splits(turns, splits)
        stats["splits"] = {}
        for name, ids in splits.items():
            subset = [turn for cid in ids for turn in grouped[cid]]
            stats["splits"][name] = {
                "conversations": len(ids),
                "turns": len(subset),
                "nonempty_turns": sum(bool(turn["utterance"].strip()) for turn in subset),
                "generation_examples": len(build_examples(subset)),
            }
    return stats
