"""Data integrity and feature/target separation checks."""

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from e_therapist.data import (
    build_classifier_examples,
    build_examples,
    conversation_profile,
    dataset_stats,
    load_examples,
    load_turns,
    make_splits,
    select_split,
    validate_splits,
    validate_turns,
)
from e_therapist.schema import LABEL_VOCAB, TURN_FIELDS, build_classifier_input


def turn(cid, tid, speaker, text, **labels):
    result = dict.fromkeys(TURN_FIELDS)
    result.update(conversation_id=str(cid), turn_id=tid, speaker=speaker, utterance=text)
    result.update(labels)
    return result


@pytest.fixture
def dialogue():
    return [
        turn(1, 1, "therapist", "Welcome, opening only.", politeness="polite", ipc="helpful"),
        turn(
            1,
            2,
            "patient",
            "I feel sad.",
            gender="female",
            age="adult",
            persona="openness",
            issue="depression",
            sentiment="negative",
            approach="directive",
        ),
        turn(
            1,
            3,
            "therapist",
            "Try this unique response.",
            politeness="moderately_polite",
            ipc="directing",
        ),
        turn(1, 4, "patient", "This future text must not leak.", sentiment="positive"),
        turn(1, 5, "therapist", "Another response.", politeness="polite", ipc="understanding"),
    ]


def test_generation_has_only_past_context_and_user_metadata(dialogue):
    examples = build_examples(dialogue, context_window=2)
    assert [item["turn_id"] for item in examples] == [3, 5]
    example = examples[0]
    assert example["profile"]["gender_age"] == "female_adult"
    assert example["profile"]["persona"] == "openness"
    assert example["user_sentiment"] == "negative"
    assert example["labels"]["ipc"] == "directing"
    assert example["response"] == "Try this unique response."
    for forbidden in (
        example["response"],
        "future text",
        "moderately_polite",
        "directing",
        "directive",
    ):
        assert forbidden not in example["prompt"]
    assert "sentiment=negative" in example["prompt"]
    assert all(item["turn_id"] < example["turn_id"] for item in example["context"])


def test_missing_labels_are_not_invented(dialogue):
    dialogue[1]["approach"] = None
    dialogue[2]["ipc"] = None
    assert build_examples(dialogue)[0]["labels"]["approach"] is None
    assert build_examples(dialogue)[0]["labels"]["ipc"] is None
    assert build_classifier_examples(dialogue, "approach") == []
    assert [row["turn_id"] for row in build_classifier_examples(dialogue, "ipc")] == [5]
    stats = dataset_stats(dialogue)
    assert stats["missing_labels"]["approach_conversations"] == 1
    assert stats["missing_labels"]["ipc_nonempty_therapist_turns"] == 1


def test_classifier_roles_and_inputs(dialogue):
    sentiment = build_classifier_examples(dialogue, "sentiment")
    assert [row["turn_id"] for row in sentiment] == [2, 4]
    assert sentiment[0]["text"] == "I feel sad."
    assert "negative" not in sentiment[0]["text"]
    for task in ("gender_age", "persona"):
        examples = build_classifier_examples(dialogue, task)
        assert [row["turn_id"] for row in examples] == [1, 3, 5]
        assert examples[1]["text"] == dialogue[2]["utterance"]
        assert examples[1]["label"] not in examples[1]["text"]
    for task in ("politeness", "ipc"):
        examples = build_classifier_examples(dialogue, task)
        assert [row["turn_id"] for row in examples] == [3, 5]
        assert "Patient sentiment: negative" in examples[0]["text"]
        assert examples[0]["label"] not in examples[0]["text"]
    approach = build_classifier_examples(dialogue, "approach")
    assert "Patient: I feel sad." in approach[0]["text"]
    assert approach[0]["label"] not in approach[0]["text"]


def test_missing_sentiment_skips_conditional_classifiers(dialogue):
    dialogue[1]["sentiment"] = None
    for task in ("politeness", "ipc"):
        assert [row["turn_id"] for row in build_classifier_examples(dialogue, task)] == [5]
    assert len(build_examples(dialogue)) == 2
    assert "sentiment=None" not in build_examples(dialogue)[0]["prompt"]
    with pytest.raises(ValueError, match="patient sentiment"):
        build_classifier_input("ipc", "A response", "User", None)


def test_consecutive_therapist_turns_keep_patient_and_bounded_context(dialogue):
    dialogue[3]["speaker"] = "therapist"
    examples = build_examples(dialogue, context_window=1)
    assert examples[-1]["user_turn_id"] == 2
    assert len(examples[-1]["context"]) == 1
    assert examples[-1]["context"][0]["speaker"] == "patient"


def test_blank_turn_is_retained_but_never_a_training_target(dialogue, tmp_path):
    dialogue.insert(3, turn(1, 6, "therapist", "  ", politeness=None, ipc=None))
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in dialogue))
    loaded = load_turns(path)
    assert len(loaded) == 6
    assert loaded[-1]["utterance"] == "  "
    assert len(build_examples(loaded)) == 2
    assert dataset_stats(loaded)["empty_turns"] == 1
    for task in LABEL_VOCAB:
        assert all(example["turn_id"] != 6 for example in build_classifier_examples(loaded, task))


def test_split_disjoint_complete_and_duplicate_safe():
    turns = [turn(cid, 1, "patient", f"Unique dialogue {cid}") for cid in range(1, 21)]
    turns.append(turn(21, 1, "patient", "  UNIQUE dialogue 1 "))
    splits = make_splits(turns, seed=10)
    assert splits == make_splits(list(reversed(turns)), seed=10)
    assert sorted(len(ids) for ids in splits.values()) == [2, 2, 17]
    assigned = [cid for ids in splits.values() for cid in ids]
    assert len(assigned) == len(set(assigned)) == 21
    assert next(name for name, ids in splits.items() if "1" in ids) == next(
        name for name, ids in splits.items() if "21" in ids
    )
    assert all(
        row["conversation_id"] in splits["test"] for row in select_split(turns, splits, "test")
    )
    leaking = {"train": [str(cid) for cid in range(1, 21)], "validation": ["21"], "test": []}
    with pytest.raises(ValueError, match="Duplicate transcript"):
        validate_splits(turns, leaking)


def test_invalid_records_raise(dialogue, tmp_path):
    with pytest.raises(ValueError, match="Duplicate turn"):
        validate_turns(dialogue + [dialogue[0]])
    dialogue[2]["gender"] = "male"
    with pytest.raises(ValueError, match="conflicting gender"):
        conversation_profile(dialogue)
    dialogue[2]["gender"] = None
    dialogue[2]["ipc"] = "invented"
    with pytest.raises(ValueError, match="invalid ipc"):
        validate_turns(dialogue)
    with pytest.raises(ValueError, match="positive integer"):
        build_examples(dialogue, context_window=0)
    with pytest.raises(ValueError, match="sum to one"):
        make_splits(dialogue, ratios=(0.5, 0.1, 0.1))


def test_missing_turns_break_response_context_and_classifier_conditioning(dialogue):
    dialogue[2]["turn_id"] = 4
    dialogue[3]["turn_id"] = 5
    dialogue[4]["turn_id"] = 6
    examples = build_examples(dialogue)
    assert [example["turn_id"] for example in examples] == [6]
    assert examples[0]["user_turn_id"] == 5
    assert examples[0]["segment_id"] == 4
    assert all(item["turn_id"] >= 4 for item in examples[0]["context"])
    assert "I feel sad." not in examples[0]["prompt"]
    for task in ("politeness", "ipc", "approach"):
        assert [row["turn_id"] for row in build_classifier_examples(dialogue, task)] == [6]
    # Sentiment uses patient text itself and requires no preceding turn.
    assert [row["turn_id"] for row in build_classifier_examples(dialogue, "sentiment")] == [2, 5]


def test_approach_stratification_preserves_rare_classes_and_exact_sizes():
    from collections import Counter

    turns = []
    cid = 0
    for approach, count in (("directive", 321), ("eclectic", 16), ("non_directive", 5), (None, 1)):
        for _ in range(count):
            cid += 1
            turns.append(turn(cid, 1, "patient", f"Distinct transcript {cid}", approach=approach))
    turns[1]["utterance"] = turns[0]["utterance"]
    splits = make_splits(turns, seed=10, stratify_by="approach")
    assert splits == make_splits(list(reversed(turns)), seed=10, stratify_by="approach")
    assert [len(ids) for ids in splits.values()] == [275, 34, 34]
    expected = {
        "train": {"directive": 259, "eclectic": 12, "non_directive": 3, None: 1},
        "validation": {"directive": 31, "eclectic": 2, "non_directive": 1},
        "test": {"directive": 31, "eclectic": 2, "non_directive": 1},
    }
    for name in splits:
        assert (
            Counter(row["approach"] for row in select_split(turns, splits, name)) == expected[name]
        )


def test_csv_nulls_and_split_example_loader(dialogue, tmp_path):
    with (tmp_path / "psycon.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_FIELDS)
        writer.writeheader()
        writer.writerows(dialogue)
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["1"], "validation": [], "test": []})
    )
    turns = load_turns(tmp_path)
    assert turns[0]["sentiment"] is None
    assert len(load_examples(tmp_path)) == 2


def test_shipped_dataset_and_splits_are_valid():
    directory = Path(__file__).resolve().parents[1] / "data" / "psycon"
    turns = load_turns(directory)
    splits = json.loads((directory / "splits.json").read_text())
    validate_splits(turns, splits)
    stats = dataset_stats(turns, splits)
    assert stats["conversations"] == 1020
    assert stats["turns"] == 19676
    assert stats["empty_turns"] == 2
    assert stats["duplicate_transcript_groups"] == 9
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [816, 102, 102]
    assert Counter(cid.partition("-")[0] for ids in splits.values() for cid in ids) == {
        "psycon": 343,
        "combined": 677,
    }
    assert {cid for ids in splits.values() for cid in ids if cid.startswith("psycon-")} == {
        f"psycon-{index:03d}" for index in range(1, 344)
    }
    for field in (
        "sentiment_nonempty_patient_turns",
        "politeness_nonempty_therapist_turns",
        "ipc_nonempty_therapist_turns",
    ):
        assert stats["missing_labels"][field] == 0
    assert sum(stats["speakers"].values()) == stats["turns"]
    assert stats == json.loads((directory / "stats.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, metadata in manifest["files"].items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == metadata["sha256"]
    assert manifest["stratification_column"] == "approach"
    assert manifest["shared_informative_openings_share_split"] is True
    for name in splits:
        assert {
            row["label"]
            for row in build_classifier_examples(select_split(turns, splits, name), "approach")
        } == set(LABEL_VOCAB["approach"])


def test_shipped_shared_informative_openings_never_cross_splits():
    directory = Path(__file__).resolve().parents[1] / "data" / "psycon"
    turns = load_turns(directory)
    splits = json.loads((directory / "splits.json").read_text())
    owner = {cid: name for name, ids in splits.items() for cid in ids}
    first_patient = {}
    for turn in turns:
        if turn["speaker"] == "patient" and turn["utterance"].strip():
            first_patient.setdefault(turn["conversation_id"], turn["utterance"])
    openings = defaultdict(list)
    for cid, text in first_patient.items():
        if len(text.split()) >= 25:
            normalized = " ".join(
                re.sub(r"[^\w\s]", "", unicodedata.normalize("NFKC", text).casefold()).split()
            )
            openings[normalized].append(cid)
    for ids in openings.values():
        assert len({owner[cid] for cid in ids}) == 1
    for left, right in (("674", "675"), ("699", "700"), ("732", "733")):
        assert owner[f"combined-{left}"] == owner[f"combined-{right}"]
