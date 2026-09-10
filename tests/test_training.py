"""Training boundary checks that run without downloading model weights."""

import csv
import json

import pytest

from e_therapist.metrics import classification_metrics, perplexity
from e_therapist.schema import TURN_FIELDS
from e_therapist.tokenization import collate_causal, encode_example, encode_prompt
from e_therapist.training import split_turns


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [int(word) for word in text.split()]


def test_truncation_preserves_response_supervision_and_latest_context():
    encoded = encode_example(Tokenizer(), "1 2 3 4 5 6 7 8", "21 22 23 24 25", 8, 4)
    assert len(encoded["input_ids"]) == 8
    assert encoded["input_ids"][:4] == [5, 6, 7, 8]
    assert encoded["labels"][:4] == [-100] * 4
    assert encoded["labels"][4:] == [21, 22, 23, 99]
    assert encoded["input_ids"][4:] == encoded["labels"][4:]


def test_long_context_keeps_known_profile_and_response_only_supervision():
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    words = [
        "[UNK]",
        "[EOS]",
        "Patient",
        "profile:",
        "gender=female,",
        "age=adult,",
        "persona=openness",
        "old",
        "recent",
        "Therapist:",
        "answer",
    ]
    backend = FastTokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")
    profile = "Patient profile: gender=female, age=adult, persona=openness\n"
    prompt = profile + "old " * 30 + "recent Therapist:"
    ids = encode_prompt(tokenizer, prompt, 8)
    assert tokenizer.decode(ids).startswith(profile.strip())
    assert tokenizer.decode(ids).endswith("recent Therapist:")
    assert len(ids) == 8
    encoded = encode_example(tokenizer, prompt, "answer", max_length=10, max_response_length=2)
    assert encoded["input_ids"][:8] == ids
    assert encoded["labels"] == [-100] * 8 + [10, tokenizer.eos_token_id]
    short = profile + "recent Therapist:"
    assert encode_prompt(tokenizer, short, 20) == tokenizer.encode(short, add_special_tokens=False)
    with pytest.raises(ValueError, match="too small"):
        encode_prompt(tokenizer, prompt, 5)


def test_eos_used_for_padding_does_not_remove_real_eos_supervision():
    long = encode_example(Tokenizer(), "1 2 3 4", "21 22", 10, 4)
    short = encode_example(Tokenizer(), "1", "21", 10, 4)
    batch = collate_causal([long, short], pad_token_id=99)
    real_end = len(short["input_ids"]) - 1
    assert batch["labels"][1, real_end].item() == 99
    assert batch["attention_mask"][1, real_end].item() == 1
    assert batch["attention_mask"][1, real_end + 1 :].sum().item() == 0
    assert (batch["labels"][1, real_end + 1 :] == -100).all()


def test_metrics_keep_rare_absent_classes_in_denominator():
    result = classification_metrics(["common", "common"], ["common", "common"], ["common", "rare"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 0.5
    assert result["weighted_f1"] == 1.0
    assert result["per_class"]["rare"]["support"] == 0
    assert perplexity(0.0, 3) == 1.0
    with pytest.raises(ValueError):
        perplexity(0.0, 0)


def test_training_rejects_cross_split_duplicate_transcripts(tmp_path):
    turns = []
    for cid in ("1", "2"):
        turn = dict.fromkeys(TURN_FIELDS)
        turn.update(conversation_id=cid, turn_id=1, speaker="patient", utterance="Same text")
        turns.append(turn)
    data_path = tmp_path / "psycon.csv"
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_FIELDS)
        writer.writeheader()
        writer.writerows(turns)
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"train": ["1"], "validation": ["2"], "test": []}))
    with pytest.raises(ValueError, match="Duplicate transcript"):
        split_turns({"data": str(data_path), "splits": str(split_path)}, "train")


def test_empty_generated_response_counts_as_an_attribute_failure():
    result = classification_metrics(["polite", "polite"], ["polite", None], ["impolite", "polite"])
    assert result["accuracy"] == 0.5
    assert result["per_class"]["polite"]["f1"] == pytest.approx(2 / 3)
    assert result["count"] == 2
