# PSYCON

PSYCON contains patient–therapist conversations with patient profiles, psychological issues, patient sentiment, therapist politeness, interpersonal behavior, and psychotherapeutic approach labels.

This release contains 1,020 conversations and 19,676 turns, with an 816/102/102 dialogue split. The [paper](https://aclanthology.org/2023.emnlp-main.861/) reports 25,071 turns.

## Files

| File | Contents |
| --- | --- |
| [`psycon/psycon.csv`](psycon/psycon.csv) | Canonical turn records |
| [`psycon/splits.json`](psycon/splits.json) | Conversation IDs for each split |
| [`psycon/stats.json`](psycon/stats.json) | Counts, distributions, and training example totals |
| [`psycon/manifest.json`](psycon/manifest.json) | Schema, label vocabularies, file sizes, and SHA-256 checksums |

## Splits

Splits are deterministic with seed 10 and an 80/10/10 ratio, stratified by psychotherapeutic approach. All three approach classes occur in every split. Each conversation belongs to one split; identical complete transcripts and shared patient opening prompts of at least 25 words also remain in the same split.

| Split | Conversations | Turns | Nonempty turns | Response examples |
| --- | ---: | ---: | ---: | ---: |
| Train | 816 | 15,509 | 15,507 | 7,515 |
| Validation | 102 | 2,093 | 2,093 | 1,014 |
| Test | 102 | 2,074 | 2,074 | 1,000 |
| **Total** | **1,020** | **19,676** | **19,674** | **9,529** |

There are 9,706 patient turns and 9,970 therapist turns. Two therapist turns contain no text. Empty turns retain their IDs and are excluded from training; response examples require a preceding patient utterance within a continuous sequence of turn IDs.

## Schema

| Column | Meaning |
| --- | --- |
| `conversation_id` | Unique source-prefixed dialogue ID: `psycon-NNN` or `combined-NNN`; the suffix preserves the original numeric ID |
| `turn_id` | Original positive integer turn ID within the dialogue; gaps remain explicit |
| `speaker` | `patient` or `therapist` |
| `utterance` | Utterance text |
| `issue` | Expressed psychological issue; empty when unspecified |
| `gender`, `age`, `persona` | Patient profile; shared within the dialogue |
| `sentiment` | Patient sentiment |
| `politeness` | Therapist politeness |
| `ipc` | Therapist interpersonal behavior |
| `approach` | Dialogue psychotherapeutic approach |

Empty label cells represent missing or inapplicable values and load as `None`. Profile cells can be empty on individual turns; training examples use the available dialogue profile. Classifier targets use patient turns for sentiment and therapist turns for the other tasks, skipping missing targets. Available label vocabularies are in the manifest.

## Dialogue distributions

### Psychological issue

| Label | Conversations |
| --- | ---: |
| `anxiety` | 131 |
| `bipolar_disorder` | 62 |
| `depression` | 79 |
| `disruptive_behaviour_and_dissocial_disorders` | 43 |
| `ptsd` | 64 |
| `schizophrenia` | 60 |
| `stress` | 16 |

### Gender and age

| Label | Conversations |
| --- | ---: |
| `female_adult` | 55 |
| `female_elder` | 55 |
| `female_young` | 55 |
| `male_adult` | 61 |
| `male_elder` | 57 |
| `male_young` | 60 |

### Persona

| Label | Conversations |
| --- | ---: |
| `agreeableness` | 68 |
| `conscientiousness` | 70 |
| `extraversion` | 68 |
| `neuroticism` | 69 |
| `openness` | 68 |

### Psychotherapeutic approach

| Label | Conversations |
| --- | ---: |
| `directive` | 863 |
| `eclectic` | 16 |
| `non_directive` | 54 |

### Missing dialogue values

| Label | Conversations |
| --- | ---: |
| `issue` | 565 |
| `gender` | 675 |
| `age` | 675 |
| `gender_age` | 677 |
| `persona` | 677 |
| `approach` | 87 |

## Turn distributions

Counts below use nonempty turns from the relevant speaker.

### Patient sentiment

| Label | Patient turns |
| --- | ---: |
| `negative` | 3,866 |
| `neutral` | 2,428 |
| `positive` | 3,412 |

### Therapist politeness

| Label | Therapist turns |
| --- | ---: |
| `impolite` | 91 |
| `moderately_polite` | 5,059 |
| `polite` | 4,818 |

### Therapist interpersonal behavior

| Label | Therapist turns |
| --- | ---: |
| `compliant` | 1,215 |
| `confrontational` | 35 |
| `directing` | 2,045 |
| `dissatisfied` | 54 |
| `empathetic` | 981 |
| `helpful` | 3,590 |
| `imposing` | 24 |
| `uncertain` | 227 |
| `understanding` | 1,797 |

Among nonempty turns, missing sentiment: 0; missing politeness: 0; missing interpersonal behavior: 0.
