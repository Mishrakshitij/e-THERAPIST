# Implementation guide

This repository provides a new PyTorch implementation of [e-THERAPIST](https://aclanthology.org/2023.emnlp-main.861.pdf). It includes the supervised generator, six classifiers, reward model, and NLPO training. Original trained weights and full-paper experiment outputs are not included.

## Data and supervised objectives

`data.py` groups turns by conversation, fills the available dialogue profile, and builds prompts from at most four preceding nonempty turns. Prompts include gender, age, and persona when available, together with patient sentiment. The current response, its politeness/IPC label, and the approach target never enter the prompt. Generation examples require a preceding patient utterance. A gap in source turn IDs resets context and patient conditioning; generated response history also resets at that boundary. Supervised conditioning uses the stored patient sentiment; the reward model predicts sentiment with its frozen classifier. Inference accepts supplied sentiment or classifier predictions.

`tokenization.py` tokenizes prompt and response separately, preserves the available profile header while truncating older context, retains response tokens plus EOS, and labels prompt/padding positions with `-100`. Supervised training, NLPO, and generation share this prompt truncation rule. `causal_lm_loss` applies the causal shift once. Real EOS targets remain supervised when the padding token is also EOS. Gradient accumulation weights microbatches by their target-token counts.

Each classifier has an independent RoBERTa backbone and cross-entropy objective. The label order and exact input format are shared between training and scoring in `schema.py`:

| Task | Input | Target |
| --- | --- | --- |
| Gender–age | Therapist response | Six gender–age classes |
| Persona | Therapist response | Five persona classes |
| Sentiment | Patient utterance | Three sentiment classes |
| Approach | Therapist response and preceding patient utterance | Three approach classes |
| Politeness | Therapist response and patient sentiment | Three politeness classes |
| IPC | Therapist response and patient sentiment | Nine IPC classes |

Missing targets are excluded per task. All splits are validated before training; both conversation IDs and identical complete transcripts must be disjoint. The supplied splits are stratified by dialogue approach so each split covers all three approach classes. Validation selects the supervised checkpoint; test data are used only by evaluation commands.

## Rewards and equations

`rewards.py` implements equations 2–9. Classifier inputs are probabilities of the same target class for the reference and candidate responses. The default `paper` convention retains the printed equations, including their signs:

```text
R1..R5 = p(target | reference) - alpha * p(target | candidate)
R6     = min(BSF1(context, candidate) + BSF1(user, candidate), 1) / 2
R7     = 1 / PPL(candidate) + BSF1(candidate, previous_generated_response)
RA     = sum(beta[j] * Rj), j = 1..5
RQ     = gamma[0] * R6 + gamma[1] * R7
R      = (delta[0] * RA + delta[1] * RQ) / 7
```

The coefficient defaults are beta `(0.1, 0.2, 0.2, 0.2, 0.3)`, gamma `(0.5, 0.5)`, and delta `(0.75, 0.25)`. Note that R6 is capped at **0.5**, because the cap precedes division.

Maximizing the printed attribute reward decreases candidate confidence; maximizing its R7 similarity term encourages repetition. For experiments aligned with the prose objectives, `reward_convention: sign_corrected` negates R1–R5 and replaces the R7 similarity with `1 - similarity`. This is an explicit alternative, not a claim about the original implementation.

`scoring.py` computes these signals with frozen trained classifiers, BERTScore, and the frozen supervised generator. R7 perplexity is unconditional likelihood of the generated response plus EOS, preceded by EOS. The first response has previous-response similarity zero. BERTScore uses the tuned layer count for the selected model; custom checkpoint paths require `bertscore_num_layers` and a tokenizer with a valid maximum length.

The combined dataset has incomplete profile information. The supplied configuration uses `missing_attribute_policy: mask`: absent targets contribute no attribute reward, and the beta weights of observed targets are renormalized per example. A fully labeled example retains the original formula exactly. No profile is imputed. Use `missing_attribute_policy: skip` to restrict training to examples with every active target, matching the complete-label objective. With `mix_weights: [0, 1]`, training uses only response-quality rewards and does not require classifier checkpoints.

## NLPO

`rl.py` maintains three independent models: the trainable policy/value model, a frozen supervised reference, and a frozen delayed masking policy. It uses cached autoregressive generation. The delayed model supplies a nucleus mask; top-k selects within that mask. Rollouts store the allowed token IDs, sampled actions, behavior log probabilities, and value estimates. Replay normalizes over exactly the stored support, keeping the probability ratio well defined throughout all optimization epochs.

Sequence rewards are assigned to the final sampled action. Per-token regularization uses sampled log-probability differences against the full-vocabulary reference. The adaptive KL controller uses exact state-wise KL over the allowed policy support, including the reference mass outside that support. GAE distinguishes EOS termination from a length truncation, which bootstraps the final value estimate. Padding has no loss or advantage contribution.

Dialogues are shuffled, while their response turns are visited chronologically. Three candidates are scored per turn; all enter the optimization buffer. The highest-scoring nonempty candidate supplies generated history for the next turn. Patient continuations remain the recorded dataset utterances. Empty candidates receive zero sequence reward and still participate in policy optimization. This continuation rule supplies an explicit previous generated response for R7.

The actor minimizes negative PPO-Clip; the critic uses optional clipped squared error; entropy is configurable. Dropout remains disabled during rollout and replay, including replay with gradients. The delayed mask refreshes only after a completed update, and the reference never changes. Saved NLPO checkpoints contain both the generator and value head. Checkpoints can initialize another run; optimizer/buffer state is not resumed.

## Configuration conventions

Reported settings used here include seed 10, learning rate `2e-5`, batch size 8, three candidates, context window four, top-k 20, maximum candidate length 50, discount 0.95, and policy clipping 0.2. The run configuration interprets 32,000 steps as generated candidate trajectories and 640 steps per update as trajectories per rollout buffer. Each buffer receives 20 optimization epochs.

The paper does not fix every operational detail. The following are configurable implementation choices: alpha 1; top-p 0.9; one mask refresh per update; GAE lambda 0.95; initial KL coefficient 0.1, target 6, horizon 10,000; value coefficient 0.5; value clipping 0.2; maximum encoded length 512; AdamW weight decay 0.01; gradient norm 1; supervised epochs 20; generator gradient accumulation four. Set `value_clip_range: null` to use unclipped value error.

## Evaluation and verification

Generator evaluation measures token-weighted, response-only perplexity on reference responses, unrescaled BERTScore against those references, whitespace-token response length, corpus distinct-1/2, and classifier-based attribute correctness. This specifies the evaluation choices rather than assuming unspecified paper details. Empty generated responses count as incorrect attribute predictions. Missing reference targets are excluded with the evaluated count reported.

Macro-F1 includes all canonical classes, even classes absent from a split. Support-weighted accuracy equals ordinary accuracy for these single-label tasks. Per-class support is included so rare-class coverage is visible.

`python -m pytest -q` covers numerical losses and gradients, terminal/truncation handling, masks, classifier inputs, data leakage, generated-history handling, model serialization, and frozen-policy invariants. `python -m e_therapist smoke` performs a small offline training/evaluation run with toy data. These checks establish executable behavior, not reproduction of the paper's reported quality scores.
