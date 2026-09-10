# Published results

These values are reported in Table 2 of the [EMNLP 2023 paper](https://aclanthology.org/2023.emnlp-main.861.pdf). They are reference results from the paper's experiments, not measurements produced by the checkpoints or smoke tests in this repository.

| Model | Gender–age | Persona | Approach | Politeness | IPC | PPL | BERTScore-F1 | Response length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLLM | 85.4% | 80.1% | 86.3% | 84.6% | 77.8% | 3.26 | 0.81 | 19.79 |
| SLLM+PPO | 89.0% | 83.9% | 91.5% | 91.3% | 82.3% | 2.67 | 0.89 | 23.01 |
| e-THERAPIST | 90.1% | 84.1% | 92.6% | 92.5% | 83.4% | 2.52 | 0.89 | 23.89 |

Train and evaluate your own checkpoints using the README commands. The available data release and split membership differ from the full experimental dataset; reproducing the table also requires the complete experimental data and original evaluation setup.
