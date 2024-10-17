# PrOntoQA and PrOntoQA-OOD
This repo contains PrOntoQA-OOD, as described in our papers:
 1. [Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought](https://arxiv.org/pdf/2210.01240.pdf)
 2. [Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples](https://browse.arxiv.org/pdf/2305.15269.pdf)

PrOntoQA and PrOntoQA-OOD generate question-answering examples with chains-of-thought that describe the reasoning required to answer the questions correctly. The sentences in the examples are syntactically simple and amenable to semantic parsing, and so this code can be used to formally analyze the predicted chain-of-thought from large language models.

**Note:** The `v1` branch contains the version of the repo corresponding to the original PrOntoQA paper.

**Update:** (Oct 17, 2024) The datasets in [generated_ood_data.zip](generated_ood_data.zip) were regenerated to incorporate the latest bug fixes.

If you use our code in your work, please cite our papers:
```
@inproceedings{
  PrOntoQA,
  title={Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought},
  author={Abulhair Saparov and He He},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=qFVVBzXxR2V}
}

@article{
  PrOntoQAOOD,
  title={Testing the General Deductive Reasoning Capacity of Large Language Models Using {OOD} Examples},
  author={Abulhair Saparov and
          Richard Yuanzhe Pang and
          Vishakh Padmakumar and
          Nitish Joshi and
          Seyed Mehran Kazemi and
          Najoung Kim and
          He He},
  journal={CoRR},
  volume={abs/2305.15269},
  year={2023},
  url={https://doi.org/10.48550/arXiv.2305.15269},
  doi={10.48550/arXiv.2305.15269},
  eprinttype={arXiv},
  eprint={2305.15269},
}
```

## Running experiments
To generate the examples and evaluate models, use [`run_experiment.py`](run_experiment.py). There are a number of command-line flags:
 - `--model-name [gpt3|opt|unifiedqa|dummy]` specifies the model to test. The `dummy` model is a trivial model, used for testing, that outputs nothing for any input.
 - `--model-size <size>` where `<size>` indicates the version or size of the model. For GPT-3, this must be the OpenAI identifier for the model. For example, to use the InstructGPT 350M parameter model, specify `--model-size text-ada-001`.
 - `--ordering [postorder|preorder|random]` specifies the order of the context sentences of each question.
 - `--num-trials <n>` specifies the number of examples per experiment.
 - `--few-shot-examples <n>` specifies the number of few-shot in-context examples given in each experiment example.
 - `--ontology [fictional|true|false]` indicates which ontology type to generate.
 - `--min-hops <n>`, `--max-hops <m>`, `--hops-skip <k>` specifies which hop counts to test. An experiment is run with `n` hops, then another experiment is run with `n + k` hops, `n + 2k`, and so on until the number of hops exceeds `m`.

The output of the experiments are written to a file whose name is automatically determined based on the above flag configuration.
 - `--resume` is another very useful flag that prevents the program from restarting the experiment at trial 0 if partial results already exist. Rather, the program will continue the experiment where it left off.

The model outputs from our experiments are provided in `model_outputs_v1.zip` (for the original PrOntoQA) and `model_outputs_ood.zip` (for PrOntoQA-OOD).

## Generating data without running experiments

To generate data in JSON format, use the `run_experiment.py` script with the flag `--model-name json`. See the above section for details on the other arguments.

The generated data for our experiments is available in `generated_ood_data.zip`.

## Analyzing output
Once `run_experiment.py` has saved the model predictions to files, they can be analyzed with [`analyze_results.py`](analyze_results.py). Without any arguments, this script will reproduce all results figures in our paper. The script `make_plots.py` generates all the plots in the PrOntoQA-OOD paper. To analyze the output of a single file, run `analyze_results.py <filename>`. This script supports the reading of both JSON-formatted output files as well as the log files output by `run_experiment.py`. The expected JSON format is as follows:
```
{
  "example1": {
    ...
    "test_example": {
      ...
      "model_output": <model output as a string, including the predicted label>
    }
  },
  "example2": {
    ...
    "test_example": {
      ...
      "model_output": <model output as a string, including the predicted label>
    }
  },
  ...
}
```
