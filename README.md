# Google DeepMind AI Research Foundations: Course 1 labs

Completed solutions for four labs from the official
[google-deepmind/ai-foundations](https://github.com/google-deepmind/ai-foundations/tree/main/course_1) course,
**executed end to end with their outputs saved** in `notebooks/`.

| Lab | Notebook | What was implemented | Course self-check |
|---|---|---|---|
| 1.1 | `gdm_lab_1_1_create_your_own_probability_distribution` | Hand-written next-word probability distributions that sum to 1, sampled with `random.choices` | 3 of 3 pass |
| 1.2 | `gdm_lab_1_2_experiment_with_n_gram_models` | `generate_ngrams`, `get_ngram_counts`, `build_ngram_model`, candidate-token extraction for a trigram model | 4 of 4 pass |
| 1.4 | `gdm_lab_1_4_prepare_the_dataset_for_training_a_slm` | Corpus token list, vocabulary, index-to-token mapping, and a word tokenizer on the Africa Galore dataset | 4 of 4 pass |
| 1.5 | `gdm_lab_1_5_train_your_own_small_language_model` | Sequence padding statistics, then a 200-epoch training run of a small transformer language model | 1 of 1 pass, loss 8.56 to 0.60 |

Lab 1.3 (compare n-gram and transformer models) was not requested and is not included.

## How the solutions were produced and checked

* `upstream/` holds the original notebooks, untouched.
* `tools/solve_notebooks.py` replaces only the placeholder lines (`# Add your code here`, `...`, empty values) with
  the answers and asserts that exactly one placeholder was matched per task, so nothing else in a notebook changes.
* `tools/run_notebook.py` executes a notebook headlessly and prints every cell error plus the course's own feedback
  messages (the `ai_foundations.feedback` checks that print "Nice! ..." or "You've set the probabilities ...").
* `notebooks/` contains the executed result, so the outputs you see on GitHub are real.

```bash
pip install -r requirements.txt
python tools/solve_notebooks.py          # rebuilds solved notebooks from upstream/
python tools/run_notebook.py course_1/gdm_lab_1_4_prepare_the_dataset_for_training_a_slm.ipynb
```

## Things worth knowing

* **Lab 1.2 shows one red `KeyError: 'Their name'` on purpose.** The lab itself says this cell fails because that
  bigram never occurs in the dataset, to illustrate that n-gram models cannot handle unseen contexts.
* **Lab 1.5 was trained on CPU** (one core, no GPU), about 3 seconds per epoch, roughly 9 minutes for 200 epochs.
  The model has about 3.4M parameters and the corpus has only 232 paragraphs, so the final loss of 0.60 reflects
  memorisation of a tiny dataset more than general language ability. The generated sentences are fluent but largely
  reproduce training text. Results will differ slightly on another machine because training is not bit-for-bit
  deterministic.
* The feedback checks verify correctness of the coding tasks. They do not grade the free-text reflection questions
  in the labs, which are left as in the original notebooks.
* The notebooks are subject to the licence terms of the upstream repository (Apache 2.0 for code).

## Environment

Python 3.11, JAX 0.7.2 (CPU), Keras 3.13.2 with the JAX backend. The strict pins in the upstream `pyproject.toml`
make plain `pip install` resolve slowly or fail on some machines. If that happens, install `jax==0.7.2`, `keras`,
`flax<0.12`, then `kauldron` with `--no-deps` and its requirements with `jax` held at 0.7.2, which is how these runs
were made.
