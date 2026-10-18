"""Fill in the learner tasks of the DeepMind AI Foundations Course 1 labs.

Each solution replaces the placeholder (`# Add your code here`, `...`, empty values) inside the original notebook
cell, so the notebooks stay exactly as published except for the answers. Re-run safe: it starts from the pristine
upstream notebooks in ../upstream/.
"""
import json, sys, pathlib

UP = pathlib.Path(__file__).resolve().parent.parent / "upstream"
OUT = pathlib.Path(__file__).resolve().parent.parent / "course_1"


def load(name):
    return json.loads((UP / name).read_text())


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one occurrence of {old!r}, found {text.count(old)}"
    return text.replace(old, new)


def find(nb, needle, nth=0):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and needle in src(c)]
    return hits[nth]


# ---------------------------------------------------------------- Lab 1.1
def lab_1_1(nb):
    # Hungry: food-like words get most of the mass, nonsense words get a tiny but non-zero share.
    hungry = [0.40, 0.25, 0.20, 0.04, 0.03, 0.06, 0.01, 0.01]
    # Thirsty: water dominates, food words are plausible but unlikely.
    thirsty = [0.05, 0.03, 0.02, 0.04, 0.03, 0.81, 0.01, 0.01]
    i1 = find(nb, "# prompt \"Jide was hungry so she went looking for\".")
    s = replace_once(src(nb["cells"][i1]), "your_mental_model = []", f"your_mental_model = {hungry}")
    set_src(nb["cells"][i1], s)
    i2 = find(nb, "# prompt \"Jide was thirsty so she went looking for\".")
    s = replace_once(src(nb["cells"][i2]), "your_mental_model = []", f"your_mental_model = {thirsty}")
    set_src(nb["cells"][i2], s)


# ---------------------------------------------------------------- Lab 1.2
def lab_1_2(nb):
    i = find(nb, "def generate_ngrams(text: str, n: int)")
    s = src(nb["cells"][i])
    s = replace_once(s, "tokens = ...  # Add your code here.", "tokens = space_tokenize(text)")
    s = replace_once(
        s,
        "    ngrams = []\n\n    # Add your code here.\n",
        "    ngrams = []\n\n    # Slide a window of size n over the tokens. The last n-gram starts at len(tokens) - n.\n"
        "    for i in range(0, len(tokens) - n + 1):\n        ngrams.append(tuple(tokens[i:i + n]))\n",
    )
    set_src(nb["cells"][i], s)

    i = find(nb, "def get_ngram_counts(")
    s = src(nb["cells"][i])
    s = replace_once(
        s,
        "        # Add your code here.\n        ...\n",
        "        for ngram in generate_ngrams(paragraph, n):\n"
        "            # Context = all but the last token, joined by a space (the dictionary key).\n"
        "            context = \" \".join(ngram[:-1])\n"
        "            next_token = ngram[-1]\n"
        "            ngram_counts[context][next_token] += 1\n",
    )
    set_src(nb["cells"][i], s)

    i = find(nb, "def build_ngram_model(")
    s = src(nb["cells"][i])
    s = replace_once(
        s,
        "        # Add your code here.\n        ...\n",
        "        # Count(A): how often the context appears in total.\n"
        "        context_total_count = sum(next_tokens.values())\n"
        "        ngram_model[context] = {}\n"
        "        for token, count in next_tokens.items():\n"
        "            # P(B | A) = Count(A B) / Count(A).\n"
        "            ngram_model[context][token] = count / context_total_count\n",
    )
    set_src(nb["cells"][i], s)

    i = find(nb, "candidate_tokens_probabilities = []")
    s = src(nb["cells"][i])
    s = replace_once(
        s,
        "# Add your code here.\n",
        "candidate_tokens = list(trigram_model[context].keys())\n"
        "candidate_tokens_probabilities = list(trigram_model[context].values())\n",
    )
    set_src(nb["cells"][i], s)


# ---------------------------------------------------------------- Lab 1.4
def lab_1_4(nb):
    i = find(nb, "tokens = []\n\n# Add your code here.")
    s = replace_once(
        src(nb["cells"][i]),
        "# Add your code here.\n",
        "for paragraph in dataset:\n    for token in space_tokenize(paragraph):\n        tokens.append(token)\n",
    )
    set_src(nb["cells"][i], s)

    i = find(nb, "def build_vocabulary(tokens: list[str]) -> list[str]:\n    # Build a vocabulary list")
    s = replace_once(src(nb["cells"][i]), "vocabulary = # Add your code here.", "vocabulary = list(set(tokens))")
    set_src(nb["cells"][i], s)

    i = find(nb, "index_to_token = {}\n\n# Add your code here.")
    s = replace_once(
        src(nb["cells"][i]),
        "# Add your code here.",
        "for token, index in token_to_index.items():\n    index_to_token[index] = token",
    )
    set_src(nb["cells"][i], s)


# ---------------------------------------------------------------- Lab 1.5
def lab_1_5(nb):
    i = find(nb, "shortest_paragraph_length =")
    s = src(nb["cells"][i])
    s = replace_once(s, "shortest_paragraph_length =\n", "shortest_paragraph_length = len(min(encoded_tokens, key=len))\n")
    s = replace_once(s, "longest_paragraph_length =\n", "longest_paragraph_length = len(max(encoded_tokens, key=len))\n")
    set_src(nb["cells"][i], s)


LABS = {
    "gdm_lab_1_1_create_your_own_probability_distribution.ipynb": lab_1_1,
    "gdm_lab_1_2_experiment_with_n_gram_models.ipynb": lab_1_2,
    "gdm_lab_1_4_prepare_the_dataset_for_training_a_slm.ipynb": lab_1_4,
    "gdm_lab_1_5_train_your_own_small_language_model.ipynb": lab_1_5,
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in LABS.items():
        nb = load(name)
        fn(nb)
        (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("solved", name)
