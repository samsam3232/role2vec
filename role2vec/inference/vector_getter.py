import json
import argparse
from typing import Dict, List
import numpy as np
import gensim.downloader
from collections import defaultdict
from gensim.models import Word2Vec
from spacy.tokens import Token
from tqdm import tqdm
import spacy
from spacy.symbols import ORTH


def build_tree(doc):

    heads = dict()
    for token in doc:
        heads[(token.text, token.i)] = (token.head.text, token.head.i)

    token_list = list(heads.keys())
    parents = defaultdict(lambda: list())
    for tok in token_list:
        curr_token = tok
        while True:
            if heads[curr_token] == curr_token:
                break
            parents[tok].append(heads[curr_token])
            curr_token = heads[curr_token]

    return parents

def find_distance(tree: Dict, base_token: Token, dist_token: Token):

    """
    Given a tree of parents, returns the distance between two tokens.
    """

    if base_token == dist_token:
        return 0
    elif (base_token.text, base_token.i) in tree[(dist_token.text, dist_token.i)]:
        return tree[(dist_token.text, dist_token.i)].index((base_token.text, base_token.i)) + 1
    elif (dist_token.text, dist_token.i) in tree[(base_token.text, base_token.i)]:
        return tree[(base_token.text, base_token.i)].index((dist_token.text, dist_token.i)) + 1
    else:
        for sample in tree[(base_token.text, base_token.i)]:
            if sample in tree[(dist_token.text, dist_token.i)]:
                break
        d1 = tree[(base_token.text, base_token.i)].index(sample) + 1
        d2 = tree[(dist_token.text, dist_token.i)].index(sample) + 1
        return d1 + d2


def need_vectors(line: str, index: int) -> bool:

    """
    Given a sentence following the correct format and the index of a word, returns whether we need this word's vectors
    :param line: The sentence with the desired words enclosed between ##
    :param index: The index of the word we currently check
    :return: Whether we should use the vectors for this word.
    """

    nlp = spacy.load('en_core_web_sm')
    special_case = [{ORTH: "chef's"}]
    nlp.tokenizer.add_special_case("chef's", special_case)

    split = line.split("##")
    seen = 0
    for i, part in enumerate(split):
        if len(part) > 0:
            real_part = part.strip()
            num_words = len(nlp(real_part))
            if (seen <= index < (seen + num_words)) and (i%2 == 1):
                return True
            seen += num_words
            if seen > index:
                return False

    return False


def get_role(token: Token) -> str:

    depend = token.dep_
    if depend[-4:] == "pass":
        depend = depend[:-4]
    return depend


def find_subject(line: str):

    index = len(line.split("**")[0].replace("##", "").strip().replace("  ", " ").split(" "))
    return index


def treat_line(line: str, w2v: Word2Vec, r2v: Word2Vec, distances: Dict, r2v_type: str = "merged",
               alpha: float = 0.8, beta: float = 0.2) -> List:

    """
    Receives a line, split according to the ## sign, and retrieves the vectors only for text enclosed between two ##
    :param line: The text to treat
    :param w2v: The Word2Vec model
    :param r2v: The Role2Vec model
    :param distances: The distance vectors
    :param alpha: The factor of the role vector
    :param beta: The factor of the distance vector
    :return: A list of all the words and their vectors
    """

    nlp = spacy.load('en_core_web_sm')
    special_case = [{ORTH: "chef's"}]
    nlp.tokenizer.add_special_case("chef's", special_case)

    results = list()

    subj_index = find_subject(line)
    line = line.replace("**", "").strip().replace("  ", " ")

    copied_line = line.replace("##", "").strip().replace("  ", " ")
    doc = nlp(copied_line)

    tree = build_tree(doc)
    for i, tok in enumerate(doc):
        if need_vectors(line, i):
            role = get_role(tok)
            try:
                role_vec = r2v.wv[role]
                semant_vec = w2v.get_vector(tok.text.lower().split("'")[0].replace(".", ""))
            except:
                print(tok.text)
                print(role)
                continue
            dist = distances[str(find_distance(tree, tok, doc[subj_index]))]

            role_vec = (role_vec * alpha) + (np.array(dist) * beta)
            curr_res = {"word": tok.text, "index": i+1, "semantic": semant_vec.tolist(), "syntactic": role_vec.tolist()}
            results.append(curr_res)

    return copied_line, results



def main(text_path: str, w2v_path: str, r2v_path: str, distance_path: str, output_path: str,
         r2v_type: str = "merged", alpha: float = 0.8, beta: float = 0.2):

    with open(text_path, 'r') as f:
        lines = f.readlines()

    w2v = gensim.downloader.load(w2v_path)
    r2v = Word2Vec.load(r2v_path)

    with open(distance_path, 'r') as f:
        distances = json.load(f)

    results = dict()
    for line in tqdm(lines):
        sentence, vectors = treat_line(line, w2v, r2v, distances, r2v_type, alpha, beta)
        if len(vectors) > 0:
            results[sentence] = vectors

    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Role2vec vector finder")
    parser.add_argument("-t", "--text_path", type=str, help="Path to the sentences we want to get data from.")
    parser.add_argument("-w", "--w2v_path", type=str, help="Path for Word2Vec model", default="glove-wiki-gigaword-300")
    parser.add_argument("-r", "--r2v_path", type=str, help="Path for Role2Vec model")
    parser.add_argument("-d", "--distance_path", type=str, help="Path to the distance vectors file")
    parser.add_argument("-o", "--output_path", type=str, help="Path where we keep the output")
    parser.add_argument("--r2v_type", type=str, help="Type of role2vec model", choices=["merged",  "dependence", "tag"])
    parser.add_argument("-a", "--alpha", type=float, help="Factor of the role vector")
    parser.add_argument("-b", "--beta", type=float, help="Factor of the distance vector")
    args = parser.parse_args()
    main(**vars(args))
