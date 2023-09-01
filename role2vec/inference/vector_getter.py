import json
import argparse
from typing import Dict, List
import numpy as np
from gensim.models import Word2Vec
from spacy.tokens import Token
import spacy

def need_vectors(line: str, index: int) -> bool:

    """
    Given a sentence following the correct format and the index of a word, returns whether we need this word's vectors
    :param line: The sentence with the desired words enclosed between ##
    :param index: The index of the word we currently check
    :return: Whether we should use the vectors for this word.
    """

    split = line.split("##")
    seen = 0
    for i, part in enumerate(split):
        if len(part) > 0:
            num_words = len(part.split(" "))
            if (seen <= index < (seen + num_words)) and (i%2 == 1):
                return True
            seen += num_words
            if seen > index:
                return False

    return False


def get_role(token: Token, r2v_type: str = "merged") -> str:

    if r2v_type == "merged":
        return f"{token.tag_}_{token.dep_}"
    elif r2v_type == "dependency":
        return f"{token.dep_}"
    else:
        return f"{token.tag_}"


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

    nlp = spacy.load("en_core_web_sm")
    results = list()

    copied_line = line.replace("##", "").strip().replace("  ", " ")
    doc = nlp(copied_line)
    for i, tok in enumerate(doc):
        if need_vectors(line, i):
            role = get_role(tok, r2v_type)
            role_vec = r2v(role)
            semant_vec = w2v(tok.text.lower())
            dist = distances[i]

            role_vec = (role_vec * alpha) + (np.array(dist) * beta)
            curr_res = {"word": tok.text, "index": i+1, "semantic": semant_vec, "syntactic": role_vec}
            results.append(curr_res)

    return copied_line, results



def main(text_path: str, w2v_path: str, r2v_path: str, distance_path: str, output_path: str,
         r2v_type: str = "merged", alpha: float = 0.8, beta: float = 0.2):

    with open(text_path, 'r') as f:
        lines = f.readlines()

    w2v = Word2Vec.load(w2v_path)
    r2v = Word2Vec.load(r2v_path)

    with open(distance_path, 'r') as f:
        distances = json.load(f)

    results = dict()
    for line in lines:
        sentence, vectors = treat_line(line, w2v, r2v, distances, r2v_type, alpha, beta)
        if len(vectors) > 0:
            results[sentence] = vectors

    with open(output_path, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Role2vec vector finder")
    parser.add_argument("-t", "--text_path", type=str, help="Path to the sentences we want to get data from.")
    parser.add_argument("-w", "--w2v_path", type=str, help="Path to the Word2Vec model")
    parser.add_argument("-r", "--r2v_path", type=str, help="Path to the Role2Vec model")
    parser.add_argument("-d", "--distance_path", type=str, help="Path to the distance vectors file")
    parser.add_argument("-o", "--output_path", type=str, help="Path where we keep the output")
    parser.add_argument("-r2v_type", type=str, help="Type of role2vec model", choices=["merged",  "dependence", "tag"])
    parser.add_argument("-a", "--alpha", type=float, help="Factor of the role vector")
    parser.add_argument("-b", "--beta", type=float, help="Factor of the distance vector")
    args = parser.parse_args()
    main(**vars(args))