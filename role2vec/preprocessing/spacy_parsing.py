import argparse
import spacy
import os
import re
from tqdm import tqdm

def treat_single_file(file_path: str, output_path: str):

    nlp = spacy.load("en_core_web_sm")

    with open(file_path, 'r') as f:
        data = f.readlines()


    par, par_dep, par_tag = list(), list(), list()
    for line in tqdm(data):
        line = line.replace(' ? ', '? ')
        line = line.replace(' ! ', '! ')
        sents, sents_dep, sents_tag = list(), list(), list()
        sentences = re.split('[?.!]', line)
        for sent in tqdm(sentences):
            merged, dep, tag = list(), list(), list()
            doc = nlp(sent)
            for tok in doc:
                merged.append(f"{tok.tag_}_{tok.dep_}")
                dep.append(tok.dep_)
                tag.append(tok.tag_)
            sents.append(' '.join(merged))
            sents_dep.append(' '.join(dep))
            sents_tag.append(' '.join(tag))

        par.append('. '.join(sents))
        par_dep.append('. '.join(sents_dep))
        par_tag.append('. '.join(sents_tag))

    with open(output_path, 'w') as f:
        f.write('\n'.join(par))

    with open(output_path.replace('merged/', 'dep/'), 'w') as f:
        f.write('\n'.join(par_dep))

    with open(output_path.replace('merged/', 'tag/'), 'w') as f:
        f.write('\n'.join(par_tag))


def main(input_folder: str, output_folder: str):

    filenames = os.listdir(input_folder)
    for fname in filenames:
        treat_single_file(os.path.join(input_folder, fname), os.path.join(output_folder, fname))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Spacy preprocessing")
    parser.add_argument('-i', '--input_folder', type=str, help="Path to where the text files are kept")
    parser.add_argument('-o', '--output_folder', type=str)
    args = parser.parse_args()
    main(**vars(args))