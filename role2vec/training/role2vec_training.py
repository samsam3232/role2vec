import argparse
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from tqdm import tqdm
import multiprocessing


def main(input_path: str, output_path: str):

    dir_lists = os.listdir(input_path)
    for dir in tqdm(dir_lists):
        model = Word2Vec(PathLineSentences(os.path.join(input_path, dir)),
                         vector_size=300,
                         window=5,
                         workers=multiprocessing.cpu_count() // 2,
                         compute_loss=True,
                         epochs=5)
        model.save(os.path.join(output_path, f"{dir}.json"))

    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Role2vec training")
    parser.add_argument('-i', '--input_path', type=str, help="Path to where the results of preprocessing is kept")
    parser.add_argument('-o', '--output_path', type=str, help="Path to where we keep the trained model")
    args = parser.parse_args()
    main(**vars(args))