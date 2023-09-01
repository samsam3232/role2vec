import argparse
import os
import zipfile
from tqdm import tqdm

def fix_text(txt: str):

    txt = txt.replace(' . ', '. ').replace(' , ', ', ')
    txt = txt.replace("n't", 'not').replace(" ( ", " (").replace(" ) ", ") ")
    txt = txt.replace("<p>", '\n').replace(" '", "'")
    return txt


def main(input_folder: str, output_path: str):

    input_files = [os.path.join(input_folder, k) for k in os.listdir(input_folder)]
    for in_file in tqdm(input_files):
        with zipfile.ZipFile(in_file) as z:
            for filename in tqdm(z.namelist()):
                if not os.path.isdir(filename):
                    if os.path.exists(os.path.join(output_path, filename)):
                        continue
                    try:
                        texts = list()
                        with z.open(filename) as f:
                            for line in f:
                                try:
                                    txt = line.decode()
                                    if "@" not in txt.split('\t')[1][:2]:
                                        texts.append(txt.split('\t')[1])
                                except:
                                    continue
                            txt = ' '.join(texts)
                            txt = fix_text(txt)
                            with open(os.path.join(output_path, filename), 'w') as f:
                                f.write(txt)
                    except:
                        continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parser for the COCA preprocessing")
    parser.add_argument('-i', '--input_folder', type=str, help="Path to where the COCA dataset is kept")
    parser.add_argument('-o', '--output_path', type=str, help="Path to where we keep the preprocessed data")
    args = parser.parse_args()
    main(**vars(args))