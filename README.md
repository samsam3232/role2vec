# Role2vec

Role2vec is a model that mimics the Word2vec model for syntactic roles.  
Pretrained Role2vec models are provided in the Role2vec/models/ directory.  
An inference runner is provided to run the Role2vec models. It saves the 
syntactic and the semantic vectors for the words in a .txt file.  
The .txt file should have the following format:  
- Two different sentences are separated by a line break sign ("\n"), and are therefore on two different lines.
- Within a sentence, we will retrieve the vectors for words enclosed between two *#* signs. For example, in the sentence:   
"The # ***chef elegantly*** # cut the tomatoes with his # ***sharp knives.***#" we will save the vectors for the word 
***chef***, ***elegantly***, ***sharp*** and ***knives***.

To run the inference simply run:  
> python role2vec/inference/vector_getter.py -t path/to/the/.txt_file -w name_of_w2v_models -r path/to/role2vec/models.model -d path/to/the/distances/file/ -o output/path.json -r2v_type merged/dependence/tag -a alpha_factor_role -b beta_factor_distance