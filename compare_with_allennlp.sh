allennlp elmo _sample_dataset_file.txt elmo_pytorch_embeddings.hdf5 --all
python bilm_encode_sentenses.py -i _sample_dataset_file.txt -o elmo_chainer_embeddings.hdf5
python compare_with_allennlp.py
