import h5py
import numpy as np
DTYPE = 'float32'

ch_embedding_file = 'elmo_chainer_embeddings.hdf5'
pt_embedding_file = 'elmo_pytorch_embeddings.hdf5'
sents = [l.strip() for l in open('_sample_dataset_file.txt') if l.strip()]
with h5py.File(pt_embedding_file, 'r') as pt_fin:
    with h5py.File(ch_embedding_file, 'r') as ch_fin:
        for i in range(len(sents)):
            # allennlp uses a sentence itself as the key
            pt_embedding = pt_fin[sents[i]][...]
            # chainer uses a line index's str as the key
            ch_embedding = ch_fin[str(i)][...]

            # show shape
            print(i, pt_embedding.shape, ch_embedding.shape)

            # test equality up to desired precision, e.g. 5
            np.testing.assert_almost_equal(
                pt_embedding, ch_embedding, decimal=5)

        """
        for i in range(len(sents)):
            pt_embedding = pt_fin[sents[i]][...]
            ch_embedding = ch_fin[str(i)][...]
            # NOT completely equal from a precision, e.g. 8
            # so, this will cause error
            np.testing.assert_almost_equal(
                pt_embedding, ch_embedding, decimal=8)
        """
