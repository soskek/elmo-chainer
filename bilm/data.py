# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import numpy as np

from typing import List
import re


split_pattern = re.compile(r'([.,!?"\':;)(])')


def split_sentence_with_punctuations(sentence):
    words = []
    for word in sentence.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words


class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    '''

    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True, add_bos_eos=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            if add_bos_eos:
                word_ids = [self.eos] + word_ids + [self.bos]
            return np.array(word_ids, dtype=np.int32)
        else:
            if add_bos_eos:
                word_ids = [self.bos] + word_ids + [self.eos]
            return np.array(word_ids, dtype=np.int32)


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


# the charcter representation of the begin/end of sentence characters
def _make_bos_eos(
        c,
        pad_char,
        bow_char,
        eow_char,
        max_word_length):
    # copied from indexer
    r = np.zeros([max_word_length], dtype=np.int32)
    r[:] = pad_char
    r[0] = bow_char
    r[1] = c
    r[2] = eow_char
    return r


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """

    # original implementation has these lines in __init__,
    # though this one makes them class variables
    # for consistency with tensorflow & pytorch implemtantions

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    bos_char = 256  # <begin sentence>
    eos_char = 257  # <end sentence>
    bow_char = 258  # <begin word>
    eow_char = 259  # <end word>
    pad_char = 260  # <padding>

    _max_word_length = 50

    bos_chars = _make_bos_eos(
        bos_char,
        pad_char,
        bow_char,
        eow_char,
        _max_word_length)
    eos_chars = _make_bos_eos(
        eos_char,
        pad_char,
        bow_char,
        eow_char,
        _max_word_length)

    def __init__(self, filename, max_word_length=50, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        """
        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)
        """

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode(
            'utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True, add_bos_eos=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        if reverse:
            if add_bos_eos:
                chars_ids = [self.eos_chars] + chars_ids + [self.bos_chars]
            return np.vstack(chars_ids)
        else:
            if add_bos_eos:
                chars_ids = [self.bos_chars] + chars_ids + [self.eos_chars]
            return np.vstack(chars_ids)


class Batcher(object):
    ''' 
    Batch sentences of tokenized text into character id matrices.
    '''

    def __init__(self, lm_vocab_file, max_token_length):
        # def __init__(self, lm_vocab_file: str, max_token_length: int):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        '''
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length
        )
        self._max_token_length = max_token_length

    def batch_sentences(self, sentences, add_bos_eos=True):
        # def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) \
            + (2 if add_bos_eos else 0)

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int32
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + (2 if add_bos_eos else 0)
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False, add_bos_eos=add_bos_eos)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object):
    ''' 
    Batch sentences of tokenized text into token id matrices.
    '''

    def __init__(self, lm_vocab_file):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences, add_bos_eos=True):
        # def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) \
            + (2 if add_bos_eos else 0)

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int32)

        for k, sent in enumerate(sentences):
            length = len(sent) + (2 if add_bos_eos else 0)
            ids_without_mask = self._lm_vocab.encode(
                sent, split=False, add_bos_eos=add_bos_eos)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        return X_ids
