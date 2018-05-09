'''
ELMo usage example with pre-computed and cached context independent
token representations

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import os

import chainer

from bilm import Batcher
from bilm import Elmo
from bilm import TokenBatcher
from bilm import dump_token_embeddings

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = ['<S>', '</S>'] + tokenized_question[0]
for context_sentence in tokenized_context:
    for token in context_sentence:
        if token not in all_tokens:
            all_tokens.append(token)
vocab_file = 'vocab_small.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM.  Here we use the test fixtures.
options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = 'elmo_token_embeddings.hdf5'
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)

###########################################
"""
Differences from usage of character-elmo are only simple two points:
1. use TokenBatcher(vocab_file) instead of Batcher(vocab_file)
2. add token_embedding_file and token batcher for Elmo instantiation
"""

# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)  # REQUIRED

# Build the Elmo with biLM and weight layers.
elmo = Elmo(
    options_file,
    weight_file,
    token_embedding_file=token_embedding_file,  # REQUIRED
    token_batcher=batcher,  # REQUIRED
    num_output_representations=1,
    requires_grad=False,
    do_layer_norm=False,
    dropout=0.)

# Create batches of data.
context_token_ids = batcher.batch_sentences(
    tokenized_context, add_bos_eos=False)
question_token_ids = batcher.batch_sentences(
    tokenized_question, add_bos_eos=False)
# numpy.ndarray or cupy.ndarray
# with shape (batchsize, max_length)

# gpu id
# if you want to use cpu, set gpu=-1
# gpu = 0
gpu = -1
if gpu >= 0:
    # transfer the model to the gpu
    chainer.cuda.get_device_from_id(gpu).use()
    elmo.to_gpu()
    # transfer input data to the gpu
    context_ids = elmo.xp.asarray(context_token_ids)
    question_ids = elmo.xp.asarray(question_token_ids)

# Compute elmo outputs,
# i.e. weighted sum of multi-layer biLM's outputs.
context_embeddings = elmo.forward(context_token_ids)
question_embeddings = elmo.forward(question_token_ids)

"""
elmo's output is a dict with the following key-values:
    "elmo_representations": list of chainer.Variable.
        Each element has shape (batchsize, max_length, dim).
        i-th element represents weighted sum using the (i+1)-th weight pattern.
    "mask": numpy.ndarray with shape (batchsize, max_length).
        This mask represents the positions of padded fake values.
        The value of mask[j, k] represents
        if elmo_representations[j, k, :] is valid or not.
        For example, if 1st sentence has 9 tokens and 2nd one has 11,
        the mask is [[1 1 1 1 1 1 1 1 1 0 0]
                     [1 1 1 1 1 1 1 1 1 1 1]]
    "elmo_layers": list of chainer.Variable.
        Each element has shape (batchsize, max_length, dim).
        i-th element represents the output of i-th layer of biLM in elmo.
        Note 0th element is word embedding as input to biLM.
"""

print(len(context_embeddings['elmo_representations']),
      [x.shape for x in context_embeddings['elmo_representations']])
print(context_embeddings['elmo_representations'][0])
print(len(context_embeddings['elmo_layers']),
      [x.shape for x in context_embeddings['elmo_layers']])

print(type(context_embeddings['elmo_representations'][0]))
print(context_embeddings['elmo_representations'][0].shape)
# print(context_embeddings['elmo_representations'])
# print(context_embeddings['elmo_layers'][0])
# print(context_embeddings['elmo_layers'][1])
# print(context_embeddings['elmo_layers'][2])


"""
# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)
question_embeddings_op = bilm(question_token_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_token_ids: context_ids,
                   question_token_ids: question_ids}
    )
"""

"""
# num_output_representations represents
# the number of weighted-sum patterns.
# that is, set 1 if using elmo at the input layer in another neural model.
#          set 2 if using elmo at both input and pre-output in a neural model.

# list of list of str. (i-th batch, j-th token, token's surface string)
# [1st_sentence = [1st word, 2nd word, ...],
#  2nd_sentence = [...]]
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create batches of data.
context_ids = batcher.batch_sentences(tokenized_context, add_bos_eos=False)
question_ids = batcher.batch_sentences(tokenized_question, add_bos_eos=False)
# numpy.ndarray or cupy.ndarray
# with shape (batchsize, max_length, max_character_length)
# default max_character_length = 50

# gpu id
# if you want to use cpu, set gpu=-1
# gpu = 0
gpu = -1
if gpu >= 0:
    # transfer the model to the gpu
    chainer.cuda.get_device_from_id(gpu).use()
    elmo.to_gpu()
    # transfer input data to the gpu
    context_ids = elmo.xp.asarray(context_ids)
    question_ids = elmo.xp.asarray(question_ids)

# Compute elmo outputs,
# i.e. weighted sum of multi-layer biLM's outputs.
context_embeddings = elmo.forward(context_ids)
question_embeddings = elmo.forward(question_ids)
"""
