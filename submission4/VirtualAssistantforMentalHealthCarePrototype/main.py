import re
import random
import pandas
import numpy
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

pandas.set_option("mode.chained_assignment", None)

data = pandas.read_csv("Data/mentalhealthquestionandanswer.csv")

head = data.head(10)
# print(head)

for i in range(data.shape[0]):
    data['Answers'][i] = re.sub(r'\n', ' ', data['Answers'][i])
    data['Answers'][i] = re.sub('\(', '', data['Answers'][i])
    data['Answers'][i] = re.sub(r'\)', '', data['Answers'][i])
    data['Answers'][i] = re.sub(r',', '', data['Answers'][i])
    data['Answers'][i] = re.sub(r'-', '', data['Answers'][i])
    data['Answers'][i] = re.sub(r'/', '', data['Answers'][i])
    data['Answers'][i] = re.sub(r'/', '', data['Answers'][i])

question_and_answer = []

for i in range(data.shape[0]):
    question_and_answer.append(((data['Questions'][i]), data['Answers'][i]))

# print(question_and_answer[0])


input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for line in question_and_answer:

    input_doc, target_doc = line[0], line[1]

    # Appending each input sentence to input_docs
    input_docs.append(input_doc)

    # Splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

    # Redefine target_doc below and append it to target_docs
    target_doc = ' ' + target_doc + ' '

    target_docs.append(target_doc)

    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)

input_tokens = sorted(list(input_tokens))  # contains all words of input_docs
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# print(input_docs)
# print(target_docs)

input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

# print(input_features_dict)

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

encoder_input_data = numpy.zeros(
  (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
  dtype='float32')
decoder_input_data = numpy.zeros(
  (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
decoder_target_data = numpy.zeros(
  (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
  for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
    # Assign 1. for the current line, timestep, & word in encoder_input_data
    encoder_input_data[line, timestep, input_features_dict[token]] = 1.

  for timestep, token in enumerate(target_doc.split()):
    decoder_input_data[line, timestep, target_features_dict[token]] = 1.
    if timestep > 0:
      decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.


# print(encoder_input_data)
print(decoder_target_data)
