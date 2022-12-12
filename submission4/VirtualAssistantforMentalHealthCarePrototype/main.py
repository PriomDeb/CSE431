import re
import random
import pandas
import numpy
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model

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
# print(decoder_target_data)

dimensionality = 256  # Dimensionality
batch_size = 10  # The batch size and number of epochs
epochs = 2

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # Compiling

# print(training_model.summary())
# plot_model(training_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'],
#                        sample_weight_mode='temporal')  # Training
# history1 = training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size,
#                               epochs=epochs, validation_split=0.2)
# training_model.save('training_model.h5')

# acc = history1.history['accuracy']
# val_acc = history1.history['val_accuracy']
# loss = history1.history['loss']
# val_loss = history1.history['val_loss']
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
#
# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


training_model = load_model('training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]


decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

training_model = load_model('training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_response(test_input):
    # Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)

    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Setting the first token of target sequence with the start token
    target_seq[0, 0, target_features_dict['']] = 1.

    # A variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False
    while not stop_condition:
        # Predicting output tokens with probabilities and states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

        # Choosing the one with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # Stop if hit max length or found the stop token
        if (sampled_token == '' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence


