import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model

# Girdi ve çıktı metinleri için eşleştirme
input_texts = []
target_texts = []

# Modeli oluşturmak için gerekli hiperparametreler
num_samples = 10000
max_input_length = 20
max_target_length = 20
input_vocab_size = 20000
target_vocab_size = 2000
embedding_size = 128
latent_dim = 256

# Girdi ve çıktı metinlerini eşleştirmeye ekle
for input_text, target_text in zip(input_texts, target_texts):
    input_texts.append(input_text)
    target_texts.append("<sos> " + target_text + " <eos>")

# Girdi metinlerini kodlamak için tokenizer
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=input_vocab_size, lower=True)
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

# Çıktı metinlerini kodlamak için tokenizer
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=target_vocab_size, lower=True)
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Girdi ve çıktı metinlerini eşleştirmek için pad_sequences
input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_length)
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_length)

# Girdi için Embedding katmanı
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, embedding_size)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Çıktı için Embedding, LSTM ve Dense katmanları
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedd
decoder_embedding = Embedding(target_vocab_size, embedding_size)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modeli oluştur
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Modeli eğit
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([input_data, target_data], target_data, batch_size=64, epochs=50)

# Modeli kullanmak için Encoder ve Decoder'ı ayrı ayrı oluştur
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Chatbot'u kullanmak için fonksiyon yaz
def chatbot(input_text):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_input_length)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<sos>']
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += " " + sampled_word
        if (sampled_word == '<eos>' or len(decoded_sentence) > max_target_length):
            stop_condition = True
        target_seq = np.zer
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        return decoded_sentence
while True:
    string=input(string)
    print(chatbot(string))
