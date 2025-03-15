import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dropout, Dense

# Title of the Project
st.title('🎵 Lyrics Predictor')

# Text Input for Training
input_train = st.text_area(
    'Enter the lyrics for training (Use \n for a new line):')

if st.button('Start Training'):
    if not input_train.strip():
        st.error("Please enter some lyrics to train the model.")
    else:
        with st.spinner('Training the model...'):
            # Split the input into lines
            lyrics = input_train.split('\n')
            
            # Tokenize the text
            token = Tokenizer()
            token.fit_on_texts(lyrics)
            total_words = len(token.word_index) + 1
            
            # Generate input sequences
            input_sequences = []
            for line in lyrics:
                sequence = token.texts_to_sequences([line])[0]
                for i in range(1, len(sequence) + 1):
                    input_sequences.append(sequence[:i])
            
            # Padding sequences
            max_len = max([len(seq) for seq in input_sequences])
            sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
            
            # Split data into input and labels
            X, y = sequences[:, :-1], sequences[:, -1]
            y = to_categorical(y, num_classes=total_words)
            
            # Define the model
            model = Sequential([
                Embedding(total_words, 64, input_length=max_len - 1),
                Bidirectional(LSTM(50, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(50)),
                Dense(total_words, activation='softmax')
            ])
            
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X, y, epochs=150, verbose=1)
            
            st.success("Model training complete! You can now generate lyrics.")
            
            # Store the trained model and tokenizer
            st.session_state['model'] = model
            st.session_state['tokenizer'] = token
            st.session_state['max_len'] = max_len
            
# Prediction Section
if 'model' in st.session_state:
    st.subheader('🎤 Generate Lyrics')
    input_predict = st.text_input('Enter a starting phrase:')
    no_of_words = st.number_input('Number of words to predict:', min_value=1, value=5)
    
    if st.button('Generate Lyrics'):
        model = st.session_state['model']
        token = st.session_state['tokenizer']
        max_len = st.session_state['max_len']
        
        if not input_predict.strip():
            st.error("Please enter a phrase to start the lyrics.")
        else:
            text = input_predict
            for _ in range(no_of_words):
                sequence = token.texts_to_sequences([text])[0]
                padded = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')
                predicted_index = np.argmax(model.predict(padded, verbose=0), axis=1)[0]
                
                output_word = None
                for word, index in token.word_index.items():
                    if index == predicted_index:
                        output_word = word
                        break
                
                if output_word is None:
                    break
                
                text += ' ' + output_word
            
            st.success(f'Generated Lyrics: "{text}"')