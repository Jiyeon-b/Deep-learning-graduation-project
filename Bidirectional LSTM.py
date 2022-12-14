#세팅값
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

#모델링 준비
# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()



#결과값 이런식으로 나옴
#Model: "model"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#input_1 (InputLayer)         [(None, None)]            0         
#_________________________________________________________________
#embedding (Embedding)        (None, None, 128)         2560000   
#_________________________________________________________________
#bidirectional (Bidirectional (None, None, 128)         98816     
#_________________________________________________________________
#bidirectional_1 (Bidirection (None, 128)               98816     
#_________________________________________________________________
#dense (Dense)                (None, 1)                 129       
#=================================================================
#Total params: 2,757,761
#Trainable params: 2,757,761
#Non-trainable params: 0
#_________________________________________________________________

#IMDB 데이터 로드
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

#모델학습및평가
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
