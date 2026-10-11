import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_dcnn(input_shape=(48, 48, 1), num_conv_blocks=2, dense_neurons=128):
    """Constructs a configurable DCNN model."""
    model = Sequential()
    
    # First block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Dynamic additional convolutional blocks
    for i in range(1, num_conv_blocks):
        filters = 32 * (2 ** i)
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
    model.add(Flatten())
    model.add(Dense(dense_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Gender)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model