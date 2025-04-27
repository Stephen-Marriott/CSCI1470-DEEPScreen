import tensorflow as tf
from tensorflow.keras import layers, models
from operator import itemgetter

class CNNModel1(tf.keras.Model):
    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(CNNModel1, self).__init__()

        self.conv1 = layers.Conv2D(32, 2, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, 2, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, 2, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(64, 2, activation='relu', padding='same')
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(32, 2, activation='relu', padding='same')
        self.bn5 = layers.BatchNormalization()

        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.drop1 = layers.Dropout(drop_rate)
        self.fc1 = layers.Dense(fully_layer_1, activation='relu')
        self.drop2 = layers.Dropout(drop_rate)
        self.fc2 = layers.Dense(fully_layer_2, activation='relu')
        self.fc3 = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        x = self.pool(self.bn1(self.conv1(inputs)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.pool(self.bn3(self.conv3(x)))
        x = self.pool(self.bn4(self.conv4(x)))
        x = self.pool(self.bn5(self.conv5(x)))

        x = self.flatten(x)
        x = self.drop1(self.fc1(x), training=training)
        x = self.drop2(self.fc2(x), training=training)
        x = self.fc3(x)

        return x