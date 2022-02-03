import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class ModelMNIST(Model):
    def __init__(self):
        super(ModelMNIST, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
        
    def train(dataset):
        model = ModelMNIST()
        trainset = tf.data.Dataset.from_tensor_slices((dataset.images[..., tf.newaxis].astype("float32"), dataset.labels)).shuffle(9999).batch(32)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)
               
        EPOCHS = 5

        for epoch in range(EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()

            for img, label in trainset:
                train_step(img, label)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
            )
        return model