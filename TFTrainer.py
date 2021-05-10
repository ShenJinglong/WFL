import numpy as np

from Trainer import Trainer

class TFTrainer(Trainer):
    def __init__(self) -> None:
        import tensorflow as tf
        super().__init__()

        self.__optimizer = tf.keras.optimizers.Adam()
        self.__loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.__model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.__model.compile(
            optimizer=self.__optimizer,
            loss=self.__loss,
            metrics=['acc']
        )

    def train_with_grad(self, grad):
        self.__optimizer.apply_gradients(zip(grad, self.__model.trainable_variables))

    def compute_gradient(self, model, data, y):
        import tensorflow as tf
        self.__model.set_weights(model)
        with tf.GradientTape() as tape:
            y_pred = self.__model(data)
            loss = self.__loss(y, y_pred)
        grads = tape.gradient(loss, self.__model.trainable_variables)
        return [item.numpy() for item in grads]

    def aggregate(self, models):
        self.__model.set_weights(list(np.mean(list(models.values()), axis=0)))

    def evaluate(self, data, label):
        return self.__model.evaluate(data, label)

    def get_weights(self):
        return self.__model.get_weights()

if __name__ == '__main__':
    r = TFTrainer()