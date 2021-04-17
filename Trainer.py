
import abc

class Trainer():
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def train_with_grad(self, grad):
        pass

    @abc.abstractmethod
    def compute_gradient(self, model, data, y):
        pass

    @abc.abstractmethod
    def aggregate(self, models):
        pass

    @abc.abstractmethod
    def evaluate(self, data, label):
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass