from abc import abstractmethod, ABC

class Classifier(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
    
    @abstractmethod
    def classify(self, features):
        pass
