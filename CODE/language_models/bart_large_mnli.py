from transformers import pipeline

class BartLargeMNLI:
    """
    Class for BART model that predicts classes of a text. One can use own classes or use the default ones.
    """

    def __init__(self):
        self.pipe = None

    def initialize_model(self, device: int = -1) -> None:
        """
        Initializes the model.

        :param device: int, -1 for CPU, 0 for GPU
        """
        self.pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    
    def predict_classes(self, texts: str | list[str], classes: list[str], multi_label: bool = True) -> list:
        """
        Predicts classes of a text or list of texts.

        :param texts: str or list[str], text(s) to predict classes of
        :param classes: list[str], classes to predict
        :param multi_label: bool, whether to predict multiple classes or not
        :return: list, predictions
        """
        return self.pipe(texts, classes, multi_label=multi_label)
    
    