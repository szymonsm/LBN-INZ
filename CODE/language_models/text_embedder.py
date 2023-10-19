from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedder:

    def __init__(self, model_name: str, tokenizer_name: str, device: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device=device)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: str | list[str], max_length: int = 128) -> torch.Tensor:
        """
        Encodes text(s) into embeddings.

        :param texts: str or list[str], text(s) to encode
        :param max_length: int, max length of text
        :return: torch.Tensor, embeddings
        """
        self.model.eval()
        if isinstance(texts, str):
            text = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

