from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

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
            texts = [texts]
        print("Started tokenizing...")
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        print("Tokenizing done, now computing embeddings...")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print("Created embeddings")
        return embeddings
    
    def add_predictions_to_df(df: pd.DataFrame, predictions: torch.tensor) -> pd.DataFrame:
        """
        Adds predictions to a dataframe.

        :param df: pd.DataFrame, dataframe to add predictions to
        :param predictions: torch.tensor, embeddings to add to dataframe
        :return: pd.DataFrame, dataframe with predictions added
        """
        df["embeddings"] = predictions.tolist()
        return df

def main():

    df = pd.read_csv("DATA/alphavantage/news/BA/BA_20230315_20230430.csv")
    te = TextEmbedder('BAAI/bge-base-en-v1.5', 'BAAI/bge-base-en-v1.5', -1)
    embeddings = te.encode(list(df["summary"]))
    torch.save(embeddings, f"DATA/embeddings/BA/BA_20230315_20230430_embeddings.pt")  

if __name__ == "__main__":
    main()