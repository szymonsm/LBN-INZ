from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments
import numpy as np
import os

class TextClassificationFinetuner:
    def __init__(self, model_name: str, tokenizer_name: str, train_set: Dataset, validation_set: Dataset, output_dir: str = os.getcwd()):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_set = train_set
        self.val_set = validation_set
        self.output_dir = output_dir


    def tokenize_datasets(self, text_col_name: str, max_length: int=128):
        self.train_set = self.train_set.map(
            lambda e: self.tokenizer(e[text_col_name], truncation=True, padding='max_length', max_length=max_length),
            batched=True
        )
        self.val_set = self.val_set.map(
            lambda e: self.tokenizer(e[text_col_name], truncation=True, padding='max_length', max_length=max_length),
            batched=True
        )
        self.train_set.set_format(type='torch', columns=['label', 'input_ids', 'token_type_ids', 'attention_mask'])
        self.val_set.set_format(type='torch', columns=['label', 'input_ids', 'token_type_ids', 'attention_mask'])


    def finetune_model(self, learning_rate: float=2e-5, batch_size: int=64, epochs: int=5, weight_decay: float=0.01, metric_for_best_model: str='accuracy'):
    
        args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(predictions, labels),
                    'precision': precision_score(predictions, labels, average='macro'),
                    'recall': recall_score(predictions, labels, average='macro'),
                    'f1': f1_score(predictions, labels, average='macro')}

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            compute_metrics=compute_metrics
        )
        trainer.train()
        return trainer

    def save_model(self, trainer: Trainer) -> None:
        trainer.save_model(self.output_dir)

    def launch(self, text_col_name: str = "text") -> Trainer:
        self.tokenize_datasets(text_col_name=text_col_name)
        trainer = self.finetune_model()
        self.save_model(trainer)
        return trainer

      