import torch
from torch.utils.data import Dataset
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    pass

class SpanAspectDataset(Dataset):
    """
    Dataset for the Evidence Span Auxiliary Model.
    Formats inputs as: `[CLS] Full text with [START] evidence span [END] markers [SEP]`
    """
    def __init__(self, records: list, tokenizer_name: str = "microsoft/deberta-v3-base"):
        self.records = records
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Add special tokens for the span bounding
        special_tokens_dict = {'additional_special_tokens': ['[START]', '[END]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        record = self.records[idx]
        text = record["review_text"]
        span = record["evidence_span"]
        label = record["aspect_id"] # Pre-mapped integer label for aspect + sentiment class
        
        # Insert [START] and [END] bounds
        if span and span in text:
            marked_text = text.replace(span, f"[START] {span} [END]", 1)
        else:
            marked_text = text
            
        encoding = self.tokenizer(
            marked_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_span_model(num_labels: int, model_name: str = "microsoft/deberta-v3-base"):
    """
    Initializes DeBERTa sequence classification architecture.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # The tokenizer has 2 new special tokens which require resizing the embedding matrix
    model.resize_token_embeddings(model.config.vocab_size + 2)
    return model
