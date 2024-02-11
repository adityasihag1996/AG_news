from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
        # pre-encode the data
        self.encodings = []
        for text in tqdm(texts, desc = "Tokenising data"):
            self.encodings.append(tokenizer.encode(
                                                text,
                                                truncation = True,
                                                padding = False,
                                                max_length = self.max_len,
                                    ))
    

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        input_ids = torch.tensor(self.encodings[item])
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'labels': torch.tensor(self.labels[item], dtype = torch.long)
        }

def dynamic_padding_collator(batch, pad_token_id):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # dynamic padding of batches, according to max_len in batch
    input_ids_padded = pad_sequence(input_ids, batch_first = True, padding_value = pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first = True, padding_value = 0)
    
    labels = torch.stack(labels)
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels
    }