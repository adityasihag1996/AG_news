from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERT_CLS(nn.Module):
    def __init__ (self, model_name, dropout = 0.1, num_classes = 4):
        super(BERT_CLS, self).__init__ ()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # using the model's pooler output
        _, output = self.bert(input_ids = input_ids, attention_mask = attention_mask, return_dict = False)
        output = self.dropout(output)
        output = self.linear(output)
        return output
    

def train_one_epoch(model, dataloader, epoch, device, criterion, optimizer):
    total_loss = 0
    for batch in tqdm(dataloader, desc = f"Training Epoch {epoch + 1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device) - 1
        
        outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del batch, input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache()

    return total_loss


def get_model_and_tokenizer(model_name, device):
    model = BERT_CLS(model_name)
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer
