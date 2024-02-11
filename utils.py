import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def process_dataset(dataframe, max_len_raw = 64, min_len_raw = 32):
    dataframe['Description'] = dataframe.Description.str.replace('/', '').str.replace ('  ', ' ')

    seperator_char = " "
    
    del_index = []
    for i, row in enumerate (dataframe.iterrows ()) : 
        dataframe.loc[i, 'Description'] = dataframe.loc[i, 'Title'].lower() + seperator_char + dataframe.loc[i, 'Description'].lower()
        dataframe.loc[i, 'length'] = len(dataframe.loc[i, 'Description'].split())

        # ignoring samples with "length" > max_len_raw and < min_len_raw
        if dataframe.loc[i, 'length'] > max_len_raw or dataframe.loc[i, 'length'] < min_len_raw:
            del_index.append(i)
            
    dataframe.drop(index = del_index, inplace = True)
    
    return dataframe


def evaluate_model(model, data_loader, device):
    model.eval()

    true_labels = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) - 1

            outputs = model(input_ids, attention_mask)

            _, preds = torch.max(outputs, dim = 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            del batch, input_ids, attention_mask, labels, outputs, preds
            torch.cuda.empty_cache()

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average = 'weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }