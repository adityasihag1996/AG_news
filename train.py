from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

import config
from utils import process_dataset, evaluate_model
from NewsDataset import NewsDataset, dynamic_padding_collator
from modelling_utils import train_one_epoch, get_model_and_tokenizer


if __name__ == "__main__":
    # load the datasets
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TRAIN_PATH)
    print("Train Size - ", len(train_df))
    print("Test Size - ", len(test_df))

    # preprocess the data
    train_proc_df = process_dataset(train_df, max_len_raw = 80, min_len_raw = 5)
    test_proc_df = process_dataset(test_df, max_len_raw = 80, min_len_raw = 5)

    # create train / valid split, stratified according to class
    train_texts, val_texts, train_labels, val_labels = train_test_split(
                                                            train_proc_df['Description'], 
                                                            train_proc_df['Class Index'], 
                                                            test_size = config.VALID_SIZE,
                                                            stratify = train_proc_df['Class Index'], 
                                                            random_state = config.SEED,
                                                        )
    print("Train Size - ", len(train_texts))
    print("Valid Size - ", len(val_texts))

    # initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    
    # create datasets
    train_dataset = NewsDataset(
        texts = train_texts.to_list(),
        labels = train_labels.to_numpy(),
        tokenizer = tokenizer,
        max_len = config.MAX_LENGTH,
    )

    val_dataset = NewsDataset(
        texts = val_texts.to_list(),
        labels = val_labels.to_numpy(),
        tokenizer = tokenizer,
        max_len = config.MAX_LENGTH,
    )

    test_dataset = NewsDataset(
        texts = test_proc_df['Description'].to_list(),
        labels = test_proc_df['Class Index'].to_numpy(),
        tokenizer = tokenizer,
        max_len = config.MAX_LENGTH,
    )

    # create dataloaders
    print("Creating Dataloaders..")
    train_loader = DataLoader(
                        train_dataset,
                        batch_size = config.BATCH_SIZE,
                        shuffle = True,
                        collate_fn = partial(dynamic_padding_collator, pad_token_id = tokenizer.pad_token_id),
                        num_workers = config.NUM_WORKERS,
                    )
    val_loader = DataLoader(
                    val_dataset,
                    batch_size = config.BATCH_SIZE,
                    collate_fn = partial(dynamic_padding_collator, pad_token_id = tokenizer.pad_token_id),
                    num_workers = config.NUM_WORKERS,
                )
    test_loader = DataLoader(
                    test_dataset,
                    batch_size = config.BATCH_SIZE,
                    collate_fn = partial(dynamic_padding_collator, pad_token_id = tokenizer.pad_token_id),
                    num_workers = config.NUM_WORKERS,
                )
    
    # init optimiser, scheduler and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LR, weight_decay = config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        
        total_loss = train_one_epoch(
                        model = model,
                        dataloader = train_loader,
                        epoch = epoch,
                        device = config.DEVICE,
                        criterion = criterion,
                    )
        
        print(f"Epoch {epoch + 1}, TRAIN Loss: {total_loss / len(train_loader)}")
        valid_metrics = evaluate_model(model, val_loader, config.DEVICE)
        print("VALID Accuracy -", valid_metrics["accuracy"])

        # save model checkpoint
        torch.save(model.state_dict(), f"cls_ckpt_epoch{epoch}.pt")

    print("Training finished.")
    