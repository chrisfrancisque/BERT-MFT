from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
import torch
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_dataset(config):
    """Load and tokenize SST-2 dataset"""

    logger.info(f"Loading SST2 dataset (using {config.train_samples} samples)")

    download_config = DownloadConfig(
        force_download = True,
        resume_download = False, 
        num_proc=1
    )

    try:

        dataset = load_dataset(
            'glue',
            'sst2',
            download_config=download_config,
            verification_mode='no_checks'
        )
        logger.info("Succesfully loaded SST-2 dataset")

    except Exception as e:
        logger.warning(f"Primary loading method failed: {e}")
        logger.info("Attempting alternative loading method")

        try:
            #Alternative namespace method
            dataset = load_dataset('nyu-mll/glue', 'sst2')
            logger.info("Successfully loaded SST-2 using alternative namespace")
        except Exception as e2:
            logger.error(f"Both loading methods failed: {e2}")
            raise RuntimeError("Failed to load SST-2 Dataset")
    
    # Log dataset information
    logger.info(f"Total train samples available: {len(dataset['train'])}")
    logger.info(f"Total validation samples: {len(dataset['validation'])}")
    logger.info(f"Sample example: {dataset['train'][0]}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    logger.info(f"Loaded tokenizer: {config.model_name}")

    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_length
        )
    
    #tokenize dataset
    logger.info("Tokenizing datasets")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['idx', 'sentence']
    )

    # Rename needed for BERT compatibility
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    # Set format for PyTorch after tokenization
    tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = tokenized_datasets['train'].select(range(min(config.train_samples, len(tokenized_datasets['train']))))
    eval_dataset = tokenized_datasets['validation']

    logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    return train_dataset, eval_dataset, tokenizer

def create_dataloaders(train_dataset, eval_dataset, config):
    """
    Create PyTorch dataloaders with proper settings for gradient analysis.
    
    Args:
        train_dataset: Training dataset (1000 samples)
        eval_dataset: Validation dataset
        config: Configuration object
    
    Returns:
        train_dataloader, eval_dataloader
    """
    #No shuffling for reproducible gradient analysis
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False, # Keep all validation samples
        num_workers =0
    )

    logger.info(f"Created dataloaders - Train batches:{len(train_dataloader)}, Eval Batches: {len(eval_dataloader)}")

    return train_dataloader, eval_dataloader



