import os
import argparse
import random
import numpy as np

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch
from transformers import TrOCRProcessor

from TrOCR import TrOCR
from TrOCRDataset import TrOCRDataModule

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser(description='TrOCR for leet speak')
    
    parser.add_argument("--ices_p", default=0.2)
    parser.add_argument("--ices_topn", default=10)
    parser.add_argument("--ices_p_file", default='/home/nykim/HateSpeech/03_code/ices_perturbations.txt')
    
    parser.add_argument("--dces_p", default=0.5)
    parser.add_argument("--dces_topn", default=15)
    parser.add_argument("--dces_p_file", default='/home/nykim/HateSpeech/03_code/dces_perturbations.txt')
    
    parser.add_argument("--yamin_p", default=0.3)
    parser.add_argument("--jamo_p", default=0.3)
    
    parser.add_argument("--test_data", type=str)
    
    parser.add_argument("--model", default="team-lucid/trocr-small-korean")
    parser.add_argument("--name", type=str, default='first_test')
    
    parser.add_argument("--save_ckpt_dir", type=str, default='/home/nykim/HateSpeech/04_checkpoints')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--logging_dir", type=str, default='/home/nykim/HateSpeech/04_checkpoints')
    
    parser.add_argument("--load_ckpt_dir", type=str)
    
    args = parser.parse_args()
    
    set_random_seed(random_seed=42)
    
    device = torch.device("cuda")
    processor = TrOCRProcessor.from_pretrained(args.model)
    
    print("DM set up...")
    dm = TrOCRDataModule(args.test_data, args, processor)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_cer',
        dirpath=os.path.join(args.save_ckpt_dir, args.name),
        filename='{step:08d}-{valid_cer:.3f}',
        verbose=False,
        mode='min',
        save_top_k=5,
        every_n_train_steps=10000,
    )
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.logging_dir, args.name))
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_logger],
        default_root_dir=args.load_ckpt_dir,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=[0],
        val_check_interval=10000
    )
    
    model = TrOCR(args, processor)
    trainer.fit(model, dm)
