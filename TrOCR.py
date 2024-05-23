import pandas as pd

import torch
from torch.optim import AdamW
from torchmetrics.text import CharErrorRate
import pytorch_lightning as pl
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


class TrOCR(pl.LightningModule) :
    def __init__(self, args, processor) :
        super().__init__()
        self.learning_rate = args.learning_rate
        self.args = args
        
        self.model = VisionEncoderDecoderModel.from_pretrained(args.model)
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.config.pad_token_id = 0
        self.model.config.decoder_start_token_id = 2
        
        self.processor = processor
        
        self.cer = CharErrorRate()
        self.save_hyperparameters()
        
        self.preds = []
        self.trgts = []
        self.per_type = []
        
    def forward(
        self,
        pixel_values,
        labels
    ) :
        outputs = self.model(pixel_values, labels=labels)
        
        return outputs
    
    def default_step(self, batch, batch_idx, state=None) :
        outputs = self(
            pixel_values=batch['pixel_values'].to(self.device),
            labels=batch['label'].to(self.device)
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=2).to(self.device)
        targets = batch['label'].to(self.device)
        trgts = self.processor.tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = self.processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        cer = self.cer(preds, trgts)
        
        if state == 'valid' :
            self.preds += preds
            self.trgts += trgts
            self.per_type += batch['per_type'].tolist()
        
        self.log(f"[{state} loss]", loss, prog_bar=True)
        self.log(f"[{state} cer]", cer, prog_bar=True)
        
        return {
            'loss' : loss,
            'cer' : cer
        }

    def default_epoch_end(self, outputs, state=None) :
        loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        cer = torch.mean(torch.tensor([output['cer'] for output in outputs]))
        
        self.log(f'{state}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{state}_cer', cer, on_epoch=True, prog_bar=True)
        
    def training_step(self, batch, batch_idx, state='train') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def validation_step(self, batch, batch_idx, state='valid') :
        result = self.default_step(batch, batch_idx, state)
        return result
    
    def training_epoch_end(self, outputs, state='train') :
        self.default_epoch_end(outputs, state)
        
    def validation_epoch_end(self, outputs, state='valid') :
        self.default_epoch_end(outputs, state)
        df = {
            'trgts' : self.trgts,
            'preds' : self.preds,
            'per_type' : self.per_type
        }
        df = pd.DataFrame(df)
        df.to_csv(f'/home/nykim/HateSpeech/05_valid_outputs/{self.args.name}_{self.global_step}.csv')
        
        self.trgts.clear()
        self.preds.clear()
        self.per_type.clear()
        
    def test_step(self, batch, batch_idx, state='test') :
        generated = self.model.generate(batch['pixel_values'].to(self.device))
        
        targets = self.processor.tokenizer.batch_decode(batch['label'], skip_special_tokens=True)
        preds = self.processor.batch_decode(generated, skip_special_tokens=True)

        cer = self.cer(preds, targets)
        
        self.preds += preds
        self.trgts += targets
        self.per_type += batch['per_type'].tolist()
        
        self.log(f"[{state} cer]", cer, prog_bar=True)
        
        return {
            'cer' : cer
        }
        
    def test_epoch_end(self, outputs, state='test') :
        cer = torch.mean(torch.tensor([output['cer'] for output in outputs]))
        
        self.log(f'{state}_cer', cer, on_epoch=True, prog_bar=True)
        df = {
            'trgts' : self.trgts,
            'preds' : self.preds,
            'per_type' : self.per_type
        }
        df = pd.DataFrame(df)
        df.to_csv(f'/home/nykim/HateSpeech/06_test_outputs/{self.args.name}.csv')
        
        self.trgts.clear()
        self.preds.clear()
        self.per_type.clear()
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        
        return [optimizer], [lr_scheduler]