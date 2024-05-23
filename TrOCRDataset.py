import re
import os
import random
import numpy as np
import pandas as pd

import emoji
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from soynlp.normalizer import *
from kotka.yaminizer import encode_yamin
from kotka.shredder import shred_syllable
import jamotools
from transformers import TrOCRProcessor
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

from viper import VIPER_DCES, VIPER_ICES

class TrOCRDataset(Dataset) :
    def __init__(self, stage, args, processor, test_data=None) :
        self.stage = stage
        if stage == 'train' or stage == 'valid' :
            f = open(os.path.join('/home/nykim/HateSpeech/00_data/KcBERT', stage + '.txt'), 'r')
            self.sentences = f.readlines()
            f.close()
        else :
            self.sentences = self.get_test_sentences(test_data)
        self.ices_p = args.ices_p # ices prob
        self.dces_p = args.dces_p # dces prob
        self.yamin_p = args.yamin_p # yaminjeongum encode prob
        self.jamo_p = args.jamo_p # jamo seperate prob
        
        # ices
        self.ices = VIPER_ICES(args)
        
        # dces
        self.dces = VIPER_DCES(args)
        
        self.len = 150 # 150
        
        self.processor = processor
        
        self.arial_broken = Image.open('/home/nykim/HateSpeech/03_code/VIPER/02_arial_broken.ppm').convert("RGB")
        # self.pingfang_broken = Image.open('/home/nykim/HateSpeech/03_code/VIPER/02_pingfang_broken.ppm')
        
        self.per_types = [0, 1, 2, 3]
    
    def get_test_sentences(self, test_data) :
        sentences = None
        if test_data == 'K-MHaS' :
            sentences = pd.read_csv('/home/nykim/HateSpeech/00_data/K-MHaS/data/kmhas_test.txt', sep='\t')
            sentences.drop_duplicates(subset=['document'], inplace=True)
            sentences = list(sentences['document'])
        elif test_data == 'beep' :
            sentences = pd.read_csv('/home/nykim/HateSpeech/00_data/BEEP/labeled/dev.tsv', sep='\t')
            sentences.drop_duplicates(subset=['comments'], inplace=True)
            sentences = list(sentences['comments'])
        elif test_data == 'unsmile' :
            sentences = pd.read_csv('/home/nykim/HateSpeech/00_data/korean_unsmile_dataset/unsmile_valid_v1.0.tsv', sep='\t')
            sentences.drop_duplicates(subset=['문장'], inplace=True)
            sentences = list(sentences['문장'])
        elif test_data == 'apeach' :
            sentences = pd.read_csv('/home/nykim/HateSpeech/00_data/APEACH/APEACH/test.csv')
            sentences.drop_duplicates(subset=['text'], inplace=True)
            sentences = list(sentences['text'])
        return sentences
    
    def __getitem__(self, idx) :
        sentence = self.sentences[idx][:150]
        sentence, label = self.cleanse(sentence)
        per_type = 99
        
        label = self.processor.tokenizer(label, return_tensors='pt', max_length=128, padding='max_length', truncation=True).input_ids[0]
        if self.stage == 'test' :
            img = self.get_sentence_img(sentence).convert("RGB")
            pixel_values = self.processor(img, return_tensors="pt").pixel_values[0]
        else :
            # perturbation and get image
            per_type, perturbated = self.peturbate(sentence)
            perturb_img = self.get_sentence_img(perturbated).convert("RGB")
            pixel_values = self.processor(perturb_img, return_tensors="pt").pixel_values[0]
        
        return {
            'per_type' : per_type,
            'label' : label,
            'pixel_values' : pixel_values
        }
        
    def cleanse(self, sentence) :
        sentence = emoticon_normalize(sentence, num_repeats=2)
        sentence = repeat_normalize(sentence, num_repeats=2)
        sentence = emoji.demojize(sentence, language='ko')
        sentence = re.sub(r"[:_]", ' ', sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        label = jamotools.join_jamos(sentence)
        return sentence, label
    
    def viper_ices(self, sentence) :
        return self.ices.do_ices(sentence)
    
    def viper_dces(self, sentence) :
        return self.dces.do_dces(sentence)
    
    def yaminjeongum(self, sentence) :
        splitted = sentence.split(' ')
        for idx, item in enumerate(splitted) :
            rd = random.random()
            if rd <= self.yamin_p :
                splitted[idx] = encode_yamin(item)
        return ' '.join(splitted)
    
    def seperate_jamo(self, sentence) : 
        return shred_syllable(sentence, active_rate=self.jamo_p)
    
    def peturbate(self, sentence) :
        # perturbation
        rd = random.choice(self.per_types)
        if rd == 0 :
            sentence = self.seperate_jamo(sentence)
            per_type = 0
        elif rd == 1 :
            sentence = self.viper_dces(sentence)
            per_type = 1
        elif rd == 2 :
            sentence = self.yaminjeongum(sentence)
            per_type = 2
        else :
            sentence = self.viper_ices(sentence)
            per_type = 3
        return per_type, sentence
    
    def get_sentence_img(self, sentence) :
        total_img = Image.new("RGB", (24 * 150, 24)) 
        for idx, char in enumerate(sentence) :
            # try arial
            char_img = Image.new ("RGB", (24, 24))
            draw  =  ImageDraw.Draw (char_img)
            unicode_font = ImageFont.truetype('/usr/share/fonts/truetype/MS/Arial-Unicode-MS.ttf', 22)
            draw.text ((1,-4), char, font=unicode_font)
            diff = ImageChops.difference(char_img, self.arial_broken)
            
            if not diff.getbbox() :
                # try pingfang
                char_img = Image.new ("RGB", (24, 24))
                draw  =  ImageDraw.Draw (char_img)
                unicode_font = ImageFont.truetype('/usr/share/fonts/truetype/MS/PingFang-SC-Regular.ttf', 22)
                draw.text ((1,-4), char, font=unicode_font)
                
            total_img.paste(char_img, (24 * idx, 0)) 
        return total_img
    
    def __len__(self) :
        return len(self.sentences)
    
class TrOCRDataModule(pl.LightningDataModule) :
    def __init__(self, test_data, args, processor) :
        super().__init__()
        self.args = args
        
        self.text_data = test_data
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.processor = processor
        
        self.setup()
        
    def setup(self, stage=None) :
        self.set_train = TrOCRDataset('train', self.args, self.processor)
        self.set_valid = TrOCRDataset('valid', self.args, self.processor)
        self.set_test = TrOCRDataset('test', self.args, self.processor, test_data=self.args.test_data)
        
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
        
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test