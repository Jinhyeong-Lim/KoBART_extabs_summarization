from load_data import data_load
from train import train
from data_preprocessing import data_preprocessing
from evaluation import  eval

import torch
from kobart_transformers import get_kobart_tokenizer, get_kobart_for_conditional_generation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
text = data_load()

#data preprocessing
train_loader, valid_loader, test_loader = data_preprocessing(text)

#load Pretrained model, tokenizer
tokenizer = get_kobart_tokenizer()
model = get_kobart_for_conditional_generation()
model.to(device)

#hyperparameter
epochs = 1
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

#train
moeel = train(train_loader, valid_loader, epochs, model, tokenizer, optimizer, device)

#evaluation
Rouge1_score, Rouge2_score, Rougel_score = eval(model, tokenizer, test_loader, device)

#print performance
print(Rouge1_score, Rouge2_score, Rougel_score)

