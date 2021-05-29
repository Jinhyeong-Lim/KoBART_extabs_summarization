from load_data import data_load
from train import train
from data_preprocessing import data_preprocessing
from evaluation import eval
from make_pseudo_extractive_data import textrank
from make_pseudo_extractive_data import principal
from make_pseudo_extractive_data import lead_n
import argparse
import torch
from kobart_transformers import get_kobart_tokenizer
from kobart_transformers import get_kobart_for_conditional_generation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    # load reference data
    ext_data, abs_data = data_load()

    # stragety (make psuedo extractive data)
    if args.strategy == "textrank":
        ext_data = textrank()
    elif args.strategy == "lead_n":
        ext_data = lead_n()
    elif args.strategy == "principal":
        ext_data = principal()

    # data preprocessing
    ext_train_loader, ext_valid_loader, ext_test_loader = \
        data_preprocessing(ext_data)

    abs_train_loader, abs_valid_loader, abs_test_loader =  \
        data_preprocessing(abs_data)

    # # load Pretrained model, tokenizer
    tokenizer = get_kobart_tokenizer()
    model = get_kobart_for_conditional_generation()
    model.to(device)
    a = args.ext_epochs   # extractive summarization 학습 횟수
    b = args.abs_epochs   # abstractive summarization (k-a) 학습 횟수

    # hyperparameter
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    torch.manual_seed(args.seed)

    # train
    # 1차 미세조정
    model = train(ext_train_loader, ext_valid_loader, a, model, tokenizer,
                  optimizer, device)
    # 2차 미세조정
    model = train(abs_train_loader, abs_valid_loader, b, model, tokenizer,
                  optimizer, device)

    # evaluation
    rouge1_score, rouge2_score, rougel_score = eval(model, tokenizer,
                                                    abs_test_loader, device)

    # print performance
    print(rouge1_score, rouge2_score, rougel_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='It is code for Abstractive '
                                                 'Summarization')
    parser.add_argument("--strategy", type=str,
                        help="How to make Extracitve summary",
                        choices=['textrank', 'principal', 'lead'], default=None)
    parser.add_argument("--ext_epochs", type=int,
                        help="Enter the number of " 
                             "Extractive Summarization training epochs")
    parser.add_argument("--abs_epochs", type=int,
                        help="Enter the number of " 
                             "Abstractive Summarization training epochs")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    main(args)
