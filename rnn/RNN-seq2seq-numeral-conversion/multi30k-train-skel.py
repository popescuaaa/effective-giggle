import random
import time
import math
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Encoder, Decoder, Seq2Seq
from utils import load_configurations
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy

"""
For the requested datasets we need tokenization from spacy:
- for this we need to download the spacy cores for fr and en

python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

"""

"""
Method to initialize the weights of a model to have random values in between -0.08 and 0.08
"""


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


"""
Auxiliary methods to count the parameters of the model and return time elapsed between two epochs
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    """
    Main training function
    :param model: The used seq2seq model
    :param iterator: The dataset iterator used to produce batches of training data
    :param optimizer: The used optimizer
    :param criterion: The loss criterion
    :param clip: clip treshold for gradient stabilization
    """
    # set the model in training mode
    model.train()
    epoch_loss = 0

    # main training loop
    for _, (src, trg) in enumerate(iterator):
        # get batch of source (arabic numeral) and target (roman numeral) sequences
        src, trg = src.to(torch.device("cpu")), trg.to(torch.device("cpu"))

        optimizer.zero_grad()

        # Tensor shape indications:
        #   trg = [trg len, batch size]
        #   output = [trg len, batch size, output dim]
        # TODO: get the output predicted by the model
        output = model(src, trg)

        # TODO: Create views of the output and target tensors so as to apply the CrossEntropyLoss criterion
        #   over all tokens in the target sequence, ignoring the first. See torch.tensor.view()
        #   trg = [(trg len - 1) * batch size]
        #   output = [(trg len - 1) * batch size, output dim]
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # TODO: apply the CrossEntropyLoss between output and trg
        loss = criterion(output, trg)

        loss.backward()

        # Clip gradients if their norm is too large to ensure training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Perform one step of optimization
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """
    Method to evaluate a trained model
    :param model:
    :param iterator:
    :param criterion:
    :return:
    """
    # set model in evaluation mode
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            # Tensor shape indications
            #   trg = [trg len, batch size]
            #   output = [trg len, batch size, output dim]
            src, trg = src.to(torch.device("cpu")), trg.to(torch.device("cpu"))

            # TODO: get the output predicted by the model, WITHOUT applying teacher forcing
            output = model(src, trg, 0)

            # TODO: Obtain views of the output and target tensors as in the training case
            #   trg = [(trg len - 1) * batch size]
            #   output = [(trg len - 1) * batch size, output dim]
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # TODO: apply the CrossEntropy loss criterion
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=False, default="multi30kconfig.yml")
    args = arg_parser.parse_args()

    config = load_configurations(args.config)

    # Dataset

    # Prepare data fro translation task
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.fr.gz', 'train.en.gz')
    val_urls = ('val.fr.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.fr.gz', 'test_2016_flickr.en.gz')

    spacy_fr = spacy.load('fr_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    fr_tokenizer = lambda text: [tok.text for tok in spacy_fr.tokenizer(text)][::-1]

    en_tokenizer = lambda text: [tok.text for tok in spacy_en.tokenizer(text)]

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    fr_vocab = build_vocab(train_filepaths[0], fr_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)


    def data_process(filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_de)],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                      dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data


    train_data = data_process(train_filepaths)
    val_data = data_process(val_filepaths)
    test_data = data_process(test_filepaths)

    PAD_IDX = fr_vocab['<pad>']
    BOS_IDX = fr_vocab['<bos>']
    EOS_IDX = fr_vocab['<eos>']


    def generate_batch(data_batch):
        fr_batch, en_batch = [], []
        for (fr_item, en_item) in data_batch:
            fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return fr_batch, en_batch


    SEED = config["SEED"]
    INPUT_DIM = len(fr_vocab)
    OUTPUT_DIM = len(en_vocab)

    ENC_EMB_DIM = config["ENC_EMB_DIM"]
    DEC_EMB_DIM = config["DEC_EMB_DIM"]
    HID_DIM = config["HID_DIM"]
    N_LAYERS = config["N_LAYERS"]
    ENC_DROPOUT = config["ENC_DROPOUT"]
    DEC_DROPOUT = config["ENC_DROPOUT"]

    BATCH_SIZE = config["BATCH_SIZE"]
    N_EPOCHS = config["N_EPOCHS"]
    CLIP = config["CLIP"]

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=generate_batch)

    # set the randomness SEED to ensure repeatable results across runs of the script
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # set the device to run on
    device = torch.device('cpu')

    # create and parameterize encoder and decoder models
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    # create seq2se1 model and initialize its weights
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print(model)

    # define the optimizer
    optimizer = optim.Adam(model.parameters())


    # The loss criterion is the CrossEntropyLoss
    TRG_PAD_IDX = en_vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Run the training over several epochs
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'EPOCH: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTRAIN Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

        if (epoch + 1) % 10 == 5:
            # test the current model
            test_loss = evaluate(model, valid_iter, criterion)
            print(f'\tTEST Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # Save the trained model to be
    model_name = "multi30k-fr-en-model-sort_by_src_all-batch_%s-epochs_%s-dropout_%s.pt" \
                 % (str(BATCH_SIZE), str(N_EPOCHS), str(DEC_DROPOUT))
    torch.save(model.state_dict(), 'models/' + model_name)

    # Test the model at the end
    test_loss = evaluate(model, test_iter, criterion)
    print(f'| TEST Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
