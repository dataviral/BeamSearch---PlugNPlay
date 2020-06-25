import pandas as pd
from collections import Counter
import torch
import numpy as np

# read csv from path
def load_data(path):
    df = pd.read_csv(path, sep=',', encoding='latin1')
    print("Length of the data: {}".format(len(df)))
    
    df = df.dropna(axis=0)
    print("Length of the data after dropping nan: {}".format(len(df)))
    
    data = []
    for i in range(len(df)):
        summary = df.text.values[i]
        text = df.ctext.values[i]
        
        summary = summary.lower().split(" ")
        text = text.lower().split(" ")
        data.append([summary, text])
    return data

# word to index, index to word mappings and word counts
def get_counts(data):
    def count(word_list, cntr):
        for word in word_list:
            cntr[word] = cntr.get(word, 0) + 1

    freq_word_counter = dict()

    for i in range(len(data)):
        summary, text = data[i]
        count(summary, freq_word_counter)
        count(text, freq_word_counter)
    
    freq_word_counter = Counter(freq_word_counter)
    return freq_word_counter

def map_data(data, max_words, cntr):
    
    w2i = dict()
    i2w = dict()

    # Most common max_words 
    common_words = cntr.most_common(max_words)
    w2i = {j[0]: i for i, j in enumerate(common_words)}
    i2w = {i: j[0] for i, j in enumerate(common_words)}

    # Add SOS, EOS, UNK and PAD 
    SOS_ID, EOS_ID, UNK_ID, PAD_ID = max_words, max_words + 1, max_words + 2, max_words + 3
    SOS_TKN, EOS_TKN, UNK_TKN, PAD_TKN = "<sos>", "<eos>", "<unk>", "<pad>"

    w2i[SOS_TKN], i2w[SOS_ID] = SOS_ID, SOS_TKN
    w2i[EOS_TKN], i2w[EOS_ID] = EOS_ID, EOS_TKN
    w2i[UNK_TKN], i2w[UNK_ID] = UNK_ID, UNK_TKN
    w2i[PAD_TKN], i2w[PAD_ID] = PAD_ID, PAD_TKN
    
    for i in range(len(data)):
        data[i][0] = [w2i.get(word, UNK_ID) for word in data[i][0]]
        data[i][1] = [w2i.get(word, UNK_ID) for word in data[i][1]] 
    
    return w2i, i2w


def get_data(path, num_words):
    # Read data from csv
    data = load_data(path)

    # Count word occurrences 
    cntr = get_counts(data)

    # Map data
    w2i, i2w = map_data(data, num_words, cntr)

    return data, w2i, i2w


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, w2i, max_x_len=100, max_y_len=100, isTrain=False):
        self.data = data
        self.w2i = w2i
        self.max_x_len = max_x_len
        self.max_y_len = max_y_len
        self.isTrain = isTrain
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.isTrain:
            summary, text = self.data[idx]
        else:
            text = self.data[idx]

        # Trim to max_len (account for extra eos and sos)
        text = [self.w2i["<sos>"]] + text[:self.max_x_len - 2] + [self.w2i["<eos>"]]
        if self.isTrain:
            summary = [self.w2i["<sos>"]] + summary[:self.max_y_len - 2] + [self.w2i["<eos>"]]
        
        if self.isTrain:
            return torch.Tensor(text), torch.Tensor(summary)
        else:
            return torch.Tensor(text)

def collate_fn_train(data):
    text, summary = zip(*data)

    text_lens = [len(i) for i in text]
    summary_lens = [len(i) for i in summary]

    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    summary = torch.nn.utils.rnn.pad_sequence(summary, batch_first=True)

    return (text, text_lens), (summary, summary_lens)
        
def collate_fn_test(texts):

    text_lens = [len(i) for i in texts]

    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)

    return texts, text_lens

def index2Sent(prediction, i2w):
    return " ".join([i2w[tkn] for tkn in prediction])
