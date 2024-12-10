import torch
import torch.nn as nn
from module import *
from app import *

class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.config = load_config()
        vocab_size = self.config["model"]["VOCAB_SIZE"]
        embed_dim = self.config["model"]["EMBED_DIM"]
        hidden_dim = self.config["model"]["HIDDEN_DIM"]
        num_layers = self.config["model"]["NUM_LAYERS"]
        dropout_p = self.config["model"]["DROPOUT_P"]
        max_len = self.config["model"]["MAX_LENGTH"] # we filter out sentence pairs whose lengths exceed 50 words
        is_reverse = self.config["model"]["IS_REVERSE"]
        
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout_p, is_reverse)
        self.decoder = Decoder(vocab_size, hidden_dim, num_layers, dropout_p)
        self.linear = nn.Linear(in_features = hidden_dim, out_features = vocab_size, bias=False)
        self.init_weights()
        
    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()
        self.linear.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, src, src_len, de_input, hidden): 
        en_out, hidden = self.encoder(src, src_len, hidden) # en_out -> 원래는 decoder input
        mask = src == 0 # initialization
        de_out, hidden = self.decoder(de_input, hidden, en_out, mask)
        output = self.linear(de_out)
        return output