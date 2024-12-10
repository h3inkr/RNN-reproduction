import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from app import *

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout_p, is_reverse=False):
        self.config = load_config()
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout_p, batch_first=True)
        self.reverse_input = is_reverse

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        for layer in self.rnn.all_weights:
            for weight in layer:
                if weight.ndim == 2:  # weight matrix
                    weight.data.uniform_(-0.1, 0.1)
                else:  # bias
                    weight.data.fill_(0)

    def forward(self, src, src_len, hidden):
        if self.reverse_input:
            reversed_indices = torch.arange(src.size(1) - 1, -1, -1).to(src.device)
            src = src[:, reversed_indices]
        x = self.embedding(src)
        x = self.dropout(x)
        packed = pack_padded_sequence(input=x, lengths=src_len.tolist(), batch_first=True, enforce_sorted=False)
        output, h = self.rnn(packed, hidden)
        output_unpacked, _ = pad_packed_sequence(sequence=output, batch_first=True, total_length=src.size(1))
        return output_unpacked, h


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.config = load_config()
        self.att_score = nn.Linear(hidden_dim, self.config["model"]["MAX_LENGTH"], bias=False)
    
    def init_weights(self):
        self.att_score.weight.data.uniform_(-0.1, 0.1)

    def forward(self, h_t, mask):
        '''
        <input>
        |h_t| = (batch_size, sent_len, hidden_dim)
        |mask| = (batch_size, max_sent)
        <output>
        |attn_weight| = (batch_size, sent_len, max_len)
        '''
        attn_weight = self.att_score(h_t)  # |attn_weight| = (batch_size, sent_len, max_len)
        mask = mask.unsqueeze(1)  # (batch_size, 1, max_sent)
        attn_weight.masked_fill_(mask, -float('inf'))  # mask가 0인 부분은 무시
        attn_weight = F.softmax(attn_weight, dim=-1)  # |attn_weight| = (batch_size, sent_len, max_len)
        
        return attn_weight

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout_p):
        self.config = load_config()
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout_p)
        self.attention = Attention(hidden_dim)
        self.linear = nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=False)  # Predict vocabulary distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        for layer in self.rnn.all_weights:
            for weight in layer:
                if weight.ndim == 2:  # weight matrix
                    weight.data.uniform_(-0.1, 0.1)
                else:  # bias
                    weight.data.fill_(0)
        self.attention.init_weights()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, de_input, hidden, en_out, mask):
        x = self.embedding(de_input)  # (batch_size, sent_len, embed_dim)
        x = self.dropout(x)
        h_t, hidden = self.rnn(x, hidden)  # (batch_size, sent_len, hidden_dim), hidden = (h_dec, c_dec)
        attention_weights = self.attention(h_t, mask)
        #print(f"attention_weights shape: {attention_weights.shape}")
        #print(f"en_out shape: {en_out.shape}")
        context = torch.bmm(attention_weights, en_out)
        output = torch.cat((context, h_t), dim = -1)
        output = torch.tanh(self.linear(output))  # Vocabulary prediction
        output = self.dropout(output)
        return output, hidden

'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from app import *

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout_p):
    
        self.config = load_config()
        super(Encoder, self).__init__() # nn.Module의 생성자 상속
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx = 0) # padding index는 0
        self.dropout = nn.Dropout(p = dropout_p)
        self.rnn = nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = n_layers, dropout=dropout_p, batch_first=True)
    
    def init_weights(self):
        # Embedding layer initialization
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1) # our parameters are uniformly initialized in [-0.1, 0.1]

        # LSTM layer initialization
        for i in range(self.config["model"]["NUM_LAYERS"]):
            nn.init.uniform_(getattr(self.rnn, f'weight_ih_l{i}'), a=-0.1, b=0.1) # weight: input - hidden
            nn.init.uniform_(getattr(self.rnn, f'weight_hh_l{i}'), a=-0.1, b=0.1) # weight: hidden - hidden
            nn.init.uniform_(getattr(self.rnn, f'bias_ih_l{i}'), a=-0.1, b=0.1) # bias: input - hidden
            nn.init.uniform_(getattr(self.rnn, f'bias_hh_l{i}'), a=-0.1, b=0.1) # bias: hidden - hidden
        
    def forward(self, src, src_len, hidden):

        x = self.embedding(src) # embedding
        x = self.dropout(x)
        packed = pack_padded_sequence(input = x, lengths = src_len.tolist(), batch_first=True, enforce_sorted=False) # padding
        output, h = self.rnn(packed, hidden) # h(hidden) = (h_0, c_0)
        output_unpacked, _ = pad_packed_sequence(sequence = output, batch_first= True, total_length = self.config["model"]["MAX_LENGTH"] + 2) # total_length : <sos>, MAX_LENGTH, <eos>
        
        return output_unpacked, h
    
class Attention(nn.Module):
    def __init__(self, hidden_dim, max_sent):
        super(Attention, self).__init__()
        self.att_score = nn.Linear(hidden_dim, max_sent, bias=False)
    
    def init_weights(self):
        self.att_score.weight.data.uniform_(-0.1, 0.1)

    def forward(self, h_t, mask):

        attn_weight = self.att_score(h_t)  
        # |attn_weight| = (batch_size, sent_len, max_len)
        mask = mask.unsqueeze(1) 
        # |mask| = (batch_size, 1, max_len)
        attn_weight.masked_fill_(mask, -float('inf'))
        attn_weight = F.softmax(attn_weight, dim=-1) 
        return attn_weight
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, dropout_p, max_len):
        
        self.config = load_config()
        super(Decoder, self).__init__()
        # self.input_feed
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = n_layers, batch_first = True, dropout = dropout_p)
        self.attention = Attention(hidden_dim, max_len)
        self.linear = nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias=False) # fully-connected

    def init_weights(self):
        # Embedding layer initialization
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)
        
        # LSTM layer initialization
        for i in range(self.config["model"]["NUM_LAYERS"]):
            nn.init.uniform_(getattr(self.rnn, f'weight_ih_l{i}'), a=-0.1, b=0.1) # weight: input - hidden
            nn.init.uniform_(getattr(self.rnn, f'weight_hh_l{i}'), a=-0.1, b=0.1) # weight: hidden - hidden
            nn.init.uniform_(getattr(self.rnn, f'bias_ih_l{i}'), a=-0.1, b=0.1) # bias: input - hidden
            nn.init.uniform_(getattr(self.rnn, f'bias_hh_l{i}'), a=-0.1, b=0.1) # bias: hidden - hidden
            
        # Linear layer initialization
        nn.init.uniform_(self.linear.weight, a=-0.1, b=0.1)
    
    def forward(self, de_input, hidden, en_out, mask):
        x = self.embedding(de_input)  # (batch_size, sent_len, embed_dim)
        x = self.dropout(x)

        h_t, hidden = self.rnn(x, hidden)  # (batch_size, sent_len, hidden_dim), hidden = (h_dec, c_dec)
        attention_weights = self.attention(h_t, mask)
        context = torch.bmm(attention_weights, en_out) # attention_weights와 enc_out의 행렬 곱
        output = torch.cat((context, h_t), dim=-1) # context와 h_t를 연결하여 하나의 텐서로 합치기, dim=-1: 각 텐서의 마지막 차원인 hidden_dim을 기준으로 합치기
        output = torch.tanh(self.linear(h_t)) # Linear Transformation + Tanh Activation
        output = self.dropout(output)

        return output, hidden
'''