import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# CustomDataset: 데이터셋 인터페이스 제공, DataLoader와 함께 동작하여 배치 생성
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.en_corpus = data_path[0]
        self.de_in_corpus = data_path[1]
        self.de_out_corpus = data_path[2]
        
        self.en_corpus = np.array(self.en_corpus, dtype=object)
        self.de_in_corpus = np.array(self.de_in_corpus, dtype=object)
        self.de_out_corpus = np.array(self.de_out_corpus, dtype=object)
        
    def __len__(self): # 데이터셋의 크기
        return len(self.en_corpus)
    
    def __getitem__(self, idx): # 특정 인덱스의 데이터 반환
        return {
        "encoder": torch.LongTensor(self.en_corpus[idx]),
        "decoder_in": torch.LongTensor(self.de_in_corpus[idx]),
        "decoder_out": torch.LongTensor(self.de_out_corpus[idx])
        }

# collate_fn: 데이터를 미니배치 형태로 정리하고, 각 데이터의 길이가 다를 경우 패딩 적용하기
def collate_fn(batch):
    """
    batch: [{"encoder": tensor, "decoder_in": tensor, "decoder_out": tensor}, ...]
    """
    encoder_inputs = [item["encoder"] for item in batch]
    decoder_inputs = [item["decoder_in"] for item in batch]
    decoder_outputs = [item["decoder_out"] for item in batch]

    # 패딩 작업 (최대 길이에 맞춰 0으로 채움)
    encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
    decoder_outputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)

    return {
        "encoder": encoder_inputs,
        "decoder_in": decoder_inputs,
        "decoder_out": decoder_outputs,
    }
    
def make_padding(samples):
    def padd(samples):
        length = [len(s) for s in samples]
        max_length = 50
        batch = torch.zeros(len(length), max_length).to(torch.long)
        for idx, sample in enumerate(samples):
            if length[idx] < max_length:
                batch[idx, : length[idx]] = torch.LongTensor(sample)
            else:
                batch[idx, :max_length] = torch.LongTensor(sample[:max_length])
        return torch.LongTensor(batch)
    encoder = [sample["encoder"] for sample in samples]
    decoder_in = [sample["decoder_in"] for sample in samples]
    decoder_out = [sample["decoder_out"] for sample in samples]
    encoder = padd(encoder)
    decoder_in = padd(decoder_in)
    decoder_out = padd(decoder_out)
    return {"encoder": encoder.contiguous(), "decoder_in": decoder_in.contiguous(), "decoder_out": decoder_out.contiguous()}
