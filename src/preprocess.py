import torch
from tqdm.auto import tqdm
from mosestokenizer import *
from app import *
import pickle

# Map vocabulary: 단어 사전 생성(단어를 정수 인덱스 형태로 반환): vocab.50k.en, vocab.50k.de ➡️ vocab_map_en, vocab_map_de
def make_vocab(vocab_file):
    with open(vocab_file) as f:
        whole_file = f.readlines()
        words = [word.strip('\n') for word in whole_file] # 줄바꿈 문자(\n) 제거
        vocab_map = {} # dictionary
        vocab_map["<pad>"] = 0
        vocab_map["<unk>"] = 1
        vocab_map["<sos>"] = 2
        vocab_map["<eos>"] = 3
        for word in words:
            vocab_map[word] = len(vocab_map)
            
    print("Complete mapping vocabulary!\n")
    
    return vocab_map 

# Make dataset
def load_data(en_file, de_file, vocab_en, vocab_de):
    en_corpus = []
    de_in_corpus = []
    de_out_corpus = []
    
    with open(en_file, 'r', encoding='utf-8') as f_en:
        en_file = f_en.readlines()
    with open(de_file, 'r', encoding='utf-8') as f_de:
        de_file = f_de.readlines()
        
    # 길이가 다른 파일을 처리하기 위해 min 길이를 사용
    min_length = min(len(en_file), len(de_file))
    
    with MosesTokenizer('en') as en_tokenizer:
        with MosesTokenizer('de') as de_tokenizer:
            for sentence in tqdm(range(min_length)):  # 짧은 파일 길이에 맞춤
                en_tokenized = en_tokenizer(en_file[sentence])
                de_tokenized = de_tokenizer(de_file[sentence])

                if (len(en_tokenized) > 50) or (len(de_tokenized) > 50): # we filter out sentence pairs whose lengths exceed 50 words
                    continue
                if (len(en_tokenized) == 0) or (len(de_tokenized) == 0):
                    continue
                
                en_indices = [vocab_en[en_token] if en_token in vocab_en else vocab_en['<unk>'] for en_token in en_tokenized]
                de_indices = [vocab_de[de_token] if de_token in vocab_de else vocab_de['<unk>'] for de_token in de_tokenized]
                
                en_corpus.append(en_indices)
                de_in_corpus.append([vocab_de['<sos>']] + de_indices)
                de_out_corpus.append(de_indices + [vocab_de['<eos>']])
                
    print("Complete making dataset!\n")
                
    return en_corpus, de_in_corpus, de_out_corpus

if __name__ == '__main__':
    config = load_config()
    
    vocab_map_en = make_vocab(config["file"]["vocab_en"])
    vocab_map_de = make_vocab(config["file"]["vocab_de"])
    
    train_data = load_data(config["file"]["train_en"], config["file"]["train_de"], vocab_map_en, vocab_map_de)
    test_data = load_data(config["file"]["test_en"], config["file"]["test_de"], vocab_map_en, vocab_map_de)
    valid_data = load_data(config["file"]["valid_en"], config["file"]["valid_de"], vocab_map_en, vocab_map_de)
    
    pickle.dump(train_data, open(f'./data/preprocessed/train_data.pkl', 'wb'))
    pickle.dump(test_data, open(f'./data/preprocessed/test_data.pkl', 'wb'))
    pickle.dump(valid_data, open(f'./data/preprocessed/valid_data.pkl', 'wb'))