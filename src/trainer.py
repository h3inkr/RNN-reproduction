import os
import json
from torch.serialization import save
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
import sacrebleu
from sacremoses import MosesDetokenizer
from app import *


class Trainer:
    def __init__(self, optimizer, scheduler, device, data_loader, writer, train=True, use_gpu=True): 
        self.config = load_config()
        self.device = device   
        self.data_loader = data_loader 
        self.train = train
        self.writer = writer
        self.global_step = 0
        self.eval_step = self.config["train"]["eval_step"]
        self.n_layers = self.config["model"]["NUM_LAYERS"]
        self.batch_size = self.config["train"]["BATCH_SIZE"]
        self.hidden_dim = self.config["model"]["HIDDEN_DIM"]
        self.max_sent = self.config["model"]["MAX_LENGTH"]
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gradscaler = amp.GradScaler()
        if self.train == False:
            self.md = MosesDetokenizer(lang='du')
        self.lr = self.config["train"]["lr"]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.embeds = set(['encoder.embedding.weight', 'decoder.embedding.weight'])
        self.biasis = set()
        for i in range(self.n_layers):
            self.biasis.add('encoder.rnn.bias_ih_l{}'.format(i))
            self.biasis.add('encoder.rnn.bias_hh_l{}'.format(i))
            self.biasis.add('decoder.rnn.bias_ih_l{}'.format(i))
            self.biasis.add('decoder.rnn.bias_hh_l{}'.format(i))   

    def log_writer(self, loss, step):
        if self.train == True:
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/lr", lr, step)
            print(f"LOSS: {loss} | LR: {lr}")
        else:
            self.writer.add_scalar("test/loss", loss, step)
            print(f"Test Loss: {loss}")
        
    def train_epoch(self, model, epoch, save_path=None):
        sp = None
        if self.train == True: # train
            model.train()
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
            # reduction
            
        else: # test
            model.eval()
            total_bleu = []
            self.optimizer.zero_grad()
            
        for num, iter in enumerate(tqdm(self.data_loader)):
            with amp.autocast():
                src = iter['encoder'].to(self.device)
                tgt_in = iter['decoder_in'].to(self.device)
                tgt_out = iter['decoder_out'].to(self.device)

                src_len = torch.sum(src != 0, dim=-1).to(self.device)

                if self.train == True: # train
                    h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
                    c_0 = torch.zeros_like(h_0).to(self.device)
                    hidden = (h_0, c_0)
                    hidden = [state.detach().to(self.device) for state in hidden]

                    output = model(src, src_len, tgt_in, hidden)  # output = (batch_size, max_len, vocab_size)
                    loss = self.criterion(output.transpose(1, 2), tgt_out)  # tgt = (batch_size, max_len)
                    
                    self.global_step += 1
                    if self.global_step % self.eval_step == 0:
                        self.log_writer(loss.item(), self.global_step)
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["train"]["clip"])

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                else: # evaluation
                    with torch.no_grad():
                        h_0 = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(self.device)
                        c_0 = torch.zeros_like(h_0).to(self.device)
                        hidden = (h_0, c_0)

                        dec_input = (torch.ones(self.batch_size, 1)*2).to(torch.long).to(self.device)
                        
                        for t in range(self.max_sent):
                            out = model(src, src_len, dec_input, hidden)
                            pred_tokens = torch.max(out, dim=-1)[1]
                            dec_input = torch.cat((dec_input, pred_tokens[:, t].unsqueeze(1)), dim=1)
                            del out, pred_tokens

                        pred_tokens = dec_input[:, 1:]
                        pred_tokens = pred_tokens.detach().tolist()

                        # sentence in mini_batch
                        for i, sentence in enumerate(pred_tokens):  
                            for idx in range(len(sentence)):  # tokens 
                                if sentence[idx] == 3:  # eos
                                    sentence = sentence[:idx]
                                    break
                                
                            pred_tokens = [sp[tok] for tok in sentence]
                            pred_tokens = " ".join(pred_tokens)
                            
                            if num == 0 and i < 2:
                                print("Prediction: ", pred_tokens)
                            
                            # |tgt| = (batch_size, max_length)
                            decode_truth = tgt_out[i, :].cpu().tolist()  
                            for idx in range(len(decode_truth)):
                                if decode_truth[idx] == 3:
                                    decode_truth = decode_truth[1:idx]  
                                    break
                            
                            decode_truth = [sp[tok] for tok in decode_truth]
                            decode_truth = " ".join(decode_truth)

                            if num == 0 and i < 2:
                                print("Reference: ", decode_truth)
                    
                        del h_0, c_0, dec_input
                        
                        self.log_writer(loss.item(), self.global_step)
            
        if self.train == True:
            if epoch > 5: # 6에폭부터,, (baseline)
                self.scheduler.step() # learning rate 업데이트(halve the learning rate every epoch)
                torch.save({"epoch":epoch + 1, # ckpt 파일 저장
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_step": self.scheduler.state_dict()},
                save_path+'/ckpt_{}.pth'.format(epoch))