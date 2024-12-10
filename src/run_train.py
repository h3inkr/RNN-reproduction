import argparse
import os
from tqdm.auto import tqdm
import sacrebleu
from sacremoses import MosesDetokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Seq2seq
from trainer import Trainer
from app import *
from preprocess import make_vocab
from load_data import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    config = load_config()  # app.py
    device = torch.device('cuda:0') #n번 gpu에 연산할당
    print("device: ", device)
    train_data = config["file"]["processed_train"]
    test_data = config["file"]["processed_test"]
    
    model_dir = './save_model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    tb_writer = SummaryWriter(model_dir)  # TensorBoard 시각화

    # load vocabulary 
    vocab_de = make_vocab(config["file"]["vocab_de"])
    id_to_de = dict(zip(vocab_de.values(), vocab_de.keys()))
    print("Vocabulary is loaded!")

    # load data
    en_train, de_in_train, de_out_train = torch.load(train_data)
    train_data = CustomDataset([en_train, de_in_train, de_out_train])
    train_loader = DataLoader(train_data, batch_size=config["train"]["BATCH_SIZE"], pin_memory=True, num_workers=4, drop_last=True, collate_fn=make_padding)

    en_test, de_in_test, de_out_test = torch.load(test_data)
    test_data = CustomDataset([en_test, de_in_test, de_out_test])
    test_loader = DataLoader(test_data, batch_size=config["train"]["BATCH_SIZE"], pin_memory=True, num_workers=4, drop_last=True, collate_fn=make_padding)
    print("Data is loaded!")

    # load model
    model = Seq2seq().to(device)  # model
    print("Model is loaded!")

    # load trainer
    optimizer = torch.optim.SGD(model.parameters(), 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["train"]["GAMMA"])
    
    # option 1: start from scratch
    present_epoch = 0  # Start from epoch 0
    print("Starting training from scratch...")
    
    # option 2: start from ckpt
    '''
    ckpt = torch.load(파일 이름, map_location = device)
    present_epoch = ckpt["epoch"]
    print("Present epoch: ", present_epoch)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.lead_state_dic(ckpt["optimizer_state_dict])
    '''
    
    trainer = Trainer(optimizer, scheduler, device, train_loader, tb_writer, train=True, use_gpu=True)
    #tester = Trainer(optimizer, scheduler, device, test_loader, tb_writer, train=False, use_gpu=True)

    # Initialize lists to store losses for plotting
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(present_epoch, config["train"]["MAX_EPOCH"]):
        print("*" * 20 + f" Epoch: {epoch+1}/{config['train']['MAX_EPOCH']} " + "*" * 20)
        trainer.train_epoch(model, epoch, save_path = model_dir)
        
        '''
        # loss 그래프 그리기
        # Batch-level training with tqdm
        batch_losses = []
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Train on the batch
                optimizer.zero_grad()
                batch_loss = trainer.train_epoch(model, epoch, save_path = model_dir)  # Ensure train_epoch is implemented in Trainer
                batch_losses.append(batch_loss)

                optimizer.step()  # Update weights
                scheduler.step()  # Update learning rate

                # Update tqdm progress bar
                pbar.set_postfix(loss=f"{batch_loss:.4f}")
                pbar.update(1)

        # Epoch-level processing
        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_train_loss)

        # Evaluate on test data
        test_loss = trainer.evaluate(model, test_loader)  # Ensure evaluate is implemented in Trainer
        test_losses.append(test_loss)

        # Log to TensorBoard
        tb_writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        tb_writer.add_scalar('Loss/Test', test_loss, epoch)

        # Print epoch summary
        print(f"Epoch {epoch+1} completed with train loss: {epoch_train_loss:.4f}, test loss: {test_loss:.4f}")

        # Optional: Save the running curve
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(f"{model_dir}/loss_curve_epoch_{epoch+1}.png")
        plt.close()
        '''

    # Finish training
    print("Complete Training!")
    tb_writer.close()

