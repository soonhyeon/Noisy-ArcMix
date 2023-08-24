import torch
from model.net import TASTgramMFN
from tqdm import tqdm
from utils import get_accuracy, mixup_data, arcmix_criterion, noisy_arcmix_criterion
from losses import ASDLoss, ArcMarginProduct
from torch.cuda.amp import autocast


class Trainer:
    def __init__(self, device, mode, m, alpha, epochs=300, class_num=41, lr=1e-4):
        self.device = device
        self.epochs = epochs
        self.alpha = alpha
        self.net = TASTgramMFN(num_classes=class_num, mode=mode, use_arcface=True, m=m).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.1*float(lr))
        self.criterion = ASDLoss().to(self.device)
        self.test_criterion = ASDLoss(reduction=False).to(self.device)
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Mode should be one of [arcface, arcmix, noisy_arcmix]')
        
        print(f'{mode} mode has been selected...')
        
    def train(self, train_loader, valid_loader, save_path):
        num_steps = len(train_loader)
        min_val_loss = 1e10
        
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            sum_loss = 0.
            sum_accuracy = 0.
            
            for _, (x_wavs, x_mels, labels) in tqdm(enumerate(train_loader), total=num_steps):
                self.net.train()
                
                x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)
                
                with autocast():
                    if self.mode == 'arcface':
                        logits, _ = self.net(x_wavs, x_mels, labels)
                        loss = self.criterion(logits, labels)
                    
                    elif self.mode == 'noisy_arcmix':
                        mixed_x_wavs, mixed_x_mels, y_a, y_b, lam = mixup_data(x_wavs, x_mels, labels, self.device, alpha=self.alpha)
                        logits, _ = self.net(mixed_x_wavs, mixed_x_mels, labels)
                        loss = noisy_arcmix_criterion(self.criterion, logits, y_a, y_b, lam)
                    
                    elif self.mode == 'arcmix':
                        mixed_x_wavs, mixed_x_mels, y_a, y_b, lam = mixup_data(x_wavs, x_mels, labels, self.device, alpha=self.alpha)
                        logits, logits_shuffled, _ = self.net(mixed_x_wavs, mixed_x_mels, [y_a, y_b])
                        loss = arcmix_criterion(self.criterion, logits, logits_shuffled, y_a, y_b, lam)
                
                sum_accuracy += get_accuracy(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                sum_loss += loss.item()
            self.scheduler.step()
                
            avg_loss = sum_loss / num_steps
            avg_accuracy = sum_accuracy / num_steps
            
            valid_loss, valid_accuracy = self.valid(valid_loader)
            
            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                lr = self.scheduler.get_last_lr()[0]
                print("model has been saved!")
                print(f'lr: {lr:.7f} | EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}')
                torch.save(self.net.state_dict(), save_path)
                
    def valid(self, valid_loader):
        self.net.eval()
        
        num_steps = len(valid_loader)
        sum_loss = 0.
        sum_accuracy = 0.
        
        for (x_wavs, x_mels, labels) in valid_loader:
            x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)
            logits, _ = self.net(x_wavs, x_mels, labels, train=False)
            sum_accuracy += get_accuracy(logits, labels)
            loss = self.criterion(logits, labels)
            sum_loss += loss.item()
            
        avg_loss = sum_loss / num_steps 
        avg_accuracy = sum_accuracy / num_steps 
        return avg_loss, avg_accuracy
    
    def test(self, test_loader):
        self.net.eval()
        
        y_true = []
        y_pred = []
        
        sum_accuracy = 0.
        with torch.no_grad():
            for x_wavs, x_mels, labels, AN_N_labels in test_loader:
                x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device), AN_N_labels.to(self.device)
                
                logits, _ = self.net(x_wavs, x_mels, labels, train=False)
                score = self.test_criterion(logits, labels)
                sum_accuracy += get_accuracy(logits, labels)
                
                y_pred.extend(score.tolist())
                y_true.extend(AN_N_labels.tolist())
        auc = metrics.roc_auc_score(y_true, y_pred)
        #pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        return auc, sum_accuracy / len(test_loader)