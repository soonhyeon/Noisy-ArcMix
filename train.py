import torch 
from dataloader import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import random 
import numpy as np
import yaml

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print('Configuration...')
    print(cfg)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = './datasets'
    
    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')
    
    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list)
    
    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])
    
    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'])
    
    trainer = Trainer(device=device, name_list=name_list, alpha=cfg['alpha'], mode=cfg['mode'],
                      epochs=cfg['epoch'], class_num=cfg['num_classes'], m=cfg['m'], 
                      lr=cfg['lr'], hidden_dim=cfg['hidden_dim'], 
                      n_layers=cfg['n_layers'], n_heads=cfg['n_heads'], 
                      pf_dim=cfg['pf_dim'], dropout_ratio=cfg['dropout_ratio'])
    
    trainer.train(train_dataloader, valid_dataloader, cfg['save_path'])
    

if __name__ == '__main__':
    main()