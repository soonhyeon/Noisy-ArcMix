import torch 
import os 
import torchaudio
import glob
import itertools
import numpy as np
import sys
from tqdm import tqdm


def file_to_log_mel_spectrogram(y, sr, n_mels, n_fft, hop_length, power):
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels, power=power, pad_mode='constant', norm='slaney', mel_scale='slaney')
    mel_spectrogram = transform(y)
    log_mel_spectrogram = 20.0 / power * torch.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, test_name, name_list):
        dataset_dir = os.path.join(root_path, test_name, 'test')
        normal_files = sorted(glob.glob('{dir}/normal_*'.format(dir=dataset_dir)))
        anomaly_files = sorted(glob.glob('{dir}/anomaly_*'.format(dir=dataset_dir)))
        
        self.test_files = np.concatenate((normal_files, anomaly_files), axis=0)
        
        normal_labels = np.zeros(len(normal_files))
        anomaly_labels = np.ones(len(anomaly_files))
        self.y_true = torch.LongTensor(np.concatenate((normal_labels, anomaly_labels), axis=0))
        
        target_idx = name_list.index(test_name)
        
        label_init_num = 0
        for i, name in enumerate(name_list):
            if i == target_idx:
                break
            label_init_num+=len(self._get_label_list(name))
            
        self.labels = []
        label_list = self._get_label_list(test_name)
        for file_name in self.test_files:
            for idx, label_idx in enumerate(label_list):
                if label_idx in file_name:
                    self.labels.append(idx + label_init_num)
        
        self.labels = torch.LongTensor(self.labels)
        
        
        self.y_list = []
        self.y_spec_list = []
        
        for i in tqdm(range(len(self.test_files))):
            y, sr = self._file_load(self.test_files[i])
            y_specgram = file_to_log_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128, power=2)
            self.y_list.append(y)
            self.y_spec_list.append(y_specgram)
    
    def __getitem__(self, idx):
        anomal_label = self.y_true[idx]
        label = self.labels[idx]
        return self.y_list[idx], self.y_spec_list[idx], label, anomal_label 

    def __len__(self):
        return len(self.test_files)
    
    def _file_load(self, file_name):
        try:
            y, sr = torchaudio.load(file_name)
            y = y[..., :sr * 10]
            return y, sr
        except:
            print("file_broken or not exists!! : {}".format(file_name))
    
    def _get_label_list(self, name):
        if name == 'ToyConveyor':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06'] 
    
        elif name == 'ToyCar':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']
        
        else:
            label_list = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
            
        return label_list 


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, name_list): 
        data_path = [os.path.join(root_path, name, 'train') for name in name_list]
        
        files_list= [self._file_list_generator(target_path) for target_path in data_path]
        
        self.labels = []

        maximum = 0
        for i, files in enumerate(files_list):
            label_list = self._get_label_list(name_list[i])
            for file_name in files:
                for idx, label_idx in enumerate(label_list):
                    if label_idx in file_name:
                        self.labels.append(idx + maximum)
            maximum = max(self.labels)+1

        self.unrolled_files_list = list(itertools.chain.from_iterable(files_list))
        self.labels = torch.LongTensor(self.labels)
        
        self.y_list = []
        self.y_spec_list = []
        
        for i in tqdm(range(len(self.unrolled_files_list))):
            y, sr = self._file_load(self.unrolled_files_list[i])
            y_specgram = file_to_log_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128, power=2)
            self.y_list.append(y)
            self.y_spec_list.append(y_specgram)
    
    def __getitem__(self, idx):
        return self.y_list[idx], self.y_spec_list[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.unrolled_files_list)
      
    def _get_label_list(self, name):
        if name == 'ToyConveyor':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06'] 
    
        elif name == 'ToyCar':
            label_list = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07']
        
        else:
            label_list = ['id_00', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
            
        return label_list 
    
    def _file_list_generator(self, target_dir):
        training_list_path = os.path.abspath('{dir}/*.wav'.format(dir=target_dir))
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            print('no_wav_file!!')
        return files
    
    def _file_load(self, file_name):
        try:
            y, sr = torchaudio.load(file_name)
            y = y[..., :sr * 10]
            return y, sr
        except:
            print("file_broken or not exists!! : {}".format(file_name))


if __name__ == '__main__':
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    
    root_path = './datasets'
    test_dataset = test_dataset(root_path, name_list[0], name_list)
    
    




