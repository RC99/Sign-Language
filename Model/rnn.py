from fastai.vision.all import *
from torch.nn import CrossEntropyLoss

class FrameSequencesDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.classes = sorted(os.listdir(data_folder))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform
    
    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.data_folder))
    
    def __getitem__(self, idx):
        class_folders = os.listdir(self.data_folder)
        class_folder = class_folders[idx % len(class_folders)]
        class_idx = self.class_to_idx[class_folder]
        
        sequence_files = os.listdir(os.path.join(self.data_folder, class_folder))
        sequence_file = sequence_files[idx // len(class_folders)]
        sequence_path = os.path.join(self.data_folder, class_folder, sequence_file)
        
        frames = []  # Load frames from sequence_path
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        return frames, class_idx

def create_dataloaders(data_folder):
    data = DataBlock(blocks=(ImageBlock, CategoryBlock),
                     get_items=get_image_files,
                     splitter=RandomSplitter(valid_pct=0.2, seed=42),
                     get_y=parent_label,
                     item_tfms=Resize((224, 224)),
                     batch_tfms=None)
    
    dls = data.dataloaders(data_folder)
    return dls

class FrameLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(FrameLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size() if len(x.size()) == 5 else (*x.size(), 1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    
        x = x.view(batch_size, seq_len, -1)
    
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    data_folder = '/Users/reetvikchatterjee/Desktop/Dataset'
    dls = create_dataloaders(data_folder)

    model = FrameLSTM(input_size=224*224, hidden_size=128, num_layers=1, num_classes=len(dls.vocab))
    learn = Learner(dls, model, loss_func=CrossEntropyLoss(), metrics=accuracy)

    learn.fit_one_cycle(5)
