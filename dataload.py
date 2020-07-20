import os
from torch.utils.data import Dataset
from PIL import Image


class CreateNiiDataset(Dataset):
    def __init__(self,dirname,transform=None):
        self.pathdir = dirname
        self.imagelist = os.listdir(dirname)
        self.transform = transform
    def __getitem__(self, item):
        # target has no transform
        # because we want the images have shape[N,C,H,W] but the masks have shape[N,H,W]
        if self.transform :
            return self.transform(Image.open(self.pathdir+self.imagelist[item]))
        else:
            return Image.open(self.pathdir+self.imagelist[item])

    def __len__(self):
        length = len(self.imagelist)
        return length