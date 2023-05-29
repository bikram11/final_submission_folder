import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from grid_sequencer import GridSequencer

class BDD_Dataloader(Dataset):

    def __init__(self, subset, file, feature_path, threshold, gazemap_path, seqlen):
        self.subset = subset
        self.file = file
        self.feature_path = feature_path
        self.gazemap_path = gazemap_path
        self.threshold = threshold
        self.mean = torch.zeros(1024)
        self.std = torch.ones(1024)
        self.seqlen = seqlen
        self._parse_list()
        self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([36,64]),
                torchvision.transforms.ToTensor()])

    def _parse_list(self):
        self.img_list = []
        self.img_dict = {}

        with open(self.file, 'r') as f:
            tmp = [x.strip().split(',') for x in f.readlines()]

        for item in tmp:
            img_name = item[0].split('.')[0]
            feature_name = img_name + ".pt"
            grid = [float(x) for x in item[1:]]

            feature_path = os.path.join(self.feature_path, feature_name)
            if os.path.exists(feature_path) and not all(math.isnan(y) for y in grid):
                self.img_list.append(GridSequencer(item))

                clip = item[0].split('.')[0].split('_')[0]
                img_nr = item[0].split('.')[0].split('_')[1]
                if clip not in self.img_dict:
                    self.img_dict[clip] = []
                self.img_dict[clip].append(img_nr)
            else:
                print('Error loading feature:', feature_path)

        for key in self.img_dict:
            self.img_dict[key].sort()

        print('Video number in %s: %d' % (self.subset, len(self.img_list)))




    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        record = self.img_list[index]
        img_name = record.img_number.split('.')[0]
        feature_name = img_name + ".pt"
        feature_path = os.path.join(self.feature_path, feature_name)
        feature = torch.load(feature_path)


        clip = record.img_number.split('.')[0].split('_')[0]
        img_nr = record.img_number.split('.')[0].split('_')[1]
        dict_idx = self.img_dict[clip].index(img_nr)
        # create list with previous features, last one is original
        feature_list = []
        first = dict_idx - (self.seqlen - 1)
        
        for idx in range(first, dict_idx+1):
            feature_name2 = clip + '_' + self.img_dict[clip][idx] + ".pt"
            feature_path2 = os.path.join(self.feature_path, feature_name2)
            feature2 = torch.load(feature_path2)
            feature_list.append(feature2)
        feature = torch.stack(feature_list)

        grid = np.array(record.grid_matrix)
        grid[grid > self.threshold] = 1.0
        grid[grid <= self.threshold] = 0.0
        grid = grid.astype(np.float32)

        if self.subset == 'test':
            gaze_gt = Image.open(os.path.join(self.gazemap_path, record.img_number + '.jpg')).convert('L').crop((0, 96, 1024, 672)) #left, top, right, bottom
            gaze_gt = self.transform(gaze_gt)
            return feature, grid, gaze_gt, img_name
        else:
            return feature, grid

