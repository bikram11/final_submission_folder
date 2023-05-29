# %%
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Create square gridmap txt file from gridmap images')


parser.add_argument('--gazemap-path', default='raw_data/training/gazemap_images',metavar=str, help='path for the directory where to store gazemap image frames')
parser.add_argument('--txtgrid', default='grids/grid1616', type=str, metavar='PATH', help='path to txt with gaze map grid entries')
parser.add_argument('--task', default='train', type=str, metavar='PATH', help='specifies the task between train, test, and validation (val)')

def main():
    args = parser.parse_args()
    # %%
    entries = os.listdir(args.gazemap_path)
    entries.sort()

    # %%
    def form_grid(grid_array, dimensions):
        grid = np.zeros(dimensions[0]*dimensions[1], dtype=float)

        Y,X = np.nonzero(grid_array)

        x_axis = grid_array.shape[1]//dimensions[1]
        y_axis = grid_array.shape[0]//dimensions[0]

        for x,y in zip(X,Y):
            grid_index = y//y_axis * dimensions[1]+ x//x_axis
            grid[grid_index]+=1
        grid = grid/np.sum(grid)

        return grid

    # %%
    def normalize(arr):
        normalized_array = (arr - np.min(arr))/((np.max(arr)-np.min(arr))*1.0)
        return normalized_array

    # %%
    with open(args.txtgrid+"/"+args.task+".txt",'w') as f:
        for each_gazemap in entries:

            grid_prefix_video_number =[]

            grid_prefix_video_number.append(each_gazemap[:-4])

            ground_truth = np.array(Image.open(args.gazemap_path+"/"+each_gazemap).convert('L').crop((0,96,1024,672)))


            normalized_ground_truth = normalize(ground_truth)
            # # print(np.count_nonzero(normalized_ground_truth))
            # plt.imshow(normalized_ground_truth, interpolation='nearest')
            # plt.show()

            ground_truth_grid = form_grid(normalized_ground_truth, [16,16])

            grid_prefix_video_number.extend(ground_truth_grid)

            s = ','.join(map(str,grid_prefix_video_number))
            f.write(s+'\n')


# %%



