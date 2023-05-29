# https://gmuedu-my.sharepoint.com/:u:/g/personal/badhika5_gmu_edu/EUlCIlqulltJvh-ZmIjfiRQBDPZ8CiS9SHRN844ovQoWXw?e=gK4WMn to download the BDD dataset

import zipfile
import argparse


parser = argparse.ArgumentParser(description='Unzip the BDD dataset')


parser.add_argument('--bdda-path', default='BDDA.zip',metavar=str, help='path for the directory where to zipped BDDA is located')


def main():
    args = parser.parse_args()  
    with zipfile.ZipFile(args.bdda_path,"r") as zip_ref:
        zip_ref.extractall("")