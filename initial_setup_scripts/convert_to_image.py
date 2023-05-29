import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Convert each video file into corresponding frames')

parser.add_argument('--basepath', default='BDDA/training', metavar=str, help='path for the directory with video files')

parser.add_argument('--camera-image-path', default='raw_data/training/camera_images',metavar=str, help='path for the directory where to store camera image frames')

parser.add_argument('--gazemap-path', default='raw_data/training/gazemap_images',metavar=str, help='path for the directory where to store gazemap image frames')

def main():
    args = parser.parse_args()
    entries = os.listdir(args.basepath)
    entries.sort()

    if not os.path.exists(args.gazemap_path):
        os.makedirs(args.gazemap_path)


    if not os.path.exists(args.camera_image_path):
        os.makedirs(args.camera_image_path)

    for each_folder in entries:
        if "videos" in each_folder:
            source_prefix = each_folder.split('_')
            inner_entries = os.listdir(args.basepath+"/"+each_folder)
            inner_entries.sort()
            for inner_files in inner_entries:
                videocap = cv2.VideoCapture(args.basepath+"/"+each_folder+"/"+inner_files)

                count = 0
                while(True):
                    ret, frame = videocap.read()
                    if ret:
                        if(source_prefix[0]=="gazemap"):
                            cv2.imwrite(args.gazemap_path+'/%d_%d.jpg'%(int(inner_files[:-12]),count),frame)
                        else:
                            cv2.imwrite(args.camera_image_path+'/%d_%d.jpg'%(int(inner_files[:-4]),count),frame)
                        count +=1
                    else:
                        break
                    
                videocap.release()

     
            





