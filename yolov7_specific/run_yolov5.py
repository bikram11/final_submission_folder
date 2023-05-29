from subprocess import call
call(["python","yolov5/featuextractor.py", 
      "--weights", "yolov5s.pt", 
      "--save-txt", 
      "--temp_feature_dir", "features_checked/test", 
      "--source", "raw_data/test/camera_images",
      "--device","cpu"])

# !python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source 'raw_data/test/camera_images'