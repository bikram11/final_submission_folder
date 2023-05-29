Initial Git repo for the machine learning project, 


Setup the virtual environment using the environment.yaml
Or you can set up the learning specific environments as per the requirements under yolov7 repository : https://github.com/WongKinYiu/yolov7.git
__________________________________________________________________________________________________________________________
First start with initial setup scripts under the folder of the same name

the download link for the dataset is provided in unzip_dataset.py
the same file also unzips the folder into the project directory


once downloaded convert video frames to image in convert_to_image.py

when video frames have been converted to image frames proceed with grid map calculation using the create_gridmap_from_grid_image.py
____________________________________________________________________________________________________________________________
Next, download Yolov7 repository from the github https://github.com/WongKinYiu/yolov7.git

Also download the yolov7 pretrained model form the same repository and place it in the root directory

Now place the files under yolov7_specific folder into the yolov7 folder

Run run_yolov7.py, change the parameters in the file as specific to the folder structure that you have


****** Only for the test instance
To extract bounding box labels for the test data, run the commented out section in the run_yolov7.py but do change the folder parameters

___________________________________________________________________________________________________________________________________________

Once the features have been extracted run proceed on to the lstm_code_Section folder

The model code is located in the lstm_model.py folder,

The code to run is test_lstm.py but do change the input parameters as specific to your folder structure

____________________________________________________________________________________________________________________________________________
