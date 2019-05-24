##################################################
This belongs to Britton Chance Center for Biomedical Photonics, Wuhan National Laboratory for Optoelectronics-Huazhong University of Science and Technology, Hubei, China
Author: Huang

Usage:
1.First use the Split_Dataset.py file in the "dataset" folder to generate your own 3D patches(which are random chosen with defined rules).
  Replace the first_train_path and first_val_path using your own image train_path and val_path.
  "Image" and "Lable" folders are contained in the train or val path folder.
  The coressponding image and label share the same filename, and are placed in the "Image" and "Lable" folder respectively.
  Format: tifffile.

2. Train your own network:
   Use the Train_UnSupervise.py to train the network.
   The corresponding parameters are set in the config.py.

3. Retrain the network with refined labels .
   The Generated_Refined_Samples.py is used for label refinement.
   Using the refined labels to replace the original label.
   Then Run Train_UnSupervise.py iteratively.

4.Prediction for new dataset. Use Predict_10Data.py.

You may need to change the default settings for your own dataset, including batch_size, num_workers.



  

