# Hexmap-Predictor
A VGG-16 network to predict the hexmap id of each square in CCZ gaming maps.

![Example Image]([https://example.com/example.png](https://puluo.top/show/img/r8.bmp))

# The environment
Cuda:    12.1<br />
Python:  3.11<br />
Pytorch: 2.1.0<br />

# How to predict the hexmap of gaming maps?
1. Place your .pt model files into 'models' folders.
2. Run the 'main.py', and you would find a window.
3. Select the model file you want to use, select the images you want to predict.
4. Finally, you would find the hexmap result and the compare result files in your image folders.

# How to train my own dataset?
1. Pick all the gaming maps of one mod into 'origin_map' folder, and then pick the 'hexmap.e5' file into 'origin_map' folder too.
2. In the 'DataMaker.py', you can modify the 'mod_id', such as 0. And then run the 'DataMaker.py'.<br />
   In the 'training_data' folder, you would find that each map will be splited into 48×48 images, and a new file called 'train.csv' would be created to save the hex label of each splited map.
3. If you want to pick the maps of second or more mod, you should update the 'origin_map' folder and change the 'mod_id' in 'DataMaker.py'. This operation is to prevent that new mod maps cover the old map data.
4. Run the 'train.py'. Some argument should be state:<br />
    --batch_size       the number of images in one batch.<br />
    --num_workers      the number of workders to deal with the dataloader.<br />
    --epoch_num        how many epoches will be trained.<br />
    --batch_print      how many batches to print the loss result.<br />
    --epoch_save       how many epoches to save the model weights.<br />
    --resume_train     if you want to resumt a training task, you should state the path of the last model weight file.<br />

   
