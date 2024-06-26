# Hexmap-Predictor
A VGG-16 network to predict the hexmap id of each square in CCZ gaming maps.

![Example Image](https://puluo.top/show/img/r8.bmp)

# The environment
Cuda:    12.1<br />
Python:  3.11<br />
Pytorch: 2.1.0<br />

# How to predict the hexmap of gaming maps?
1. Place your .pt model files into 'models' folders.<br />
   You can download a new_style model and a old_style model I have trained in https://pan.baidu.com/s/1aAVgn2xXw4dFYHoT6kOtdA?pwd=lcfg , the password is lcfg.
2. Run the 'main.py', and you would find a window.
3. Select the model file you want to use, select the images you want to predict.
4. Finally, you would find the hexmap result and the compare result files in your image folders.

# How to train my own dataset?
1. Pick all the gaming maps of one mod into 'origin_map' folder, and then pick the 'hexmap.e5' file into 'origin_map' folder too.
2. In the 'DataMaker.py', you can modify the 'mod_id', such as 0. And then run the 'DataMaker.py'.<br />
   In the 'training_data' folder, you would find that each map will be splited into 48×48 images, and a new file called 'train.csv' would be created to save the hex label of each splited map.
3. If you want to pick the maps of second or more mod, you should update the 'origin_map' folder and change the 'mod_id' in 'DataMaker.py'. This operation is to prevent that new mod maps cover the old map data.
4. Run the 'train.py'. Some argument should be stated:<br />
    --batch_size    &emsp;   the number of images in one batch.<br />
    --num_workers   &emsp;   the number of workders to deal with the dataloader.<br />
    --epoch_num     &emsp;   how many epoches will be trained.<br />
    --batch_print   &emsp;   how many batches to print the loss result.<br />
    --epoch_save    &emsp;   how many epoches to save the model weights.<br />
    --resume_train  &emsp;   if you want to resumt a training task, you should state the path of the last model weight file.<br />

   
