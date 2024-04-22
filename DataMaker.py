import cv2
import os

current_path = os.getcwd()
map_path = current_path + "/origin_map"
mod_id = 0

with open(current_path + "/training_data/train.csv", "a+") as label:
    with open(map_path + "/Hexzmap.e5", "rb") as hex:
        #遍历每一张图片
        for root, dirs, files in os.walk(map_path):
            for file in files:
                #找到这张图片的地形数据地址
                map_id = int(file[1]) * 100 + (int(file[2])  * 10) + int(file[3]) 
                hex.seek(0x110 + 12 * map_id + 8)
                addr = int.from_bytes(hex.read(4), byteorder='big')
                addr += 2

                #开始分解每一块地形像素
                img = cv2.imread(map_path + "/" + file)
                height, width, _ = img.shape
                sum = 0
                for i in range(int(height / 48)):
                    for j in range(int(width / 48)):
                        split_img = img[i*48:(i+1)*48, j*48:(j+1)*48]
                        file_name = str(mod_id) + "_" + str(map_id) + "_" + str(sum) + ".bmp"
                        cv2.imwrite(current_path + "/data/" + file_name, split_img)

                        hex.seek(addr + sum)
                        label_res = hex.read(1)
                        label.write(file_name + "," + str(ord(label_res)) + "\n")
                        sum += 1
