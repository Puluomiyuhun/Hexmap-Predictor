import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QComboBox
from PyQt5.QtGui import QPixmap
from predict import Predictor

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('地形预测器')
        self.resize(250, 60)
        self.init_ui()

    def init_ui(self):
        self.image1_label = QLabel(self)
        self.image2_label = QLabel(self)
        
        self.combobox = QComboBox(self)
        self.populate_combobox()
        self.load_button = QPushButton('选择图片', self)
        self.load_button.clicked.connect(self.load_images)
        self.checkbox = QCheckBox('显示对比结果', self)
        self.label = QLabel('', self)

        vbox = QVBoxLayout()
        vbox.addWidget(self.combobox)
        vbox.addWidget(self.load_button)
        vbox.addWidget(self.checkbox)
        vbox.addWidget(self.label)

        self.setLayout(vbox)

    def populate_combobox(self):
        folder_path = './models'  # 指定文件夹路径
        files = os.listdir(folder_path)
        self.combobox.addItems(files)

    def load_images(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        pre = Predictor(self.combobox.currentText(), self.checkbox.isChecked())
        for i in range(0,len(file_paths)):
            pre.predict(file_paths[i])
            self.label.setText(os.path.basename(file_paths[i] + "已完成"))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
