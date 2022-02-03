import sys
import os

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from dataset import *
from canvas import *
from UI.ui_dataset_viewer import *
from ModelMNIST import *

class AppWindow(Ui_MainWindow):
    def __init__(self, model, dataset):
        super().__init__()
        self.dataset = dataset
        self.model = None
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        self.canvas = Canvas(self.frame, [200, 200], self.dataset, self.lcdLabel)
        self.lcdLabel.display(self.dataset.labels[0])
        
        self.draw_canvas = DrawCanvas(self.drawFrame, [200, 200], 28)
        
        self.nextBtn.clicked.connect(self.canvas.nextImage)
        self.previousBtn.clicked.connect(self.canvas.prevImage)
        self.trainBtn.clicked.connect(self.sig_train)
        self.predictBtn.clicked.connect(self.sig_predict)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.draw_canvas.update)
        self.timer.timeout.connect(self.sig_draw_predict)
        self.timer.start(20)
        

    def sig_train(self):
        self.model = ModelMNIST.train(self.dataset)
    
    def sig_predict(self):
        if self.model != None:
            imgs = self.dataset.images[self.canvas.imgDraw % self.dataset.img_count]
            x = self.model(imgs[tf.newaxis, ..., tf.newaxis].astype("float32"), training=False)
            
            self.lcdAI.display(np.argmax(x))
    
    def sig_draw_predict(self):
        if self.model != None:
            x = self.model(self.draw_canvas.image[tf.newaxis, ..., tf.newaxis].astype("float32"), training=False)
            self.lcdAI.display(np.argmax(x))

if __name__ == '__main__':
    mnist = Dataset('datasets/t10k-labels.idx1-ubyte', 'datasets/t10k-images.idx3-ubyte')
    mnist.load(9999)
    
    model = ModelMNIST()   
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    window = AppWindow(model, mnist)
    window.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
