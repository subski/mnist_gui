from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np

class Canvas(QOpenGLWidget):
    def __init__(self, parent, size, mnist, lcd):
        super().__init__(parent)

        self.size = size
        self.mnist = mnist
        self.lcd = lcd

        self.setGeometry(QRect(0, 0, size[0], size[0]))

        self.imgDraw = 0

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)

        qp.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        qp.drawLine(200, 0, 200, 200)

        qp.setBrush(QColor(200, 0, 0))
        qp.drawRect(0, 0, 10, 10)

        
        qp.setBrush(QColor(200, 0, 0))
        qp.drawRect(190, 190, 210, 210)

        tab = []

        grid_length = self.mnist.img_size[0]
        square_size = self.size[0] / grid_length
        
        image = self.mnist.images[self.imgDraw % self.mnist.img_count]
        # print(image)

        for r in range(grid_length):
            for c in range(grid_length):
                qp.setBrush(QColor(image[r][c]*255, image[r][c]*255, image[r][c]*255))
                qp.drawRect(int(c * square_size), int(r * square_size), int(c * square_size + square_size), int(r * square_size + square_size) )

        qp.end()
    
    def update(self):
        self.lcd.display(self.mnist.labels[self.imgDraw % self.mnist.img_count])
        
        super().update()
    
    def nextImage(self):
        self.imgDraw += 1
        self.update()
    
    def prevImage(self):
        self.imgDraw -= 1
        self.update()

class DrawCanvas(QOpenGLWidget):
    def __init__(self, parent, size, res):
        super().__init__(parent)

        self.size = size
        self.setGeometry(QRect(0, 0, size[0], size[0]))
        
        self.resolution = res
        
        self.mouse = [0, 0]
        self.isPressed = False
        
        self.image = np.zeros((res, res))
                
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.isPressed = True
        elif event.button() == Qt.RightButton:
            self.image = np.zeros((self.resolution, self.resolution))
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.isPressed = False
    
    def mouseMoveEvent(self, event):
        if self.isPressed:
            x = event.pos().x()
            y = event.pos().y()
                        
            self.image[int(y/self.size[0]*self.resolution)][int(x/self.size[0]*self.resolution)] = 1
    
    def update(self):
        super().update()
    
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        
        square_size = self.size[0] / self.resolution
        
        for r in range(self.resolution):
            for c in range(self.resolution):
                qp.setBrush(QColor(self.image[r][c]*255, self.image[r][c]*255, self.image[r][c]*255))
                qp.drawRect(int(c * square_size), int(r * square_size), int(c * square_size + square_size), int(r * square_size + square_size) )
        qp.end()