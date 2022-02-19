import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QPen,QPolygon,QFont
from PyQt5.QtCore import Qt,QPoint,QRectF,QThread
import time
from multiprocessing import Process,Queue


COL_SIZE = 70

BACKGROUND_COLOR_DICT = {0:"#92877d",2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
                         16: "#f59563", 32: "#f67c5f", 64: "#fa5e3f",
                         128: "#edcf72", 256: "#edcc61", 512: "#edc850",
                         1024: "#edc53f", 2048: "#edc22e",

                         4096: "#f02010", 8192: "#edc22e", 16384: "#f2b179",
                         32768: "#f59563", 65536: "#f67c5f", }

CELL_COLOR_DICT = {0:"#f9f6f2",2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2",
                   256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
                   2048: "#f9f6f2",

                   4096: "#f9f6f2", 8192: "#f9f6f2", 16384: "#f9f6f2",
                   32768: "#f9f6f2", 65536: "#f9f6f2", }

class MapWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent=parent)
        self.initUI()
        self.game_map = [[0]*4]*4
        #print("self.game_map",self.game_map)

        # self.game_map = [[0,2,4,8],
        #                  [16,32,64,128],
        #                  [256,512,1024,2048],
        #                  [4096,8192,16384,32768]]
        self.show()

    def initUI(self):
        self.setGeometry(300, 300, COL_SIZE*4, COL_SIZE*4)
        #self.setWindowTitle("2048 Game")


    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.render_map(qp)
        qp.end()



    def render_map(self, qp):
        for i in range(4):
            for j in range(4):
                qp.setBrush(QColor(BACKGROUND_COLOR_DICT[self.game_map[i][j]]))
                rect = QRectF(j*COL_SIZE,i*COL_SIZE,COL_SIZE,COL_SIZE)
                qp.drawRect(rect)
                qp.setPen(QColor(CELL_COLOR_DICT[self.game_map[i][j]]))
                qp.setFont(QFont("Helvetica", 20))
                qp.drawText(rect, Qt.AlignCenter, str(self.game_map[i][j]))

    def set_game_map(self,game_map):
        self.game_map = game_map
        self.update()


class UpdateThread(QThread):
    def __init__(self,game_engine,queue):
        super(UpdateThread, self).__init__()
        self.queue = queue
        self.game_engine = game_engine

    def run(self):
        while True:
            item = self.queue.get()
            self.game_engine.set_game_map(item)


def start_render(queue):
    app = QApplication(sys.argv)
    game_engine = MapWidget()
    thread = UpdateThread(game_engine,queue)
    thread.start()
    app.exec_()

if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # ex = MapWidget()
    # sys.exit(app.exec_())
    start_render(Queue())
    while True:
        time.sleep(1)
