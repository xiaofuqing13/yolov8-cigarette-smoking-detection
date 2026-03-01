import sys


from PyQt5 import QtCore, QtGui, QtWidgets
import bi

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()    # 创建窗体类对象--窗口类型对象
    ui = bi.Ui_MainWindow()                    # 创建PyQT设计的窗体对象--该类用于初始化任何类型的窗口设置
    ui.setupUi(MainWindow)                  # 初始化MainWindow窗口设置
    MainWindow.show()                       # 显示窗口
    sys.exit(app.exec_())