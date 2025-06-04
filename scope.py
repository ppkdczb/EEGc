from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import pyqtgraph as pg
import numpy as np
import pygds


class DataAcquisitionThread(QThread):
    """
    采集线程，使用pygds.GDS接口实时获取数据
    采集到数据后发射信号传递给主线程
    """
    data_ready = pyqtSignal(np.ndarray)  # 发送形状为 (channels, samples) 的数据

    def __init__(self, scan_count=500):
        super().__init__()
        self.running = True
        self.scan_count = scan_count
        self.d = pygds.GDS()
        pygds.configure_demo(self.d)
        self.d.SetConfiguration()

    def run(self):
        def more_callback(samples):
            """
            回调函数，samples形状为 (scan_count, channels)
            转置后变为 (channels, scan_count)
            """
            if not self.running:
                return False  # 停止采集

            data = samples.T.copy()[:8, :]  # 转置为 (channels, samples)
            print(data.shape)
            self.data_ready.emit(data)
            return True  # 继续采集

        self.d.GetData(5, more_callback)
        self.d.Close()

    def stop(self):
        del self.d
        self.running = False


class EEGscope(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Real-time Monitor")
        self.setGeometry(100, 100, 1000, 800)

        # 中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 绘图部件
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', 'uV')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)

        # 通道选择
        self.channel_combo = QComboBox()
        self.channels = [f"Channel {i + 1}" for i in range(8)]
        self.channel_combo.addItems(self.channels)
        layout.addWidget(self.channel_combo)

        # 显示参数配置
        self.display_seconds = 5  # 显示时间窗口（秒）
        self.sampling_rate = 250  # 采样率（Hz）
        self.buffer_size = self.display_seconds * self.sampling_rate  # 缓冲区大小

        # 初始化数据缓冲区
        self.data_buffer = np.zeros((8, self.buffer_size))  # 8通道的循环缓冲区
        self.time_axis = np.linspace(-self.display_seconds, 0, self.buffer_size)

        # 初始化绘图曲线
        self.plot_curve = self.plot_widget.plot(pen='y')
        self.plot_widget.setXRange(-self.display_seconds, 0)
        self.plot_widget.setYRange(-1000, 1000)  # EEG信号范围±100uV，调整为合适范围

        # 连接信号
        self.channel_combo.currentIndexChanged.connect(self.update_display)
        self.current_channel = 0

        # 采集线程
        self.data_thread = DataAcquisitionThread(scan_count=250)
        self.data_thread.data_ready.connect(self.receive_data)
        self.data_thread.start()

        # 定时器，用于刷新绘图
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 50Hz刷新率

    def receive_data(self, new_data):
        """
        接收采集线程传来的新数据，更新缓冲区
        new_data形状为 (channels, samples)
        """
        roll_size = new_data.shape[1]
        self.data_buffer = np.roll(self.data_buffer, -roll_size, axis=1)
        self.data_buffer[:, -roll_size:] = new_data

    def update_plot(self):
        """更新绘图数据"""
        self.plot_curve.setData(self.time_axis,
                                self.data_buffer[self.current_channel])

    def update_display(self):
        """切换显示通道"""
        self.current_channel = self.channel_combo.currentIndex()
        # 立即更新显示
        self.plot_curve.setData(self.time_axis,
                                self.data_buffer[self.current_channel])

    def closeEvent(self, event):
        """关闭窗口时，停止采集线程"""
        self.data_thread.stop()
        self.data_thread.wait()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = EEGscope()
    main_window.show()
    sys.exit(app.exec_())
