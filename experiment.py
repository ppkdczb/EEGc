import sys
import time
import numpy as np
from threading import Lock
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, pyqtSlot, QTimer)  # Added QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QVBoxLayout, QWidget, QMessageBox)  # Added QMessageBox
from PyQt5.QtGui import QPixmap, QPainter
import pickle
import os  # For path operations
import datetime  # For timestamped filenames
import random

# --- Configuration ---
USE_GDS_DEVICE = False  # << SET THIS TO True TO USE REAL G-TEC DEVICE, False FOR SIMULATION

# GDS Device Specific (if USE_GDS_DEVICE = True)
GDS_SAMPLING_RATE = 128  # Hz
GDS_N_CHANNELS = 8  # Number of EEG channels for g.tec
SAMPLES_PER_EMIT_GDS = 64  # How many samples to collect before emitting a signal (from GDS thread)
# Should be a multiple of d.NumberOfScans typically.

# Simulation Specific (if USE_GDS_DEVICE = False)
SIMULATED_SAMPLING_RATE = 128  # Hz (can be different from GDS for testing)
SIMULATED_N_CHANNELS = 8
SIMULATED_CHUNK_DURATION_SEC = 0.1  # Emit simulated data in chunks of this duration
SAMPLES_PER_EMIT_SIMULATED = int(SIMULATED_SAMPLING_RATE * SIMULATED_CHUNK_DURATION_SEC)

# Experiment Paradigm
TRIALS_LEFT = 6  # As per your request
TRIALS_RIGHT = 7  # As per your request
TOTAL_TRIALS = TRIALS_LEFT + TRIALS_RIGHT

# Durations in seconds
DURATION_READY_SEC = 2.0
DURATION_ACTION_SEC = 2.0
DURATION_IMAGE_SEC = 4.0
DURATION_REST_SEC = 2.0  # Fixed rest, or you can make it random: random.uniform(2.0, 3.0)

IMAGE_DIR = "./tips/"  # Directory for cue images
OUTPUT_DATA_DIR = "./experiment_output/"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DATA_DIR):
    os.makedirs(OUTPUT_DATA_DIR)

# --- GDS Acquisition Thread ---
if USE_GDS_DEVICE:
    try:
        import pygds
    except ImportError:
        print("ERROR: pygds library not found. Cannot use GDS_DEVICE.")
        USE_GDS_DEVICE = False  # Fallback to simulation

if USE_GDS_DEVICE:
    class GDSDataAcquisitionThread(QThread):
        data_acquired = pyqtSignal(np.ndarray)  # Emits chunk of data: (samples, channels)
        acquisition_error = pyqtSignal(str)

        def __init__(self, sampling_rate, num_channels, samples_per_emit):
            super().__init__()
            self.running = True
            self.lock = Lock()
            self.d = None
            self.gds_sampling_rate = sampling_rate
            self.gds_num_channels = num_channels
            self.samples_per_emit = samples_per_emit  # How many samples before callback

        def _configure_gtec_device(self):
            """Configures the g.tec device for specific settings."""
            if not self.d:
                self.acquisition_error.emit("g.tec device not connected.")
                return False

            print(f"Configuring g.tec device: {self.d.Name}")
            config = self.d.Configs[0] if self.d.ConfigCount > 0 else self.d

            config.SamplingRate = self.gds_sampling_rate
            config.NumberOfScans_calc()  # Let pygds calculate optimal NumberOfScans

            config.Counter = False
            config.Trigger = False
            if hasattr(config, 'InternalSignalGenerator'):
                config.InternalSignalGenerator.Enabled = False

            available_channels_info = self.d.GetAvailableChannels(combine=False)
            actual_channels_on_device = sum(available_channels_info[0]) if available_channels_info else len(
                config.Channels)

            channels_to_configure = min(self.gds_num_channels, actual_channels_on_device)
            if channels_to_configure < self.gds_num_channels:
                print(
                    f"Warning: Requested {self.gds_num_channels} channels, but device supports {channels_to_configure}.")

            for i in range(len(config.Channels)):
                config.Channels[i].Acquire = i < channels_to_configure
                if config.DeviceType == pygds.DEVICE_TYPE_GNAUTILUS:
                    config.Channels[i].BipolarChannel = -1
                else:
                    config.Channels[i].BipolarChannel = 0
                config.Channels[i].BandpassFilterIndex = -1
                config.Channels[i].NotchFilterIndex = -1

            print(f"Attempting to set SamplingRate: {config.SamplingRate}, NumberOfScans: {config.NumberOfScans}")
            self.d.SetConfiguration()
            self.d.N_ch_calc()  # Update d.N_ch
            print(f"Device configured to acquire {self.d.N_ch} channels at {self.d.SamplingRate} Hz.")

            if self.d.N_ch != channels_to_configure:
                self.acquisition_error.emit(
                    f"Configuration mismatch: Expected {channels_to_configure} ch, got {self.d.N_ch}.")
                return False
            return True

        def run(self):
            try:
                self.d = pygds.GDS()
                print(f"Connected to g.tec device: {self.d.Name}")

                if not self._configure_gtec_device():
                    self.d.Close()
                    self.d = None
                    return

                def more_callback(samples):
                    with self.lock:
                        if not self.running:
                            return False
                    # samples shape is (self.samples_per_emit, N_ch_actually_acquired)
                    self.data_acquired.emit(samples.copy())
                    return True  # True to continue acquisition

                # Start acquisition - GetData will call more_callback
                # The first argument to GetData is the number of samples to acquire in total
                # if the 'more' callback is not used or returns False at some point.
                # When 'more' is used, GetData calls it with 'self.samples_per_emit' chunks.
                # To run indefinitely until stop(), pass a very large number or handle it.
                # For simplicity, we'll rely on 'more_callback' returning False on stop().
                # pygds GetData with a 'more' callback typically runs until 'more' returns False or an error.
                print(f"Starting GDS data acquisition with emit size: {self.samples_per_emit} samples.")
                self.d.GetData(self.samples_per_emit, more_callback)  # Will loop internally based on more_callback

            except pygds.GDSError as e:
                print(f"GDS Acquisition Thread GDSError: {e}")
                self.acquisition_error.emit(f"g.tec API Error: {e}")
            except Exception as e:
                print(f"GDS Acquisition Thread Error: {e}")
                self.acquisition_error.emit(f"Unexpected Error: {e}")
            finally:
                if self.d:
                    print("Closing g.tec device in thread.")
                    self.d.Close()
                    self.d = None
                print("GDS Acquisition Thread finished.")

        def stop(self):
            print("GDS Acquisition Thread: stop called.")
            with self.lock:
                self.running = False


# --- Simulated Data Acquisition Thread ---
class SimulatedDataAcquisitionThread(QThread):
    data_acquired = pyqtSignal(np.ndarray)  # Emits chunk of data: (samples, channels)

    def __init__(self, sampling_rate, num_channels, samples_per_emit, chunk_duration_sec):
        super().__init__()
        self.running = True
        self.lock = Lock()
        self.sim_sampling_rate = sampling_rate
        self.sim_num_channels = num_channels
        self.samples_per_emit = samples_per_emit  # Number of samples in each emitted chunk
        self.interval_sec = chunk_duration_sec  # How often to emit a chunk

    def _generate_chunk(self):
        """Generates a chunk of simulated EEG data."""
        # Simple random data for simulation
        return np.random.rand(self.samples_per_emit, self.sim_num_channels) * 100 - 50  # uV scale

    def run(self):
        print(f"Starting Simulated Data Acquisition: {self.sim_sampling_rate} Hz, {self.sim_num_channels} Ch, "
              f"{self.samples_per_emit} samples every {self.interval_sec:.3f}s.")
        while True:
            with self.lock:
                if not self.running:
                    break

            start_time = time.perf_counter()
            data_chunk = self._generate_chunk()
            self.data_acquired.emit(data_chunk)

            elapsed_time = time.perf_counter() - start_time
            sleep_time = self.interval_sec - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Simulated Data Acquisition Thread finished.")

    def stop(self):
        print("Simulated Acquisition Thread: stop called.")
        with self.lock:
            self.running = False


# --- Main BCI Experiment Window ---
class BCIExperiment(QMainWindow):
    def __init__(self, use_gds):
        super().__init__()

        self.use_gds = use_gds
        if self.use_gds:
            self.sampling_rate = GDS_SAMPLING_RATE
            self.n_channels = GDS_N_CHANNELS
            self.samples_per_emit_thread = SAMPLES_PER_EMIT_GDS
        else:
            self.sampling_rate = SIMULATED_SAMPLING_RATE
            self.n_channels = SIMULATED_N_CHANNELS
            self.samples_per_emit_thread = SAMPLES_PER_EMIT_SIMULATED

        self.stage_durations_samples = {
            'ready': int(DURATION_READY_SEC * self.sampling_rate),
            'cue': int(DURATION_ACTION_SEC * self.sampling_rate),
            'image': int(DURATION_IMAGE_SEC * self.sampling_rate),
            'rest': int(DURATION_REST_SEC * self.sampling_rate)
        }

        self.all_eeg_data_chunks = []  # Store incoming data chunks
        self.events = []  # Store (sample_index, trial_num, hand_type, event_label)
        self.data_lock = Lock()

        self.current_total_sample_count = 0  # Overall samples received
        self.current_state = None
        self.current_hand = None
        self.state_start_sample = 0  # Sample count at the beginning of the current state

        self.trial_list = []
        self.current_trial_idx = -1  # Index for trial_list

        self.states_sequence = ['ready', 'cue', 'image', 'rest']
        self.current_state_idx = -1

        self.initUI()
        self.load_images()
        self.prepare_trials()

        if self.use_gds:
            self.data_thread = GDSDataAcquisitionThread(
                GDS_SAMPLING_RATE, GDS_N_CHANNELS, SAMPLES_PER_EMIT_GDS
            )
            self.data_thread.acquisition_error.connect(self.handle_acquisition_error)
        else:
            self.data_thread = SimulatedDataAcquisitionThread(
                SIMULATED_SAMPLING_RATE, SIMULATED_N_CHANNELS,
                SAMPLES_PER_EMIT_SIMULATED, SIMULATED_CHUNK_DURATION_SEC
            )

        self.data_thread.data_acquired.connect(self.process_acquired_data)

        # QTimer to start the experiment after GUI is shown
        QTimer.singleShot(100, self.start_experiment)

    def initUI(self):
        self.setWindowTitle(f'Motor Imagery Experiment ({"GDS Device" if self.use_gds else "Simulated"})')
        self.setGeometry(100, 100, 800, 600)  # Default size
        # self.showFullScreen() # Uncomment for full screen

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.central_widget.setLayout(layout)

    def load_images(self):
        self.images = {}
        required_images = ['ready', 'left', 'right', 'rest', 'image']
        for img_name in required_images:
            path = os.path.join(IMAGE_DIR, f"{img_name}.png")
            pixmap = QPixmap(path)
            if pixmap.isNull():
                print(f"Warning: Could not load image {path}")
                # Fallback text if image loading fails
                label = QLabel(img_name.upper(), self)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("font-size: 72px; color: white; background-color: black;")
                # Render QLabel to QPixmap
                fallback_pixmap = QPixmap(
                    self.image_label.size() / 2 if not self.image_label.size().isEmpty() else QPixmap(400, 300))
                fallback_pixmap.fill(Qt.black)
                painter = QPainter(fallback_pixmap)  # Requires from PyQt5.QtGui import QPainter
                label.render(painter)
                painter.end()
                self.images[img_name] = fallback_pixmap

            else:
                self.images[img_name] = pixmap
        if not all(img in self.images for img in required_images):
            QMessageBox.warning(self, "Image Loading Error",
                                f"One or more images could not be loaded from {IMAGE_DIR}. Using text fallbacks.")

    def prepare_trials(self):
        self.trial_list = ['left'] * TRIALS_LEFT + ['right'] * TRIALS_RIGHT
        random.shuffle(self.trial_list)
        print(f"Prepared {len(self.trial_list)} trials. Left: {TRIALS_LEFT}, Right: {TRIALS_RIGHT}")

    def start_experiment(self):
        if not self.images:
            QMessageBox.critical(self, "Error", "Images not loaded. Cannot start experiment.")
            self.close()
            return

        print("Starting experiment...")
        self.current_trial_idx = -1  # Will be incremented in next_trial
        self.data_thread.start()
        self.next_trial()

    def next_trial(self):
        self.current_trial_idx += 1
        if self.current_trial_idx >= len(self.trial_list):
            self.end_experiment()
            return

        self.current_hand = self.trial_list[self.current_trial_idx]
        self.current_state_idx = 0  # Start with 'ready' state

        trial_num_display = self.current_trial_idx + 1
        print(
            f"\n--- Starting Trial {trial_num_display}/{len(self.trial_list)}: Hand - {self.current_hand.upper()} ---")
        self.record_event('trial_start')
        self.update_state_display()

    def update_state_display(self):
        self.current_state = self.states_sequence[self.current_state_idx]
        self.state_start_sample = self.current_total_sample_count

        print(
            f"  Entering State: {self.current_state.upper()} (Trial {self.current_trial_idx + 1}) at sample {self.current_total_sample_count}")
        self.record_event(f'state_{self.current_state}_start')

        # 修改点: 根据新的状态显示对应图片
        if self.current_state == 'ready':
            self.show_image('ready')
        elif self.current_state == 'cue':  # 行动提示阶段
            self.show_image(self.current_hand)  # 'left' 或 'right'
        elif self.current_state == 'image':  # 想象阶段
            self.show_image('image')  # 显示 "imagine.png"
        elif self.current_state == 'rest':
            self.show_image('rest')

    def show_image(self, image_key):
        if image_key in self.images:
            scaled_pixmap = self.images[image_key].scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            print(f"Error: Image key '{image_key}' not found.")
            self.image_label.setText(f"{image_key.upper()}\n(Image Missing)")

    @pyqtSlot(np.ndarray)
    def process_acquired_data(self, data_chunk):
        """Processes data chunk from acquisition thread."""
        with self.data_lock:
            self.all_eeg_data_chunks.append(data_chunk)
            num_samples_in_chunk = data_chunk.shape[0]
            self.current_total_sample_count += num_samples_in_chunk

        if self.current_state is None:  # Experiment might not have fully started next_trial
            return

        # Check if current state duration is exceeded
        samples_in_current_state = self.current_total_sample_count - self.state_start_sample

        if samples_in_current_state >= self.stage_durations_samples[self.current_state]:
            self.record_event(f'state_{self.current_state}_end')  # Record end of previous state
            self.advance_to_next_state_or_trial()

    def advance_to_next_state_or_trial(self):
        self.current_state_idx += 1
        if self.current_state_idx < len(self.states_sequence):
            self.update_state_display()  # Move to next state in the current trial
        else:
            self.record_event('trial_end')
            self.next_trial()  # Current trial finished, move to next trial

    def record_event(self, event_label):
        event_info = {
            'sample_index': self.current_total_sample_count,
            'trial_num': self.current_trial_idx + 1,  # 1-based
            'hand_type': self.current_hand if self.current_hand else "N/A",
            'event_label': event_label
        }
        with self.data_lock:
            self.events.append(event_info)
        # print(f"Event: {event_label} (Trial {event_info['trial_num']}, Hand {event_info['hand_type']}) at sample {event_info['sample_index']}")

    @pyqtSlot(str)
    def handle_acquisition_error(self, error_message):
        QMessageBox.critical(self, "Acquisition Error", error_message)
        self.end_experiment(error_occurred=True)

    def end_experiment(self, error_occurred=False):
        print("\n--- Experiment Ending ---")
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()
            self.data_thread.wait(5000)  # Wait up to 5 seconds for thread to finish

        if not error_occurred and self.all_eeg_data_chunks:
            try:
                # Concatenate all data chunks
                final_eeg_data = np.concatenate(self.all_eeg_data_chunks, axis=0)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DATA_DIR, f'bci_mi_data_{timestamp}.pkl')

                data_to_save = {
                    'eeg_data': final_eeg_data,  # Shape: (total_samples, n_channels)
                    'events': self.events,  # List of event dictionaries
                    'sampling_rate': self.sampling_rate,
                    'n_channels': self.n_channels,
                    'trial_stimuli_order': self.trial_list,  # The actual order of 'left'/'right' stimuli
                    'gds_device_used': self.use_gds
                }
                with open(filename, 'wb') as f:
                    pickle.dump(data_to_save, f)
                print(f"Data saved successfully to {filename}")
                QMessageBox.information(self, "Experiment Complete", f"Data saved to {filename}")
            except Exception as e:
                print(f"Error saving data: {e}")
                QMessageBox.critical(self, "Save Error", f"Could not save data: {e}")
        elif error_occurred:
            print("Experiment ended due to an error. Data might not be saved.")
        else:
            print("No data collected or experiment ended prematurely.")

        self.image_label.setText("Experiment Finished.\nPlease close the window.")
        self.image_label.setStyleSheet("font-size: 30px; color: white; background-color: black;")
        QTimer.singleShot(3000, self.close_application)  # Close app after 3s

    def close_application(self):
        QApplication.instance().quit()

    def closeEvent(self, event):
        """Handle window close event."""
        if self.data_thread and self.data_thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "Experiment is running. Are you sure you want to exit? Data might not be saved.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.end_experiment(error_occurred=True)  # Treat as premature end
                event.accept()
            else:
                event.ignore()
        else:  # If thread not running, just accept
            self.end_experiment()  # Try to save if any data exists
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Make sure to import QPainter if you use the image fallback text rendering
    # from PyQt5.QtGui import QPainter # (Already implicitly imported by QPixmap usually)

    # Initialize pygds if using the real device
    # This should be done once for the application
    if USE_GDS_DEVICE:
        try:
            if not pygds.Initialize():  # pygds.Initialize is from the provided pygds.py text
                QMessageBox.critical(None, "pygds Error",
                                     "Failed to initialize pygds library. Check DLL paths and g.NEEDaccess installation.")
                sys.exit(1)  # Exit if pygds cannot be initialized
        except NameError:  # pygds might not be defined if import failed
            QMessageBox.critical(None, "pygds Error", "pygds module not found or Initialize function missing.")
            sys.exit(1)

    window = BCIExperiment(use_gds=USE_GDS_DEVICE)
    window.show()
    sys.exit(app.exec_())
