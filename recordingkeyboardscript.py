import time
import pandas as pd
import numpy as np
from pynput import keyboard
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from threading import Thread, Event
from queue import Queue

# Experiment Parameters
lsl_out = False
save_dir = 'data/misc/'  # Directory to save data
run = 1  # Run number for tracking
save_file_aux = save_dir + f'aux_run-{run}.npy'

# Initialize BrainFlow Synthetic Board
CYTON_BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
params = BrainFlowInputParams()
board = BoardShim(CYTON_BOARD_ID, params)
board.prepare_session()
board.start_stream()

# Data Storage
timestamps = []
labels = []
queue_in = Queue()
stop_event = Event()

def get_data(queue_in):
    while not stop_event.is_set():
        data_in = board.get_board_data()
        timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
        eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
        aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
        if len(timestamp_in) > 0:
            queue_in.put((eeg_in, aux_in, timestamp_in))
        time.sleep(0.1)

def on_press(key):
    global timestamps, labels
    try:
        if key.char == '1':
            label = "Lost Focus"
        elif key.char == '2':
            label = "Focused Again"
        elif key.char == '3':
            label = "Lecture Started"
        elif key.char == '4':
            label = "Lecture Paused"
        else:
            return
        
        timestamp = time.time()
        timestamps.append(timestamp)
        labels.append(label)
        print(f"[{time.strftime('%H:%M:%S', time.localtime(timestamp))}] {label}")
    except AttributeError:
        pass

def main():
    print("Press 1 for 'Lost Focus', 2 for 'Focused Again', 3 for 'Lecture Started', 4 for 'Lecture Paused'")
    
    cyton_thread = Thread(target=get_data, args=(queue_in,))
    cyton_thread.daemon = True
    cyton_thread.start()
    
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            while True:
                time.sleep(0.1)  # Keep the program running
        except KeyboardInterrupt:
            print("Stopping recording...")
            stop_event.set()
            listener.stop()
            board.stop_stream()
            board.release_session()
            
            # Save attention tracking data
            df = pd.DataFrame({"Timestamp": timestamps, "Label": labels})
            df.to_csv("attention_tracking.csv", index=False)
            print("Data saved to attention_tracking.csv")
            
            # Save EEG auxiliary data
            os.makedirs(save_dir, exist_ok=True)
            aux_data = np.hstack([queue_in.get()[1] for _ in range(queue_in.qsize())])
            np.save(save_file_aux, aux_data)
            print(f"Auxiliary data saved to {save_file_aux}")

if __name__ == "__main__":
    main()
