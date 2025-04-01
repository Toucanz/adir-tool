import tkinter as tk
import pyaudio
import wave
import threading

#audiorecorder class


class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()   # <- This is the missing piece
        self.stream = None
        self.frames = []
        self.recording = False

    def start_recording(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024)
        self.recording = True
        self.frames = []

        while self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)

    def stop_recording(self, file_path):
        if self.stream:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

            # Save to .wav file
            import wave
            wf = wave.open(file_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()

        print(f'recording saved as {filename}')
    #for button
class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.recorder = AudioRecorder()

        #Button  start
        self.start_button = tk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)


        #button for stop
        self.stop_button = tk.Button(self.root, text="Stop Recording", command =self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)


    def start_recording(self):
        #disable start and enable stop
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)


        self.recording_thread = threading.Thread(target=self.recorder.start_recording)
        self.recording_thread.start()

    def stop_recording(self):
        # Disable the stop button and enable the start button
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

        # Stop the recording and save the file
        
        file_path = "/home/nick_z/ADIRproject/adir-tool/Audiointerface/recordedaudio.wav"

        self.recorder.stop_recording(file_path)



if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()









