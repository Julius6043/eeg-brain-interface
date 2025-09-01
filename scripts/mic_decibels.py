"""
Standalone mic -> dBFS -> LSL streamer.

Dependencies:
    pip install sounddevice numpy pylsl

Run:
    python mic_decibels.py
    Press Ctrl+C to stop.

device = none param ->  uses the device's default audio
                        recording device, change
                        the name if using the other mic

import sounddevice as sd
print(sd.query_devices()) -> run this code separately if not sure
                            what devices are available
"""

import math
import numpy as np
import csv
import sounddevice as sd
from pylsl import StreamInfo, StreamOutlet, local_clock

# redundancy -> saving it as a csv file as well
db_log = [] # list of (timestamp, decibel)

def start_mic_decibel_stream(
    device=None,   
    samplerate=44100,   # standard CD audio quality
    blocksize=1024      # refreshes every 23ms
):
    """
    Opens the default mic, computes RMS (root mean square) 
    converted to dB per block, and pushes
    one float32 sample per block to an LSL stream named 'MicDecibels'.
    Returns the InputStream so you can stop() it later.
    """
    info = StreamInfo(
        name='MicDecibels',       # lsl streamname
        type='Audio',
        channel_count=1,
        nominal_srate=0,          # irregular rate
        channel_format='float32',
        source_id='mic_stream_001'
    )
    outlet = StreamOutlet(info)

    # Callback: compute RMS = dB, push to LSL
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", flush=True)
        # take first channel
        samples = indata[:, 0]
        rms = np.sqrt(np.mean(np.square(samples)))
        db  = 20.0 * math.log10(rms) if rms > 0 else -np.inf

        # use the audio ADC timestamp for LSL
        ts = time_info.inputBufferAdcTime
        outlet.push_sample([db], ts)

    # Open & start the stream
    stream = sd.InputStream(
        device=device,
        channels=1,
        samplerate=samplerate,
        blocksize=blocksize,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    return stream

def main():
    print("Starting microphone -> dB -> LSL stream.")
    print("Press Ctrl+C to stop.")

    # launch the streaming
    stream = start_mic_decibel_stream(
        samplerate=44100,
        blocksize=1024
    )

    try:
        # keep the main thread alive
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping...")
    finally:
        stream.stop()
        stream.close()
        print("Stream closed.")

    # Save CSV redundancy
        fname = "mic_dB_log.csv"
        print(f"Saving CSV log to '{fname}' …")
        with open(fname, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "decibels"])
            writer.writerows(db_log)
        print("CSV saved.")


if __name__ == "__main__":
    main()