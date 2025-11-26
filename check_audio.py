import sounddevice as sd

print("------------------------------------------------------------------")
print(f"Host APIs: {len(sd.query_hostapis())}")
print("------------------------------------------------------------------")

devices = sd.query_devices()
for i, dev in enumerate(devices):
    # Filter out devices that are output only
    if dev['max_input_channels'] > 0:
        print(f"Index {i}: {dev['name']}")
        print(f"  - Max Input Channels: {dev['max_input_channels']}")
        print(f"  - Default Sample Rate: {dev['default_samplerate']}")
        print("------------------------------------------------------------------")
