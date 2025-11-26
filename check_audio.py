import pyaudio

p = pyaudio.PyAudio()

print("------------------------------------------------------------------")
print(f"Host APIs: {p.get_host_api_count()}")
print("------------------------------------------------------------------")

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    # Filter out devices that are output only
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']}")
        print(f"  - Max Input Channels: {dev['maxInputChannels']}")
        print(f"  - Default Sample Rate: {dev['defaultSampleRate']}")
        print("------------------------------------------------------------------")

p.terminate()
