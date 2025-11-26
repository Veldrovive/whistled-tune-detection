from pattern_event_handler import PatternEventListener
import asyncio
import argparse
import os
from kasa.discover import Discover
from kasa import Device
import traceback

async def find_devices():
    devices = await Discover.discover()
    print(devices)
    return devices

async def set_devices_on_state(device_list, on):
    for device in device_list:
        try:
            print(f"Setting {device.host} to {on}")
            await device.turn_on() if on else await device.turn_off()
            print(f"Turned {device.host} to {on}")
            await device.update()
            print(f"Updated device: {device}")
        except Exception as e:
            print(f"Error setting device {device.host}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Kasa Light Control Demo")
    
    # Audio Device
    parser.add_argument("--device", type=str, default=os.environ.get("AUDIO_DEVICE", "MacBook Pro Microphone"), help="Name of the input audio device")
    
    # Kasa Lights
    default_macs = os.environ.get("KASA_LIGHT_MACS", "e0:d3:62:83:62:13,98:03:8E:76:25:BE")
    parser.add_argument("--mac-addresses", type=str, default=default_macs, help="Comma-separated list of Kasa light MAC addresses")
    
    # Audio Processing Parameters
    parser.add_argument("--chunk-size", type=int, default=int(os.environ.get("AUDIO_CHUNK_SIZE", 1024)), help="Audio chunk size")
    parser.add_argument("--rate", type=int, default=int(os.environ.get("AUDIO_RATE", 0)) or None, help="Sample rate (0 or None for default)")
    parser.add_argument("--n-fft", type=int, default=int(os.environ.get("AUDIO_N_FFT", 2048)), help="FFT size")
    parser.add_argument("--hop-len", type=int, default=int(os.environ.get("AUDIO_HOP_LEN", 512)), help="Hop length")
    parser.add_argument("--buffer-duration", type=float, default=float(os.environ.get("AUDIO_BUFFER_DURATION", 5.0)), help="Buffer duration in seconds")
    parser.add_argument("--n-mels", type=int, default=int(os.environ.get("AUDIO_N_MELS", 128)), help="Number of Mel bands")
    parser.add_argument("--fmin", type=int, default=int(os.environ.get("AUDIO_FMIN", 300)), help="Minimum frequency")
    parser.add_argument("--fmax", type=int, default=int(os.environ.get("AUDIO_FMAX", 3500)), help="Maximum frequency")
    parser.add_argument("--max-curve-jump", type=int, default=int(os.environ.get("AUDIO_MAX_CURVE_JUMP", 2)), help="Max curve jump")
    parser.add_argument("--min-curve-len", type=int, default=int(os.environ.get("AUDIO_MIN_CURVE_LEN", 5)), help="Min interesting curve length")
    parser.add_argument("--refresh-rate", type=int, default=int(os.environ.get("REFRESH_RATE_HZ", 100)), help="Refresh rate in Hz")

    args = parser.parse_args()

    # Parse MAC addresses
    target_macs = [mac.strip() for mac in args.mac_addresses.split(",")]
    print(f"Targeting MAC addresses: {target_macs}")

    print("Discovering Kasa devices...")
    devices = await find_devices()
    light_ips = [ip for ip, device in devices.items() if device.mac in target_macs]
    
    if not light_ips:
        print("No matching Kasa devices found.")
    
    light_devices = []
    for ip in light_ips:
        try:
            dev = await Device.connect(host=ip)
            light_devices.append(dev)
            print(f"Connected to {dev.alias} ({ip})")
        except Exception as e:
            print(f"Failed to connect to {ip}: {e}")

    async def on_rising(detection):
        await set_devices_on_state(light_devices, True)
        print("Turning on lights")

    async def on_falling(detection):
        await set_devices_on_state(light_devices, False)
        print("Turning off lights")

    # Create the listener
    print(f"Initializing PatternEventListener with device: {args.device}")
    listener = PatternEventListener(
        device_name=args.device,
        chunk_size=args.chunk_size,
        rate=args.rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        buffer_duration=args.buffer_duration,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        max_curve_jump=args.max_curve_jump,
        min_interesting_curve_len=args.min_curve_len,
        refresh_rate_hz=args.refresh_rate
    )

    # Get the current event loop
    loop = asyncio.get_running_loop()

    def on_rising_sync(detection):
        print("Rising pattern detected, scheduling async task...")
        asyncio.run_coroutine_threadsafe(on_rising(detection), loop)

    def on_falling_sync(detection):
        print("Falling pattern detected, scheduling async task...")
        asyncio.run_coroutine_threadsafe(on_falling(detection), loop)

    # Connect functions to patterns
    listener.on("rising", on_rising_sync)
    listener.on("falling", on_falling_sync)

    # Start the loop in a separate thread
    print("\nWhistle your patterns! Press Ctrl+C to stop.")
    try:
        await asyncio.to_thread(listener.start)
    except asyncio.CancelledError:
        print("Listener task cancelled")
    finally:
        listener.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
