from pattern_event_handler import PatternEventListener
import asyncio
import argparse
import os
from kasa.discover import Discover
from kasa import Device
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def find_devices():
    devices = await Discover.discover()
    logging.info(devices)
    return devices

async def set_devices_on_state(device_list, on):
    for device in device_list:
        try:
            logging.info(f"Setting {device.host} to {on}")
            await device.turn_on() if on else await device.turn_off()
            logging.info(f"Turned {device.host} to {on}")
            await device.update()
            logging.info(f"Updated device: {device}")
        except Exception as e:
            logging.error(f"Error setting device {device.host}: {e}")

async def heartbeat():
    """Logs 'alive' every 10 seconds."""
    while True:
        logging.info("alive")
        await asyncio.sleep(10)

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
    parser.add_argument("--max-gap-frames", type=int, default=int(os.environ.get("AUDIO_MAX_GAP_FRAMES", 3)), help="Max frames to bridge a gap in a curve")
    parser.add_argument("--ignore-freq-bands", type=str, default=os.environ.get("AUDIO_IGNORE_FREQ_BANDS", ""), help="Comma-separated list of frequency bands to ignore (e.g. '1000-1020,2000-2050')")
    parser.add_argument("--refresh-rate", type=int, default=int(os.environ.get("REFRESH_RATE_HZ", 100)), help="Refresh rate in Hz")

    args = parser.parse_args()

    # Parse MAC addresses
    target_macs = [mac.strip() for mac in args.mac_addresses.split(",")]
    target_macs = [mac.strip() for mac in args.mac_addresses.split(",")]
    logging.info(f"Targeting MAC addresses: {target_macs}")

    # Parse ignore frequency bands
    ignore_freq_bands = []
    if args.ignore_freq_bands:
        try:
            for band in args.ignore_freq_bands.split(","):
                if "-" in band:
                    min_f, max_f = map(float, band.split("-"))
                    ignore_freq_bands.append((min_f, max_f))
            logging.info(f"Ignoring frequency bands: {ignore_freq_bands}")
        except ValueError:
            logging.error(f"Invalid format for ignore-freq-bands: {args.ignore_freq_bands}")

    logging.info("Discovering Kasa devices...")
    devices = await find_devices()
    light_ips = [ip for ip, device in devices.items() if device.mac in target_macs]
    
    if not light_ips:
        logging.warning("No matching Kasa devices found.")
    
    light_devices = []
    for ip in light_ips:
        try:
            dev = await Device.connect(host=ip)
            light_devices.append(dev)
            logging.info(f"Connected to {dev.alias} ({ip})")
        except Exception as e:
            logging.error(f"Failed to connect to {ip}: {e}")

    # Start heartbeat
    asyncio.create_task(heartbeat())

    async def on_rising(detection):
        await set_devices_on_state(light_devices, True)
        logging.info("Turning on lights")

    async def on_falling(detection):
        await set_devices_on_state(light_devices, False)
        logging.info("Turning off lights")

    # Create the listener
    logging.info(f"Initializing PatternEventListener with device: {args.device}")
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
        min_interesting_curve_len=args.min_curve_len,
        max_curve_jump=args.max_curve_jump,
        refresh_rate_hz=args.refresh_rate,
        ignore_freq_bands=ignore_freq_bands,
        max_gap_frames=args.max_gap_frames
    )

    # Get the current event loop
    loop = asyncio.get_running_loop()

    def on_rising_sync(detection):
        logging.info("Rising pattern detected, scheduling async task...")
        asyncio.run_coroutine_threadsafe(on_rising(detection), loop)

    def on_falling_sync(detection):
        logging.info("Falling pattern detected, scheduling async task...")
        asyncio.run_coroutine_threadsafe(on_falling(detection), loop)

    # Connect functions to patterns
    listener.on("rising", on_rising_sync)
    listener.on("falling", on_falling_sync)

    # Start the loop in a separate thread
    logging.info("\nWhistle your patterns! Press Ctrl+C to stop.")
    try:
        await asyncio.to_thread(listener.start)
    except asyncio.CancelledError:
        logging.info("Listener task cancelled")
    finally:
        listener.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nStopped by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
