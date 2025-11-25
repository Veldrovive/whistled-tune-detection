from pattern_event_handler import PatternEventListener
import asyncio
import argparse
from kasa.discover import Discover
import time
from kasa import Device

NORMAL_LIGHT_MAC_ADDRESSES = ["e0:d3:62:83:62:13", "98:03:8E:76:25:BE"]

async def find_devices():
    devices = await Discover.discover()
    print(devices)
    return devices

async def set_devices_on_state(device_list, on):
    for device in device_list:
        print(f"Setting {device.host} to {on}")
        await device.turn_on() if on else await device.turn_off()
        print(f"Turned {device.host} to {on}")
        await device.update()
        print(f"Updated device: {device}")

async def main():
    parser = argparse.ArgumentParser(description="Kasa Light Control Demo")
    parser.add_argument("--device", type=str, default="MacBook Pro Microphone", help="Name of the input audio device")
    args = parser.parse_args()

    devices = await find_devices()
    light_ips = [ip for ip, device in devices.items() if device.mac in NORMAL_LIGHT_MAC_ADDRESSES]
    light_devices = [await Device.connect(host=ip) for ip in light_ips]

    async def on_rising(detection):
        await set_devices_on_state(light_devices, True)
        print("Turning on lights")

    async def on_falling(detection):
        await set_devices_on_state(light_devices, False)
        print("Turning off lights")

    # Create the listener
    listener = PatternEventListener(device_name=args.device)

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
    await asyncio.to_thread(listener.start)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
