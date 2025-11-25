from pattern_event_handler import PatternEventListener
import datetime
import argparse

def on_rising(detection):
    print(f"\n[EVENT] 📈 RISING pattern detected! (Score: {detection['score']:.2f})")
    print(f"        Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    # Do something cool here, like turn on a light

def on_falling(detection):
    print(f"\n[EVENT] 📉 FALLING pattern detected! (Score: {detection['score']:.2f})")
    print(f"        Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    # Do something else cool, like turn off a light

def main():
    parser = argparse.ArgumentParser(description="Whistle Pattern Listener Demo")
    parser.add_argument("--device", type=str, default="MacBook Pro Microphone", help="Name of the input audio device")
    args = parser.parse_args()

    # Create the listener
    listener = PatternEventListener(device_name=args.device)

    # Connect functions to patterns
    listener.on("rising", on_rising)
    listener.on("falling", on_falling)

    # Start the loop
    print("\nWhistle your patterns! Press Ctrl+C to stop.")
    listener.start()

if __name__ == "__main__":
    main()
