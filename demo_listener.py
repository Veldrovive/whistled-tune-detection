from pattern_event_handler import PatternEventListener
import datetime

def on_rising(detection):
    print(f"\n[EVENT] 📈 RISING pattern detected! (Score: {detection['score']:.2f})")
    print(f"        Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    # Do something cool here, like turn on a light

def on_falling(detection):
    print(f"\n[EVENT] 📉 FALLING pattern detected! (Score: {detection['score']:.2f})")
    print(f"        Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
    # Do something else cool, like turn off a light

def main():
    # Create the listener
    listener = PatternEventListener()

    # Connect functions to patterns
    listener.on("rising", on_rising)
    listener.on("falling", on_falling)

    # Start the loop
    print("\nWhistle your patterns! Press Ctrl+C to stop.")
    listener.start()

if __name__ == "__main__":
    main()
