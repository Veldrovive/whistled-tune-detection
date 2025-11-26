import argparse
import logging
import os
import sys
import time
from homeassistant_api import WebsocketClient

from pattern_event_handler import PatternEventListener

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Whistle to Home Assistant Controller")
    
    # HASS connection args
    parser.add_argument("--hass-url", type=str, default=os.environ.get("HASS_URL"), help="Home Assistant URL (e.g. http://homeassistant.local:8123)")
    parser.add_argument("--hass-token", type=str, default=os.environ.get("HASS_TOKEN"), help="Long-lived access token")
    
    # Audio args
    parser.add_argument("--device", type=str, default=os.environ.get("AUDIO_DEVICE", "MacBook Pro Microphone"), help="Name of the input audio device")
    parser.add_argument("--model-dir", type=str, default="pattern_models", help="Directory containing pattern models")
    parser.add_argument("--n-mels", type=int, default=int(os.environ.get("AUDIO_N_MELS", 128)), help="Number of Mel bands")
    parser.add_argument("--fmin", type=int, default=int(os.environ.get("AUDIO_FMIN", 300)), help="Minimum frequency")
    parser.add_argument("--fmax", type=int, default=int(os.environ.get("AUDIO_FMAX", 3500)), help="Maximum frequency")
    parser.add_argument("--max-curve-jump", type=int, default=int(os.environ.get("AUDIO_MAX_CURVE_JUMP", 2)), help="Max curve jump")
    parser.add_argument("--min-curve-len", type=int, default=int(os.environ.get("AUDIO_MIN_CURVE_LEN", 5)), help="Min interesting curve length")
    parser.add_argument("--max-gap-frames", type=int, default=int(os.environ.get("AUDIO_MAX_GAP_FRAMES", 3)), help="Max frames to bridge a gap in a curve")
    parser.add_argument("--ignore-freq-bands", type=str, default=os.environ.get("AUDIO_IGNORE_FREQ_BANDS", ""), help="Comma-separated list of frequency bands to ignore")
    parser.add_argument("--refresh-rate", type=int, default=int(os.environ.get("REFRESH_RATE_HZ", 100)), help="Refresh rate in Hz")
    parser.add_argument("--threshold-sigma", type=float, default=float(os.environ.get("THRESHOLD_SIGMA", 3.0)), help="Detection threshold (sigma)")
    parser.add_argument("--logging-threshold-sigma", type=float, default=float(os.environ.get("LOGGING_THRESHOLD_SIGMA", 6.0)), help="Logging threshold (sigma)")

    args = parser.parse_args()

    if not args.hass_url or not args.hass_token:
        logging.error("HASS_URL and HASS_TOKEN must be provided via arguments or environment variables.")
        sys.exit(1)

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

    # Prepare WS URL
    ws_url = args.hass_url
    
    # Handle scheme replacement
    if ws_url.startswith("https://"):
        ws_url = ws_url.replace("https://", "wss://")
    elif ws_url.startswith("http://"):
        ws_url = ws_url.replace("http://", "ws://")
        
    if not ws_url.endswith("/api/websocket"):
        if ws_url.endswith("/"):
            ws_url = ws_url + "api/websocket"
        else:
            ws_url = ws_url + "/api/websocket"
    
    logging.info(f"Connecting to Home Assistant Websocket at {ws_url}...")

    try:
        # Initialize Client
        with WebsocketClient(ws_url, args.hass_token) as client:
            logging.info("Connected to Home Assistant.")

            def on_pattern_detected(detection):
                pattern_name = detection['name']
                score = detection['score']
                logging.info(f"Pattern '{pattern_name}' detected (score={score:.2f}). Firing event.")
                
                try:
                    # Fire event
                    client.fire_event("WHISTLE_DETECTED", name=pattern_name, score=score)
                    logging.info(f"Fired WHISTLE_DETECTED event for {pattern_name}")
                except Exception as e:
                    logging.error(f"Failed to fire event: {e}")

            # Initialize Listener with default callback
            listener = PatternEventListener(
                model_dir=args.model_dir,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
                max_curve_jump=args.max_curve_jump,
                min_interesting_curve_len=args.min_curve_len,
                refresh_rate_hz=args.refresh_rate,
                device_name=args.device,
                ignore_freq_bands=ignore_freq_bands,
                max_gap_frames=args.max_gap_frames,
                threshold_sigma=args.threshold_sigma,
                logging_threshold_sigma=args.logging_threshold_sigma,
                default_callback=on_pattern_detected
            )

            # Start Listening
            logging.info("Starting audio listener...")
            listener.start()
            
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if 'listener' in locals():
            listener.stop()

if __name__ == "__main__":
    main()