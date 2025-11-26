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
    
    # Entities
    parser.add_argument("--light-entity", type=str, default=os.environ.get("HASS_LIGHT_ENTITY", "light.aidan_room_standing_lamp"), help="Light entity ID")
    parser.add_argument("--plug-entity", type=str, default=os.environ.get("HASS_PLUG_ENTITY", "switch.aidan_room_colored_lights"), help="Plug entity ID")
    
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

    # Initialize Listener
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
        max_gap_frames=args.max_gap_frames
    )

    try:
        with WebsocketClient(ws_url, args.hass_token) as client:
            logging.info("Connected to Home Assistant.")

            # Define Callbacks (using the client from context)
            def on_falling(detection):
                logging.info(f"Pattern FALLING detected (score={detection['score']:.2f}). Turning OFF light and plug.")
                try:
                    client.get_domain("light").turn_off(entity_id=args.light_entity)
                    client.get_domain("switch").turn_off(entity_id=args.plug_entity)
                except Exception as e:
                    logging.error(f"Failed to execute HASS commands: {e}")

            def on_rising(detection):
                logging.info(f"Pattern RISING detected (score={detection['score']:.2f}). Turning ON light (warm/bright) and plug.")
                try:
                    client.get_domain("light").turn_on(entity_id=args.light_entity, brightness=255, color_temp=370)
                    client.get_domain("switch").turn_on(entity_id=args.plug_entity)
                except Exception as e:
                    logging.error(f"Failed to execute HASS commands: {e}")

            def on_placeholder_1(detection):
                logging.info(f"Pattern PLACEHOLDER_1 detected. Light OFF, Plug ON.")
                try:
                    client.get_domain("light").turn_off(entity_id=args.light_entity)
                    client.get_domain("switch").turn_on(entity_id=args.plug_entity)
                except Exception as e:
                    logging.error(f"Failed to execute HASS commands: {e}")

            def on_placeholder_2(detection):
                logging.info(f"Pattern PLACEHOLDER_2 detected. Light ON (dim), Plug OFF.")
                try:
                    client.get_domain("light").turn_on(entity_id=args.light_entity, brightness=10)
                    client.get_domain("switch").turn_off(entity_id=args.plug_entity)
                except Exception as e:
                    logging.error(f"Failed to execute HASS commands: {e}")

            # Register callbacks
            listener.on("falling", on_falling)
            listener.on("rising", on_rising)
            listener.on("placeholder_1", on_placeholder_1)
            listener.on("placeholder_2", on_placeholder_2)

            # Start Listening
            logging.info("Starting audio listener...")
            listener.start()
            
    except KeyboardInterrupt:
        logging.info("Stopped by user.")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        listener.stop()

if __name__ == "__main__":
    main()