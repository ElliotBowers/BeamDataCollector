import redis
import json
import numpy as np
import time
import yaml


def generate_test_signal(length=500, noise_level=5, pulse_height=100, pulse_width=20):
    """Generate a signal with noise and a single pulse in the center."""
    signal = np.random.normal(0, noise_level, size=length)

    start = length // 2 - pulse_width // 2
    end = start + pulse_width
    signal[start:end] += pulse_height

    return signal


def stream_test_signals(r, out_key="beam:raw", interval=1.0):
    """Simulate streaming of signals into Redis using XADD."""
    i = 0
    while True:
        signal = generate_test_signal()
        r.xadd(out_key, {'signal': json.dumps(signal.tolist())}, maxlen=10)
        print(f"Streamed test signal {i} to {out_key}")
        i += 1
        time.sleep(interval)

if __name__ == "__main__":
    try:
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        # Accessing variables
        redis_host = config['redis_host']
        redis_port = config['redis_port']
        input_key = config['input_key']
        function_name = config['function_name']
        output_key = config['output_key']


    except FileNotFoundError:
        print("Error: config.yml not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YML file: {e}")

    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )
    stream_test_signals(r, out_key=input_key, interval=1.0)