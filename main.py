import BeamDataCollector
import redis
import yaml
import time
from threading import Thread
from pydapter_wraptor_py import pydapter_wraptor

# if __name__ == "__main__":

#     try:
#         with open('config.yml', 'r') as file:
#             config = yaml.safe_load(file)

#         # Accessing variables
#         redis_host = config['redis_host']
#         redis_port = config['redis_port']
#         input_key = config['input_key']
#         function_name = config['function_name']
#         output_key = config['output_key']
#     except FileNotFoundError:
#         print("Error: config.yml not found.")
#     except yaml.YAMLError as e:
#         print(f"Error parsing YML file: {e}")

#     conn_options = pydapter_wraptor.RedisConnectionOptions(
#         host="blongd",  # Redis server host
#         port=6379,         # Redis server port
#         timeout=500,       # Connection timeout in milliseconds
#         size=5             # Connection pool size
#     )

#     ra_options = pydapter_wraptor.RA_Options(
#         cxn=conn_options,   # Redis connection options
#         dogname="example_dogtag",   # Watchdog name
#         workers=2,          # Number of worker threads
#         readers=2           # Number of reader threads
#     )

#     r = pydapter_wraptor.RedisConnection(options=conn_options)
#     print("Checking connection status...")
#     if not r.ping():
#         print("Failed to connect to Redis server.")
#     else:
#         print("Connected to Redis server successfully. The adapter is ready for use.")

#     # r = redis.Redis(
#     #     host=redis_host,
#     #     port=redis_port,
#     #     decode_responses=False
#     # )
#     # base_key = "amplitudes"  # e.g., amplitudes:0, amplitudes:1, etc.
#     # out_key_base = "processed"  # saves to processed:signal:N, processed:current:N, etc.
#     # daemon_loop(base_key, out_key_base, r)
#     # BeamDataCollector.daemon_loop_sync_overlap(base_key, out_key_base, r)

#     # r = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
#     try:
#         callback_fn = getattr(BeamDataCollector, function_name)
#     except AttributeError:
#         raise ValueError(f"Function '{function_name}' not found in BeamDataCollector module.")

#     BeamDataCollector.measure(r, in_key=input_key, callback=callback_fn, out_key=output_key)



# def run_measure(r, in_key, callback, out_key):
#     BeamDataCollector.measure(r, in_key=in_key, callback=callback, out_key=out_key)

# if __name__ == "__main__":

#     try:
#         with open('config.yml', 'r') as file:
#             config = yaml.safe_load(file)

#         redis_host = config['redis_host']
#         redis_port = config['redis_port']
#         input_key = config['input_key']
#         output_key = config['output_key']
#         function_names = config['function_names']
#     except FileNotFoundError:
#         print("Error: config.yml not found.")
#         exit(1)
#     except yaml.YAMLError as e:
#         print(f"Error parsing YML file: {e}")
#         exit(1)

#     r = redis.Redis(
#         host=redis_host,
#         port=redis_port,
#         decode_responses=False
#     )

#     threads = []
#     for fn_name in function_names:
#         try:
#             callback_fn = getattr(BeamDataCollector, fn_name)
#         except AttributeError:
#             raise ValueError(f"Function '{fn_name}' not found in BeamDataCollector module.")

#         # Optionally, you can customize out_key per function if needed
#         t = Thread(target=run_measure, args=(r, input_key, callback_fn, output_key))
#         t.daemon = True
#         t.start()
#         threads.append(t)

#     # Keep main thread alive
#     while True:
#         time.sleep(1)



import BeamDataCollector
import redis
import yaml
import time

if __name__ == "__main__":

    try:
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        redis_host = config['redis_host']
        redis_port = config['redis_port']
        input_key = config['input_key']
        output_key = config['output_key']
        function_names = config['function_names']
    except FileNotFoundError:
        print("Error: config.yml not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YML file: {e}")
        exit(1)

    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=False
    )

    # Prepare callback functions
    callbacks = []
    for fn_name in function_names:
        try:
            callbacks.append(getattr(BeamDataCollector, fn_name))
        except AttributeError:
            raise ValueError(f"Function '{fn_name}' not found in BeamDataCollector module.")

    last_id = '$'
    while True:
        entries = r.xread({input_key: last_id}, block=0, count=1)
        if entries:
            for stream_name, messages in entries:
                for msg_id, fields in messages:
                    try:
                        y = BeamDataCollector.extract_signal_from_binary(fields)
                        combined_result = {}
                        for fn in callbacks:
                            result = fn(y)
                            if not isinstance(result, dict):
                                raise ValueError("Callback did not return a dict")
                            combined_result.update(result)
                        r.xadd(output_key, combined_result, maxlen=10)
                        last_id = msg_id
                        print(f"Processed entry {msg_id} from {input_key}")
                    except Exception as e:
                        print(f"Error processing entry {msg_id}: {e}")
        else:
            time.sleep(0.1)