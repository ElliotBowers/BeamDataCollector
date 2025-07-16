import json
import numpy as np
# import pydapter_wraptor_py.pydapter_wraptor
from scipy import sparse
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import spsolve
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
# import pydapter_wraptor_py


# Calculate the baseline using Asymmetric Least Squares Smoothing (ALS)
def als_baseline(y, lam=1e6, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        A = Z.tocsr()
        z = spsolve(A, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def als_baseline_corrected(z):
    baseline = als_baseline(z)
    corrected = z - baseline
    return {
        'corrected': json.dumps(corrected.tolist())
    }

def squarify(y):
    _, corrected, peak_value, left_index, right_index = get_prereqs(y)
    corrected[left_index:right_index + 1] = peak_value
    corrected[right_index:] = 0
    corrected[:left_index] = 0
    return {
        'squarified': json.dumps(corrected.tolist())
    }
    

def width(y):
    left, right = get_prereqs(y)[3:]
    return {
        'width': right - left
    }


def current(y, sampling_rate=5_000_000, f_corner=250, peak_fraction=0.9):
    corrected = get_prereqs(y, sampling_rate, f_corner)[1]

    peak_index = np.argmax(corrected)
    peak_value = corrected[peak_index]
    threshold = peak_fraction * peak_value

    # Define top region: points above threshold
    top_indices = np.where(corrected >= threshold)[0]

    if len(top_indices) == 0:
        raise ValueError("No values found above threshold; check pulse shape or threshold level.")

    avg_current = np.mean(corrected[top_indices])
    # pulse_width = len(top_indices)
    # total_charge = avg_current * pulse_width / sampling_rate  # A × s = Coulombs

    return {
        'current': round(float(avg_current), 4),
        # 'duration': round(pulse_width / sampling_rate, 6),  # in seconds
        # 'charge': round(total_charge, 6),  # in Coulombs
    }


def intensity(y):
    _, corrected, _, left_index, right_index = get_prereqs(y)
    my_range = right_index - left_index
    ten_percent = int(my_range * 0.1)
    start = max(left_index - ten_percent, 0)
    end = min(right_index + ten_percent, len(corrected))

    y_slice = corrected[start:end]
    x_vals = np.arange(len(y_slice))

    area = np.trapezoid(y=y_slice, x=x_vals)
    return {
        'intensity': round(float(area), 3)  # ensure clean Redis output
    }


def mode_filter(signal, window_size=3):
    s = pd.Series(signal)
    filtered = s.rolling(window=window_size, center=True, min_periods=1)\
                .apply(lambda x: x.mode().iloc[0])
    return filtered.to_numpy()


"""ALS version"""    
# def get_prereqs(y):
#     y = mode_filter(y, window_size=5)  # Apply mode filter to smooth the signal
#     baseline = als_baseline(y)
#     corrected = y - baseline

#     peak_index = np.argmax(corrected)
#     peak_value = corrected[peak_index]
#     half_max = 0.5 * peak_value

#     left_index = peak_index
#     while left_index > 0 and corrected[left_index] > half_max:
#         left_index -= 1

#     right_index = peak_index
#     while right_index < len(corrected) - 1 and corrected[right_index] > half_max:
#         right_index += 1
    
#     return baseline, corrected, peak_value, left_index, right_index

def get_prereqs(y, sampling_rate=5_000_000, f_corner=100):
    droop = droop_correct(y, sampling_rate, f_corner)
    smoothed = mode_filter(droop, window_size=5)
    baseline = als_baseline(smoothed)
    corrected = smoothed - baseline

    peak_index = np.argmax(corrected)
    peak_value = corrected[peak_index]
    half_max = 0.5 * peak_value

    left_index = peak_index
    while left_index > 0 and corrected[left_index] > half_max:
        left_index -= 1

    right_index = peak_index
    while right_index < len(corrected) - 1 and corrected[right_index] > half_max:
        right_index += 1

    return baseline, corrected, peak_value, left_index, right_index


"""Brianna's version of droop correction"""
# def droop_correct(y, sampling_rate=100_000, f_corner=70):
#     dt = 1 / sampling_rate
#     droop_const = 2 * np.pi * f_corner
#     corrected = np.zeros_like(y, dtype=float)
#     integrated = 0.0

#     for i in range(len(y) - 1):  # Avoid out of bounds
#         integrated += y[i] * dt
#         corrected[i] = y[i] + integrated * droop_const

#     # Shift so that the signal starts near zero
#     corrected -= np.min(corrected)
#     return corrected


def droop_correct(y, sampling_rate=5_000_000, f_corner=250):
    dt = 1 / sampling_rate
    droop_const = 2 * np.pi * f_corner

    # Cumulative integral of the signal
    integral = cumulative_trapezoid(y, dx=dt, initial=0)

    # Apply correction
    corrected = y + droop_const * integral

    # Optional: shift to start near 0
    corrected -= np.min(corrected)

    return corrected

"""Brianna version"""
# def get_prereqs(y):
#     corrected = droop_correct(y)

#     peak_index = np.argmax(corrected)
#     peak_value = corrected[peak_index]
#     half_max = 0.5 * peak_value

#     left_index = peak_index
#     while left_index > 0 and corrected[left_index] > half_max:
#         left_index -= 1

#     right_index = peak_index
#     while right_index < len(corrected) - 1 and corrected[right_index] > half_max:
#         right_index += 1
    

#     return None, corrected, peak_value, left_index, right_index



# def extract_first_signal_entry(fields):
#     """Attempt to parse the first JSON-parsable field as a NumPy array."""
#     for value in fields.values():
#         try:
#             data = json.loads(value)
#             if isinstance(data, list):  # ensure it looks like signal data
#                 return np.array(data)
#         except json.JSONDecodeError:
#             continue
#     raise ValueError("No valid JSON signal data found in stream entry.")


# def measure(r, in_key, callback, out_key):
#     """Continuously stream from Redis using XREAD and apply a processing callback."""
#     last_id = '$'  # '$' starts from new entries
#     while True:
#         print("Listening for data...")
#         entries = r.xread({in_key: last_id}, block=1000, count=1)
#         if entries:
#             print("Got entries...")
#             for stream_name, messages in entries:
#                 for msg_id, fields in messages:
#                     try:
#                         y = extract_first_signal_entry(fields)
#                         result = callback(y)  # Must return a dictionary
#                         r.xadd(out_key, result, maxlen=10)
#                         last_id = msg_id  # move forward in the stream
#                         print(f"Processed entry {msg_id} from {in_key}")
#                     except Exception as e:
#                         print(f"Error processing entry {msg_id}: {e}")

def extract_signal_from_binary(fields):
    """Extracts the first binary field and converts it into a NumPy array of int16."""
    for value in fields.values():
        if isinstance(value, bytes):
            try:
                # Interpret the byte string as little-endian int16 values
                data = np.frombuffer(value, dtype='<i2')
                return data
            except Exception as e:
                print(f"Error decoding binary field: {e}")
                continue
    raise ValueError("No valid binary signal data found in stream entry.")


def measure(r, in_key, callback, out_key):
    """Continuously stream from Redis using XREAD and apply a processing callback."""
    last_id = '$'  # '$' starts from new entries
    while True:
        print("Listening for data...")
        entries = r.xread({in_key: last_id}, block=0, count=1)
        print(in_key)
        # entries = r.xreadMultiBlock({in_key: last_id}, 2000)
        print("entries: ", entries, type(entries))
        if entries:
            print("Got entries...")
            for stream_name, messages in entries:
                for msg_id, fields in messages:
                    try:
                        y = extract_signal_from_binary(fields)  # <- updated to binary
                        result = callback(y)  # Must return a dictionary
                        # r.xadd(out_key, "*", result)
                        r.xadd(out_key, result, maxlen=10)
                        last_id = msg_id  # move forward in the stream
                        print(f"Processed entry {msg_id} from {in_key}")
                    except Exception as e:
                        print(f"Error processing entry {msg_id}: {e}")


def process_pulse(y):
    baseline, corrected, peak_value, left_index, right_index = get_prereqs(y)

    flattened = corrected.copy()
    flattened[left_index:right_index + 1] = peak_value
    flattened[:left_index] = 0
    flattened[right_index:] = 0
    rectified = flattened + baseline

    pulse_width = int(right_index - left_index)
    current = peak_value
    intensity = current * pulse_width

    return {
        'signal': json.dumps(rectified.tolist()),
        'current': round(current, 4),
        'intensity': round(intensity, 4),
        'width': pulse_width
    }



# def daemon_loop(base_key, out_key_base, r, chunk_size=500, sleep_time=0.5):
#     last_index = 0
#     while True:
#         key = f"{base_key}:{last_index}"
#         raw = r.get(key)
#         if raw:
#             y = np.array(json.loads(raw))
#             processed, current, intensity, width = process_pulse(y)
#             r.set(f"{out_key_base}:signal:{last_index}", json.dumps(processed))
#             r.set(f"{out_key_base}:current:{last_index}", json.dumps(current))
#             r.set(f"{out_key_base}:intensity:{last_index}", json.dumps(intensity))
#             r.set(f"{out_key_base}:width:{last_index}", json.dumps(width))
#             print(f"Processed chunk {last_index}")
#             last_index += 1
#         else:
#             time.sleep(sleep_time)  # Avoid CPU spamming

# def wait_until(target_ts):
#     """Wait until the system time reaches target_ts (seconds since epoch)."""
#     while True:
#         now = time.time()
#         if now >= target_ts:
#             break
#         time.sleep(min(0.01, target_ts - now))

# def daemon_loop_sync_overlap(base_key, out_key_base, r, chunk_size=500, overlap=50):
#     step_size = chunk_size - overlap
#     index = 0

#     while True:
#         key = f"{base_key}:{index}"
#         ts_key = f"{base_key}:ts:{index}"
#         raw = r.get(key)
#         ts_raw = r.get(ts_key)

#         if raw and ts_raw:
#             target_ts = float(ts_raw)
#             wait_until(target_ts)

#             y = np.array(json.loads(raw))
            
#             # Re-fetch previous samples for overlap if needed
#             if index > 0:
#                 prev_key = f"{base_key}:{index-1}"
#                 prev_raw = r.get(prev_key)
#                 if prev_raw:
#                     prev_y = np.array(json.loads(prev_raw))
#                     y = np.concatenate([prev_y[-overlap:], y])

#             # Process the (overlapped) chunk
#             processed, current, intensity, width = process_pulse(y)
#             r.set(f"{out_key_base}:signal:{index}", json.dumps(processed))
#             r.set(f"{out_key_base}:current:{index}", json.dumps(current))
#             r.set(f"{out_key_base}:intensity:{index}", json.dumps(intensity))
#             r.set(f"{out_key_base}:width:{index}", json.dumps(width))
#             print(f"[{time.strftime('%H:%M:%S')}] Processed chunk {index} at t={target_ts}")
#             index += 1
#         else:
#             time.sleep(0.01)


