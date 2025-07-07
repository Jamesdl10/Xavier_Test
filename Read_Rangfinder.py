from pymavlink import mavutil
import time

# 1. Connect to the Pixhawk over the correct serial port and baud rate
# Replace with the appropriate device (e.g. /dev/ttyUSB0 or /dev/ttyTHS2)
master = mavutil.mavlink_connection('/dev/ttyTHS0', baud=57600)

# 2. Wait for a heartbeat to confirm connection
master.wait_heartbeat()
print(f"Connected to system {master.target_system} component {master.target_component}")

# 3. OPTIONAL: Request DISTANCE_SENSOR message rate (10 Hz = 100,000 µs)
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,  # message_id
    100000,  # interval in microseconds
    0, 0, 0, 0, 0
)

# 4. Receive and print rangefinder data
try:
    while True:
        msg = master.recv_match(type='DISTANCE_SENSOR', blocking=False)
        if msg:
            distance_m = msg.current_distance /100
            print(f"[{msg.time_boot_ms} ms] Rangefinder: {distance_m:.3f} m")
        time.sleep(0.05)  # Limit polling rate
except KeyboardInterrupt:
    print("Stopped.")
