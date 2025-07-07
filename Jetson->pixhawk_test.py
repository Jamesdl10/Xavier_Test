from pymavlink import mavutil
import time

# Connect to TELEM2 with correct baud and a separate system ID
master = mavutil.mavlink_connection(
    '/dev/ttyTHS0', baud=57600, source_system=2  # Pixhawk is system 1
)

# Wait for heartbeat from Pixhawk (confirms the link is live)
master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

# Send Jetson's own heartbeat so Pixhawk sees us as a participant
master.mav.heartbeat_send(
    type=mavutil.mavlink.MAV_TYPE_GCS,
    autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
    base_mode=0,
    custom_mode=0,
    system_status=0
)

# Wait a moment so Pixhawk can register us
time.sleep(2)

# Send a STATUSTEXT message
master.mav.statustext_send(
    severity=3,
    text="✅ Jetson system 2 reporting in! it works!!".encode('utf-8')
)

print("STATUSTEXT sent.")
