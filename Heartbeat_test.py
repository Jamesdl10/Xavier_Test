from pymavlink import mavutil

# Establish connection to the Pixhawk
master = mavutil.mavlink_connection('/dev/ttyTHS0', baud=57600)

# Send an initial heartbeat to prompt the Pixhawk
master.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_GCS,          # Type of the MAV (ground control station)
    mavutil.mavlink.MAV_AUTOPILOT_INVALID, # Autopilot type (not applicable here)
    0, 0, 0                                # Mode flags, custom mode, system status
)

# Wait for the first heartbeat from the Pixhawk
print("Waiting for heartbeat from system...")
master.wait_heartbeat()
print(f"Heartbeat received from system (system {master.target_system} component {master.target_component})")
