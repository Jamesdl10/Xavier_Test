from pymavlink import mavutil

# Establish connection to the Pixhawk via serial port
master = mavutil.mavlink_connection('/dev/ttyTHS0', baud=57600)

# Optional: Wait for the first heartbeat to ensure communication is established
master.wait_heartbeat()
print(f"Heartbeat from system (system {master.target_system} component {master.target_component})")

# Continuously read and print all incoming MAVLink messages
while True:
    msg = master.recv_match(blocking=True)
    if msg:
        print(msg)
