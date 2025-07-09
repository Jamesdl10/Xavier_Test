#!/usr/bin/env python3
"""
Comprehensive Ground Test Suite - NO PROPELLERS REQUIRED
Tests all systems without any flight risk
"""

import asyncio
import time
import logging
import threading
from dataclasses import dataclass
from pymavlink import mavutil
import json
from datetime import datetime

# ============================================================================
# TEST DATA COLLECTION AND VALIDATION
# ============================================================================

@dataclass
class TestResults:
    """Store test results for analysis"""
    test_name: str
    passed: bool
    data_collected: dict
    errors: list
    timestamp: float
    duration: float

class DataCollector:
    """Collect and validate streaming data"""
    def __init__(self):
        self.collected_data = {
            'LOCAL_POSITION_NED': [],
            'ATTITUDE': [],
            'DISTANCE_SENSOR': [],
            'BATTERY_STATUS': [],
            'HEARTBEAT': [],
            'COMMAND_ACK': []
        }
        self.collection_active = False
        self.start_time = 0
        
    def start_collection(self):
        """Start data collection"""
        self.collection_active = True
        self.start_time = time.time()
        for key in self.collected_data:
            self.collected_data[key] = []
    
    def stop_collection(self):
        """Stop data collection"""
        self.collection_active = False
    
    def add_message(self, msg):
        """Add message to collection"""
        if not self.collection_active:
            return
            
        msg_type = msg.get_type()
        if msg_type in self.collected_data:
            timestamp = time.time() - self.start_time
            
            if msg_type == 'LOCAL_POSITION_NED':
                data = {
                    'timestamp': timestamp,
                    'x': msg.x, 'y': msg.y, 'z': msg.z,
                    'vx': msg.vx, 'vy': msg.vy, 'vz': msg.vz
                }
            elif msg_type == 'ATTITUDE':
                data = {
                    'timestamp': timestamp,
                    'roll': msg.roll, 'pitch': msg.pitch, 'yaw': msg.yaw,
                    'rollspeed': msg.rollspeed, 'pitchspeed': msg.pitchspeed, 'yawspeed': msg.yawspeed
                }
            elif msg_type == 'DISTANCE_SENSOR':
                data = {
                    'timestamp': timestamp,
                    'distance_cm': msg.current_distance,
                    'distance_m': msg.current_distance / 100.0
                }
            elif msg_type == 'BATTERY_STATUS':
                voltage = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else 0
                data = {
                    'timestamp': timestamp,
                    'voltage': voltage,
                    'remaining': msg.battery_remaining,
                    'current': msg.current_battery / 100.0 if msg.current_battery != -1 else 0
                }
            elif msg_type == 'HEARTBEAT':
                data = {
                    'timestamp': timestamp,
                    'custom_mode': msg.custom_mode,
                    'base_mode': msg.base_mode,
                    'armed': bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                }
            elif msg_type == 'COMMAND_ACK':
                data = {
                    'timestamp': timestamp,
                    'command': msg.command,
                    'result': msg.result
                }
            else:
                data = {'timestamp': timestamp, 'raw': str(msg)}
            
            self.collected_data[msg_type].append(data)
    
    def get_analysis(self):
        """Analyze collected data"""
        analysis = {}
        
        for msg_type, messages in self.collected_data.items():
            if messages:
                count = len(messages)
                duration = messages[-1]['timestamp'] - messages[0]['timestamp'] if count > 1 else 0
                frequency = count / duration if duration > 0 else 0
                
                analysis[msg_type] = {
                    'message_count': count,
                    'duration_seconds': duration,
                    'frequency_hz': frequency,
                    'first_message_time': messages[0]['timestamp'],
                    'last_message_time': messages[-1]['timestamp']
                }
            else:
                analysis[msg_type] = {
                    'message_count': 0,
                    'frequency_hz': 0,
                    'status': 'NO_DATA'
                }
        
        return analysis

# ============================================================================
# GROUND TEST CLASSES
# ============================================================================

class GroundTestSuite:
    """Comprehensive ground testing without propellers"""
    
    def __init__(self):
        self.master = None
        self.connected = False
        self.data_collector = DataCollector()
        self.test_results = []
        self.logger = logging.getLogger(__name__)
        
        # Test configuration
        self.connection_params = {
            'device': '/dev/ttyTHS0',
            'baudrate': 57600,
            'source_system': 2,
            'timeout': 10
        }
        
        # Expected data rates (Hz)
        self.expected_rates = {
            'LOCAL_POSITION_NED': 20,
            'ATTITUDE': 20,
            'DISTANCE_SENSOR': 20,
            'BATTERY_STATUS': 5,
            'HEARTBEAT': 1
        }
    
    def connect_mavlink(self) -> bool:
        """Establish MAVLink connection for testing"""
        try:
            print(f"🔌 Connecting to {self.connection_params['device']} at {self.connection_params['baudrate']} baud...")
            
            self.master = mavutil.mavlink_connection(
                self.connection_params['device'],
                baud=self.connection_params['baudrate'],
                source_system=self.connection_params['source_system']
            )
            
            print("⏳ Waiting for heartbeat...")
            self.master.wait_heartbeat()
            
            print(f"✅ Connected to system {self.master.target_system}, component {self.master.target_component}")
            
            # Send our heartbeat
            self.master.mav.heartbeat_send(
                type=mavutil.mavlink.MAV_TYPE_GCS,
                autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                base_mode=0, custom_mode=0, system_status=0
            )
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect_mavlink(self):
        """Close MAVLink connection"""
        if self.master:
            self.master.close()
            self.connected = False
            print("🔌 Disconnected from MAVLink")
    
    # ========================================================================
    # TEST 1: BASIC COMMUNICATION TEST
    # ========================================================================
    
    def test_basic_communication(self) -> TestResults:
        """Test 1: Basic MAVLink communication"""
        print("\n" + "="*60)
        print("TEST 1: BASIC MAVLINK COMMUNICATION")
        print("="*60)
        
        start_time = time.time()
        errors = []
        data = {}
        
        try:
            if not self.connect_mavlink():
                return TestResults("Basic Communication", False, {}, ["Connection failed"], time.time(), 0)
            
            # Test heartbeat reception
            print("📡 Testing heartbeat reception...")
            heartbeat_count = 0
            test_duration = 5
            start = time.time()
            
            while time.time() - start < test_duration:
                msg = self.master.recv_match(type='HEARTBEAT', blocking=False, timeout=1)
                if msg:
                    heartbeat_count += 1
                    data['heartbeat_received'] = True
                    data['system_id'] = msg.get_srcSystem()
                    data['component_id'] = msg.get_srcComponent()
                time.sleep(0.1)
            
            heartbeat_rate = heartbeat_count / test_duration
            data['heartbeat_rate'] = heartbeat_rate
            
            print(f"📊 Heartbeat rate: {heartbeat_rate:.1f} Hz (expected: ~1 Hz)")
            
            # Test message sending
            print("📤 Testing message sending...")
            test_message = "✅ Ground Test 1 - Basic Communication"
            self.master.mav.statustext_send(6, test_message.encode('utf-8'))
            data['message_sent'] = True
            
            # Verify connection health
            if heartbeat_rate > 0.5:
                print("✅ Communication healthy")
                passed = True
            else:
                print("❌ Communication unreliable")
                errors.append("Low heartbeat rate")
                passed = False
            
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Basic Communication", passed, data, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    # ========================================================================
    # TEST 2: DATA STREAMING VALIDATION
    # ========================================================================
    
    def test_data_streaming(self) -> TestResults:
        """Test 2: Validate all required data streams"""
        print("\n" + "="*60)
        print("TEST 2: DATA STREAMING VALIDATION")
        print("="*60)
        
        start_time = time.time()
        errors = []
        passed = True
        
        if not self.connected:
            if not self.connect_mavlink():
                return TestResults("Data Streaming", False, {}, ["Connection failed"], time.time(), 0)
        
        try:
            # Request data streams at specified rates
            print("📡 Requesting data streams...")
            self._request_all_data_streams()
            
            # Allow streams to stabilize
            print("⏳ Waiting for streams to stabilize (3 seconds)...")
            time.sleep(3)
            
            # Collect data for analysis
            print("📊 Collecting data for 10 seconds...")
            self.data_collector.start_collection()
            
            collection_start = time.time()
            message_count = 0
            
            while time.time() - collection_start < 10:
                msg = self.master.recv_match(blocking=False, timeout=0.1)
                if msg:
                    self.data_collector.add_message(msg)
                    message_count += 1
                    
                    # Print live updates
                    if message_count % 50 == 0:
                        elapsed = time.time() - collection_start
                        print(f"   📈 Collected {message_count} messages in {elapsed:.1f}s")
            
            self.data_collector.stop_collection()
            
            # Analyze collected data
            print("\n📊 ANALYZING COLLECTED DATA:")
            analysis = self.data_collector.get_analysis()
            
            for msg_type, stats in analysis.items():
                expected_rate = self.expected_rates.get(msg_type, 0)
                actual_rate = stats['frequency_hz']
                count = stats['message_count']
                
                print(f"\n📋 {msg_type}:")
                print(f"   Messages: {count}")
                print(f"   Rate: {actual_rate:.1f} Hz (expected: {expected_rate} Hz)")
                
                # Validate data content
                if msg_type in self.data_collector.collected_data and self.data_collector.collected_data[msg_type]:
                    self._validate_message_content(msg_type, self.data_collector.collected_data[msg_type])
                
                # Check if rate is acceptable (within 50% of expected)
                if expected_rate > 0:
                    rate_ok = actual_rate >= expected_rate * 0.5
                    status = "✅ OK" if rate_ok else "❌ LOW"
                    print(f"   Status: {status}")
                    
                    if not rate_ok:
                        errors.append(f"{msg_type} rate too low: {actual_rate:.1f} Hz")
                        passed = False
                else:
                    print(f"   Status: ✅ RECEIVED")
            
            # Summary
            total_messages = sum(stats['message_count'] for stats in analysis.values())
            print(f"\n📊 SUMMARY:")
            print(f"   Total messages: {total_messages}")
            print(f"   Message types: {len([s for s in analysis.values() if s['message_count'] > 0])}")
            print(f"   Data collection: {'✅ SUCCESS' if total_messages > 0 else '❌ FAILED'}")
            
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Data Streaming", passed, analysis, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"\n⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    def _request_all_data_streams(self):
        """Request all required data streams"""
        streams = [
            (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20),  # 20Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20),           # 20Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 20),    # 20Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 5),      # 5Hz
            (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5),         # 5Hz
        ]
        
        for msg_id, rate_hz in streams:
            interval_us = int(1e6 / rate_hz)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0, msg_id, interval_us, 0, 0, 0, 0, 0
            )
            print(f"   📡 Requested {msg_id} at {rate_hz} Hz")
    
    def _validate_message_content(self, msg_type, messages):
        """Validate message content for sanity"""
        if not messages:
            return
        
        latest = messages[-1]
        
        if msg_type == 'LOCAL_POSITION_NED':
            # Check for reasonable position values (should be near 0 on ground)
            if abs(latest['x']) > 100 or abs(latest['y']) > 100:
                print(f"   ⚠️  Large position values: x={latest['x']:.1f}, y={latest['y']:.1f}")
        
        elif msg_type == 'ATTITUDE':
            # Check for reasonable attitude values
            roll_deg = latest['roll'] * 57.3
            pitch_deg = latest['pitch'] * 57.3
            if abs(roll_deg) > 45 or abs(pitch_deg) > 45:
                print(f"   ⚠️  Large attitude values: roll={roll_deg:.1f}°, pitch={pitch_deg:.1f}°")
        
        elif msg_type == 'BATTERY_STATUS':
            # Check for reasonable battery values
            voltage = latest['voltage']
            if voltage < 10 or voltage > 20:
                print(f"   ⚠️  Unusual battery voltage: {voltage:.1f}V")
        
        elif msg_type == 'DISTANCE_SENSOR':
            # Check for reasonable distance values
            distance = latest['distance_m']
            if distance < 0.1 or distance > 20:
                print(f"   ⚠️  Unusual distance reading: {distance:.1f}m")
    
    # ========================================================================
    # TEST 3: GUIDED MODE TESTING (NO MOVEMENT)
    # ========================================================================
    
    def test_guided_mode(self) -> TestResults:
        """Test 3: Guided mode commands without movement"""
        print("\n" + "="*60)
        print("TEST 3: GUIDED MODE TESTING (NO MOVEMENT)")
        print("="*60)
        print("⚠️  ENSURE PROPELLERS ARE REMOVED!")
        
        response = input("Are propellers REMOVED and drone SAFE for ground testing? (yes/no): ")
        if response.lower() != 'yes':
            return TestResults("Guided Mode", False, {}, ["Safety not confirmed"], time.time(), 0)
        
        start_time = time.time()
        errors = []
        data = {}
        passed = True
        
        if not self.connected:
            if not self.connect_mavlink():
                return TestResults("Guided Mode", False, {}, ["Connection failed"], time.time(), 0)
        
        try:
            # Get initial mode
            print("📊 Getting current flight mode...")
            initial_mode = self._get_current_mode()
            data['initial_mode'] = initial_mode
            print(f"   Current mode: {initial_mode}")
            
            # Test mode change to GUIDED
            print("🎮 Testing mode change to GUIDED...")
            if self._set_mode_and_verify('GUIDED'):
                print("   ✅ Successfully set GUIDED mode")
                data['guided_mode_set'] = True
            else:
                print("   ❌ Failed to set GUIDED mode")
                errors.append("Failed to set GUIDED mode")
                passed = False
            
            # Test position commands (at current location)
            if passed:
                print("📍 Testing position commands...")
                success = self._test_position_commands()
                data['position_commands_sent'] = success
                if success:
                    print("   ✅ Position commands sent successfully")
                else:
                    print("   ❌ Position command failed")
                    errors.append("Position command failed")
                    passed = False
            
            # Test other mode changes
            if passed:
                print("🔄 Testing mode changes...")
                modes_to_test = ['LOITER', 'ALT_HOLD', 'STABILIZE']
                for mode in modes_to_test:
                    if self._set_mode_and_verify(mode):
                        print(f"   ✅ {mode} mode set successfully")
                        data[f'{mode.lower()}_mode_set'] = True
                    else:
                        print(f"   ❌ Failed to set {mode} mode")
                        errors.append(f"Failed to set {mode} mode")
                        # Don't fail test for other modes, but log error
            
            # Return to initial mode
            print(f"🔄 Returning to initial mode: {initial_mode}")
            self._set_mode_and_verify(initial_mode)
            
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Guided Mode", passed, data, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"\n⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    def _get_current_mode(self) -> str:
        """Get current flight mode"""
        msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if msg:
            mode_mapping = {
                0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO',
                4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 7: 'CIRCLE',
                9: 'LAND', 17: 'BRAKE', 18: 'THROW'
            }
            return mode_mapping.get(msg.custom_mode, f'UNKNOWN_{msg.custom_mode}')
        return 'UNKNOWN'
    
    def _set_mode_and_verify(self, mode: str) -> bool:
        """Set mode and verify it was set"""
        mode_mapping = {
            'STABILIZE': 0, 'ACRO': 1, 'ALT_HOLD': 2, 'AUTO': 3,
            'GUIDED': 4, 'LOITER': 5, 'RTL': 6, 'CIRCLE': 7,
            'LAND': 9, 'BRAKE': 17, 'THROW': 18
        }
        
        if mode not in mode_mapping:
            return False
        
        # Send mode change command
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_mapping[mode]
        )
        
        # Wait and verify
        time.sleep(2)
        current_mode = self._get_current_mode()
        return current_mode == mode
    
    def _test_position_commands(self) -> bool:
        """Test position commands at current location"""
        try:
            # Get current position first
            pos_msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=5)
            if not pos_msg:
                print("   ❌ No position data available")
                return False
            
            current_x = pos_msg.x
            current_y = pos_msg.y
            current_z = pos_msg.z
            
            print(f"   📍 Current position: N={current_x:.2f}, E={current_y:.2f}, D={current_z:.2f}")
            
            # Send position commands at current location (no movement)
            for i in range(3):
                self.master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000111111111000,  # type_mask (position only)
                    current_x,  # x (North)
                    current_y,  # y (East) 
                    current_z,  # z (Down)
                    0, 0, 0,    # velocity
                    0, 0, 0,    # acceleration
                    0, 0        # yaw, yaw_rate
                )
                print(f"   📤 Position command {i+1} sent")
                time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Position command error: {e}")
            return False
    
    # ========================================================================
    # TEST 4: BATTERY MONITORING (4S SPECIFIC)
    # ========================================================================
    
    def test_battery_monitoring(self) -> TestResults:
        """Test 4: 4S Battery monitoring and thresholds"""
        print("\n" + "="*60)
        print("TEST 4: 4S BATTERY MONITORING")
        print("="*60)
        
        start_time = time.time()
        errors = []
        data = {}
        passed = True
        
        # 4S Battery thresholds
        battery_thresholds = {
            'nominal_voltage': 14.8,
            'full_voltage': 16.8,
            'critical_voltage': 14.0,
            'warning_voltage': 14.8,
            'min_takeoff_voltage': 15.6
        }
        
        if not self.connected:
            if not self.connect_mavlink():
                return TestResults("Battery Monitoring", False, {}, ["Connection failed"], time.time(), 0)
        
        try:
            # Request battery status
            print("🔋 Requesting battery status...")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
                500000,  # 2Hz
                0, 0, 0, 0, 0
            )
            
            # Collect battery data
            print("📊 Collecting battery data for 15 seconds...")
            battery_readings = []
            collection_start = time.time()
            
            while time.time() - collection_start < 15:
                msg = self.master.recv_match(type='BATTERY_STATUS', blocking=False, timeout=1)
                if msg:
                    if msg.voltages[0] != 65535:  # Valid voltage reading
                        voltage = msg.voltages[0] / 1000.0
                        current = msg.current_battery / 100.0 if msg.current_battery != -1 else 0
                        remaining = msg.battery_remaining
                        
                        battery_readings.append({
                            'timestamp': time.time() - collection_start,
                            'voltage': voltage,
                            'current': current,
                            'remaining': remaining
                        })
                        
                        print(f"   🔋 V: {voltage:.2f}V, I: {current:.1f}A, Remaining: {remaining}%")
                
                time.sleep(0.5)
            
            if not battery_readings:
                errors.append("No battery data received")
                passed = False
            else:
                # Analyze battery data
                latest_reading = battery_readings[-1]
                avg_voltage = sum(r['voltage'] for r in battery_readings) / len(battery_readings)
                voltage_stability = max(r['voltage'] for r in battery_readings) - min(r['voltage'] for r in battery_readings)
                
                data['latest_voltage'] = latest_reading['voltage']
                data['latest_remaining'] = latest_reading['remaining']
                data['average_voltage'] = avg_voltage
                data['voltage_stability'] = voltage_stability
                data['reading_count'] = len(battery_readings)
                
                print(f"\n📊 BATTERY ANALYSIS:")
                print(f"   Latest voltage: {latest_reading['voltage']:.2f}V")
                print(f"   Average voltage: {avg_voltage:.2f}V")
                print(f"   Voltage stability: {voltage_stability:.3f}V")
                print(f"   Remaining: {latest_reading['remaining']}%")
                print(f"   Readings collected: {len(battery_readings)}")
                
                # Check against 4S thresholds
                print(f"\n🔍 4S THRESHOLD ANALYSIS:")
                for threshold_name, threshold_value in battery_thresholds.items():
                    if 'voltage' in threshold_name:
                        status = "✅ OK" if latest_reading['voltage'] >= threshold_value else "❌ LOW"
                        print(f"   {threshold_name}: {threshold_value}V - {status}")
                        
                        if threshold_name == 'critical_voltage' and latest_reading['voltage'] < threshold_value:
                            errors.append(f"Battery below critical voltage: {latest_reading['voltage']:.2f}V")
                
                # Check if suitable for flight testing
                flight_ready = latest_reading['voltage'] >= battery_thresholds['min_takeoff_voltage']
                data['flight_ready'] = flight_ready
                
                print(f"\n🚁 FLIGHT READINESS:")
                print(f"   Battery suitable for takeoff: {'✅ YES' if flight_ready else '❌ NO'}")
                
                if voltage_stability > 0.5:
                    errors.append("High voltage instability")
                    print(f"   ⚠️  High voltage instability: {voltage_stability:.3f}V")
        
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Battery Monitoring", passed, data, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"\n⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    # ========================================================================
    # TEST 5: SENSOR VALIDATION
    # ========================================================================
    
    def test_sensor_validation(self) -> TestResults:
        """Test 5: Validate all sensors and their data quality"""
        print("\n" + "="*60)
        print("TEST 5: SENSOR VALIDATION")
        print("="*60)
        
        start_time = time.time()
        errors = []
        data = {}
        passed = True
        
        if not self.connected:
            if not self.connect_mavlink():
                return TestResults("Sensor Validation", False, {}, ["Connection failed"], time.time(), 0)
        
        try:
            # Test GPS
            print("🛰️  Testing GPS...")
            gps_result = self._test_gps()
            data.update(gps_result)
            if not gps_result.get('gps_healthy', False):
                errors.append("GPS not healthy")
            
            # Test IMU/Attitude
            print("📐 Testing IMU/Attitude...")
            imu_result = self._test_imu()
            data.update(imu_result)
            if not imu_result.get('imu_healthy', False):
                errors.append("IMU not healthy")
            
            # Test Rangefinder
            print("📏 Testing Rangefinder...")
            range_result = self._test_rangefinder()
            data.update(range_result)
            if not range_result.get('rangefinder_healthy', False):
                errors.append("Rangefinder not healthy")
            
            # Overall sensor health
            sensor_count = sum(1 for key in data.keys() if key.endswith('_healthy') and data[key])
            total_sensors = 3
            
            data['healthy_sensor_count'] = sensor_count
            data['total_sensor_count'] = total_sensors
            
            print(f"\n📊 SENSOR HEALTH SUMMARY:")
            print(f"   Healthy sensors: {sensor_count}/{total_sensors}")
            
            if sensor_count < total_sensors:
                passed = False
                print(f"   ❌ Some sensors unhealthy")
            else:
                print(f"   ✅ All sensors healthy")
        
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Sensor Validation", passed, data, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"\n⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    def _test_gps(self) -> dict:
        """Test GPS functionality"""
        try:
            msg = self.master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=10)
            if msg:
                fix_type = msg.fix_type
                satellites = msg.satellites_visible
                hdop = msg.eph / 100.0 if msg.eph != 65535 else 999
                
                print(f"   Fix type: {fix_type} (3=3D fix)")
                print(f"   Satellites: {satellites}")
                print(f"   HDOP: {hdop:.1f}")
                
                healthy = fix_type >= 3 and satellites >= 6 and hdop < 5.0
                status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
                print(f"   Status: {status}")
                
                return {
                    'gps_fix_type': fix_type,
                    'gps_satellites': satellites,
                    'gps_hdop': hdop,
                    'gps_healthy': healthy
                }
            else:
                print("   ❌ No GPS data received")
                return {'gps_healthy': False}
        except Exception as e:
            print(f"   ❌ GPS test error: {e}")
            return {'gps_healthy': False}
    
    def _test_imu(self) -> dict:
        """Test IMU/Attitude functionality"""
        try:
            # Collect multiple attitude readings
            attitudes = []
            for _ in range(10):
                msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=2)
                if msg:
                    attitudes.append({
                        'roll': msg.roll,
                        'pitch': msg.pitch,
                        'yaw': msg.yaw
                    })
                time.sleep(0.1)
            
            if attitudes:
                latest = attitudes[-1]
                roll_deg = latest['roll'] * 57.3
                pitch_deg = latest['pitch'] * 57.3
                yaw_deg = latest['yaw'] * 57.3
                
                print(f"   Roll: {roll_deg:.1f}°")
                print(f"   Pitch: {pitch_deg:.1f}°")
                print(f"   Yaw: {yaw_deg:.1f}°")
                
                # Check for reasonable values (drone on ground)
                healthy = abs(roll_deg) < 45 and abs(pitch_deg) < 45
                status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
                print(f"   Status: {status}")
                
                return {
                    'imu_roll_deg': roll_deg,
                    'imu_pitch_deg': pitch_deg,
                    'imu_yaw_deg': yaw_deg,
                    'imu_healthy': healthy
                }
            else:
                print("   ❌ No IMU data received")
                return {'imu_healthy': False}
        except Exception as e:
            print(f"   ❌ IMU test error: {e}")
            return {'imu_healthy': False}
    
    def _test_rangefinder(self) -> dict:
        """Test rangefinder functionality"""
        try:
            # Collect multiple rangefinder readings
            distances = []
            for _ in range(10):
                msg = self.master.recv_match(type='DISTANCE_SENSOR', blocking=True, timeout=2)
                if msg:
                    distance_m = msg.current_distance / 100.0
                    distances.append(distance_m)
                time.sleep(0.1)
            
            if distances:
                latest_distance = distances[-1]
                avg_distance = sum(distances) / len(distances)
                distance_stability = max(distances) - min(distances)
                
                print(f"   Distance: {latest_distance:.2f}m")
                print(f"   Average: {avg_distance:.2f}m")
                print(f"   Stability: {distance_stability:.2f}m")
                
                # Check for reasonable values
                healthy = 0.1 < latest_distance < 20 and distance_stability < 0.5
                status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
                print(f"   Status: {status}")
                
                return {
                    'rangefinder_distance': latest_distance,
                    'rangefinder_stability': distance_stability,
                    'rangefinder_healthy': healthy
                }
            else:
                print("   ❌ No rangefinder data received")
                return {'rangefinder_healthy': False}
        except Exception as e:
            print(f"   ❌ Rangefinder test error: {e}")
            return {'rangefinder_healthy': False}
    
    # ========================================================================
    # TEST 6: COMMAND ACKNOWLEDGMENT
    # ========================================================================
    
    def test_command_acknowledgment(self) -> TestResults:
        """Test 6: Verify command acknowledgments work"""
        print("\n" + "="*60)
        print("TEST 6: COMMAND ACKNOWLEDGMENT")
        print("="*60)
        
        start_time = time.time()
        errors = []
        data = {}
        passed = True
        
        if not self.connected:
            if not self.connect_mavlink():
                return TestResults("Command Acknowledgment", False, {}, ["Connection failed"], time.time(), 0)
        
        try:
            # Request command acknowledgments
            print("📤 Testing command acknowledgments...")
            
            commands_to_test = [
                {
                    'name': 'Set Message Interval',
                    'command': mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    'params': [mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1000000, 0, 0, 0, 0, 0]
                },
                {
                    'name': 'Request Autopilot Version',
                    'command': mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
                    'params': [1, 0, 0, 0, 0, 0, 0]
                }
            ]
            
            for cmd_test in commands_to_test:
                print(f"\n📋 Testing: {cmd_test['name']}")
                
                # Send command
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    cmd_test['command'],
                    0,  # confirmation
                    *cmd_test['params']
                )
                
                # Wait for acknowledgment
                ack_received = False
                start_wait = time.time()
                
                while time.time() - start_wait < 3:
                    msg = self.master.recv_match(type='COMMAND_ACK', blocking=False, timeout=0.1)
                    if msg and msg.command == cmd_test['command']:
                        ack_received = True
                        result = msg.result
                        
                        result_names = {
                            0: 'ACCEPTED',
                            1: 'TEMPORARILY_REJECTED',
                            2: 'DENIED',
                            3: 'UNSUPPORTED',
                            4: 'FAILED'
                        }
                        
                        result_name = result_names.get(result, f'UNKNOWN_{result}')
                        print(f"   📨 ACK received: {result_name}")
                        
                        data[f'{cmd_test["name"].lower().replace(" ", "_")}_ack'] = result_name
                        break
                
                if not ack_received:
                    print(f"   ❌ No ACK received")
                    errors.append(f"No ACK for {cmd_test['name']}")
                    data[f'{cmd_test["name"].lower().replace(" ", "_")}_ack'] = 'NO_ACK'
            
            # Check if at least one ACK was received
            ack_count = sum(1 for key in data.keys() if '_ack' in key and data[key] != 'NO_ACK')
            data['total_acks_received'] = ack_count
            
            if ack_count == 0:
                passed = False
                print("\n❌ No command acknowledgments received")
            else:
                print(f"\n✅ Received {ack_count} command acknowledgments")
        
        except Exception as e:
            errors.append(str(e))
            passed = False
            print(f"❌ Test failed: {e}")
        
        duration = time.time() - start_time
        result = TestResults("Command Acknowledgment", passed, data, errors, time.time(), duration)
        self.test_results.append(result)
        
        print(f"\n⏱️  Test duration: {duration:.1f}s")
        print(f"📊 Result: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    # ========================================================================
    # COMPREHENSIVE TEST RUNNER
    # ========================================================================
    
    def run_all_tests(self):
        """Run all ground tests in sequence"""
        print("🧪 COMPREHENSIVE GROUND TEST SUITE")
        print("=" * 80)
        print("⚠️  ENSURE PROPELLERS ARE REMOVED!")
        print("⚠️  DRONE SHOULD BE ON STABLE SURFACE!")
        print("=" * 80)
        
        confirm = input("\nPropellers removed and drone secured? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ TESTS ABORTED - Safety not confirmed")
            return
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_basic_communication,
            self.test_data_streaming,
            self.test_guided_mode,
            self.test_battery_monitoring,
            self.test_sensor_validation,
            self.test_command_acknowledgment
        ]
        
        print(f"\n🏁 Starting {len(tests)} tests...")
        
        for i, test_func in enumerate(tests, 1):
            print(f"\n{'='*20} Test {i}/{len(tests)} {'='*20}")
            try:
                result = test_func()
                if not result.passed:
                    print(f"\n⚠️  Test {i} failed - continuing with remaining tests...")
            except Exception as e:
                print(f"\n💥 Test {i} crashed: {e}")
        
        # Generate final report
        self._generate_final_report()
        
        # Cleanup
        self.disconnect_mavlink()
        
        total_duration = time.time() - start_time
        print(f"\n⏱️  Total test duration: {total_duration:.1f}s")
    
    def _generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📋 FINAL TEST REPORT")
        print("="*80)
        
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        print(f"📊 SUMMARY:")
        print(f"   Total tests: {len(self.test_results)}")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        print(f"   Success rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test.test_name}")
                for error in test.errors:
                    print(f"     - {error}")
        
        print(f"\n✅ PASSED TESTS:")
        for test in passed_tests:
            print(f"   • {test.test_name}")
        
        # Overall readiness assessment
        critical_tests = ['Basic Communication', 'Data Streaming', 'Guided Mode']
        critical_passed = all(any(r.test_name == ct and r.passed for r in self.test_results) for ct in critical_tests)
        
        print(f"\n🚁 FLIGHT READINESS:")
        if critical_passed and len(failed_tests) == 0:
            print("   ✅ READY FOR FLIGHT TESTING")
            print("   All systems operational, proceed to test flight")
        elif critical_passed:
            print("   ⚠️  CONDITIONAL READY")
            print("   Critical systems OK, but some issues detected")
            print("   Review failed tests before flight")
        else:
            print("   ❌ NOT READY FOR FLIGHT")
            print("   Critical system failures detected")
            print("   Fix issues before attempting flight")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/ground_test_report_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests)/len(self.test_results)*100,
                'flight_ready': critical_passed and len(failed_tests) == 0
            },
            'tests': [
                {
                    'name': r.test_name,
                    'passed': r.passed,
                    'duration': r.duration,
                    'errors': r.errors,
                    'data': r.data_collected
                }
                for r in self.test_results
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Report saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save report: {e}")

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    """Main test execution"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create test suite
    test_suite = GroundTestSuite()
    
    print("Choose test option:")
    print("1. Run individual test")
    print("2. Run all tests")
    print("3. Quick connectivity check")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        print("\nAvailable tests:")
        print("1. Basic Communication")
        print("2. Data Streaming")
        print("3. Guided Mode (No propellers!)")
        print("4. Battery Monitoring")
        print("5. Sensor Validation")
        print("6. Command Acknowledgment")
        
        test_choice = input("Enter test number (1-6): ")
        
        test_map = {
            "1": test_suite.test_basic_communication,
            "2": test_suite.test_data_streaming,
            "3": test_suite.test_guided_mode,
            "4": test_suite.test_battery_monitoring,
            "5": test_suite.test_sensor_validation,
            "6": test_suite.test_command_acknowledgment
        }
        
        if test_choice in test_map:
            test_map[test_choice]()
            test_suite.disconnect_mavlink()
        else:
            print("Invalid test choice")
    
    elif choice == "2":
        test_suite.run_all_tests()
    
    elif choice == "3":
        test_suite.test_basic_communication()
        test_suite.disconnect_mavlink()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()