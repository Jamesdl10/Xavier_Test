
import os
import sys
import subprocess
import time
from pathlib import Path

class SetupValidator:
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def check_python_environment(self):
        """Check Python and PyMAVLink installation"""
        self.print_header("PYTHON ENVIRONMENT CHECK")
        
        # Check Python version
        print("1. Checking Python version...")
        python_version = sys.version_info
        print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
            self.issues.append("Python 3.6+ required")
            print("   ❌ Python version too old")
        else:
            print("   ✅ Python version OK")
        
        # Check PyMAVLink
        print("\n2. Checking PyMAVLink...")
        try:
            import pymavlink
            print(f"   ✅ PyMAVLink {pymavlink.__version__} installed")
        except ImportError:
            self.issues.append("PyMAVLink not installed")
            print("   ❌ PyMAVLink not found")
            print("   Install with: pip3 install pymavlink")
        
        # Check other required modules
        print("\n3. Checking required modules...")
        required_modules = ['asyncio', 'threading', 'json', 'logging', 'dataclasses']
        for module in required_modules:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError:
                self.issues.append(f"Missing module: {module}")
                print(f"   ❌ {module}")
    
    def check_hardware_connections(self):
        """Check UART device and permissions"""
        self.print_header("HARDWARE CONNECTION CHECK")
        
        # Check UART device
        print("1. Checking UART device...")
        uart_devices = ['/dev/ttyTHS0', '/dev/ttyTHS1', '/dev/ttyUSB0', '/dev/ttyACM0']
        found_device = None
        
        for device in uart_devices:
            if os.path.exists(device):
                print(f"   ✅ Found: {device}")
                if device == '/dev/ttyTHS0':
                    found_device = device
                    break
                elif found_device is None:
                    found_device = device
        
        if found_device is None:
            self.issues.append("No UART device found")
            print("   ❌ No suitable UART device found")
            print("   Available devices:")
            try:
                result = subprocess.run(['ls', '/dev/tty*'], capture_output=True, text=True)
                for line in result.stdout.split('\n')[:10]:  # Show first 10
                    if 'tty' in line:
                        print(f"     {line}")
            except:
                pass
        else:
            print(f"   ✅ Will use: {found_device}")
        
        # Check permissions
        print("\n2. Checking permissions...")
        if found_device and os.access(found_device, os.R_OK | os.W_OK):
            print(f"   ✅ Read/write access to {found_device}")
        else:
            self.issues.append("No UART permissions")
            print(f"   ❌ No access to {found_device}")
            print("   Fix with: sudo chmod 666 /dev/ttyTHS0")
            print("   And: sudo usermod -a -G dialout $USER")
        
        # Check for conflicting processes
        print("\n3. Checking for conflicts...")
        try:
            if found_device:
                result = subprocess.run(['sudo', 'lsof', found_device], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    self.warnings.append("Other processes using UART")
                    print(f"   ⚠️  Other processes using {found_device}:")
                    print(f"     {result.stdout}")
                else:
                    print(f"   ✅ {found_device} available")
        except:
            print("   ⚠️  Cannot check for conflicts (need sudo)")
    
    def check_directory_structure(self):
        """Check directory structure and files"""
        self.print_header("DIRECTORY STRUCTURE CHECK")
        
        # Check current directory
        print("1. Checking current directory...")
        current_dir = Path.cwd()
        print(f"   Current: {current_dir}")
        
        # Check for logs directory
        logs_dir = current_dir / 'logs'
        if logs_dir.exists():
            print("   ✅ logs/ directory exists")
        else:
            print("   ⚠️  logs/ directory missing")
            print("   Creating logs directory...")
            try:
                logs_dir.mkdir()
                print("   ✅ logs/ directory created")
            except Exception as e:
                self.warnings.append("Cannot create logs directory")
                print(f"   ❌ Failed to create logs/: {e}")
        
        # Check for test script
        test_script = current_dir / 'ground_test_suite.py'
        if test_script.exists():
            print("   ✅ ground_test_suite.py found")
        else:
            self.warnings.append("Test script not found")
            print("   ⚠️  ground_test_suite.py not found")
            print("   Save the test script as ground_test_suite.py")
    
    def test_basic_connection(self):
        """Test basic MAVLink connection"""
        self.print_header("BASIC CONNECTION TEST")
        
        try:
            from pymavlink import mavutil
            
            print("1. Attempting connection...")
            print("   Device: /dev/ttyTHS0")
            print("   Baud: 57600")
            print("   System ID: 2")
            
            # Create connection
            master = mavutil.mavlink_connection(
                '/dev/ttyTHS0', 
                baud=57600, 
                source_system=2
            )
            
            print("\n2. Waiting for heartbeat...")
            start_time = time.time()
            
            # Wait for heartbeat with timeout
            try:
                master.wait_heartbeat(timeout=10)
                connection_time = time.time() - start_time
                
                print(f"   ✅ Connected in {connection_time:.1f}s")
                print(f"   Target System: {master.target_system}")
                print(f"   Target Component: {master.target_component}")
                
                # Send test message
                print("\n3. Sending test message...")
                master.mav.statustext_send(6, b"Setup validation test")
                print("   ✅ Test message sent")
                
                # Try to receive a few messages
                print("\n4. Testing message reception...")
                message_count = 0
                test_start = time.time()
                
                while time.time() - test_start < 3 and message_count < 5:
                    msg = master.recv_match(blocking=False, timeout=0.1)
                    if msg:
                        message_count += 1
                        print(f"   📨 Received: {msg.get_type()}")
                
                print(f"   ✅ Received {message_count} messages in 3 seconds")
                
                # Close connection
                master.close()
                print("\n   ✅ CONNECTION TEST PASSED")
                
            except Exception as e:
                self.issues.append("Connection timeout")
                print(f"   ❌ Connection timeout: {e}")
                print("   Check:")
                print("     - Pixhawk is powered on")
                print("     - UART cable is connected")
                print("     - Pixhawk SERIAL1_PROTOCOL = 2")
                print("     - Pixhawk SERIAL1_BAUD = 57")
                
        except ImportError:
            self.issues.append("PyMAVLink import failed")
            print("   ❌ Cannot import PyMAVLink")
        except Exception as e:
            self.issues.append(f"Connection test failed: {e}")
            print(f"   ❌ Connection test failed: {e}")
    
    def check_pixhawk_parameters(self):
        """Check critical Pixhawk parameters"""
        self.print_header("PIXHAWK PARAMETER CHECK")
        
        try:
            from pymavlink import mavutil
            
            master = mavutil.mavlink_connection('/dev/ttyTHS0', baud=57600, source_system=2)
            master.wait_heartbeat(timeout=5)
            
            # Parameters to check
            critical_params = {
                'SERIAL1_PROTOCOL': 2,    # MAVLink 2
                'SERIAL1_BAUD': 57,       # 57600 baud
                'BATT_MONITOR': 4,        # Analog voltage/current
                'GPS_TYPE': 1,            # Auto-detect
            }
            
            print("Checking critical parameters...")
            
            for param_name, expected_value in critical_params.items():
                print(f"\n   Checking {param_name}...")
                
                # Request parameter
                master.mav.param_request_read_send(
                    master.target_system,
                    master.target_component,
                    param_name.encode('utf-8'),
                    -1
                )
                
                # Wait for response
                msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
                
                if msg and msg.param_id.decode().strip() == param_name:
                    actual_value = msg.param_value
                    if abs(actual_value - expected_value) < 0.1:
                        print(f"     ✅ {param_name}: {actual_value} (expected: {expected_value})")
                    else:
                        self.warnings.append(f"{param_name} not optimal")
                        print(f"     ⚠️  {param_name}: {actual_value} (expected: {expected_value})")
                else:
                    self.warnings.append(f"Cannot read {param_name}")
                    print(f"     ❌ Cannot read {param_name}")
            
            master.close()
            
        except Exception as e:
            self.warnings.append("Parameter check failed")
            print(f"   ⚠️  Parameter check failed: {e}")
            print("   (Parameters can be checked later via Mission Planner)")
    
    def generate_report(self):
        """Generate final validation report"""
        self.print_header("VALIDATION REPORT")
        
        print(f"📊 SUMMARY:")
        print(f"   Issues found: {len(self.issues)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES (must fix before testing):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS (recommended to fix):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Overall status
        if not self.issues:
            if not self.warnings:
                print(f"\n🎉 SETUP VALIDATION PASSED")
                print(f"   ✅ Ready to run ground test suite!")
                print(f"\n   Next step: python3 ground_test_suite.py")
            else:
                print(f"\n✅ SETUP MOSTLY READY")
                print(f"   Ground tests should work, but review warnings")
                print(f"\n   Next step: python3 ground_test_suite.py")
        else:
            print(f"\n❌ SETUP NOT READY")
            print(f"   Fix critical issues before running tests")
        
        return len(self.issues) == 0

def main():
    """Main validation function"""
    print("🔍 JETSON GROUND TEST SETUP VALIDATOR")
    print("This script checks if your system is ready for ground testing")
    
    validator = SetupValidator()
    
    # Run all validation checks
    validator.check_python_environment()
    validator.check_hardware_connections()
    validator.check_directory_structure()
    validator.test_basic_connection()
    validator.check_pixhawk_parameters()
    
    # Generate final report
    success = validator.generate_report()
    
    if success:
        print(f"\n🚀 READY TO PROCEED!")
        run_tests = input(f"\nRun ground test suite now? (y/n): ")
        if run_tests.lower() == 'y':
            try:
                import subprocess
                subprocess.run(['python3', 'ground_test_suite.py'])
            except FileNotFoundError:
                print("❌ ground_test_suite.py not found in current directory")
            except Exception as e:
                print(f"❌ Error running test suite: {e}")
    else:
        print(f"\n🔧 FIX ISSUES FIRST")
        print(f"   Re-run this validator after fixing issues")

if __name__ == "__main__":
    main()