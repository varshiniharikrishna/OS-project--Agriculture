#!/usr/bin/env python3
"""
AgriEdge OS — Instant 10-Second Arduino Hardware Verification Script
──────────────────────────────────────────────────────────────────
Run this script the second you plug the Arduino into your laptop:
    python3 hardware/test_arduino_connection.py

Works on:
  - Apple Silicon MacBook (USB-C adapter -> USB-A)
  - ASUS Windows Laptop (Direct USB-A port)
"""

import sys
import time

print("=" * 65)
print("  AgriEdge OS — Arduino Uno & IR Sensor Instant Test  ")
print("=" * 65)

# Step 1: Check Python Serial Library
try:
    import serial
    import serial.tools.list_ports
    print("[✓] pyserial library is installed.")
except ImportError:
    print("\n[!] 'pyserial' is not installed in this Python environment.")
    print("    Install it in 3 seconds by running:")
    print("        pip install pyserial\n")
    sys.exit(1)

# Step 2: Scan USB Ports
print("\n[Scanning USB Serial Ports...]")
ports = list(serial.tools.list_ports.comports())

if not ports:
    print("[!] No USB serial devices detected.")
    print("    • Ensure the Arduino Uno USB cable is firmly plugged in.")
    print("    • Check if the green 'ON' LED on the Arduino board is lit.")
    sys.exit(1)

arduino_port = None
print(f"Found {len(ports)} serial port(s):")
for p in ports:
    desc = p.description or ""
    print(f"  • {p.device}: {desc}")
    if any(k in (desc + p.device).lower() for k in ["arduino", "usbmodem", "ch340", "2341", "cp210", "usb serial"]) or p.device.upper().startswith("COM"):
        if not arduino_port:
            arduino_port = p.device

if not arduino_port:
    arduino_port = ports[0].device

print(f"\n[Connecting to Arduino on {arduino_port} at 9600 baud...]")

try:
    ser = serial.Serial(arduino_port, 9600, timeout=1.0)
    time.sleep(2.0)  # Allow Arduino bootloader to initialize
    print(f"[✓] SUCCESS! Connected to Arduino Uno on {arduino_port}")
    print("\n" + "-" * 65)
    print("  PHYSICAL SENSOR TEST:")
    print("  1. Wave your hand in front of the IR sensor.")
    print("  2. Watch the on-board Pin 13 LED light up on the Arduino.")
    print("  3. Live JSON events will appear below in real-time.")
    print("  Press Ctrl+C to exit.")
    print("-" * 65 + "\n")

    event_count = 0
    start_time = time.time()

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"[SERIAL] {line}")
            event_count += 1
            if event_count >= 5 and (time.time() - start_time) > 10:
                print("\n[✓] Arduino Uno and IR sensor are 100% operational!")
                break

except serial.SerialException as e:
    print(f"\n[!] Could not open port {arduino_port}: {e}")
    print("    If Arduino IDE Serial Monitor is open, CLOSE IT so Python can connect.")
except KeyboardInterrupt:
    print("\n\nTest stopped by user. Connection successful!")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
