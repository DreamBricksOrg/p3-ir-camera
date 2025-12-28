# P3 USB Protocol

Protocol documentation for P3-series USB thermal cameras.

> **Note**: Protocol details were determined through USB traffic analysis and
> experimentation. This is an independent documentation effort.

## Device

- VID: 0x3474
- PID: 0x45A2
- Native resolution: 256×192
- Frame rate: ~25 fps

## USB Setup

```
Interface 0: Control commands
Interface 1, Alt 0: Inactive (streaming stopped)
Interface 1, Alt 1: Active (streaming enabled)
```

Detach kernel drivers, claim both interfaces before use.

## Control Transfers

| bRequest | bmRequestType | wIndex | Purpose |
|----------|---------------|--------|---------|
| 0x20 | 0x41 | 0 | Send 18-byte command |
| 0x21 | 0xC1 | 0 | Read response data |
| 0x22 | 0xC1 | 0 | Read status (1 byte) |
| 0xEE | 0x40 | 1 | Start streaming |

### USB Request Type Constants

```
bmRequestType = Direction | Type | Recipient
Direction: USB_DIR_OUT=0x00, USB_DIR_IN=0x80
Type:      USB_TYPE_STANDARD=0x00, USB_TYPE_CLASS=0x20, USB_TYPE_VENDOR=0x40
Recipient: USB_RECIP_DEVICE=0x00, USB_RECIP_INTERFACE=0x01

Common values:
  0x41 = OUT | VENDOR | INTERFACE (write vendor command)
  0xC1 = IN  | VENDOR | INTERFACE (read vendor response)
  0x40 = OUT | VENDOR | DEVICE (device-level write)
```

## Command Format (18 bytes)

```
Offset  Size  Description
0       2     Command type (LE)
2       2     Parameter (usually 0x0081)
4       2     Register ID (LE)
6       8     Reserved (zeros)
14      2     Response length (LE)
16      2     CRC16 checksum (LE)
```

Command types:
- 0x0101: Read register
- 0x1021: Status check
- 0x012f: Stream control
- 0x0136: Shutter/NUC

## Pre-computed Commands (with CRC)

```python
COMMANDS = {
    'read_name':    bytes.fromhex('0101810001000000000000001e0000004f90'),
    'read_version': bytes.fromhex('0101810002000000000000000c0000001f63'),
    'read_model':   bytes.fromhex('010181000600000000000000400000004f65'),
    'read_serial':  bytes.fromhex('01018100070000000000000040000000104c'),
    'status':       bytes.fromhex('102181000000000000000000020000009501'),
    'start_stream': bytes.fromhex('012f8100000000000000000001000000493f'),
    'gain_low':     bytes.fromhex('012f41000000000000000000000000003c3a'),
    'gain_high':    bytes.fromhex('012f4100010000000000000000000000493f'),
    'shutter':      bytes.fromhex('01364300000000000000000000000000cd0b'),
}
```

## Initialization Sequence

```python
# 1. Detach kernel drivers and claim interfaces
dev.set_configuration()
usb.util.claim_interface(dev, 0)
usb.util.claim_interface(dev, 1)

# 2. Read device info (optional)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['read_name'], 1000)
time.sleep(0.02)
name = dev.ctrl_transfer(0xC1, 0x21, 0, 0, 30, 1000)

# 3. Status check
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['status'], 1000)
dev.ctrl_transfer(0xC1, 0x22, 0, 0, 1, 1000)

# 4. Enable streaming interface
dev.set_interface_altsetting(interface=1, alternate_setting=1)

# 5. Start stream
dev.ctrl_transfer(0x40, 0xEE, 0, 1, None, 1000)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['start_stream'], 1000)
```

## Stop Streaming

```python
dev.set_interface_altsetting(interface=1, alternate_setting=0)
```

## Reading Frames

Read from bulk endpoint 0x81 in chunks:

```python
data = b''
while len(data) < 196620:  # 12 + 256*384*2
    chunk = dev.read(0x81, 16384, 1000)
    data += bytes(chunk)
```

## Frame Structure

Total frame: 196,620 bytes

```
Offset  Size    Description
0       12      Header/metadata
12      196608  Pixel data (256×384 × 2 bytes)
```

The 256×384 pixel buffer contains:
- Rows 0-191: IR brightness data (hardware AGC'd, for display)
- Rows 192-193: Metadata
- Rows 194-383: Temperature data (190 rows × 256 cols)

## Column Alignment Quirk

The camera has a hardware quirk where the first 12 columns of each row are
transmitted at the end of the previous row's USB data. Fix with:

```python
def fix_alignment(img):
    # Roll left by 12 columns
    result = np.roll(img, -12, axis=1)
    # Fix edge columns (shifted up by one row)
    result[:, -12:] = np.roll(result[:, -12:], -1, axis=0)
    # Hide garbage in last row
    result[-1, -12:] = result[-2, -12:]
    return result
```

## Temperature Conversion

Raw 16-bit values are in 1/64 Kelvin units:

```python
SCALE = 64  # 1/64 Kelvin units

def raw_to_celsius(raw):
    return (raw / SCALE) - 273.15

def celsius_to_raw(celsius):
    return int((celsius + 273.15) * SCALE)
```

## Gain Modes

- **High gain**: -20°C to 150°C (higher sensitivity)
- **Low gain**: 0°C to 550°C (extended range)

```python
# Set low gain (extended range)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['gain_low'], 1000)

# Set high gain (higher sensitivity)
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['gain_high'], 1000)
```

## Shutter/NUC Calibration

Trigger shutter calibration (audible click):

```python
dev.ctrl_transfer(0x41, 0x20, 0, 0, COMMANDS['shutter'], 1000)
```

## Display Pipeline

```python
import numpy as np
import cv2

# 1. Parse frame
pixels = np.frombuffer(data[12:], dtype='<u2')
full = pixels.reshape((384, 256))

# 2. Extract and align thermal data
thermal = full[194:384, :].copy()
thermal = fix_alignment(thermal)

# 3. Normalize to 8-bit (percentile-based AGC)
low = np.percentile(thermal, 1)
high = np.percentile(thermal, 99)
img_8 = np.clip((thermal - low) / (high - low) * 255, 0, 255).astype(np.uint8)

# 4. Apply colormap
img_color = cv2.applyColorMap(img_8, cv2.COLORMAP_INFERNO)
```

## Troubleshooting

**Camera in degraded state (purple/garbage image):**
USB reset and reinitialize, or replug the device.

**Frame sync lost:**
Reset camera and reinitialize streaming.

**Temperature drift:**
Trigger shutter calibration, allow camera to warm up (~5 minutes).

## CRC16 Reference

CRC16-CCITT with polynomial 0x1021, initial value 0x0000:

```python
def crc16_ccitt(data):
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc
```
