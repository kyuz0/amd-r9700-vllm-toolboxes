# AMD Radeon 9700 AI PRO (Navi 48) Tuning Guide

This guide show you how to enable undervolting and raise the power limit on the Sapphire Radeon PRO R9700 on Fedora 34. It might work with other R9700 cards (it doesn't with the ASUS one) and other Linux distributions.

## 1. Prerequisites (Enable Overclocking)

The `ppfeaturemask` must be set to unlock voltage control.

```bash
sudo grubby --update-kernel=ALL --args="amdgpu.ppfeaturemask=0xffffffff"
```

**Reboot your system after running this.**

## 2. Identify Your GPU

You need the **PCI Bus ID** of the target card (e.g., `07:00.0`).

Run this command to list your AMD GPUs:

```bash
lspci -nn | grep "VGA" | grep "AMD"
```

*Example output:*
`07:00.0 VGA compatible controller...`
*(In this example, your ID is `0000:07:00.0`)*

## 3. Tuning Script

Save this script as `tune_r9700.sh`.

**Important:** Edit the `PCI_ID` variable at the top to match the ID you found in Step 2.

```bash
#!/bin/bash

# --- CONFIGURATION ---
# Replace this with your specific PCI ID from 'lspci'
# Format must be 0000:XX:XX.X
PCI_ID="0000:07:00.0"
# ---------------------

# 1. Robustly Find the Card Name (e.g., card1) directly from the PCI Bus
# We look inside the PCI device's 'drm' folder for a folder starting with 'card' followed only by numbers.
if [ ! -d "/sys/bus/pci/devices/$PCI_ID/drm" ]; then
    echo "Error: PCI Device $PCI_ID not found or has no DRM driver attached."
    exit 1
fi

# This finds 'card1' but ignores 'card1-DP-6'
CARD_NAME=$(ls "/sys/bus/pci/devices/$PCI_ID/drm" | grep -E '^card[0-9]+$' | head -n 1)

if [ -z "$CARD_NAME" ]; then
    echo "Error: Could not determine card name for $PCI_ID"
    exit 1
fi

# Construct the clean path (e.g., /sys/class/drm/card1)
CARD_PATH="/sys/class/drm/$CARD_NAME"

echo "Tuning GPU: $CARD_NAME (at $CARD_PATH)..."

# 2. Force Manual Performance Level (Required for UV)
# We use 'tee' without pipe to ensure exact errors are caught, but your 'echo | sudo tee' method is fine.
echo "manual" | sudo tee "$CARD_PATH/device/power_dpm_force_performance_level" > /dev/null
if [ $? -ne 0 ]; then echo "Failed to set Manual mode. Check permissions/path."; exit 1; fi

# 3. Apply -75mV Undervolt
echo "vo -75" | sudo tee "$CARD_PATH/device/pp_od_clk_voltage" > /dev/null
echo "c" | sudo tee "$CARD_PATH/device/pp_od_clk_voltage" > /dev/null
echo "Applied Undervolt (-75mV)"

# 4. Set Power Limit to 315W
# Find the hwmon directory strictly inside the device
HWMON_DIR=$(find "$CARD_PATH/device/hwmon/" -maxdepth 1 -name "hwmon*" | head -n 1)
if [ -n "$HWMON_DIR" ]; then
    echo "315000000" | sudo tee "$HWMON_DIR/power1_cap" > /dev/null
    echo "Applied Power Limit (315W)"
else
    echo "Error: Could not find hwmon directory for power limit."
fi
```

### Usage

```bash
chmod +x tune_r9700.sh
sudo ./tune_r9700.sh
```

## 4\. Verification

Run these commands to confirm settings are active.

**Check Undervolt:**

```bash
# Look for: OD_VDDGFX_OFFSET: -75mV
cat /sys/class/drm/card1/device/pp_od_clk_voltage 
```

**Check Power Limit:**

```bash
# Should be ~315 W
cat /sys/class/drm/card1/device/hwmon/hwmon*/power1_cap 
```