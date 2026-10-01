# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:10:55 2026

@author: leeau
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to Agilent
SHOW_PLOT = True           # Set to True to display plot in Spyder

# -----------------------------
# User-editable PUND parameters
# -----------------------------
zero_before_s = 10e-6            # Flat baseline at 0 V before the reset pulse
ramp_time_s = 10e-6             # Rise/fall time, shared by all pulses
pulse_length_s = 1e-3           # Flat-top duration of each positive (P/U) pulse
gap_reset_to_positive_s = 5e-3  # Gap between the reset pulse and the first positive pulse
gap_positive_to_positive_s = 3e-3  # Gap between the two positive pulses
zero_after_s = 5e-3             # Flat baseline at 0 V after the last pulse
amplitude_v = 0.8                # Peak pulse amplitude (V)

arb_sample_rate_hz = 20e6
maximum_arb_points = 1_000_000
max_allowed_amplitude_v = 1.5

#awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR" # Keysight
awg_address = "USB0::2391::9991::MY52300442::0::INSTR" # Agilent
awg_timeout_ms = 30000


def segment(value, duration_s):
    """Create a flat segment at a given value."""
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * arb_sample_rate_hz)))
    return np.full(point_count, value, dtype=float)


def ramp(start_value, stop_value, duration_s):
    """Create a linear ramp between two values."""
    if duration_s < 0:
        raise ValueError("Ramp time cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * arb_sample_rate_hz)))
    return np.linspace(start_value, stop_value, point_count, endpoint=False)


def pulse(polarity, width_s):
    """Create a pulse with rise, flat-top, and fall."""
    return np.concatenate((
        ramp(0.0, polarity, ramp_time_s),
        segment(polarity, width_s),
        ramp(polarity, 0.0, ramp_time_s),
    ))


def build_pund_waveform():
    """Build the complete PUND waveform."""
    waveform = np.concatenate((
        segment(0.0, zero_before_s),
        pulse(-1.0, 2 * pulse_length_s),
        segment(0.0, gap_reset_to_positive_s),
        pulse(1.0, pulse_length_s),
        segment(0.0, gap_positive_to_positive_s),
        pulse(1.0, pulse_length_s),
        segment(0.0, zero_after_s),
    ))
    
    if len(waveform) > maximum_arb_points:
        raise ValueError(f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}")
    
    total_duration_s = len(waveform) / arb_sample_rate_hz
    return waveform, total_duration_s


def plot_waveform(waveform, total_duration_s, amplitude_v):
    """Plot the waveform with two views: full and zoomed."""
    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3
    voltage = waveform * amplitude_v  # Scale to actual voltage
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Full waveform view
    ax1.plot(time_ms, voltage, linewidth=1.5, color='blue')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Full PUND Waveform: {len(waveform)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlim([0, total_duration_s * 1e3])
    
    # Calculate where pulses are for annotations
    t_current = zero_before_s
    
    # Reset pulse
    reset_start = t_current
    reset_duration = 2 * ramp_time_s + 2 * pulse_length_s
    reset_center = reset_start + reset_duration / 2
    t_current += reset_duration
    
    # Gap
    t_current += gap_reset_to_positive_s
    
    # First positive pulse
    pos1_start = t_current
    pos1_duration = 2 * ramp_time_s + pulse_length_s
    pos1_center = pos1_start + pos1_duration / 2
    t_current += pos1_duration
    
    # Gap
    t_current += gap_positive_to_positive_s
    
    # Second positive pulse
    pos2_start = t_current
    pos2_duration = 2 * ramp_time_s + pulse_length_s
    pos2_center = pos2_start + pos2_duration / 2
    
    # Add vertical lines and labels
    ax1.axvline(x=reset_center*1e3, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(reset_center*1e3, -amplitude_v*0.9, 'Reset', 
            rotation=0, fontsize=10, ha='center', color='red', fontweight='bold')
    
    ax1.axvline(x=pos1_center*1e3, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(pos1_center*1e3, amplitude_v*0.9, 'P', 
            rotation=0, fontsize=10, ha='center', color='green', fontweight='bold')
    
    ax1.axvline(x=pos2_center*1e3, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(pos2_center*1e3, amplitude_v*0.9, 'U', 
            rotation=0, fontsize=10, ha='center', color='blue', fontweight='bold')
    
    # Zoomed view of PUND sequence (exclude most of the zero_after)
    zoom_duration_s = total_duration_s - zero_after_s + 0.5e-3  # Show 0.5 ms of the zero_after
    zoom_points = int(zoom_duration_s * arb_sample_rate_hz)
    zoom_points = min(zoom_points, len(waveform))
    
    ax2.plot(time_ms[:zoom_points], voltage[:zoom_points], linewidth=2, color='green')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.set_title(f'Zoomed PUND Sequence ({zoom_duration_s*1e3:.2f} ms)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add annotations for zoomed view
    ax2.axvline(x=reset_center*1e3, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(reset_center*1e3, -amplitude_v*1.1, 'Reset\n(N)', 
            rotation=0, fontsize=11, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.axvline(x=pos1_center*1e3, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(pos1_center*1e3, amplitude_v*1.1, 'Positive\n(P)', 
            rotation=0, fontsize=11, ha='center', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.axvline(x=pos2_center*1e3, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(pos2_center*1e3, amplitude_v*1.1, 'Unswitched\n(U)', 
            rotation=0, fontsize=11, ha='center', color='blue', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax2.set_xlim([0, zoom_duration_s * 1e3])
    
    plt.tight_layout()
    plt.show()


def check_errors(awg, context):
    """Check for errors on the AWG."""
    while True:
        response = awg.query("SYST:ERR?").strip()
        code_str = response.split(",", 1)[0]
        if int(code_str) == 0:
            break
        print(f"AWG error after '{context}': {response}")


def setup_awg(waveform, amplitude_v):
    """Upload waveform to AWG and configure output."""
    amplitude_v = float(amplitude_v)
    if amplitude_v <= 0 or amplitude_v > max_allowed_amplitude_v:
        raise ValueError("Pulse amplitude is outside the allowed range")
    
    arb_name = "PUND1"
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    awg.write_termination = "\n"
    awg.read_termination = "\n"
    
    try:
        print("\nConfiguring AWG...")
        awg.write("*RST")
        awg.write("*CLS")

        awg.write("SOURCE:DATA:VOLATILE:CLEAR")
        check_errors(awg, "SOURCE:DATA:VOLATILE:CLEAR")

        print(f"Uploading waveform ({len(waveform)} points) via binary transfer...")
        awg.write_binary_values(
            f"SOURCE:DATA:ARBITRARY {arb_name},",
            waveform,
            datatype="f",
            is_big_endian=True,
        )
        awg.write("*WAI")
        check_errors(awg, f"SOURCE:DATA:ARBITRARY {arb_name} (upload)")

        awg.write(f"FUNC:ARB {arb_name}")
        check_errors(awg, f"FUNC:ARB {arb_name}")

        awg.write("FUNC ARB")
        check_errors(awg, "FUNC ARB")

        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        check_errors(awg, f"FUNC:ARB:SRATE {arb_sample_rate_hz}")

        awg.write(f"VOLT {2.0 * amplitude_v}")
        check_errors(awg, f"VOLT {2.0 * amplitude_v}")

        awg.write("VOLT:OFFS 0")
        check_errors(awg, "VOLT:OFFS 0")

        awg.write("OUTP ON")
        check_errors(awg, "OUTP ON")
        
        print("✓ Waveform uploaded and AWG output enabled")
    finally:
        awg.close()
        rm.close()


def turn_off_awg():
    """Turn off AWG output."""
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    try:
        awg.write("OUTP OFF")
        print("AWG output turned OFF.")
    finally:
        awg.close()
        rm.close()


def main():
    """Main function to build, plot, and/or upload waveform."""
    print("="*60)
    print("PUND Waveform Generator (Binary Upload)")
    print("="*60)
    
    # Build waveform
    waveform, total_duration_s = build_pund_waveform()
    
    # Print configuration
    print(f"\nWaveform Configuration:")
    print(f"  Zero before: {zero_before_s*1e3:.3f} ms")
    print(f"  Ramp time: {ramp_time_s*1e6:.3f} µs")
    print(f"  Pulse length: {pulse_length_s*1e3:.3f} ms")
    print(f"  Gap (reset to positive): {gap_reset_to_positive_s*1e3:.3f} ms")
    print(f"  Gap (positive to positive): {gap_positive_to_positive_s*1e3:.3f} ms")
    print(f"  Zero after: {zero_after_s*1e3:.3f} ms")
    print(f"  Amplitude: {amplitude_v:.3f} V")
    print(f"\nWaveform Details:")
    print(f"  Total points: {len(waveform)} / {maximum_arb_points}")
    print(f"  Memory used: {len(waveform)/maximum_arb_points*100:.1f}%")
    print(f"  Sample rate: {arb_sample_rate_hz/1e6:.1f} MS/s")
    print(f"  Total duration: {total_duration_s*1e3:.3f} ms")
    print(f"  Repetition rate: {1/total_duration_s:.3f} Hz")
    print(f"  Vpp on AWG: {2.0*amplitude_v:.3f} V")
    
    # Plot if requested
    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_waveform(waveform, total_duration_s, amplitude_v)
    
    # Send to instrument if requested
    if SEND_TO_INSTRUMENT:
        setup_awg(waveform, amplitude_v)
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")
    
    print("="*60)


if __name__ == "__main__":
    main()