# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:16:53 2026

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
# Reset pulse (negative)
reset_amplitude = -2.0         # V
reset_width = 1000e-9           # s
reset_rise_time = 20e-9        # s

# Positive pulses
pos_amplitude = 1            # V
pos_width = 500e-9            # s
pos_rise_time = 20e-9         # s

# Negative pulses
neg_amplitude = -1.5           # V
neg_width = 500e-9             # s
neg_rise_time = 20e-9          # s

# Gaps
gap_after_reset = 500e-9       # s
gap_between_pos = 500e-9        # s
gap_pos_to_neg = 500e-9        # s
gap_between_neg = 500e-9        # s

# Holds
init_hold = 50e-9              # s
final_hold = 50e-9             # s

# Repetition control
num_pund_repeats = 1          # Number of times PUND repeats in one 60 Hz cycle
target_period = 1/60        # 60 Hz = 16.67e-3       # Target period (s) for 60 Hz repetition

# Gap between PUND repetitions (if multiple PUNDs per cycle)
gap_between_punds = 100e-6     # s (only used if num_pund_repeats > 1)

# System parameters
arb_sample_rate_hz = 5.7e7
maximum_arb_points = 1000000
max_allowed_amplitude_v = 10.0  # Adjust based on your system

#awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR" # Keysight
awg_address = "USB0::2391::9991::MY52300442::0::INSTR" # Agilent
awg_timeout_ms = 30000


def segment(value, duration_s, sample_rate):
    """Create a flat segment at a given value."""
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * sample_rate)))
    return np.full(point_count, value, dtype=float)


def ramp(start_value, stop_value, duration_s, sample_rate):
    """Create a linear ramp between two values."""
    if duration_s < 0:
        raise ValueError("Ramp time cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * sample_rate)))
    return np.linspace(start_value, stop_value, point_count, endpoint=False)


def pulse(amplitude, width_s, rise_time_s, sample_rate):
    """Create a pulse with rise, flat-top, and fall."""
    return np.concatenate((
        ramp(0.0, amplitude, rise_time_s, sample_rate),
        segment(amplitude, width_s, sample_rate),
        ramp(amplitude, 0.0, rise_time_s, sample_rate),
    ))

def build_single_pund():
    """Build a single PUND sequence (without zero padding)."""
    # Normalize all amplitudes to -1 to +1 range
    max_amplitude = max(abs(reset_amplitude), abs(pos_amplitude), abs(neg_amplitude))
    
    reset_norm = reset_amplitude / max_amplitude
    pos_norm = pos_amplitude / max_amplitude
    neg_norm = neg_amplitude / max_amplitude
    
    # Build single PUND waveform
    single_pund = np.concatenate((
        # Initial hold
        segment(0.0, init_hold, arb_sample_rate_hz),
        
        # Reset pulse (negative)
        pulse(reset_norm, reset_width, reset_rise_time, arb_sample_rate_hz),
        segment(0.0, gap_after_reset, arb_sample_rate_hz),
        
        # First positive pulse
        pulse(pos_norm, pos_width, pos_rise_time, arb_sample_rate_hz),
        segment(0.0, gap_between_pos, arb_sample_rate_hz),
        
        # Second positive pulse
        pulse(pos_norm, pos_width, pos_rise_time, arb_sample_rate_hz),
        segment(0.0, gap_pos_to_neg, arb_sample_rate_hz),
        
        # First negative pulse
        pulse(neg_norm, neg_width, neg_rise_time, arb_sample_rate_hz),
        segment(0.0, gap_between_neg, arb_sample_rate_hz),
        
        # Second negative pulse
        pulse(neg_norm, neg_width, neg_rise_time, arb_sample_rate_hz),
        
        # Final hold
        segment(0.0, final_hold, arb_sample_rate_hz),
    ))
    
    # Calculate single PUND duration
    single_pund_duration = (init_hold + 
                           2*reset_rise_time + reset_width + gap_after_reset +
                           2*(2*pos_rise_time + pos_width) + gap_between_pos + gap_pos_to_neg +
                           2*(2*neg_rise_time + neg_width) + gap_between_neg +
                           final_hold)
    
    return single_pund, single_pund_duration, max_amplitude


def build_pund_burst_waveform():
    """
    Build PUND burst waveform with repetitions:
    Repeats PUND sequence num_pund_repeats times, then fills remainder 
    of target_period with 0V to achieve 60 Hz repetition rate.
    """
    
    # Build single PUND
    single_pund, single_pund_duration, max_amplitude = build_single_pund()
    
    # Build repeated PUND sequences
    if num_pund_repeats == 1:
        # Single PUND, no inter-PUND gaps
        pund_sequences = single_pund
        total_pund_time = single_pund_duration
    else:
        # Multiple PUNDs with gaps between them
        pund_list = []
        for i in range(num_pund_repeats):
            pund_list.append(single_pund)
            if i < num_pund_repeats - 1:  # Don't add gap after last PUND
                pund_list.append(segment(0.0, gap_between_punds, arb_sample_rate_hz))
        
        pund_sequences = np.concatenate(pund_list)
        total_pund_time = num_pund_repeats * single_pund_duration + (num_pund_repeats - 1) * gap_between_punds
    
    # Calculate required zero duration to reach target period
    zero_duration = target_period - total_pund_time
    
    if zero_duration < 0:
        raise ValueError(
            f"PUND sequences ({total_pund_time*1e3:.3f} ms) exceed target period "
            f"({target_period*1e3:.3f} ms)!\n"
            f"Reduce num_pund_repeats, pulse widths, or gaps."
        )
    
    # Build complete waveform
    waveform = np.concatenate((
        pund_sequences,
        segment(0.0, zero_duration, arb_sample_rate_hz),
    ))
    
    if len(waveform) > maximum_arb_points:
        raise ValueError(
            f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}\n"
            f"Reduce sample rate or number of repetitions."
        )
    
    total_duration_s = len(waveform) / arb_sample_rate_hz
    
    return waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude


def plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude):
    """Plot the PUND waveform with two views: full cycle and zoomed on first PUND."""
    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3
    voltage = waveform * max_amplitude  # Scale to actual voltage
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Full waveform view (entire 60 Hz cycle)
    ax1.plot(time_ms, voltage, linewidth=1.5, color='blue')
    ax1.axvline(x=total_pund_time*1e3, color='red', linestyle='--', 
               linewidth=2, label=f'PUND sequences end / Zero section ({zero_duration*1e3:.2f} ms)')
    
    # Mark each PUND repetition
    for i in range(num_pund_repeats):
        pund_start = i * (single_pund_duration + gap_between_punds) if i > 0 else 0
        if i < num_pund_repeats - 1:
            pund_end = pund_start + single_pund_duration
            ax1.axvspan(pund_start*1e3, pund_end*1e3, alpha=0.1, color='green')
            ax1.text((pund_start + single_pund_duration/2)*1e3, max_amplitude*0.5, 
                    f'PUND {i+1}', ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax1.axvspan(pund_start*1e3, total_pund_time*1e3, alpha=0.1, color='green')
            ax1.text((pund_start + single_pund_duration/2)*1e3, max_amplitude*0.5, 
                    f'PUND {i+1}', ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'Full 60 Hz Cycle: {len(waveform)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s | {num_pund_repeats} PUND(s) per cycle', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, total_duration_s * 1e3])
    
    # Zoomed view of first PUND sequence only
    if num_pund_repeats == 1:
        first_pund_points = int(round(total_pund_time * arb_sample_rate_hz))
    else:
        first_pund_points = int(round(single_pund_duration * arb_sample_rate_hz))
    
    # Add safety margin
    first_pund_points = min(first_pund_points + 20, len(waveform))
    
    # Create time array in microseconds
    time_us = np.arange(first_pund_points) / arb_sample_rate_hz * 1e6
    
    ax2.plot(time_us, voltage[:first_pund_points], linewidth=2, color='green', 
             marker='o', markersize=3, markevery=max(1, first_pund_points//50))
    ax2.set_xlabel('Time (µs)', fontsize=12)
    ax2.set_ylabel('Voltage (V)', fontsize=12)
    ax2.set_title(f'Zoomed: First PUND Sequence ({single_pund_duration*1e9:.1f} ns, {first_pund_points} points)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # FIXED: Calculate pulse centers correctly using explicit tracking
    print("\n=== Pulse Timing Calculation ===")
    
    t = 0.0  # Current time in seconds
    
    # Initial hold
    t += init_hold
    print(f"After init_hold: {t*1e9:.1f} ns")
    
    # Reset pulse (negative)
    t += reset_rise_time
    reset_center = t + reset_width / 2
    print(f"Reset center: {reset_center*1e9:.1f} ns (at {reset_center*1e6:.3f} µs)")
    t += reset_width + reset_rise_time
    print(f"After reset pulse: {t*1e9:.1f} ns")
    
    # Gap after reset
    t += gap_after_reset
    print(f"After gap_after_reset: {t*1e9:.1f} ns")
    
    # First positive pulse
    t += pos_rise_time
    pos1_center = t + pos_width / 2
    print(f"Pos1 center: {pos1_center*1e9:.1f} ns (at {pos1_center*1e6:.3f} µs)")
    t += pos_width + pos_rise_time
    
    # Gap between positive pulses
    t += gap_between_pos
    print(f"After gap_between_pos: {t*1e9:.1f} ns")
    
    # Second positive pulse
    t += pos_rise_time
    pos2_center = t + pos_width / 2
    print(f"Pos2 center: {pos2_center*1e9:.1f} ns (at {pos2_center*1e6:.3f} µs)")
    t += pos_width + pos_rise_time
    
    # Gap positive to negative
    t += gap_pos_to_neg
    print(f"After gap_pos_to_neg: {t*1e9:.1f} ns")
    
    # First negative pulse
    t += neg_rise_time
    neg1_center = t + neg_width / 2
    print(f"Neg1 center: {neg1_center*1e9:.1f} ns (at {neg1_center*1e6:.3f} µs)")
    t += neg_width + neg_rise_time
    
    # Gap between negative pulses
    t += gap_between_neg
    print(f"After gap_between_neg: {t*1e9:.1f} ns")
    
    # Second negative pulse
    t += neg_rise_time
    neg2_center = t + neg_width / 2
    print(f"Neg2 center: {neg2_center*1e9:.1f} ns (at {neg2_center*1e6:.3f} µs)")
    t += neg_width + neg_rise_time
    print(f"After neg2 pulse: {t*1e9:.1f} ns")
    
    # Final hold
    t += final_hold  # ← THIS WAS MISSING OR NOT BEING COUNTED
    print(f"After final_hold: {t*1e9:.1f} ns")
    
    print(f"\nTotal time calculated: {t*1e9:.1f} ns ({t*1e6:.3f} µs)")
    print(f"Expected PUND duration: {single_pund_duration*1e9:.1f} ns ({single_pund_duration*1e6:.3f} µs)")
    print(f"Difference: {abs(single_pund_duration - t)*1e9:.1f} ns")
    
    if abs(single_pund_duration - t) < 1e-12:
        print("✓ Times match!")
    else:
        print("⚠ Times don't match - check calculation")
    print("="*50)
    
    # Print final pulse centers
    print(f"\nPulse Center Times (µs):")
    print(f"  Reset:  {reset_center*1e6:.3f}")
    print(f"  Pos1:   {pos1_center*1e6:.3f}")
    print(f"  Pos2:   {pos2_center*1e6:.3f}")
    print(f"  Neg1:   {neg1_center*1e6:.3f}")
    print(f"  Neg2:   {neg2_center*1e6:.3f}")
    
    # Print debug info
    print(f"\nPulse Center Times (µs):")
    print(f"  Reset:  {reset_center*1e6:.3f}")
    print(f"  Pos1:   {pos1_center*1e6:.3f}")
    print(f"  Pos2:   {pos2_center*1e6:.3f}")
    print(f"  Neg1:   {neg1_center*1e6:.3f}")
    print(f"  Neg2:   {neg2_center*1e6:.3f}")
    print(f"  Total calculated time: {t*1e6:.3f} µs")
    print(f"  Expected PUND duration: {single_pund_duration*1e6:.3f} µs")
    
    # Determine appropriate y-offset for labels
    label_offset = max(abs(reset_amplitude), abs(pos_amplitude), abs(neg_amplitude)) * 0.25
    
    # Add annotations with CORRECTED positions
    # Reset pulse
    ax2.axvline(x=reset_center*1e6, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(reset_center*1e6, reset_amplitude - label_offset, 'Reset', 
            rotation=0, fontsize=9, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # First positive pulse
    ax2.axvline(x=pos1_center*1e6, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(pos1_center*1e6, pos_amplitude + label_offset, 'P', 
            rotation=0, fontsize=9, ha='center', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Second positive pulse
    ax2.axvline(x=pos2_center*1e6, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(pos2_center*1e6, pos_amplitude + label_offset, 'U', 
            rotation=0, fontsize=9, ha='center', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # First negative pulse
    ax2.axvline(x=neg1_center*1e6, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(neg1_center*1e6, neg_amplitude - label_offset, 'N', 
            rotation=0, fontsize=9, ha='center', color='orange', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Second negative pulse
    ax2.axvline(x=neg2_center*1e6, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(neg2_center*1e6, neg_amplitude - label_offset, 'D', 
            rotation=0, fontsize=9, ha='center', color='orange', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Set x-limit based on actual data
    ax2.set_xlim([-25e-9, time_us[-1]])
    
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


def setup_awg(waveform, max_amplitude):
    """Upload waveform to AWG and configure output."""
    if max_amplitude <= 0 or max_amplitude > max_allowed_amplitude_v:
        raise ValueError(f"Pulse amplitude {max_amplitude}V is outside the allowed range (0 to {max_allowed_amplitude_v}V)")
    
    arb_name = "PUND_BURST"
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

        awg.write(f"VOLT {2.0 * max_amplitude}")
        check_errors(awg, f"VOLT {2.0 * max_amplitude}")

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
    """Main function to build, plot, and/or upload PUND burst waveform."""
    print("="*60)
    print("PUND Burst Waveform Generator (Binary Upload)")
    print("="*60)
    
    # Build waveform
    try:
        waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude = build_pund_burst_waveform()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Print configuration
    print(f"\nWaveform Configuration:")
    print(f"  Reset pulse: {reset_amplitude:.2f} V, {reset_width*1e6:.1f} µs width, {reset_rise_time*1e6:.1f} µs rise/fall")
    print(f"  Positive pulses: {pos_amplitude:.2f} V, {pos_width*1e6:.1f} µs width, {pos_rise_time*1e6:.1f} µs rise/fall")
    print(f"  Negative pulses: {neg_amplitude:.2f} V, {neg_width*1e6:.1f} µs width, {neg_rise_time*1e6:.1f} µs rise/fall")
    print(f"\nGaps:")
    print(f"  After reset: {gap_after_reset*1e6:.1f} µs")
    print(f"  Between positive pulses: {gap_between_pos*1e6:.1f} µs")
    print(f"  Positive to negative: {gap_pos_to_neg*1e6:.1f} µs")
    print(f"  Between negative pulses: {gap_between_neg*1e6:.1f} µs")
    if num_pund_repeats > 1:
        print(f"  Between PUND sequences: {gap_between_punds*1e6:.1f} µs")
    print(f"\nHolds:")
    print(f"  Initial hold: {init_hold*1e6:.1f} µs")
    print(f"  Final hold: {final_hold*1e6:.1f} µs")
    print(f"\nRepetition:")
    print(f"  Number of PUNDs per cycle: {num_pund_repeats}")
    print(f"  Single PUND duration: {single_pund_duration*1e6:.1f} µs")
    print(f"  Total PUND time: {total_pund_time*1e3:.3f} ms")
    print(f"  Zero section (auto-calculated): {zero_duration*1e3:.3f} ms")
    print(f"  Target period (60 Hz): {target_period*1e3:.3f} ms")
    print(f"\nWaveform Details:")
    print(f"  Total duration: {total_duration_s*1e3:.3f} ms")
    print(f"  Total points: {len(waveform)} / {maximum_arb_points}")
    print(f"  Memory used: {len(waveform)/maximum_arb_points*100:.1f}%")
    print(f"  Sample rate: {arb_sample_rate_hz/1e6:.1f} MS/s")
    print(f"  Actual repetition rate: {1/total_duration_s:.3f} Hz")
    print(f"  Max amplitude: {max_amplitude:.3f} V")
    print(f"  Vpp on AWG: {2.0*max_amplitude:.3f} V")
    
    # Plot if requested
    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_pund_waveform(waveform, total_duration_s, total_pund_time, single_pund_duration, zero_duration, max_amplitude)
    
    # Send to instrument if requested
    if SEND_TO_INSTRUMENT:
        setup_awg(waveform, max_amplitude)
    else:
        print("\n(SEND_TO_INSTRUMENT = False, not uploading to AWG)")
    
    print("="*60)


if __name__ == "__main__":
    main()