# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:09:27 2026

@author: leeau
"""

import numpy as np
import pyvisa
import matplotlib.pyplot as plt
from scipy import signal

# -----------------------------
# CONFIGURATION FLAGS
# -----------------------------
SEND_TO_INSTRUMENT = False
SHOW_PLOT = True

# -----------------------------
# User-editable Parameters
# -----------------------------

# --- Square Reset Pulse ---
square_amplitude    = 2.0       # Amplitude of square pulse (V), POSITIVE value
                                # Triangle will always be NEGATIVE
square_rise_fall    = 50e-6     # Rise and fall time (s)
square_pulse_width  = 1e-3      # Flat-top duration (s)

# --- Gap between square pulse and triangle ---
gap1_duration       = 500e-6    # Zero-volt gap after square pulse (s)

# --- Negative Triangle Arch ---
triangle_amplitude  = 1.0       # Peak amplitude of triangle (V), POSITIVE value
                                # Output will be negative (0 -> -peak -> 0)
triangle_apparent_freq = 3000   # Hz — as if it were a full triangle period;
                                # only the negative half-arch is shown

# --- Gap between triangle and next repeat ---
gap2_duration       = 500e-6    # Zero-volt gap after triangle (s)

# --- Repetitions ---
num_repeats         = 3         # How many times to repeat [square, gap1, triangle, gap2]
                                # within the 60 Hz window

# --- 60 Hz Cycle ---
target_period       = 1 / 60   # 16.67 ms

# --= Initial Hold ---
initial_hold_duration =  1e-3 # Zero volt hold before first square pulse (s)

# -----------------------------
# System Parameters
# -----------------------------
arb_sample_rate_hz  = 3.8e6
maximum_arb_points  = 1000000
max_allowed_amplitude_v = 10.0

#awg_address = "USB0::0x0957::0x2707::MY57301577::INSTR" # Keysight
awg_address    = "USB0::2391::9991::MY52300442::0::INSTR" # Agilent
awg_timeout_ms = 30000


# -----------------------------
# Helper: flat segment
# -----------------------------
def segment(value, duration_s, sample_rate):
    """Create a flat segment at a given voltage value."""
    if duration_s < 0:
        raise ValueError(f"Duration cannot be negative (got {duration_s:.6f} s)")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * sample_rate)))
    return np.full(point_count, value, dtype=float)


# -----------------------------
# Helper: linear ramp
# -----------------------------
def ramp(start_value, end_value, duration_s, sample_rate):
    """Create a linear ramp from start_value to end_value."""
    if duration_s < 0:
        raise ValueError(f"Ramp duration cannot be negative (got {duration_s:.6f} s)")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * sample_rate)))
    return np.linspace(start_value, end_value, point_count, dtype=float)


# -----------------------------
# Helper: build one negative triangle arch (0 -> -peak -> 0)
# -----------------------------
def build_triangle_arch(apparent_freq, amplitude_v, sample_rate):
    """
    Build one negative arch: 0 V -> -amplitude_v -> 0 V.

    Duration = 1 / (2 * apparent_freq)
    This equals half the period of apparent_freq, so the arch
    looks like the negative half of a full triangle period.

    Built directly with two linear ramps — no sawtooth/roll tricks:
      - Ramp DOWN from 0 to -amplitude_v  (first half of arch)
      - Ramp UP   from -amplitude_v to 0  (second half of arch)
    """
    arch_duration = 1.0 / (2.0 * apparent_freq)   # total arch duration (s)
    n_total       = int(round(arch_duration * sample_rate))

    # Split evenly into descending and ascending ramp
    # Using n_total // 2 + 1 with [:-1] trick avoids duplicating
    # the shared midpoint where the two ramps meet
    n_half = n_total // 2

    ramp_down = np.linspace(0.0,          -amplitude_v, n_half,          endpoint=False)
    ramp_up   = np.linspace(-amplitude_v,  0.0,         n_total - n_half, endpoint=True)

    arch = np.concatenate([ramp_down, ramp_up])

    return arch


# -----------------------------
# Waveform Builder
# -----------------------------
def build_waveform():
    """
    Build full waveform:

    Repeat num_repeats times:
        [rise | flat top | fall | gap1 | negative triangle arch | gap2]

    Then pad with zeros to fill the 60 Hz target_period.
    """
    
    # --- Initial hold ---
    initial_hold_section = segment(0.0, initial_hold_duration, arb_sample_rate_hz)

    # --- Build one square pulse ---
    rise_section     = ramp(0.0, square_amplitude, square_rise_fall, arb_sample_rate_hz)
    flat_top_section = segment(square_amplitude, square_pulse_width, arb_sample_rate_hz)
    fall_section     = ramp(square_amplitude, 0.0, square_rise_fall, arb_sample_rate_hz)

    square_duration  = square_rise_fall + square_pulse_width + square_rise_fall

    # --- Build gap1 ---
    gap1_section     = segment(0.0, gap1_duration, arb_sample_rate_hz)

    # --- Build negative triangle arch ---
    triangle_arch    = build_triangle_arch(
        triangle_apparent_freq, triangle_amplitude, arb_sample_rate_hz
    )
    triangle_duration = len(triangle_arch) / arb_sample_rate_hz

    # --- Build gap2 ---
    gap2_section     = segment(0.0, gap2_duration, arb_sample_rate_hz)

    # --- One full repeat block ---
    one_repeat = np.concatenate([
        rise_section,
        flat_top_section,
        fall_section,
        gap1_section,
        triangle_arch,
        gap2_section,
    ])

    one_repeat_duration = len(one_repeat) / arb_sample_rate_hz

    # --- Check repeats fit within target period ---
    total_active_duration = one_repeat_duration * num_repeats
    zero_pad_duration     = target_period - total_active_duration

    if zero_pad_duration < 0:
        raise ValueError(
            f"\nInitial hold ({initial_hold_duration*1e3:.4f} ms) + "
            f"\n{num_repeats} repeats × {one_repeat_duration*1e3:.4f} ms per repeat "
            f"= {total_active_duration*1e3:.4f} ms\n"
            f"This EXCEEDS the target period ({target_period*1e3:.4f} ms) by "
            f"{abs(zero_pad_duration)*1e3:.4f} ms.\n"
            f"Options: reduce num_repeats, shorten square_pulse_width, "
            f"reduce gap durations, or increase triangle_apparent_freq."
        )

    # --- Tile repeats + zero pad ---
    repeated_section = np.tile(one_repeat, num_repeats)
    zero_pad_section = segment(0.0, zero_pad_duration, arb_sample_rate_hz)

    waveform = np.concatenate([initial_hold_section, repeated_section, zero_pad_section])

    if len(waveform) > maximum_arb_points:
        raise ValueError(
            f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}.\n"
            f"Reduce sample rate, durations, or number of repeats."
        )

    total_duration_s = len(waveform) / arb_sample_rate_hz

    return (
        waveform,
        total_duration_s,
        one_repeat_duration,
        square_duration,
        triangle_duration,
        zero_pad_duration,
    )


# -----------------------------
# Plotter
# -----------------------------
def plot_waveform(waveform, total_duration_s, one_repeat_duration,
                  square_duration, triangle_duration, zero_pad_duration):

    time_ms = np.arange(len(waveform)) / arb_sample_rate_hz * 1e3
    voltage  = waveform

    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    # ------------------------------------------------------------------
    # Plot 1: Full 60 Hz cycle with shaded regions for each repeat
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(time_ms, voltage, linewidth=1.2, color='steelblue')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.4)

    colors_sq  = ['#ff9999', '#ff4444']  # alternating shades for square
    colors_tri = ['#99ff99', '#22aa22']  # alternating shades for triangle

    for i in range(num_repeats):
        block_start = initial_hold_duration + i * one_repeat_duration

        sq_start  = block_start
        sq_end    = block_start + square_duration
        tri_start = block_start + square_duration + gap1_duration
        tri_end   = tri_start  + triangle_duration

        ci = i % 2  # alternate shading so adjacent blocks are visually distinct

        ax.axvspan(sq_start  * 1e3, sq_end  * 1e3,
                   alpha=0.25, color=colors_sq[ci],
                   label='Square pulse' if i == 0 else None)
        ax.axvspan(tri_start * 1e3, tri_end * 1e3,
                   alpha=0.25, color=colors_tri[ci],
                   label='Triangle arch' if i == 0 else None)

    # Shade zero pad
    active_end_ms = num_repeats * one_repeat_duration * 1e3
    ax.axvspan(active_end_ms, total_duration_s * 1e3,
               alpha=0.10, color='gray', label='Zero pad')

    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(
        f'Full 60 Hz Cycle — {num_repeats} repeats — '
        f'{len(waveform)} points @ {arb_sample_rate_hz/1e6:.1f} MS/s',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, total_duration_s * 1e3])

    # ------------------------------------------------------------------
    # Plot 2: Zoomed — first square pulse only
    # ------------------------------------------------------------------
    ax = axes[1]
    sq_start_pt = int(round(initial_hold_duration * arb_sample_rate_hz))
    sq_end_pt   = sq_start_pt + int(round(square_duration * arb_sample_rate_hz))
    sq_end_pt   = min(sq_end_pt, len(waveform))
    
    # reset time axis to 0 for this zoomed window
    t_sq = np.arange(sq_end_pt - sq_start_pt) / arb_sample_rate_hz * 1e3
    v_sq = voltage[sq_start_pt:sq_end_pt]

    ax.plot(t_sq, v_sq, linewidth=2, color='red')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.4)

    rise_end_ms = square_rise_fall * 1e3
    flat_end_ms = rise_end_ms + square_pulse_width * 1e3
    sq_end_ms   = square_duration * 1e3

    ax.axvspan(0,            rise_end_ms, alpha=0.25, color='orange',
               label=f'Rise ({square_rise_fall*1e6:.0f} µs)')
    ax.axvspan(rise_end_ms,  flat_end_ms, alpha=0.25, color='red',
               label=f'Flat top ({square_pulse_width*1e3:.2f} ms)')
    ax.axvspan(flat_end_ms,  sq_end_ms,   alpha=0.25, color='orange',
               label=f'Fall ({square_rise_fall*1e6:.0f} µs)')

    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(
        f'Zoomed: Square Reset Pulse  (amplitude = +{square_amplitude:.2f} V)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, sq_end_ms])

    # ------------------------------------------------------------------
    # Plot 3: Zoomed — first triangle arch only
    # ------------------------------------------------------------------
    ax = axes[2]
    tri_start_pt = int(round(
        (initial_hold_duration + square_duration + gap1_duration) * arb_sample_rate_hz
    ))
    tri_end_pt = tri_start_pt + int(round(triangle_duration * arb_sample_rate_hz))
    tri_end_pt = min(tri_end_pt, len(waveform))

    t_tri = np.arange(tri_end_pt - tri_start_pt) / arb_sample_rate_hz * 1e3
    v_tri = voltage[tri_start_pt:tri_end_pt]

    ax.plot(t_tri, v_tri, linewidth=2, color='green')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.4)

    half_period_ms = (1.0 / (2.0 * triangle_apparent_freq)) * 1e3
    mid_ms         = half_period_ms / 2

    ax.annotate(
        '', xy=(half_period_ms, -triangle_amplitude * 0.6),
        xytext=(0, -triangle_amplitude * 0.6),
        arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2)
    )
    ax.text(
        mid_ms, -triangle_amplitude * 0.75,
        f'Half-period = {half_period_ms*1e3:.1f} µs\n'
        f'Apparent freq = {triangle_apparent_freq} Hz',
        ha='center', fontsize=9, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Voltage (V)', fontsize=11)
    ax.set_title(
        f'Zoomed: Negative Triangle Arch  '
        f'(amplitude = -{triangle_amplitude:.2f} V, apparent freq = {triangle_apparent_freq} Hz)',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, triangle_duration * 1e3])

    plt.tight_layout()
    plt.show()


# -----------------------------
# AWG Functions
# -----------------------------
def check_errors(awg, context):
    while True:
        response = awg.query("SYST:ERR?").strip()
        if int(response.split(",", 1)[0]) == 0:
            break
        print(f"AWG error after '{context}': {response}")


def setup_awg(waveform):
    peak_v = np.max(np.abs(waveform))
    if peak_v <= 0 or peak_v > max_allowed_amplitude_v:
        raise ValueError(f"Peak amplitude {peak_v:.3f} V is outside allowed range.")

    arb_name      = "RST_TRI"
    rm            = pyvisa.ResourceManager()
    awg           = rm.open_resource(awg_address)
    awg.timeout          = awg_timeout_ms
    awg.write_termination = "\n"
    awg.read_termination  = "\n"

    norm_waveform = waveform / peak_v   # normalize to [-1, +1] for upload

    try:
        print("\nConfiguring AWG...")
        awg.write("*RST")
        awg.write("*CLS")
        awg.write("SOURCE:DATA:VOLATILE:CLEAR")
        check_errors(awg, "SOURCE:DATA:VOLATILE:CLEAR")

        print(f"Uploading {len(norm_waveform)} points...")
        awg.write_binary_values(
            f"SOURCE:DATA:ARBITRARY {arb_name},",
            norm_waveform, datatype="f", is_big_endian=True,
        )
        awg.write("*WAI")
        check_errors(awg, f"SOURCE:DATA:ARBITRARY {arb_name}")

        awg.write(f"FUNC:ARB {arb_name}")
        awg.write("FUNC ARB")
        awg.write(f"FUNC:ARB:SRATE {arb_sample_rate_hz}")
        awg.write(f"VOLT {2.0 * peak_v}")
        awg.write("VOLT:OFFS 0")
        awg.write("OUTP ON")
        check_errors(awg, "OUTP ON")

        print("✓ Waveform uploaded and AWG output enabled.")
    finally:
        awg.close()
        rm.close()


def turn_off_awg():
    rm  = pyvisa.ResourceManager()
    awg = rm.open_resource(awg_address)
    awg.timeout = awg_timeout_ms
    try:
        awg.write("OUTP OFF")
        print("AWG output turned OFF.")
    finally:
        awg.close()
        rm.close()


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print("Reset Pulse + Negative Triangle Waveform Generator")
    print("=" * 60)

    try:
        (waveform, total_duration_s, one_repeat_duration,
         square_duration, triangle_duration, zero_pad_duration) = build_waveform()
    except ValueError as e:
        print(f"\n ERROR: {e}")
        return
    
    print(f"\n--- Initial Hold ---")
    print(f"  Duration         : {initial_hold_duration*1e3:.4f} ms")

    print(f"\n--- Square Reset Pulse ---")
    print(f"  Amplitude        : +{square_amplitude:.3f} V")
    print(f"  Rise / Fall time : {square_rise_fall*1e6:.1f} µs each")
    print(f"  Flat-top width   : {square_pulse_width*1e3:.3f} ms")
    print(f"  Total duration   : {square_duration*1e3:.4f} ms")

    print(f"\n--- Gap 1 (square -> triangle) ---")
    print(f"  Duration         : {gap1_duration*1e6:.1f} µs")

    print(f"\n--- Negative Triangle Arch ---")
    print(f"  Amplitude        : -{triangle_amplitude:.3f} V")
    print(f"  Apparent freq    : {triangle_apparent_freq} Hz")
    print(f"  Half-period shown: {1/(2*triangle_apparent_freq)*1e6:.2f} µs")
    print(f"  Duration         : {triangle_duration*1e3:.4f} ms")

    print(f"\n--- Gap 2 (triangle -> next repeat) ---")
    print(f"  Duration         : {gap2_duration*1e6:.1f} µs")

    print(f"\n--- Timing ---")
    print(f"  One repeat block : {one_repeat_duration*1e3:.4f} ms")
    print(f"  Num repeats      : {num_repeats}")
    print(f"  Total active     : {one_repeat_duration*num_repeats*1e3:.4f} ms")
    print(f"  Zero pad         : {zero_pad_duration*1e3:.4f} ms")
    print(f"  Target period    : {target_period*1e3:.4f} ms  (60 Hz)")
    print(f"  Total points     : {len(waveform)} / {maximum_arb_points} "
          f"({len(waveform)/maximum_arb_points*100:.1f}% memory)")

    if SHOW_PLOT:
        print("\nGenerating plot...")
        plot_waveform(waveform, total_duration_s, one_repeat_duration,
                      square_duration, triangle_duration, zero_pad_duration)

    if SEND_TO_INSTRUMENT:
        setup_awg(waveform)
    else:
        print("\n(SEND_TO_INSTRUMENT = False — not uploading to AWG)")

    print("=" * 60)


if __name__ == "__main__":
    main()