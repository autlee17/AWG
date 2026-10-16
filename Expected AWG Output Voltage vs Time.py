# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:16:46 2026

@author: leeau
"""

"""
Expected AWG output voltage vs. time delay for PUND or triangle burst waveforms,
with optional integration-gate averaging.

Time delay is measured from the external trigger edge (t = 0).
The waveform repeats every target_period (one burst per trigger).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# WAVEFORM SELECTION
# =============================================================
WAVEFORM_TYPE = "Triangle"          # "PUND" or "TRIANGLE"

# =============================================================
# COMMON SETTINGS (match your AWG script)
# =============================================================
voltage_offset_v = 0.0          # DC offset (V); shifts everything, including the idle level
trigger_delay_s = 0.0           # AWG TRIG:DEL setting (s)
trigger_latency_s = 0.0         # Measured trigger-edge-to-output latency (s); 0 = ideal
target_period = 1/60            # External trigger period (s)

# =============================================================
# DELAYS TO EVALUATE (relative to the trigger edge)
# =============================================================
delay_start_s = 0.0
delay_stop_s = 600e-6
delay_step_s = 10e-6
custom_delays_s = None          # e.g. [0.5e-6, 1.2e-6, 2.0e-6] overrides the range above

# =============================================================
# INTEGRATION GATE
# =============================================================
integration_gate_s = 300e-9     # Gate width (s). 0 = instantaneous voltage only
gate_alignment = "center"       # "center": gate = [d - g/2, d + g/2]
                                # "start":  gate = [d, d + g]   (gate opens at the delay)
                                # "end":    gate = [d - g, d]   (gate closes at the delay)

# =============================================================
# OUTPUT OPTIONS
# =============================================================
SHOW_PLOT = True
SAVE_CSV = False
csv_filename = "expected_voltages.csv"

# =============================================================
# TRIANGLE PARAMETERS
# =============================================================
triangle_duration = 4e-3        # s
triangle_freq = 10000            # Hz
triangle_amplitude = 3.0        # Peak V
triangle_phase_deg = 90         # 90 = starts at 0 V and rises (matches the +pi/2 fix)
                                # 0  = starts at -peak (scipy sawtooth default)

# =============================================================
# PUND PARAMETERS
# =============================================================
reset_amplitude = -2.0          # V
reset_width = 1000e-9           # s
reset_rise_time = 20e-9         # s

pos_amplitude = 1.0             # V
pos_width = 500e-9              # s
pos_rise_time = 20e-9           # s

neg_amplitude = -1.5            # V
neg_width = 500e-9              # s
neg_rise_time = 20e-9           # s

gap_after_reset = 500e-9        # s
gap_between_pos = 500e-9        # s
gap_pos_to_neg = 500e-9         # s
gap_between_neg = 500e-9        # s

init_hold = 50e-9               # s
final_hold = 50e-9              # s

num_pund_repeats = 1
gap_between_punds = 100e-6      # s (only used if num_pund_repeats > 1)


# =============================================================
# WAVEFORM CONSTRUCTION (piecewise-linear breakpoints)
# =============================================================
class WaveformBuilder:
    """Builds a piecewise-linear waveform as (time, voltage) breakpoints plus named regions."""

    def __init__(self):
        self.t = [0.0]
        self.v = [0.0]
        self.regions = []       # list of (start_s, end_s, name)

    @property
    def now(self):
        return self.t[-1]

    def hold(self, duration, name=None):
        if duration < 0:
            raise ValueError("Durations cannot be negative")
        start = self.now
        if duration > 0:
            self.t.append(start + duration)
            self.v.append(self.v[-1])
            if name:
                self.regions.append((start, self.now, name))

    def ramp_to(self, value, duration, name=None):
        if duration < 0:
            raise ValueError("Ramp time cannot be negative")
        start = self.now
        self.t.append(start + duration)   # duration 0 -> instantaneous step
        self.v.append(value)
        if name and duration > 0:
            self.regions.append((start, self.now, name))

    def pulse(self, amplitude, width, rise_time, name):
        self.ramp_to(amplitude, rise_time, f"{name} rise")
        self.hold(width, f"{name} top")
        self.ramp_to(0.0, rise_time, f"{name} fall")


def build_pund():
    """PUND burst, time relative to burst start."""
    b = WaveformBuilder()
    for i in range(num_pund_repeats):
        tag = f"#{i+1} " if num_pund_repeats > 1 else ""
        b.hold(init_hold, f"{tag}init hold")
        b.pulse(reset_amplitude, reset_width, reset_rise_time, f"{tag}Reset")
        b.hold(gap_after_reset, f"{tag}gap")
        b.pulse(pos_amplitude, pos_width, pos_rise_time, f"{tag}P")
        b.hold(gap_between_pos, f"{tag}gap")
        b.pulse(pos_amplitude, pos_width, pos_rise_time, f"{tag}U")
        b.hold(gap_pos_to_neg, f"{tag}gap")
        b.pulse(neg_amplitude, neg_width, neg_rise_time, f"{tag}N")
        b.hold(gap_between_neg, f"{tag}gap")
        b.pulse(neg_amplitude, neg_width, neg_rise_time, f"{tag}D")
        b.hold(final_hold, f"{tag}final hold")
        if i < num_pund_repeats - 1:
            b.hold(gap_between_punds, "gap between PUNDs")
    return np.array(b.t), np.array(b.v), b.regions


def triangle_shape(t):
    """Normalized symmetric triangle (same as scipy sawtooth(width=0.5) with phase offset)."""
    phase = 2 * np.pi * triangle_freq * np.asarray(t, dtype=float) + np.deg2rad(triangle_phase_deg)
    p = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    return np.where(p < 0.5, -1 + 4 * p, 3 - 4 * p)


def build_triangle():
    """Triangle burst, time relative to burst start. Breakpoints at every peak/trough."""
    w = 2 * np.pi * triangle_freq
    phi = np.deg2rad(triangle_phase_deg)
    # Peaks/troughs occur where the phase = k*pi
    k = np.arange(np.floor(phi / np.pi), np.ceil((w * triangle_duration + phi) / np.pi) + 1)
    corners = (k * np.pi - phi) / w
    eps = 1e-15
    corners = corners[(corners > eps) & (corners < triangle_duration - eps)]
    t = np.concatenate(([0.0], corners, [triangle_duration]))
    v = triangle_amplitude * triangle_shape(t)
    return t, v, [(0.0, triangle_duration, "triangle")]


def assemble_period(t_rel, v_rel, regions_rel):
    """Place the burst inside one full trigger period: idle -> burst -> idle, then add offset."""
    shift = trigger_delay_s + trigger_latency_s
    burst_end = shift + t_rel[-1]
    if burst_end >= target_period:
        raise ValueError(
            f"Burst ends at {burst_end*1e3:.3f} ms, which is not before the next trigger "
            f"({target_period*1e3:.3f} ms)."
        )
    t = np.concatenate(([0.0, shift], shift + t_rel, [burst_end, target_period]))
    v = np.concatenate(([0.0, 0.0], v_rel, [0.0, 0.0]))   # steps handled by duplicate times
    regions = [(s + shift, e + shift, n) for s, e, n in regions_rel]
    return t, v + voltage_offset_v, regions


# =============================================================
# EVALUATION ENGINE
# =============================================================
class PeriodicPWL:
    """Periodic piecewise-linear waveform with exact value and integral evaluation."""

    def __init__(self, t, v):
        self.t = np.asarray(t, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.period = self.t[-1]
        seg_area = 0.5 * (self.v[1:] + self.v[:-1]) * np.diff(self.t)
        self.cum = np.concatenate(([0.0], np.cumsum(seg_area)))
        self.area_per_period = self.cum[-1]

    def _locate(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        xw = np.mod(x, self.period)
        i = np.searchsorted(self.t, xw, side="right") - 1   # right-continuous at steps
        i = np.clip(i, 0, len(self.t) - 2)
        dt = self.t[i + 1] - self.t[i]
        frac = np.divide(xw - self.t[i], dt, out=np.zeros_like(xw), where=dt > 0)
        val = self.v[i] + frac * (self.v[i + 1] - self.v[i])
        return x, xw, i, val

    def value(self, x):
        """Instantaneous voltage at time(s) x after the trigger."""
        return self._locate(x)[3]

    def integral(self, x):
        """Exact integral of V(t) from 0 to x (V*s)."""
        x, xw, i, val = self._locate(x)
        n_periods = np.floor(x / self.period)
        return (n_periods * self.area_per_period
                + self.cum[i] + (xw - self.t[i]) * (self.v[i] + val) / 2)

    def gate_average(self, delays, gate, alignment="center"):
        """Average voltage over the integration gate for each delay."""
        d = np.atleast_1d(np.asarray(delays, dtype=float))
        if alignment == "center":
            g1, g2 = d - gate / 2, d + gate / 2
        elif alignment == "start":
            g1, g2 = d, d + gate
        elif alignment == "end":
            g1, g2 = d - gate, d
        else:
            raise ValueError("gate_alignment must be 'center', 'start', or 'end'")
        if gate <= 0:
            return self.value(d), d, d
        avg = (self.integral(g2) - self.integral(g1)) / (g2 - g1)
        return avg, g1, g2


def region_at(x, regions, period):
    """Name of the waveform region containing time x."""
    xw = x % period
    for start, end, name in regions:
        if start <= xw < end:
            return name
    return "idle"


def build_waveform():
    """Build the full-period waveform model for the selected WAVEFORM_TYPE."""
    kind = WAVEFORM_TYPE.upper()
    if kind == "PUND":
        t_rel, v_rel, reg = build_pund()
    elif kind == "TRIANGLE":
        t_rel, v_rel, reg = build_triangle()
    else:
        raise ValueError("WAVEFORM_TYPE must be 'PUND' or 'TRIANGLE'")
    t, v, regions = assemble_period(t_rel, v_rel, reg)
    return PeriodicPWL(t, v), regions


def compute_expected(delays, gate=None, alignment=None):
    """
    Convenience function for interactive use.
    Returns (instantaneous_V, gate_avg_V, gate_start_s, gate_end_s, region_names).
    """
    gate = integration_gate_s if gate is None else gate
    alignment = gate_alignment if alignment is None else alignment
    wf, regions = build_waveform()
    delays = np.atleast_1d(np.asarray(delays, dtype=float))
    inst = wf.value(delays)
    avg, g1, g2 = wf.gate_average(delays, gate, alignment)
    names = [region_at(d, regions, wf.period) for d in delays]
    return inst, avg, g1, g2, names


# =============================================================
# OUTPUT HELPERS
# =============================================================
def choose_time_unit(span_s):
    if span_s < 1e-6:
        return 1e9, "ns"
    if span_s < 1e-3:
        return 1e6, "µs"
    return 1e3, "ms"


def get_delays():
    if custom_delays_s is not None:
        return np.asarray(custom_delays_s, dtype=float)
    return np.arange(delay_start_s, delay_stop_s + delay_step_s / 2, delay_step_s)


def main():
    wf, regions = build_waveform()
    delays = get_delays()
    inst = wf.value(delays)
    avg, g1, g2 = wf.gate_average(delays, integration_gate_s, gate_alignment)

    scale, unit = choose_time_unit(max(delays.max() - delays.min(), integration_gate_s, 1e-9))

    print("=" * 78)
    print(f"Expected AWG output: {WAVEFORM_TYPE.upper()} | offset {voltage_offset_v:.3f} V | "
          f"gate {integration_gate_s*1e9:.1f} ns ({gate_alignment})")
    print("=" * 78)
    print(f"{'Delay ('+unit+')':>14} {'V_inst (V)':>11} {'V_gate (V)':>11}   Region")
    print("-" * 78)

    rows = []
    for d, vi, va, a, b in zip(delays, inst, avg, g1, g2):
        r_start = region_at(a, regions, wf.period)
        r_end = region_at(b - 1e-15, regions, wf.period) if b > a else r_start
        region = region_at(d, regions, wf.period)
        flag = f"  ⚠ gate spans {r_start} -> {r_end}" if r_start != r_end else ""
        print(f"{d*scale:14.4f} {vi:11.4f} {va:11.4f}   {region}{flag}")
        rows.append([d, vi, va, a, b, region, r_start != r_end])

    if SAVE_CSV:
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["delay_s", "V_instantaneous", "V_gate_avg",
                             "gate_start_s", "gate_end_s", "region", "gate_spans_regions"])
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {csv_filename}")

    if SHOW_PLOT:
        pad = max(integration_gate_s, 0.05 * (delays.max() - delays.min()), 1e-9)
        lo, hi = delays.min() - pad, delays.max() + pad
        # Dense time axis plus exact breakpoints so sharp edges render correctly
        bp = wf.t[(wf.t >= lo) & (wf.t <= hi)]
        tt = np.unique(np.concatenate((np.linspace(lo, hi, 5000), bp, bp - 1e-15)))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(tt * scale, wf.value(tt), color="blue", lw=1.5, label="AWG output (ideal)")
        ax.plot(delays * scale, inst, "o", color="black", ms=4, label="Instantaneous")
        if integration_gate_s > 0:
            ax.plot(delays * scale, avg, "s", color="red", ms=5, mfc="none",
                    label=f"Gate average ({integration_gate_s*1e9:.0f} ns, {gate_alignment})")
            ax.axvspan(g1[0] * scale, g2[0] * scale, color="orange", alpha=0.25,
                       label="Gate (first delay)")
        ax.axhline(voltage_offset_v, color="purple", ls="--", lw=1,
                   label=f"Idle level ({voltage_offset_v:.2f} V)")
        ax.set_xlabel(f"Delay from trigger ({unit})", fontsize=12)
        ax.set_ylabel("Voltage (V)", fontsize=12)
        ax.set_title(f"Expected {WAVEFORM_TYPE.upper()} voltage vs. delay", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()