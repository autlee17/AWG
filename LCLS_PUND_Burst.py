# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:23:43 2026

@author: leeau
"""
import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import time

# CONFIGURATION FLAGS
SEND_TO_INSTRUMENT = False  # Set to False to only plot, True to send to Agilent
SHOW_PLOT = True           # Set to True to display plot in Spyder

# Connect to your AWG using the VISA protocol
rm = pyvisa.ResourceManager()
fg = None
if SEND_TO_INSTRUMENT:
    # Keysight USB address
    #fg = rm.open_resource('USB0::0x0957::0x2707::MY57301577::INSTR')  # Replace with your VISA address
    # Agilent 33500B USB address
    fg = rm.open_resource('USB0::2391::9991::MY52300442::0::INSTR')
    fg.timeout = 20000  # 20 seconds (in milliseconds)
    
def generate_pund_burst(
    # Reset pulse parameters
    reset_amplitude=-1.0,        # Negative reset pulse amplitude (V at sample)
    reset_width=100e-6,          # Reset pulse width (s)
    reset_rise_time=10e-6,       # Reset pulse rise/fall time (s)
    
    # Positive pulse parameters  
    pos_amplitude=1.0,           # Positive pulse amplitude (V at sample)
    pos_width=100e-6,            # Positive pulse width (s)
    pos_rise_time=10e-6,         # Positive pulse rise/fall time (s)
    
    # Negative pulse parameters
    neg_amplitude=-1.0,          # Negative pulse amplitude (V at sample)
    neg_width=100e-6,            # Negative pulse width (s)
    neg_rise_time=10e-6,         # Negative pulse rise/fall time (s)
    
    # Gap parameters
    gap_after_reset=100e-6,      # Gap after reset pulse (s)
    gap_between_pos=100e-6,      # Gap between two positive pulses (s)
    gap_pos_to_neg=100e-6,       # Gap from positive to negative pulses (s)
    gap_between_neg=100e-6,      # Gap between two negative pulses (s)
    
    # Initial and final holds
    init_hold=50e-6,             # Initial hold at 0V (s)
    final_hold=50e-6,            # Final hold at 0V before zero section (s)
    
    # Zero section (idle time)
    zero_duration=12.7e-3,       # Duration of 0V after PUND (s)
    
    # System parameters
    op_amp_ratio=20,             # Op-amp gain
    sample_rate=3.8e6,           # Sampling rate (S/s)
    send_to_fg=True,
    show_plot=True,
    max_points=60000
):
    """
    Generate a PUND waveform followed by zero voltage:
    
    Structure:
    [init_hold] → [Reset pulse] → [gap] → 
    [Pos pulse 1] → [gap] → [Pos pulse 2] → [gap] →
    [Neg pulse 1] → [gap] → [Neg pulse 2] → 
    [final_hold] → [zero_duration at 0V]
    
    Then repeats continuously.
    """
    
    if send_to_fg and fg is not None:
        fg.write("DATA:VOLatile:CLEar")
    
    # Calculate number of points for each section
    n_init_hold = int(init_hold * sample_rate)
    
    # Reset pulse
    n_reset_rise = int(reset_rise_time * sample_rate)
    n_reset_width = int(reset_width * sample_rate)
    n_gap_reset = int(gap_after_reset * sample_rate)
    
    # Positive pulses
    n_pos_rise = int(pos_rise_time * sample_rate)
    n_pos_width = int(pos_width * sample_rate)
    n_gap_between_pos = int(gap_between_pos * sample_rate)
    n_gap_pos_to_neg = int(gap_pos_to_neg * sample_rate)
    
    # Negative pulses
    n_neg_rise = int(neg_rise_time * sample_rate)
    n_neg_width = int(neg_width * sample_rate)
    n_gap_between_neg = int(gap_between_neg * sample_rate)
    
    n_final_hold = int(final_hold * sample_rate)
    n_zero = int(zero_duration * sample_rate)
    
    # Calculate total points
    total_points = (n_init_hold + 
                   3*n_reset_rise + n_reset_width + n_gap_reset +
                   2*(3*n_pos_rise + n_pos_width) + n_gap_between_pos + n_gap_pos_to_neg +
                   2*(3*n_neg_rise + n_neg_width) + n_gap_between_neg +
                   n_final_hold + n_zero)
    
    # Calculate PUND duration (everything except zero section)
    pund_duration = (init_hold + 
                    reset_rise_time*3 + reset_width + gap_after_reset +
                    2*(pos_rise_time*3 + pos_width) + gap_between_pos + gap_pos_to_neg +
                    2*(neg_rise_time*3 + neg_width) + gap_between_neg +
                    final_hold)
    
    # Check if within instrument limits
    print(f"\n{'='*60}")
    print(f"PUND Waveform Configuration:")
    print(f"  PUND sequence duration: {pund_duration*1e3:.3f} ms")
    print(f"  Zero section: {zero_duration*1e3:.1f} ms")
    print(f"  Total duration: {(pund_duration + zero_duration)*1e3:.2f} ms")
    print(f"  Total points: {total_points}")
    print(f"  Instrument limit: {max_points} points")
    
    if total_points > max_points:
        print(f"\n⚠ WARNING: {total_points} points exceeds {max_points} limit!")
        suggested_sample_rate = sample_rate * max_points / total_points * 0.95
        print(f"  Recommended sample_rate: {suggested_sample_rate/1e6:.2f} MS/s")
        print(f"{'='*60}\n")
        return np.array([]), np.array([])
    else:
        print(f"  ✓ Waveform fits ({total_points/max_points*100:.1f}% of memory used)")
        print(f"{'='*60}\n")
    
    # Normalize amplitudes to -1 to +1 range
    # Find max absolute amplitude to normalize properly
    max_amplitude = max(abs(reset_amplitude), abs(pos_amplitude), abs(neg_amplitude))
    
    reset_norm = reset_amplitude / max_amplitude
    pos_norm = pos_amplitude / max_amplitude
    neg_norm = neg_amplitude / max_amplitude
    
    # Build waveform segments
    segments = []
    segment_labels = []
    
    # Initial hold
    segments.append(np.zeros(n_init_hold))
    segment_labels.append("Init hold")
    
    # Reset pulse (typically negative)
    segments.append(np.linspace(0, reset_norm, n_reset_rise))
    segment_labels.append("Reset rise")
    segments.append(np.ones(n_reset_width) * reset_norm)
    segment_labels.append("Reset pulse")
    segments.append(np.linspace(reset_norm, 0, n_reset_rise))
    segment_labels.append("Reset fall")
    segments.append(np.zeros(n_gap_reset))
    segment_labels.append("Gap after reset")
    
    # First positive pulse
    segments.append(np.linspace(0, pos_norm, n_pos_rise))
    segment_labels.append("Pos1 rise")
    segments.append(np.ones(n_pos_width) * pos_norm)
    segment_labels.append("Pos1 pulse")
    segments.append(np.linspace(pos_norm, 0, n_pos_rise))
    segment_labels.append("Pos1 fall")
    segments.append(np.zeros(n_gap_between_pos))
    segment_labels.append("Gap between pos")
    
    # Second positive pulse
    segments.append(np.linspace(0, pos_norm, n_pos_rise))
    segment_labels.append("Pos2 rise")
    segments.append(np.ones(n_pos_width) * pos_norm)
    segment_labels.append("Pos2 pulse")
    segments.append(np.linspace(pos_norm, 0, n_pos_rise))
    segment_labels.append("Pos2 fall")
    segments.append(np.zeros(n_gap_pos_to_neg))
    segment_labels.append("Gap pos to neg")
    
    # First negative pulse
    segments.append(np.linspace(0, neg_norm, n_neg_rise))
    segment_labels.append("Neg1 rise")
    segments.append(np.ones(n_neg_width) * neg_norm)
    segment_labels.append("Neg1 pulse")
    segments.append(np.linspace(neg_norm, 0, n_neg_rise))
    segment_labels.append("Neg1 fall")
    segments.append(np.zeros(n_gap_between_neg))
    segment_labels.append("Gap between neg")
    
    # Second negative pulse
    segments.append(np.linspace(0, neg_norm, n_neg_rise))
    segment_labels.append("Neg2 rise")
    segments.append(np.ones(n_neg_width) * neg_norm)
    segment_labels.append("Neg2 pulse")
    segments.append(np.linspace(neg_norm, 0, n_neg_rise))
    segment_labels.append("Neg2 fall")
    
    # Final hold
    segments.append(np.zeros(n_final_hold))
    segment_labels.append("Final hold")
    
    # Zero section
    segments.append(np.zeros(n_zero))
    segment_labels.append("Zero section")
    
    # Concatenate all segments
    y = np.concatenate(segments)
    
    # Create time array (in milliseconds)
    t_full = np.arange(len(y)) / sample_rate * 1e3
    
    # Calculate actual output amplitude (accounting for op-amp)
    output_amplitude = max_amplitude / op_amp_ratio
    
    # Plotting
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Full waveform view
        ax1.plot(t_full, y, linewidth=1.5, color='blue')
        ax1.axvline(x=pund_duration*1e3, color='red', linestyle='--', 
                   linewidth=2, label='PUND/Zero boundary')
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Normalized Amplitude', fontsize=12)
        ax1.set_title(f'Full PUND Waveform: {len(y)} points @ {sample_rate/1e6:.1f} MS/s', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, (pund_duration + zero_duration)*1e3])
        
        # Zoomed view of PUND sequence only
        pund_points = len(y) - n_zero
        ax2.plot(t_full[:pund_points], y[:pund_points], linewidth=2, color='green')
        
        # Add markers for pulse boundaries
        current_time = 0
        colors = ['red', 'blue', 'blue', 'orange', 'orange']
        labels_added = set()
        
        pulse_times = [
            (init_hold + reset_rise_time + reset_width/2, 'Reset', 'red'),
            (init_hold + reset_rise_time*3 + reset_width + gap_after_reset + 
             pos_rise_time + pos_width/2, 'Pos 1', 'blue'),
            (init_hold + reset_rise_time*3 + reset_width + gap_after_reset + 
             pos_rise_time*3 + pos_width + gap_between_pos + 
             pos_rise_time + pos_width/2, 'Pos 2', 'blue'),
            (init_hold + reset_rise_time*3 + reset_width + gap_after_reset + 
             2*(pos_rise_time*3 + pos_width) + gap_between_pos + gap_pos_to_neg +
             neg_rise_time + neg_width/2, 'Neg 1', 'orange'),
            (init_hold + reset_rise_time*3 + reset_width + gap_after_reset + 
             2*(pos_rise_time*3 + pos_width) + gap_between_pos + gap_pos_to_neg +
             neg_rise_time*3 + neg_width + gap_between_neg +
             neg_rise_time + neg_width/2, 'Neg 2', 'orange')
        ]
        
        for t_marker, label, color in pulse_times:
            ax2.axvline(x=t_marker*1e3, color=color, linestyle=':', 
                       linewidth=1.5, alpha=0.7)
            ax2.text(t_marker*1e3, max(y[:pund_points])*0.9, label, 
                    rotation=0, fontsize=10, ha='center', color=color)
        
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Normalized Amplitude', fontsize=12)
        ax2.set_title(f'PUND Sequence Detail ({pund_duration*1e6:.1f} µs)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_xlim([0, pund_duration*1e3])
        
        plt.tight_layout()
        plt.show()
    
    # Calculate waveform frequency (repetition rate)
    waveform_frequency = sample_rate / len(y)
    
    # Send to function generator
    if send_to_fg and fg is not None:
        fg.write('*RST')
        fg.write('DATA VOLATILE, ' + ','.join(map(str, y)))
        
        # Check for errors
        try:
            error = fg.query('SYST:ERR?')
            if not error.startswith('+0') and not error.startswith('0'):
                print(f"⚠ Instrument error: {error}")
                return y, t_full
        except:
            pass
        
        fg.write('FUNC:USER VOLATILE')
        fg.write('FUNC USER')
        fg.write(f'FREQ {waveform_frequency}')
        fg.write(f'VOLT {output_amplitude * 2}')  # Vpp = 2 × amplitude
        fg.write(f'VOLT:OFFS 0')
        fg.write('OUTP ON')
        
        print(f"✓ Waveform sent to instrument")
    
  # Summary
    print(f"\nPUND Waveform Summary:")
    print(f"  PUND sequence: {pund_duration*1e6:.1f} µs")
    print(f"  Zero section: {zero_duration*1e3:.1f} ms")
    print(f"  Total duration: {(pund_duration + zero_duration)*1e3:.2f} ms")
    print(f"  Sample rate: {sample_rate/1e6:.1f} MS/s")
    print(f"  Function generator output: {output_amplitude*2:.4f} Vpp")
    print(f"  After {op_amp_ratio}× amplification: {max_amplitude*2:.2f} Vpp at sample")
    print(f"  Waveform repetition rate: {waveform_frequency:.3f} Hz ({1/waveform_frequency*1e3:.2f} ms period)")
    print(f"\nPulse Details:")
    print(f"  Reset: {reset_amplitude:.2f} V, {reset_width*1e6:.1f} µs width, {reset_rise_time*1e6:.1f} µs rise/fall")
    print(f"  Positive: {pos_amplitude:.2f} V, {pos_width*1e6:.1f} µs width, {pos_rise_time*1e6:.1f} µs rise/fall")
    print(f"  Negative: {neg_amplitude:.2f} V, {neg_width*1e6:.1f} µs width, {neg_rise_time*1e6:.1f} µs rise/fall")
    
    return y, t_full


# Example usage for PUND:
if __name__ == "__main__":
    SEND_TO_INSTRUMENT = False
    SHOW_PLOT = True
    
    if SEND_TO_INSTRUMENT and fg is None:
        fg = rm.open_resource('USB0::2391::9991::MY52300442::0::INSTR')
        fg.timeout = 20000
    
    # Generate PUND waveform
    waveform, time = generate_pund_burst(
        # Reset pulse
        reset_amplitude=-2.0,        # -2V reset pulse
        reset_width=200e-6,          # 200 µs
        reset_rise_time=10e-6,       # 10 µs rise/fall
        
        # Positive pulses
        pos_amplitude=1.5,           # +1.5V 
        pos_width=100e-6,            # 100 µs
        pos_rise_time=10e-6,         # 10 µs rise/fall
        
        # Negative pulses
        neg_amplitude=-1.5,          # -1.5V
        neg_width=100e-6,            # 100 µs
        neg_rise_time=10e-6,         # 10 µs rise/fall
        
        # Gaps
        gap_after_reset=100e-6,      # 100 µs
        gap_between_pos=50e-6,       # 50 µs
        gap_pos_to_neg=100e-6,       # 100 µs
        gap_between_neg=50e-6,       # 50 µs
        
        # Holds
        init_hold=50e-6,             # 50 µs
        final_hold=50e-6,            # 50 µs
        
        # Zero section
        zero_duration=12.7e-3,       # 12.7 ms at 0V
        
        # System parameters
        op_amp_ratio=20,
        sample_rate=3.8e6,           # 3.8 MS/s
        send_to_fg=SEND_TO_INSTRUMENT,
        show_plot=SHOW_PLOT,
        max_points=65534
    )
    
    if len(waveform) > 0:
        print("\n✓ PUND waveform ready!")
    else:
        print("\n✗ PUND waveform generation failed")
    
    if SEND_TO_INSTRUMENT:
        # Close connection when done
        fg.close()
        rm.close()