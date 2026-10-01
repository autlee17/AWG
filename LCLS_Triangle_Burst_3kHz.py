# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:12:42 2026

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

def generate_triangular_burst(
    triangle_duration=4e-3,      # 4 ms of triangle wave
    zero_duration=12.7e-3,       # 12.7 ms of zero voltage
    triangle_freq=3000,          # 3 kHz triangle wave
    triangle_amplitude=1.0,      # Peak amplitude in volts (before op-amp)
    op_amp_ratio=20,             # Op-amp gain
    sample_rate=60e6,            # 60 MS/s (max for your instrument)
    send_to_fg=True,
    show_plot=True,
    max_points=60000
):
    """
    Generate a waveform with:
    - First 4 ms: 3 kHz triangular wave
    - Last 12.7 ms: 0V
    Total duration: 16.7 ms
    """
    
    if send_to_fg and fg is not None:
        fg.write("DATA:VOLatile:CLEar")
    
    # Calculate number of points for each section
    n_triangle = int(triangle_duration * sample_rate)
    n_zero = int(zero_duration * sample_rate)
    total_points = n_triangle + n_zero
    
    # Check if within instrument limits
    print(f"\n{'='*60}")
    print(f"Waveform Configuration:")
    print(f"  Triangle section: {triangle_duration*1e3:.1f} ms ({n_triangle} points)")
    print(f"  Zero section: {zero_duration*1e3:.1f} ms ({n_zero} points)")
    print(f"  Total: {(triangle_duration + zero_duration)*1e3:.1f} ms ({total_points} points)")
    print(f"  Instrument limit: {max_points} points")
    
    if total_points > max_points:
        print(f"\n⚠ WARNING: {total_points} points exceeds {max_points} limit!")
        suggested_sample_rate = sample_rate * max_points / total_points * 0.95
        print(f"  Recommended sample_rate: {suggested_sample_rate/1e6:.2f} MS/s")
        print(f"{'='*60}\n")
        return None
    else:
        print(f"  ✓ Waveform fits ({total_points/max_points*100:.1f}% of memory used)")
        print(f"{'='*60}\n")
    
    # Generate time arrays
    t_triangle = np.linspace(0, triangle_duration, n_triangle)
    t_zero = np.linspace(triangle_duration, triangle_duration + zero_duration, n_zero)
    
    # Generate triangular wave
    # Triangle wave: sawtooth with width=0.5 makes symmetric triangle
    from scipy import signal
    triangle_wave = signal.sawtooth(2 * np.pi * triangle_freq * t_triangle, width=0.5)
    
    # Alternative without scipy (pure numpy):
    # period = 1 / triangle_freq
    # phase = (t_triangle % period) / period  # 0 to 1
    # triangle_wave = np.where(phase < 0.5, 
    #                          4 * phase - 1,      # Rising edge: -1 to +1
    #                          3 - 4 * phase)      # Falling edge: +1 to -1
    
    # Zero section
    zero_section = np.zeros(n_zero)
    
    # Concatenate
    y = np.concatenate([triangle_wave, zero_section])
    
    # Create full time array for plotting (in milliseconds)
    t_full = np.arange(len(y)) / sample_rate * 1e3
    
    # Calculate actual output amplitude (accounting for op-amp)
    output_amplitude = triangle_amplitude / op_amp_ratio
    
    # Plotting
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Full waveform view
        ax1.plot(t_full, y, linewidth=1.0, color='blue')
        ax1.axvline(x=triangle_duration*1e3, color='red', linestyle='--', 
                   linewidth=2, label='Triangle/Zero boundary')
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Normalized Amplitude', fontsize=12)
        ax1.set_title(f'Full Waveform: {len(y)} points @ {sample_rate/1e6:.1f} MS/s', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, (triangle_duration + zero_duration)*1e3])
        
        # Zoomed view of first few triangle cycles
        zoom_duration = min(3 / triangle_freq, triangle_duration)  # Show ~3 cycles
        zoom_points = int(zoom_duration * sample_rate)
        ax2.plot(t_full[:zoom_points], y[:zoom_points], linewidth=1.5, 
                color='green', marker='o', markersize=2, markevery=max(1, zoom_points//100))
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Normalized Amplitude', fontsize=12)
        ax2.set_title(f'Zoomed: First {zoom_duration*1e3:.2f} ms of Triangle Wave ({triangle_freq/1e3:.1f} kHz)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
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
                return None
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
    print(f"\nWaveform Summary:")
    print(f"  Total duration: {(triangle_duration + zero_duration)*1e3:.2f} ms")
    print(f"  Triangle wave: {triangle_freq} Hz for {triangle_duration*1e3:.1f} ms")
    print(f"  Number of triangle cycles: {triangle_freq * triangle_duration:.1f}")
    print(f"  Sample rate: {sample_rate/1e6:.1f} MS/s")
    print(f"  Points per triangle cycle: {sample_rate/triangle_freq:.0f}")
    print(f"  Function generator output: {output_amplitude*2:.4f} Vpp")
    print(f"  After {op_amp_ratio}× amplification: {triangle_amplitude*2:.2f} Vpp at sample")
    print(f"  Waveform repetition rate: {waveform_frequency:.3f} Hz ({1/waveform_frequency*1e3:.2f} ms period)")
    
    return y, t_full


# Example usage:
if __name__ == "__main__":
    # Test with plotting only (no instrument)
    SEND_TO_INSTRUMENT = False
    SHOW_PLOT = True
    
    if SEND_TO_INSTRUMENT and fg is None:
        fg = rm.open_resource('USB0::2391::9991::MY52300442::0::INSTR')
        fg.timeout = 20000
    
    # Generate the waveform
    waveform, time = generate_triangular_burst(
        triangle_duration=4e-3,      # 4 ms
        zero_duration=12.7e-3,       # 12.7 ms
        triangle_freq=3000,          # 3 kHz
        triangle_amplitude=1.0,      # ±1V at sample (after 20× amp)
        op_amp_ratio=20,
        sample_rate=3.8e6,            # 60 MS/s (your max)
        send_to_fg=SEND_TO_INSTRUMENT,
        show_plot=SHOW_PLOT,
        max_points=65534
    )
    
    # Check if generation was successful
    if len(waveform) > 0:
        print("\n✓ Waveform ready!")
    else:
        print("\n✗ Waveform generation failed")
    
    if SEND_TO_INSTRUMENT:
        close()