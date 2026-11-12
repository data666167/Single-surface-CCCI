import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

def fft_procedure(time:np.array ,signal: np.array) :
    """Wrapper function for the fft of the signal and calculates the nyquist frequency 

    Parameters
    ----------
    time : np.array
        array of time values
    signal : np.array
        array signal evaluated per each time value 

    Returns
    -------
    Tuple[np.array,np.array,float,float]
        index 0: frequencies that FFT is calculated over  
        index 1: FFT over the range of frequencies  
        index 2: frequency resolution
        index 3: Nyquist frequency  
    """
    
    step_size = time[1]-time[0]
    fft_signal = np.fft.fftshift(np.fft.fft(signal)) 
    fft_freqs = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(time),step_size)) 
    
    sampling_frequency = 1/step_size
    
    nyquist_frequency = sampling_frequency/2
    frequency_resolution = 2*np.pi*(sampling_frequency/len(time))

    return(fft_freqs,fft_signal,frequency_resolution ,nyquist_frequency) 

def fourier_transforms(signal: np.array , t: np.array, delta_t:float, freq:float) -> float:
    """Performs the discrete fourier transform of a signal for a given frequency 

    Parameters
    ----------
    signal : np.array
        1D array of signal evaluated at all over range specified in the time array   
    t : np.array
        1D array of evenly distanced points over a time interval 
    delta_t : float
        Difference between 2 time points 
    freq : float
        Frequency in which fourier transform is performed for 

    Returns
    -------
    float
        Fourier transform of the signal for the given frequency  
    """
    # expression_1 = np.trapezoid(np.exp(1j * freq * t) * np.conjugate(signal), t)
    # expression_2 = np.trapezoid(np.exp(-1j * freq * t) * signal, t)
    # residual = 0.5*(expression_1 + expression_2)
    return np.trapezoid(signal * np.exp(-1j*freq*t), dx=t[1]-t[0])/t[-1]

def first_derivative_pwr_fourier_transforms(signal, t, delta_t, freq):
    F = np.trapezoid(signal * np.exp(-1j*freq*t),t)
    F_prime = -1j * np.trapezoid(t * signal * np.exp(-1j*freq*t),t)
    return 2 * np.real(F_prime * np.conj(F))  # REAL derivative

def second_derivative_pwr_fourier_transforms(signal, t, delta_t, freq):
    F = np.trapezoid(signal * np.exp(-1j*freq*t),t)
    F_prime = -1j * np.trapezoid(t * signal * np.exp(-1j*freq*t),t)
    F_dbl_prime = -np.trapezoid(t**2 * signal * np.exp(-1j*freq*t),t)
    # breakpoint()
    return 2 * (np.real(F_dbl_prime * np.conj(F)) + np.abs(F_prime)**2)  # REAL



def first_derivative_abs_FT(signal,t,freq):
    FT = np.trapezoid(np.exp(-1j*freq*t)*signal, t)
    FT_conj = np.conj(FT)
    FT_prime = -1j*np.trapezoid(t*np.exp(-1j*freq*t)*signal,t)
    FT_prime_conj = 1j*np.trapezoid(t*np.exp(1j*freq*t)*signal,t)
    FT_abs = np.abs(FT)
    
    dFT_abs= np.real(FT_prime*FT_conj)/(FT_abs)
    return dFT_abs

def second_derivative_abs_FT(signal,t,freq):
    FT = np.trapezoid(np.exp(-1j*freq*t)*signal, t)
    FT_conj = np.conj(FT)
    
    FT_prime = -1j*np.trapezoid(t*np.exp(-1j*freq*t)*signal,t)
    FT_prime_conj = 1j*np.trapezoid(t*np.exp(1j*freq*t)*signal,t)
    
    F_dbl_prime = ((-1j)**2)*np.trapezoid((t**2)*np.exp(-1j*freq*t)*signal,t)
    F_dbl_prime_conj = ((1j)**2)*np.trapezoid((t**2)*np.exp(1j*freq*t)*signal,t) 
    FT_abs = np.abs(FT)
    
    dFT2_abs = (-(FT_abs**(-3))*(np.real(FT_prime*FT_conj)**2)
                  + (FT_abs**(-1))*np.real(F_dbl_prime*FT_conj)
                  + (FT_abs**(-1))*np.real(np.abs(FT_prime)**2) )
    return dFT2_abs


def first_derivative_re_fourier_transforms(signal:np.array , t: np.array, delta_t:float , freq: float) -> float:
    """Performs the first derivative of the discrete fourier transform of a signal for a given frequency with respect to frequency 

    Parameters
    ----------
    signal : np.array
        1D array of signal evaluated at all over range specified in the time array   
    t : np.array
        1D array of evenly distanced points over a time interval 
    delta_t : float
        Difference between 2 time points 
    freq : float
        Frequency in which fourier transform is performed for 

    Returns
    -------
    float
        First derivative of the discrete fourier transform of a signal for a given frequency with respect to frequency 
    """
    expression_1 = np.trapezoid(1j * t * np.exp(1j * freq * t) * np.conjugate(signal), t)
    expression_2 = np.trapezoid(-1j * t * np.exp(-1j * freq * t) * signal, t)
    # breakpoint()
    residual = 0.5 * (expression_1 + expression_2)
    return residual    

def second_derivative_re_fourier_transforms(signal: np.array, t: np.array, delta_t:float, freq:float) -> float:
    """Performs the second derivative of the discrete fourier transform of a signal for a given frequency with respect to frequency 

    Parameters
    ----------
    signal : np.array
        1D array of signal evaluated at all over range specified in the time array   
    t : np.array
        1D array of evenly distanced points over a time interval 
    delta_t : float
        Difference between 2 time points 
    freq : float
        Frequency in which fourier transform is performed for 

    Returns
    -------
    float
        Second derivative of the discrete fourier transform of a signal for a given frequency with respect to frequency 
    """
    expression_1 = np.trapezoid(-t**2 * np.exp(1j * freq * t) * np.conjugate(signal), t)
    expression_2 = np.trapezoid(-t**2 * np.exp(-1j * freq * t) * signal, t)
    residual = 0.5 * (expression_1 + expression_2)
    return residual

# def objective_Re(signal, time, delta_t, freq):
#     F = fourier_transforms(signal, time, delta_t, freq)
#     return -np.real(F)   # minimize this → maximize Re F


# def rfo_step_1d(g, h):
#     """Return RF step candidates s_minus, s_plus for scalar g, h (S=1)."""
#     disc = np.sqrt(h*h + 4.0*g*g)
#     s_minus = (h - disc) / (2.0*g)
#     s_plus  = (h + disc) / (2.0*g)
#     return s_minus, s_plus
# q

def objective_Re(signal, time, delta_t, freq):
    # f(ω) = -Re F(ω)  (minimize this → maximize Re F)
    F = fourier_transforms(signal, time, delta_t, freq)
    return -np.real(F)

def objective_pwr(signal, time, delta_t, freq):
    # f(ω) = -Re F(ω)  (minimize this → maximize Re F)
    F = fourier_transforms(signal, time, delta_t, freq)
    return -(np.abs(F)**2)


def rfo_step_1d(g, h):
    """Return RF step candidates s_minus, s_plus for scalar g, h (S=1)."""
    disc = np.sqrt(h*h + 4.0*g*g)
    s_minus = (h - disc) / (2.0*g)
    s_plus  = (h + disc) / (2.0*g)
    return s_minus, s_plus


def optimize_Re_peak(signal, time, delta_t,
                     freq0, freq_left, freq_right,
                     max_iters=50):
    g_tol    = 1e-6     # looser so we actually declare convergence
    h_tol    = 1e-12
    step_tol = 1e-8
    Delta_TR = 0.05     # trust-region radius

    freq = float(freq0)
    history = []

    for it in range(max_iters):
        # --- 1) Evaluate f, g, h at current freq ---
        f_val = objective_Re(signal, time, delta_t, freq)

        g = -first_derivative_re_fourier_transforms(signal, time, delta_t, freq)   # f'(ω)
        h = -second_derivative_re_fourier_transforms(signal, time, delta_t, freq)  # f''(ω)

        print(f"\nIter {it}")
        print("  freq =", freq)
        print("  f    =", f_val)
        print("  g    =", g)
        print("  h    =", h)

        # Convergence: small gradient
        if np.abs(g) < g_tol:
            print("  |g| < g_tol, converged")
            break

        # Avoid near-zero curvature
        if np.abs(h) < h_tol:
            print("  |h| < h_tol, break (flat region)")
            break

        # --- 2) RF step candidates ---
        s_minus, s_plus = rfo_step_1d(g, h)
        print("  s_minus =", s_minus)
        print("  s_plus  =", s_plus)
        print("  g*s_minus =", g * s_minus)
        print("  g*s_plus  =", g * s_plus)

        # Descent candidates: g*s < 0  (for MINIMIZATION)
        candidates = [s for s in (s_minus, s_plus) if np.real(g*s) < 0]

        if not candidates:
            print("  No descent RF root, fallback to small gradient step")
            stp_raw = -np.sign(np.real(g)) * 1e-4
        else:
            s_newton = -g / h
            print("  s_newton =", s_newton)
            # choose the descent root closest to Newton step
            stp_raw = min(candidates, key=lambda s: np.abs(s - s_newton))

        print("  stp_raw =", stp_raw)

        # --- 3) Enforce trust region + bounds ---
        Delta_left  = freq - freq_left
        Delta_right = freq_right - freq

        max_left  = max(0.0, min(Delta_left,  Delta_TR))
        max_right = max(0.0, min(Delta_right, Delta_TR))

        print("  Delta_left, Delta_right =", Delta_left, Delta_right)
        print("  max_left, max_right     =", max_left, max_right)

        stp = np.clip(stp_raw, -max_left, max_right)
        freq_trial = freq + stp

        print("  stp after clip =", stp)
        print("  freq_trial     =", freq_trial)

        # --- 4) Accept/reject based on f ---
        f_trial = objective_Re(signal, time, delta_t, freq_trial)
        print("  f_old, f_trial =", f_val, f_trial)

        if f_trial < f_val:
            # Accept step
            print("  ACCEPT step")
            freq = freq_trial
            history.append(freq)
            # Optional: modestly grow trust radius
            Delta_TR = min(Delta_TR * 1.5, 0.5)
        else:
            # Reject step and shrink trust region
            print("  REJECT step, shrink Delta_TR")
            Delta_TR *= 0.5
            # freq unchanged

        # Stop if steps are tiny and we're not improving
        if np.abs(stp) < step_tol:
            print("  |stp| < step_tol, stop")
            break

    return freq, history


def optimize_pwr_peak(signal, time, delta_t,
                      freq0, freq_left, freq_right,
                      max_iters=50):
    # Relative/absolute gradient tolerances
    g_abs_tol = 1e-6
    g_rel_tol = 1e-6
    h_tol     = 1e-12
    step_tol  = 1e-8
    Delta_min = 1e-6
    Delta_max = 0.1

    freq = float(freq0)
    history = []

    # start with some generic trust radius; will be reset from Newton step
    Delta_TR = 0.01

    for it in range(max_iters):
        # --- 1) Evaluate f, g, h at current freq ---
        f_val = objective_pwr(signal, time, delta_t, freq)

        g = -first_derivative_pwr_fourier_transforms(signal, time, delta_t, freq)   # f'(ω)
        h = -second_derivative_pwr_fourier_transforms(signal, time, delta_t, freq)  # f''(ω)

        print(f"\n[POWER] Iter {it}")
        print("  freq =", freq)
        print("  f    =", f_val)
        print("  g    =", g)
        print("  h    =", h)

        # Convergence: relative/absolute gradient
        if (np.abs(g) < g_abs_tol or
            np.abs(g) < g_rel_tol * (1.0 + np.abs(f_val))):
            print("  gradient small enough, converged")
            break

        # Avoid near-zero curvature
        if np.abs(h) < h_tol:
            print("  |h| < h_tol, break (flat region)")
            break

        # --- 2) RF step candidates ---
        s_minus, s_plus = rfo_step_1d(g, h)
        print("  s_minus =", s_minus)
        print("  s_plus  =", s_plus)
        print("  g*s_minus =", g * s_minus)
        print("  g*s_plus  =", g * s_plus)

        # Descent candidates: g*s < 0  (for MINIMIZATION)
        candidates = [s for s in (s_minus, s_plus) if np.real(g*s) < 0]

        if not candidates:
            print("  No descent RF root, fallback to small gradient step")
            # if trust region already tiny, treat as converged
            if Delta_TR < 1e-5:
                print("  Tiny trust region and no descent root → converged")
                break
            stp_raw = -np.sign(np.real(g)) * 1e-4
        else:
            s_newton = -g / h
            print("  s_newton =", s_newton)
            # choose the descent root closest to Newton step
            stp_raw = min(candidates, key=lambda s: np.abs(s - s_newton))

        print("  stp_raw =", stp_raw)

        # --- 3) Adaptive trust region from Newton scale + bounds ---
        # local natural step scale
        s_scale = np.abs(-g / h)
        # base trust radius proportional to that, clipped globally
        Delta_TR = np.clip(5.0 * s_scale, Delta_min, Delta_max)

        Delta_left  = freq - freq_left
        Delta_right = freq_right - freq

        max_left  = max(0.0, min(Delta_left,  Delta_TR))
        max_right = max(0.0, min(Delta_right, Delta_TR))

        print("  Delta_left, Delta_right =", Delta_left, Delta_right)
        print("  max_left, max_right     =", max_left, max_right)

        stp = np.clip(stp_raw, -max_left, max_right)
        freq_trial = freq + stp

        print("  stp after clip =", stp)
        print("  freq_trial     =", freq_trial)

        # --- 4) Accept/reject based on *power* objective ---
        f_trial = objective_pwr(signal, time, delta_t, freq_trial)
        print("  f_old, f_trial =", f_val, f_trial)

        if f_trial < f_val:
            print("  ACCEPT step")
            freq = freq_trial
            history.append(freq)
        else:
            print("  REJECT step, shrink Delta_TR")
            Delta_TR *= 0.5

        if np.abs(stp) < step_tol:
            print("  |stp| < step_tol, stop")
            break

    return freq, history


def newtons_method(signal, time, freq_guess, freq_left, freq_right, max_iters=60):
    delta_t = time[1] - time[0]
    freq = freq_guess
    damping = 1.0
    converged = False
    prev_mag = 0
    mag_history = []
    
    # refine grid for better start
    omega = np.linspace(freq_left,freq_right,200) 
    ft = [fourier_transforms(signal,time,delta_t,i) for i in omega]    
    max_peak_idx = np.argmax(ft)
    new_freq_guess = omega[max_peak_idx]
    new_left_freq = omega[max_peak_idx-1]
    new_right_freq = omega[max_peak_idx+1]
    
    # plt.plot(omega,ft)
    # plt.vlines([new_left_freq,new_right_freq,new_freq_guess],[0,0,0],[1,1,1])
    # plt.show()
    # breakpoint()
    
    omega_new = np.linspace(new_left_freq,new_right_freq,50)
    ft_new = [fourier_transforms(signal,time,delta_t,i) for i in omega_new]
    
    freq = new_freq_guess.copy()
    freq_candidates = [new_freq_guess]
    for iterations in range(max_iters):
        # Calculate derivatives and magnitude
        f1 = first_derivative_re_fourier_transforms(signal, time, delta_t, freq)
        f2 = second_derivative_re_fourier_transforms(signal, time, delta_t, freq)

        step = -f1 / f2
        freq_candidate = freq + damping * step
        freq_candidates.append(freq_candidate)
        
        
        freq = freq_candidate
        if not(new_left_freq < freq_candidate < new_right_freq):
            breakpoint()
        
        if (np.abs(f1) < 1e-10):
            converged = True
            break
    if converged == False:
        print('converged:',converged,"\n",'f1:',f1)
        # breakpoint()    
    print(freq)
    return freq, converged

def newtons_method_pwr(signal, time, freq_guess, freq_left, freq_right, max_iters=60):
    delta_t = time[1] - time[0]
    freq = freq_guess
    damping = 1.0
    converged = False
    prev_mag = 0
    mag_history = []
    
    # refine grid for better start
    omega = np.linspace(freq_left,freq_right,200) 
    ft = [np.abs(fourier_transforms(signal,time,delta_t,i)) for i in omega]    
    max_peak_idx = np.argmax(ft)
    new_freq_guess = omega[max_peak_idx]
    new_left_freq = omega[max_peak_idx-1]
    new_right_freq = omega[max_peak_idx+1]
    
    # plt.plot(omega,ft)
    # plt.vlines([new_left_freq,new_right_freq,new_freq_guess],[0,0,0],[1,1,1])
    # plt.show()
    # breakpoint()
    
    omega_new = np.linspace(new_left_freq,new_right_freq,50)
    ft_new = [fourier_transforms(signal,time,delta_t,i) for i in omega_new]
    
    freq = new_freq_guess.copy()
    freq_candidates = [new_freq_guess]
    for iterations in range(max_iters):
        # Calculate derivatives and magnitude
        # f1 = first_derivative_pwr_fourier_transforms(signal, time, delta_t, freq)
        # f2 = second_derivative_pwr_fourier_transforms(signal, time, delta_t, freq)

        f1 = first_derivative_abs_FT(signal,time,freq)
        f2 = second_derivative_abs_FT(signal,time,freq)


        step = -f1 / f2
        freq_candidate = freq + damping * step
        freq_candidates.append(freq_candidate)
        
        
        freq = freq_candidate
        if not(new_left_freq < freq_candidate < new_right_freq):
            breakpoint()
        
        if (np.abs(f1) < 1e-10):
            converged = True
            break
    if converged == False:
        print('converged:',converged,"\n",'f1:',f1)
        # breakpoint()    
    print(freq)
    return freq, converged

def peak_detector(time:np.array,signal:np.array,peak_height_minimum=0.0005,energy_threshold_ratio=0.1):
    """Detects peaks in the fft of a signal and removes them until signal energy is bellow a threshold 

    Parameters
    ----------
    time : np.array
        equidistant time point 
    signal : np.array
        signal the same size as time array 
    peak_height_minimum : float, optional
        minimum height for a peak to be detected in the fft, by default 0.0005
    energy_threshold_ratio : float, optional
        threshold of original signal that must be made before ending peak detection algorithm , by default 0.01

    Returns
    -------
    Tuple[list[float],list[float]]
        Optimized peak frequencies and original peak frequencies 
    """
    # initialization of freqs
    peak_frequencies = []
    original_guess = []
    
    original_energy = np.sum(np.abs(signal)**2)
    current_energy = original_energy
    threshold_energy = energy_threshold_ratio * original_energy

    fft_frequency_range, fft_signal ,_ , _=fft_procedure(time,signal)
    
    # normalization of fft
    fft_signal = np.abs(fft_signal / len(signal)) 
    
    # initial signals
    fft_frequency_initial = np.copy(fft_frequency_range)
    fft_signal_initial = np.copy(np.abs(fft_signal))
    
    peak_height_maximum = None
    count = 0
    while current_energy > threshold_energy:
        
        # Create FFT spectra and frequency range
        fft_frequency_range, fft_signal ,_ , _=fft_procedure(time,signal)
        
        # Normalize FFT spectra 
        fft_signal = np.abs(fft_signal/len(signal))   
           
        # Get coarse peak information from fft data
        peaks_index = sp.signal.find_peaks(fft_signal,height=(peak_height_minimum,peak_height_maximum))[0]
    
        if not peaks_index.size > 0:
            print('No more peaks above minimum height')
            break 
        
        peaks_frequency = fft_frequency_range[peaks_index]
        peaks_heights = fft_signal[peaks_index]
        max_peak_index = np.argmax(peaks_heights)
        max_peak_frequency = peaks_frequency[max_peak_index]
        upper_bound = fft_frequency_range[peaks_index[max_peak_index]+1]
        lower_bound = fft_frequency_range[peaks_index[max_peak_index]-1]
        # breakpoint()
        # Use Newton's method to optimize the peak frequency location 
        # w = np.blackman(len(signal))
        optimized_peak_maximum_frequency,conv = newtons_method(signal, time,max_peak_frequency,lower_bound,upper_bound)

        # Determine intensity of the signal (this is the Fourier coefficient)
        optimized_intensity = fourier_transforms(signal,time,time[1],optimized_peak_maximum_frequency)
        unoptimized_intensity = fourier_transforms(signal,time,time[1],max_peak_frequency) 
        
        # Create signal corresponding to found peak frequency being removed from the original signal  
        optimized_peak_signal = optimized_intensity*np.exp(1j * optimized_peak_maximum_frequency * time)
        
        # Subtract the optimized peak from the original signal
        new_signal = signal - optimized_peak_signal
        
        # Append guess and optimized frequencies
        # negative signs introduced to match eigenvalue actual sign 
        original_guess.append(-max_peak_frequency)
        peak_frequencies.append(-optimized_peak_maximum_frequency)
        
        # Update signal 
        signal = new_signal
        current_energy = np.sum(np.abs(signal)**2)
        
        # only for plotting purposes to debug
        fft_frequency_range, fft_signal ,_ , _ = fft_procedure(time,new_signal)
        fft_signal = fft_signal/len(new_signal)
        
        print('opt',optimized_intensity)
        print('unopt',unoptimized_intensity)
        print('fft height',np.max(peaks_heights))
        print('guess freq',max_peak_frequency)
        print(conv)
        
        
        # negative signs introduced to match eigenvalue actual value
        print(count)
        # plt.plot(-fft_frequency_initial,fft_signal_initial,label="Initial spectra")
        # plt.plot(-fft_frequency_range,np.abs(fft_signal),':',label = "Final spectra")
        # plt.xlabel('Energy (eV)')
        # plt.ylabel(r'$S(E)$')
        # plt.xlim(0, 4)
        # plt.legend()
        # plt.show()
        
        count += 1
        # breakpoint()
        #print(peak_frequencies,original_guess)
           
    plt.plot(-fft_frequency_initial,fft_signal_initial,label="Initial spectra")
    plt.plot(-fft_frequency_range,np.abs(fft_signal),label = "Final spectra")
    plt.title('Final FFT after peak search and peak deletion')
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()
    print('Successfully Completed')
    
    return peak_frequencies, original_guess

