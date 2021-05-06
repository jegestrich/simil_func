"""
Functions to fit similarity spectra to data
"""

import numpy as np
import scipy
import matplotlib
import obspy as obs


def poly3d(m, f):
    """
    function to calculate 3rd order polynomial for a frequency band
    :param m: array_like, shape (4,): model parameters for 3rd order polynomial x**3*m[0] + x**2*m[1] + x*m[2] + m[3]
    :param f: array_like, shape (M,): frequency array
    :return:
    :param F: amplitude in Hz/Pa (not dB)
    """
    Flog = np.log10(f) ** 3 * m[0] + np.log10(f) ** 2 * m[1] + np.log10(f) * m[2] + m[3]
    F = 10 ** (Flog / 10)
    return F


def GF(f, fL=None, fF=None, output='GF'):
    '''
    Helper function to calculate the parameters F and G from 
    Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (6)
    :param f: array_like, shape (M,): frequency array
    :param fL: (optional) float: peak frequency of large scale turbulence spectrum (LST), 
                required when output='F' or output='GF'
    :param fF: (optional) float: peak frequency of fine scale turbulence spectrum (FST), 
                required when output='G' or output='GF'
    :param output: (optional) string: 'G' returns only the parameter for fine scale turbulence, 
                    'F' returns parameter for large scale turbulence, 
                    'GF' returns both parameters, default: 'GF'

    :return: 
    :param G or F (if output='G' or output='F'): array_like, shape (M,)
    :param G, F (if output='GF'): array_like, (M,),(M,) 
    Note: the output is not in dB scale as in equation (5) and (6) but just G and F!
    '''
    g1 = lambda fF: f[f >= 30 * fF] / fF
    g2 = lambda fF: f[np.all([f < 30 * fF, f >= 10 * fF], axis=0)] / fF
    g3 = lambda fF: f[np.all([f < 10 * fF, f >= 1 * fF], axis=0)] / fF
    g4 = lambda fF: f[np.all([f < 1 * fF, f >= 0.15 * fF], axis=0)] / fF
    g5 = lambda fF: f[np.all([f < 0.15 * fF, f >= 0.05 * fF], axis=0)] / fF
    g6 = lambda fF: f[f < 0.05 * fF] / fF

    G1 = lambda fF: 29.77786 - (38.16739 * np.log10(g1(fF)))
    G2 = lambda fF: -11.8 - (27.2523 + 0.8091863 * (np.log10(g2(fF)) - 1) \
                             + 14.851964 * (np.log10(g2(fF)) - 1) ** 2) * (np.log10(g2(fF)) - 1)
    G2 = lambda fF: -11.8 - (27.2523 + 0.8091863 * np.log10(.1 * g2(fF)) \
                             + 14.851964 * (np.log10(.1 * g2(fF))) ** 2) * np.log10(.1 * g2(fF))
    G3 = lambda fF: -(8.1476823 + 3.6523177 * np.log10(g3(fF))) * ((np.log10(g3(fF))) ** 2)
    G4 = lambda fF: (-1.0550362 + 4.9774046 * np.log10(g4(fF))) * ((np.log10(g4(fF))) ** 2)
    G5 = lambda fF: -3.5 + (11.874876 + 2.1202444 * np.log10((20 / 3) * g5(fF)) \
                            + 7.5211814 * ((np.log10((20. / 3) * g5(fF))) ** 2)) * np.log10((20 / 3) * g5(fF))
    G6 = lambda fF: 9.9 + 14.91126 * np.log10(g6(fF))

    f1 = lambda fL: f[f >= 2.5 * fL] / fL
    f2 = lambda fL: f[np.all([f < 2.5 * fL, f >= 1 * fL], axis=0)] / fL
    f3 = lambda fL: f[np.all([f < 1 * fL, f >= 0.5 * fL], axis=0)] / fL
    f4 = lambda fL: f[f < 0.5 * fL] / fL

    F1 = lambda fL: 5.64174 - (27.7472 * np.log10(f1(fL)))
    F2 = lambda fL: (1.06617 - (45.29940 * np.log10(f2(fL))) + (21.40972 * (np.log10(f2(fL))) ** 2)) * (
        np.log10(f2(fL)))
    F3 = lambda fL: -38.19338 * ((np.log10(f3(fL))) ** 2) - 16.91175 * ((np.log10(f3(fL))) ** 3)
    F4 = lambda fL: 2.53895 + 18.4 * np.log10(f4(fL))

    if output == 'G':
        Glog = np.concatenate([G6(fF), G5(fF), G4(fF), G3(fF), G2(fF), G1(fF)])
        G = 10 ** (Glog / 10)
        return G

    if output == 'F':
        Flog = np.concatenate([F4(fL), F3(fL), F2(fL), F1(fL)])
        F = 10 ** (Flog / 10)
        return F

    if output == 'GF':
        Glog = np.concatenate([G6(fF), G5(fF), G4(fF), G3(fF), G2(fF), G1(fF)])
        Flog = np.concatenate([F4(fL), F3(fL), F2(fL), F1(fL)])
        G = 10 ** (Glog / 10)
        F = 10 ** (Flog / 10)
        return G, F


def simil_func(m, f, p=20e-6, output='dB', model='LSTFST', **kwargs):
    '''
    Function to calculate the similarity spectrum for combined contributions of fine scale turbulence (FST)
    and large scale turbulence (LST) from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from
    Supersonic Jets" equation (2)
    INPUT:
    m: [array shape (4,1)] model vector with m[0]=ln(A*C**(-2)), m[1]=ln(B*C**(-2))), m[2]=fL, m[3]=fF with C=ln(r/Dj)
        (for LST and FST), or x**3*m[0] + x**2*m[1] + x*m[2] + m[3] for poly3d
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (2)), 'orig' spectrum (as in eq. (1)), default: 'dB'
    model: [string] specifies the model used: 'LSTFST' (default, calculates combined LST&FST spectrum), 'LST', 'FST'
        or 'poly3d' (calculates spectrum for 3rd order polynomial)
    OUTPUT:
    SdB (if output='dB') [array like, equal length to f] spectrum
    S (if output='orig') [array like, equal length to f] spectrum
    '''

    if model == 'LSTFST':
        a = m[0]
        b = m[1]
        fL = m[2]
        fF = m[3]
        G, F = GF(f, fL=fL, fF=fF)
        if output == 'dB':
            SdB = 10 * np.log10(np.exp(a) * F + np.exp(b) * G) - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S = (np.exp(a) * F + np.exp(b) * G)
            return S

    if model == 'LST':
        if len(m) == 4:
            a = m[0]
            b = m[1]
            fL = m[2]
            fF = m[3]
        elif len(m) == 2:
            a = m[0]
            fL = m[1]
        else:
            print('model vector must have length of 2 or 4')
        # print(m)
        F = GF(f, fL=fL, output='F')
        if output == 'dB':
            SdB = 10 * np.log10(np.exp(a) * F) - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S = np.exp(a) * F
            return S

    if model == 'FST':
        if len(m) == 4:
            a = m[0]
            b = m[1]
            fL = m[2]
            fF = m[3]
        elif len(m) == 2:
            b = m[0]
            fF = m[1]
        G = GF(f, fF=fF, output='G')
        if output == 'dB':
            SdB = 10 * np.log10(np.exp(b) * G) - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S = np.exp(b) * G
            return S

    if model == 'poly3d':
        F = poly3d(m, f)
        if output == 'dB':
            SdB = 10 * np.log10(F)  # - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S = F
            return S

    # if model == 'poly2fun':
    #     a = m[0]
    #     m0 = m[1:]
    #     F = poly2fun(m0,  f)
    #     if output == 'dB':
    #         SdB = 10 * np.log10(10**(a) * F) - 20 * np.log10(p)
    #         return SdB
    #     elif output == 'orig':
    #         S = 10**(a) * F
    #         return S


def misfit(m, f, d, model='LSTFST', **kwargs):
    '''
    Function to calculate misfit (std) between data and synthetic similarity spectrum
    INPUT:
    m: [array shape (4,1)] model vector with m[0]=ln(A*C**(-2)), m[1]=ln(B*C**(-2))), m[2]=fL, m[3]=fF with C=ln(r/Dj)
        (for LST and FST), or x**3*m[0] + x**2*m[1] + x*m[2] + m[3] for poly3d
    f: [array like] frequency
    d: [array like, same shape as f] data vector
    model: [string] 'LSTFST' (default), 'LST', 'FST or 'poly3d'
    **kwargs: keyword arguments for simil_func
    OUTPUT:
    M: [array like, equal length to f and d] std between data and similarity spectrum
    '''
    # if input == 'total':
    S = simil_func(m, f, model=model, **kwargs)
    S = np.reshape(S, np.shape(d))
    # sig = (1.1 - scipy.signal.gaussian(len(d), std=len(d)/3))
    # sig = sig/ np.sum(sig) * len(d)
    M = ((1 / len(d)) ** (1 / 2) * np.abs(d - S)).squeeze() #/ sig
    return M


def misfit_peak(m0, f, d, fc, model='LSTFST', **kwargs):
    '''
    Function to calculate misfit (std) between data and synthetic similarity spectrum by keeping the peak
    frequency constant (only works for LSTFST, LST and FST)
    INPUT:
    m: [array_like, (2,)] model vector with m[0]=ln(A*C**(-2)), m[1]=ln(B*C**(-2)))
    f: [array like] frequency
    d: [array like, same shape as f] data vector
    fc: [array_like, (2,)] constant model vector with fc[0]=fL and fc[1]=fF
    model: [string] 'LSTFST' (default), 'LST' or 'FST'
    **kwargs: keyword arguments for simil_func    OUTPUT:
    M: [array like, equal length to f and d] std between data and similarity spectrum
    '''
    m = np.append(m0, fc)
    # print(m0)
    # print(fc)
    # print(m)
    S = simil_func(m, f, model=model, **kwargs)

    S = np.reshape(S, np.shape(d))
    M = ((1 / len(d)) ** (1 / 2) * (d - S)).squeeze()
    return M


def myinterp(x, y, der=0, s=0):
    '''
    Function to interpolate linearly spaced points to points speced equally in log10 space
    INPUT:
    x: [array like] axis to be scaled in log10
    y: [array like, length as x] data points for each x to be interpolated for new x positions
    der: order of derivative for spline calculation in `scipy.interpolate.splev`, default: der=0
    s: smoothing condition for `scipy.interpolate.splrep`, default: s=0
    OUTPUT:
    xnew: [array like, length as x] axis scales linearly in log10 space
    ynew: [array like, length as x] data points interpolated for xnew positions
    '''
    tck = scipy.interpolate.splrep(x, y, s=s)
    xnew = 10 ** (np.arange(np.log10(x[0]), np.log10(x[-1]), np.log10(x[-1] / x[0]) / x.size))
    ynew = scipy.interpolate.splev(xnew, tck, der=der)
    return xnew, ynew





FREQMAX = 10
FREQMIN = 0.3
F8BAZ = 121
M0 = np.array([np.log(400), np.log(300), 10 ** (np.linspace(np.log10(FREQMIN), np.log10(FREQMAX), 3))[1],
               10 ** (np.linspace(np.log10(FREQMIN), np.log10(FREQMAX), 3))[1]])
B1 = np.array([-np.inf, -np.inf, 1e-5, 1e-5])
B2 = np.array([np.inf, np.inf, 20, 30])


def simil_fit(stream=None, PSD_f=None, model='LSTFST', freqmin=FREQMIN, freqmax=FREQMAX, baz=F8BAZ, m0=None, PSD_win='1min',
              peaks='variable', bounds=(B1, B2), reject_band=None, response=None):
    '''
    Tool for automated fitting of similarity spectra to a given spectrum using non-linear least squares fitting and root-mean-square error aas misfit function.
    INPUT:
    st: [Obspy stream object] Stream with at least 4 traces (array elements)
    Optional:
    model: [string] defines model for calculateion, options: 'LSTFST' (defualt), 'LST', 'FST' and  'poly3d'
    freqmin: [float] lower bound of frequency range for fitting the similarity spectrum (waveform will not be filtered)
        , default: FREQMIN (defined above)
    freqmax: [float] upper bound of frequency range for fitting the similarity spectrum (waveform will not be filtered)
        , default: FREQMAX (defined above)
    baz: [float] backazimuth, used for beamforming, default: F8BAZ=121 (Kilauea fissure 8)
    m0: [array, len(m0)=5] initial model parameters with m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF
        , default: M0 (defined above)
    PSD_win: [float] number of points per psd window (they overlap by 50% and will be averaged)
        , (equals nperseg for scipy.signal.welch), default: 1 minute windows (60 * sampling rate)
    peaks: [string] option to bound the peak frequency ('bound') when bounds are given
        , option to set peaks at 'constant' value and use b1[-2] for fL and b1[-1] for fF, default='variable'
    bounds: [tuple of arrays ((4,),(4,)) bounds for model parameters with bounds=(b1,b2) and b1 being the lower bound
        for all 4 model parameters and b2 the upper bound
    '''
    from array_processing.tools import beamForm
    from array_processing.algorithms.helpers import getrij
    import pandas as pd
    if type(model) == str:
        model = [model]
    for i in range(len(model)):
        if model[i] not in ['LSTFST', 'LST', 'FST']:
            print(model[i], ' is not a valid model')
    ##### Defaults: ##############
    if peaks!='variable':
        b1, b2 = bounds
    if np.all(m0 == None):
        m0 = np.array([np.log(400), np.log(300), 10 ** (np.linspace(np.log10(freqmin), np.log10(freqmax), 3))[1],
                       10 ** (np.linspace(np.log10(freqmin), np.log10(freqmax), 3))[1]])
    pref = 20e-6
    ############################
    if np.all([PSD_f==None, stream==None]):
        print('Give either steam or PSD')
    elif PSD_f==None: #calculate PSD
        st = stream
        if PSD_win == '1min':
            if type(stream) == obs.core.trace.Trace:
                sampling_rate = st.stats.sampling_rate
            else:
                sampling_rate = st[0].stats.sampling_rate
            PSD_win = int(60 * sampling_rate)
        if type(stream) == obs.core.trace.Trace:
            tvec = matplotlib.dates.date2num(st.stats.starttime.datetime) + st.times() / 86400  # datenum time vector
        else:
            tvec = matplotlib.dates.date2num(st[0].stats.starttime.datetime) + st[0].times() / 86400  # datenum time vector

        stf = st.copy()  ## filter for sound pressure level
        stf.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)
        stf.taper(max_percentage=.01)

        if type(stream) == obs.core.trace.Trace:
            beam = st.data
            beamf = stf.data
        else:
            latlist = []
            lonlist = []
            [latlist.append(st[i].stats.latitude) for i in range(len(st))]
            [lonlist.append(st[i].stats.longitude) for i in range(len(st))]
            rij = getrij(latlist, lonlist)

            data = np.zeros((len(st[0].data), len(st)))
            for i in range(len(st)):
                data[:, i] = st[i].data
            beam = beamForm(data, rij, sampling_rate, baz, M=len(tvec))  # unfiltered beamformed data

        #     dataf = np.zeros((len(stf[0].data), len(stf)))
        #     for i in range(len(stf)):
        #         dataf[:, i] = stf[i].data
        #     beamf = beamForm(dataf, rij, stf[0].stats.sampling_rate, baz,
        #                      M=len(tvec))  # filtered beamformed data (used for SPL)
        #
        # p_rms = np.sqrt(np.nanmean(beamf[beamf != 0] ** 2))
        # SPL = 10 * np.log10(p_rms ** 2 / pref ** 2)  # sound pressure level

        ## Calculate PSD
        fpsd_o, PSD_o = scipy.signal.welch(beam, sampling_rate,
                                           nperseg=PSD_win)  # calculate PSD with Welch's method
        PSDdB_o = 10 * np.log10(abs(PSD_o) / pref ** 2)  # converting to decibel
        fpsd, PSD = myinterp(fpsd_o[1:], PSDdB_o[
                                         1:])  # interpolateing to have equal spacing in log10 frequency space (important for equal fitting of low and high frequencies)
        mnum = np.argmin(np.abs((np.log10(fpsd) - np.log10(fpsd[0])) - 0.125))
        # print('mnum= ' + str(mnum))
        # print(np.shape(PSD))
        # return PSD
        PSD[mnum:-mnum] = pd.DataFrame(PSD).rolling(mnum, axis=0, center=True).median().to_numpy()[mnum:-mnum].squeeze()


    elif stream == None:
        fpsd = PSD_f[0]
        PSD = PSD_f[1]

    if np.any(response):
        res_f_temp = response[0]
        res_v_temp = response[1]
        res_const = np.min(1/res_v_temp)
        tck = scipy.interpolate.splrep(res_f_temp, res_v_temp, s=0)
        res_v = scipy.interpolate.splev(fpsd, tck, der=0)
        res_v = 1.0 / res_v ** 2
        PSD = 10*np.log10(10**(PSD/10) * res_v)
        if PSD_f == None:
            beam = beam * res_const
    # choose frequencies and PSD between frequency bounds for fitting
    f = fpsd[np.all([fpsd > freqmin, fpsd < freqmax], axis=0)]
    d = np.reshape(PSD[np.all([fpsd > freqmin, fpsd < freqmax], axis=0)], (len(f), 1))

    if np.any([reject_band is not None]):
        f_keep = np.any([f < reject_band[0], f > reject_band[1]], axis=0)
        f = f[f_keep]
        d = d[f_keep]

    m = m0
    ARGS = [f, d]
    M_func = misfit

    if peaks == 'constant':
        fc = np.array([b1[-2], b1[-1]])
        m = m0[:2]
        b2 = b2[:2]
        b1 = b1[:2]
        ARGS = [f, d, fc]
        M_func = misfit_peak
        for i in range(len(model)):
            if model[i] == 'poly3d':
                z = np.polyfit(np.log10(f.squeeze()), d.squeeze(), 3)
                sol_temp = np.asarray(np.poly1d(z))
                norm_temp = np.asarray(
                    [np.dot(misfit(sol_temp, f, d, model='poly3d'), misfit(sol_temp, f, d, model='poly3d')) ** (
                            1 / 2)])
                # print(norm_temp)
            elif model[i] == 'LST':
                ARGS = [f, d, fc[0]]
                a0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
                m_try = np.array([a0])
                sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, method='trf',
                                                         kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = np.zeros(4)
                sol_temp[0] = sol_temp0.x
                sol_temp[2] = fc[0]

            elif model[i] == 'FST':
                ARGS = [f, d, fc[1]]
                b0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
                m_try = np.array([b0])
                sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, method='trf',
                                                         kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = np.zeros(4)
                sol_temp[1] = sol_temp0.x
                sol_temp[3] = fc[1]

            else:
                ARGS = [f, d,fc]
                sol_temp0 = scipy.optimize.least_squares(M_func, m, args=ARGS, method='trf',
                                                         kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = np.zeros(4)
                sol_temp[[0,1]] = sol_temp0.x
                sol_temp[[2,3]] = fc
                # print(np.shape(norm_temp))
            if i == 0:
                sol_all = np.array([sol_temp])
                norm_all = np.array([norm_temp])
            else:
                sol_all = np.concatenate((sol_all, np.array([sol_temp])), axis=0)
                norm_all = np.concatenate((norm_all, np.array([norm_temp])), axis=0)
        if stream == None:
            return sol_all, norm_all.squeeze()
        else:
            return beam, tvec, PSD, fpsd, sol_all, norm_all.squeeze()
    # if model == 'poly3d':
    #     z = np.polyfit(np.log10(f.squeeze()), d.squeeze(), 3)
    #     sol_all = np.asarray(np.poly1d(z))
    #     norm_m = np.transpose(
    #         [np.array([np.dot(misfit(sol_all, f, d, model='poly3d'), misfit(sol_all, f, d, model='poly3d'))])]) ** (
    #                          1 / 2)
    #     return beam, beamf, tvec, SPL, PSD, fpsd, sol_all, norm_m.squeeze()

    else:
        for i in range(len(model)):
            if model[i] == 'poly3d':
                z = np.polyfit(np.log10(f.squeeze()), d.squeeze(), 3)
                sol_temp = np.asarray(np.poly1d(z))
                norm_temp = np.asarray([np.dot(misfit(sol_temp, f, d, model='poly3d'), misfit(sol_temp, f, d, model='poly3d')) ** (
                                 1 / 2)])
                # print(norm_temp)
            elif model[i] == 'LST':

                # print(m)yY8U93z01yyh
                # print(m0[[0,2]])
                a0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
                m_try = np.array([a0, m0[2]])
                if peaks == 'variable':
                    bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
                else:
                    bound_try = (b1[[0,2]], b2[[0, 2]])
                sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, bounds=bound_try, method='trf',
                                                    kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = np.zeros(4)
                sol_temp[[0,2]] = sol_temp0.x

            elif model[i] == 'FST':
                b0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
                m_try = np.array([b0, m0[3]])
                if peaks == 'variable':
                    bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
                else:
                    bound_try = (b1[[1, 3]], b2[[1, 3]])
                sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, bounds=bound_try, method='trf',
                                                         kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = np.zeros(4)
                sol_temp[[1, 3]] = sol_temp0.x

            else:
                if peaks == 'variable':
                    bound_try = (np.array([-np.inf, -np.inf,-np.inf, -np.inf]), np.array([np.inf, np.inf,np.inf, np.inf]))
                else:
                    bound_try = (b1, b2)
                sol_temp0 = scipy.optimize.least_squares(M_func, m, args=ARGS, bounds=bound_try, method='trf',
                                                    kwargs={'model': model[i]})
                norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
                sol_temp = sol_temp0.x
                # print(np.shape(norm_temp))
            if i == 0:
                sol_all = np.array([sol_temp])
                norm_all = np.array([norm_temp])
            else:
                sol_all = np.concatenate((sol_all, np.array([sol_temp])), axis=0)
                norm_all = np.concatenate((norm_all, np.array([norm_temp])), axis=0)
        if stream == None:
            return sol_all, norm_all.squeeze()
        else:
            return beam, tvec, PSD, fpsd, sol_all, norm_all.squeeze()  # , sol_all_trf, sol_LST_trf, sol_FST_trf, ARGS, m, b1, b2


def misfit_spectrum(stream=None, PSD_f=None, FREQ_vec=None, FREQ_vec_prob=None, baz=None, peaks='bound', bounds=None, fwidth=1, fpwidth=1/2, wwidth=10 * 60,
                    overlap=0.7, model='LSTFST', response=None, **kwargs):
    '''
    :param st_day: Steam object of data (one trace for each array element)
    :param FREQ_vec: min frequency of first set of overlapping frequency bands (f_max = 10**fwidth * f_min)
    :param FREQ_vec_prob: second set of not overlapping frequency bands
    :param baz: (optional) backazimuth for array processing
    :param peaks: ['variable','bound','constant'] 'variable': no restrictions for peak frequency;
                    'bound': peak frequency is bound in a frequency band half the width of first frequency bands centered
                    'constant': peak frequency is set to the center of the frequency band (in log space)
    :param fwidth: defines width of overlqpping frequency bands (f_max = 10**fwidth * f_min)
    :param wwidth: PSD window width in  s
    :return:
    beam_all: beamformed pressure timeseries
    tvec_all: times for beamformed pressure timeseries
    P_mat: PSD matrix for frequencies `fpsd` and times `tmid`
    fpsd: frequencies for P_mat
    norm_trf: misfit values for each time tmid, each frequency band FREQ_vec and each simlarity spectrum
    tmid: times
    M: average misfit for each time tmid and each frequency in FREQ_vec_prob using combined LST&FST similarity spectrum
    M_LST: average misfit for each time tmid and each frequency in FREQ_vec_prob using LST similarity spectrum
    M_FST: average misfit for each time tmid and each frequency in FREQ_vec_prob using FST similarity spectrum
    sol_trf: solution vectors for each fitting
    '''
    import obspy as obs
    import matplotlib.dates as dates
    perc = 10
    if type(model) == str:
        model = [model]
    # METHOD = method
    if np.all(PSD_f == None):
        if type(stream) == obs.core.trace.Trace:
            tstart = stream.stats.starttime
            tend_abs = stream.stats.endtime
        else:
            tstart = stream[0].stats.starttime
            tend_abs = stream[0].stats.endtime
        tend = tstart + wwidth
    elif np.all(stream==None):
        tmid = PSD_f[0]
        fpsd = PSD_f[1]
        PSD = PSD_f[2]
        tstart = obs.UTCDateTime(dates.num2date(tmid[0])) - wwidth / 2
        tend_abs = obs.UTCDateTime(dates.num2date(tmid[-1])) + wwidth / 2
        tend = tstart + wwidth

    n = 0
    timen = (tend_abs - tstart) / (wwidth * (1 - overlap))

    while tend <= tend_abs:
        # if tmid[n] != matplotlib.dates.date2num((tstart + wwidth / 2).datetime):
        #     print('wrong time!: ', obs.UTCDateTime(matplotlib.dates.num2date(tmid[n])), (tstart + wwidth / 2).datetime)

        for iif in range(len(FREQ_vec)):
            freqmin = FREQ_vec[iif]
            freqmax = freqmin * 10 ** fwidth
            m0 = np.array([np.log(400), np.log(300), freqmin * 10 ** (fwidth / 2), freqmin * 10 ** (fwidth / 2)])
            if peaks == 'variable':
                PEAKS = 'variable'
                BOUNDS = None
            elif peaks == 'constant':
                PEAKS = 'constant'
                f_p = freqmin * 10 ** (fwidth / 2)
                if np.any([bounds==None], axis=0):
                    b1 = np.array([-np.inf, -np.inf, f_p, f_p])
                    b2 = np.array([np.inf, np.inf, f_p, f_p])
                else:
                    b1 = bounds[0]
                    b2 = bounds[1]
                m0 = np.array([0, 0, f_p, f_p])
                BOUNDS = (b1, b2)
            elif peaks == 'bound':
                PEAKS = 'bound'
                f_p = freqmin * 10 ** (fwidth / 2)
                f_p_min = f_p / 10 ** (fpwidth / 2)
                f_p_max = f_p * 10 ** (fpwidth / 2)
                if np.any([bounds==None], axis=0):
                    b1 = np.array([-np.inf,  -np.inf, f_p_min, f_p_min])
                    b2 = np.array([ np.inf, np.inf, f_p_max, f_p_max])
                else:
                    b1 = bounds[0]
                    b2 = bounds[1]
                m0 = np.array([0, 0, f_p, f_p])
                if f_p < b1[3]:
                    m0[2] = b1[2]
                    m0[3] = b1[3]
                if f_p > b2[3]:
                    m0[2] = b2[2]
                    m0[3] = b2[3]
                BOUNDS = (b1, b2)

            # print(freqmin, freqmax)
            # print(BOUNDS)
            if np.all(PSD_f == None):
                st = stream.copy()
                st.trim(starttime=tstart, endtime=tend)
                # print(m0)
                beam, tvec, PSD, fpsd, sol_m_temp, norm_m_temp = simil_fit(stream=st, response=response, freqmin=freqmin,
                                                                                   freqmax=freqmax, m0=m0, baz=baz,
                                                                                   peaks=PEAKS, model=model,
                                                                                   bounds=BOUNDS, **kwargs)
            elif np.all(stream) == None:
                sol_m_temp, norm_m_temp = simil_fit(PSD_f=[fpsd,PSD[:,n]],response=response, freqmin=freqmin,
                          freqmax=freqmax, m0=m0, baz=baz,
                          peaks=PEAKS, model=model,
                          bounds=BOUNDS, **kwargs)

            if iif == 0:
                norm_m_f = np.array([norm_m_temp]).T
                sol_m_f = np.array([sol_m_temp])
            else:
                norm_m_f = np.concatenate((norm_m_f, np.array([norm_m_temp]).T), axis=1)
                sol_m_f = np.concatenate((sol_m_f, np.array([sol_m_temp])), axis=0)

        sol_m_f = np.moveaxis(sol_m_f, 0, 1)
        if n == 0:
            sol_m = np.array([sol_m_f])
            norm_m = np.array([norm_m_f.T]).T
            if np.all(PSD_f == None):
                beam_all = beam
                tvec_all = tvec
                P_mat = np.array([PSD]).T
                tmid = matplotlib.dates.date2num((tstart + wwidth / 2).datetime)


        else:
            sol_m = np.concatenate((sol_m, np.array([sol_m_f])), axis=0)
            norm_m = np.concatenate((norm_m, np.array([norm_m_f.T]).T), axis=2)
            if np.all(PSD_f == None):
                beam_all = np.append(beam_all, [beam[tvec > tvec_all[-1]]])
                tvec_all = np.append(tvec_all, [tvec[tvec > tvec_all[-1]]])
                P_mat = np.concatenate((P_mat, np.array([PSD]).T), axis=1)
                tmid = np.append(tmid, matplotlib.dates.date2num((tstart + wwidth / 2).datetime))


        tstart = tstart + wwidth * (1 - overlap)
        tend = tstart + wwidth
        n = n + 1

        if np.around(n / timen * 100) > perc:
            print('Progess:' + str(perc) + '%')
            perc = perc + 10
    sol_m = np.moveaxis(sol_m, 0, -2)
    print(np.shape(norm_m))
    print(tend)
    print(tend_abs)
    FREQ_prob_d = np.diff(np.log10(FREQ_vec_prob))[0]
    FREQ_vec_max = FREQ_vec * 10 ** fwidth
    nf = len(FREQ_vec_prob)

    M = np.zeros((len(model), nf - 1, len(tmid)))
    nw = 100
    r = 0
    weights = scipy.signal.gaussian(nw, std=10)
    weights = weights / np.sum(weights) * nw
    for iit in range(len(tmid)):
        for i in range(nf - 1):
            mask = np.arange(0, len(FREQ_vec))[
                np.all([FREQ_vec_prob[i + 1] - FREQ_vec > 0, FREQ_vec_max - FREQ_vec_prob[i] > 0], axis=0)]
            l = 0
            for s in range(len(mask)):
                j = mask[s]
                if FREQ_vec_max[j] < FREQ_vec_prob[i + 1]:  # when frequency band ends before
                    overlap2 = (np.log10(FREQ_vec_max[j]) - np.log10(FREQ_vec_prob[i])) / FREQ_prob_d
                if FREQ_vec[j] > FREQ_vec_prob[i]:  # when frequency band starts after
                    overlap2 = (np.log10(FREQ_vec_prob[i + 1]) - np.log10(FREQ_vec[j])) / FREQ_prob_d
                else:
                    overlap2 = 1

                ind1 = np.argmin(
                    np.abs(FREQ_vec_prob[i] - 10 ** np.linspace(np.log10(FREQ_vec[j]), np.log10(FREQ_vec_max[j]), 100)))
                ind2 = np.argmin(np.abs(
                    FREQ_vec_prob[i + 1] - 10 ** np.linspace(np.log10(FREQ_vec[j]), np.log10(FREQ_vec_max[j]), 100)))
                if ind2 == 0:
                    ind2 = 1
                if ind1 == nw - 1:
                    ind1 = nw - 2
                if ind1 == ind2:
                    ind2 = ind2 + 1
                weight_temp = np.nanmean(weights[ind1:ind2])
                for im in range(len(model)):
                    #
                    # norm_w = weights * norm_m[im, j, iit]
                    # print(np.shape(M))
                    # print(np.shape(norm_m))
                    M[im, i, iit] = M[im, i, iit] + (overlap2 * weight_temp * norm_m[im, j, iit])
                l = l + overlap2 * weight_temp

            M[:, i, iit] = M[:, i, iit] / l
    print('Calculations are done.')

    if np.all(PSD_f == None):
        return beam_all, tvec_all, P_mat, fpsd, norm_m, tmid, M, sol_m
    elif np.all(stream==None):
        return norm_m, M, sol_m

def misfit_diff(M, threshold):
    '''
    
    :param M: The first dimension is 2
    :return: 
    '''
    M_diff = M[1, :, :] - M[0, :, :]
    M_diff = np.zeros(np.shape(M[1, :, :]))
    M_diff[:] = np.NaN
    M_diff[np.any([M[1, :, :] < threshold, M[0, :, :] < threshold], axis=0)] = M[1, :, :][np.any(
        [M[1, :, :] < threshold, M[0, :, :] < threshold], axis=0)] - M[0, :, :][np.any(
        [M[1, :, :] < threshold, M[0, :, :] < threshold], axis=0)]
    # M_all = np.concatenate((M, np.array([M_diff])), axis=0)  # [M[0,:,:],M_LST,M_FST,M_diff
    return M_diff

COLOR_SEQUENCE = np.array(['green', 'red', 'blue', 'magenta', 'cyan', 'orange'])


def simil_plot(beam, tvec, PSD, fpsd, tmid, norm_M, sol_lm=None, freqmin=FREQMIN, freqmax=FREQMAX,
               colorstring=np.array(['green', 'red', 'blue', 'magenta', 'cyan', 'orange']),
               labelstring=np.array(['LST&FST', 'LST', 'FST', 'polynomial']), powerlim='default'):
    import matplotlib.pyplot as plt
    import numpy.matlib
    '''
    INPUT
    beam: [array (n,)] beamformed waveform, data point for every time in tvec
    tvec: [array (n,)] time vector for beam
    SPL: [array (m,)] sound pressure level, data point for every time in tmid   
    PSD: [matrix (nf,m)] array of PSDs for times tmid
    fpsd: [array (nf,)] array of frequencies for PSD
    tmid: [array (m,)] array of times for PSD and SPL
    norm_M: [array (3,m)] misfit norm calculated with simil_fit, only use *_lm or *_trf when method includes them
    sol_lm [array (5,3,m)] solution model parameters calculated with simil_fit, only use *_lm or *_trf when method includes them
    freqmin, freqmax [float] frequency min and max for fitting (defaults to 0.3-10Hz)
    
    OUTPUT:
    fig: [matplotlib figure object]
    ax: [array of matplotlib acis objects] 
   
    '''
    if len(np.shape(norm_M)) == 1:
        norm_M = np.array([norm_M])
    if len(np.shape(sol_lm)) == 2:
        sol_lm = np.array([sol_lm])

    if powerlim == 'default':
        Pmax = np.max(np.max(PSD[np.all([fpsd < FREQMAX, fpsd > FREQMIN], axis=0), :]))
        Pmin_avg = np.median(np.min(PSD[np.all([fpsd < FREQMAX, fpsd > FREQMIN], axis=0), :]))
        powerlim = np.array([Pmin_avg, Pmax])

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    if np.any([sol_lm != None]):
        gs = fig.add_gridspec(5, 2)
    else:
        gs = fig.add_gridspec(4, 2)
    ######## Spectrogram #####################
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_yscale('log')
    im = ax0.pcolormesh(tmid, fpsd, PSD, cmap='magma')
    im.set_clim([powerlim[0], powerlim[1]])
    ax0.axis('tight')
    ax0.set_xticklabels([])
    ax0.set_yticks(np.logspace(-1, 2, 4))
    ax0.set_ylim([0.1, 50])
    ax0.plot([tmid[0], tmid[-1]], [freqmin, freqmin], 'w--', linewidth=0.8)
    ax0.plot([tmid[0], tmid[-1]], [freqmax, freqmax], 'w--', linewidth=0.8)
    Pmax_ind = np.asarray([np.argmax(PSD[np.all([fpsd < freqmax, fpsd > freqmin], axis=0), :], axis=0)[i] for i in
                           range(len(tmid))]).squeeze()
    freqmin_ind = np.where(np.abs(fpsd - freqmin) == np.min(np.abs(fpsd - freqmin)))

    ax0.plot(tmid, fpsd[Pmax_ind + freqmin_ind].squeeze(), 'w.', markersize=2)
    ax0.set_ylabel('Frequency [Hz]')
    ax0.grid()
    cax = plt.axes([0.84, 0.76, 0.025, 0.14])
    hc = plt.colorbar(im, cax)  # ,cax=cax
    hc.set_label('Power [dB]')
    ######## Waveform #######################
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(tvec, beam, 'k', linewidth=0.5)
    ax1.set_ylabel('Pa')
    ######## SPL (same subplot as waveform) ###########
    # axn = ax1.twinx()
    # axn.plot(tmid, SPL, 'yellowgreen')
    # YLIMn = axn.get_ylim()
    # axn.set_xticklabels([])
    # axn.set_xlim([tmid[0], tmid[-1]])
    # axn.grid()
    # axn.set_ylabel('SPL [dB]', color='yellowgreen')
    # axn.tick_params(axis='y', colors='yellowgreen')
    ######## Peak Frequency ##########################
    if np.any([sol_lm != None]):
        axf = fig.add_subplot(gs[2, :])
        freqmin_ind = np.where(np.abs(fpsd - freqmin) == np.min(np.abs(fpsd - freqmin)))
        for i in range(3):
            Pmax_lm_ind = np.asarray(
                [np.where(simil_func(sol_lm[:, i, j], fpsd)[np.all([fpsd < freqmax, fpsd > freqmin], axis=0)] ==
                          np.max(simil_func(sol_lm[:, i, j], fpsd)[np.all([fpsd < freqmax, fpsd > freqmin], axis=0)],
                                 axis=0))[0] for j in range(len(tmid))]).squeeze()
            fmax_lm = fpsd[freqmin_ind + Pmax_lm_ind].squeeze()
            axf.plot(tmid, fmax_lm, '.-', color=colorstring[i])
        axf.plot(tmid, fpsd[Pmax_ind + freqmin_ind].squeeze(), 'k.', label='f$_{peak}$(data)')
        axf.set_ylabel('Peak Frequency')
        axf.legend(loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
        axf.set_xlim([tmid[0], tmid[-1]])
        axf.set_xticklabels([])
    ######## Misfit ##################################
    if np.any([sol_lm != None]):
        ax2 = fig.add_subplot(gs[3:, :])
    else:
        ax2 = fig.add_subplot(gs[2:, :])
    for i in range(len(norm_M)):
        ax2.plot(tmid, norm_M[i], '-', color=colorstring[i], linewidth=1, label=labelstring[i])
    YLIM2 = ax2.get_ylim()
    ax2.set_ylim(YLIM2)
    ax2.tick_params(axis='x', which='major', labelsize=10, rotation=45, left=True)
    ax2.set_ylabel('Misfit Norm')
    ax2.grid()
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
    ax2.set_xlim([tmid[0], tmid[-1]])
    ##### time ticks ####################
    duration = (matplotlib.dates.num2date(tmid[-1]) - matplotlib.dates.num2date(tmid[0]))
    dt_sec = duration.total_seconds() / 6
    if dt_sec / (60 * 60) >= 1:
        ax0.xaxis.set_major_locator(
            matplotlib.dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
        ax2.xaxis.set_major_locator(
            matplotlib.dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
        if np.any([sol_lm != None]):
            axf.xaxis.set_major_locator(
                matplotlib.dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
    elif np.abs(dt_sec / (60 * 60) - 1) < 0.5:
        ax0.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        ax2.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        if np.any([sol_lm != None]):
            axf.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
    elif dt_sec / (10 * 60) < 1:
        ax0.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        ax2.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        if np.any([sol_lm != None]):
            axf.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
    elif dt_sec / (60 * 60) < 1:
        ax0.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
        ax2.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
        if np.any([sol_lm != None]):
            axf.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d-%H:%M"))  # tick formats
    ######################################
    if np.any([sol_lm != None]):
        ax2.get_shared_x_axes().join(ax2, ax0, axf)
    else:
        ax2.get_shared_x_axes().join(ax2, ax0)
    fig.suptitle(str(matplotlib.dates.num2date(tmid[0])).replace(':', '_')[:10], y=0.95)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.autofmt_xdate()
    if np.any([sol_lm != None]):
        ax = [ax0, ax1, axf, ax2]
    else:
        ax = [ax0, ax1, ax2]
    return fig, ax


def misfit_spec_plot(PSD, fpsd, tmid, M, FREQ_vec_prob, wf='None', powerlim='default', mid_point='default',
                     extend='default', wflim='default', show_models=False, labels=['LST & FST', 'LST', 'FST', 'FST - LST']):
    '''
    Function to plot the misfit spectra produced with misfit_spectrum
    :param PSD: Spectrogram
    :param fpsd: frequency for spectrogram
    :param tmid: time for spectrogram
    :param M: Misfit spectrogram for LST&FST
    :param M_LST: Misfit spectrogram for LST
    :param M_FST: Misfit spectrogram for FST
    :param FREQ_vec_prob: frequency array for misfit spectrograms
    :param mid_point: Misfit value to center the colorbar on
    :return:
    '''
    import matplotlib
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if wflim == 'default':
        wflim = np.array([np.nanmin(wf[1]), np.nanmax(wf[1])])
    if powerlim == 'default':
        Pmax = np.max(np.max(PSD[np.all([fpsd < FREQMAX, fpsd > FREQMIN], axis=0), :]))
        Pmin_avg = np.median(np.min(PSD[np.all([fpsd < FREQMAX, fpsd > FREQMIN], axis=0), :]))
        powerlim = np.array([Pmin_avg, Pmax])
    if mid_point == 'default':
        mid_point = np.round(np.nanmean([M])) - np.round(np.nanstd([M]))
    if extend == 'default':
        extend = 2 * np.round(np.nanstd([M]))

    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('Blues', 128)
    newcolors = np.vstack((bottom(np.linspace(1, 0, 128)), top(np.linspace(0, 1, 128))))
    FSTcmp = ListedColormap(newcolors, name='GreyBlue')

    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('Reds', 128)
    newcolors = np.vstack((bottom(np.linspace(1, 0, 128)), top(np.linspace(0, 1, 128))))
    LSTcmp = ListedColormap(newcolors, name='GreyRed')

    top = cm.get_cmap('Greys', 128)
    bottom = cm.get_cmap('Greens', 128)
    newcolors = np.vstack((bottom(np.linspace(1, 0, 128)), top(np.linspace(0, 1, 128))))
    allcmp = ListedColormap(newcolors, name='GreyGreen')

    newcmp0 = cm.get_cmap('RdYlBu', 128)
    # newcmp1 = cm.get_cmap('gist_earth', 128)
    # newcmp2 = cm.get_cmap('gist_earth', 128)
    newcmp = ListedColormap(newcmp0(np.linspace(0, 1, 128)), name='GreyGreen')
    # LSTcmp = ListedColormap(newcmp1(np.linspace(1,0,128)), name='GreyRed')
    # FSTcmp = ListedColormap(newcmp2(np.linspace(1,0,128)), name='GreyBlue')
    if show_models == True:
        CMP = [allcmp, LSTcmp, FSTcmp, 'seismic']
        vmin_vec = [np.nanmax([mid_point - extend, 0]), np.nanmax([mid_point - extend, 0]),
                    np.nanmax([mid_point - extend, 0]), -extend]
        vmax_vec = [mid_point + extend, mid_point + extend, mid_point + extend, extend]
        ticks_model = [mid_point - 1 * np.nanmin([np.round(np.nanstd([M])), np.round(extend * 0.8, decimals=1)]),
                       mid_point,
                       mid_point + 1 * np.nanmin([np.round(np.nanstd([M])), np.round(extend * 0.8, decimals=1)])]
        ticks_diff = [-0.8 * extend, 0, 0.8 * extend]
        ticks_vec = [ticks_model, ticks_model, ticks_model, ticks_diff]
    else:
        CMP = ['seismic']
        vmin_vec = [-extend]
        vmax_vec = [extend]
        ticks_diff = [-0.8 * extend, 0, 0.8 * extend]
        ticks_vec = [ticks_diff]
        labels = ['FST - LST']


    if wf != 'None':
        nplots = len(M) + 2
    else:
        nplots = len(M) + 1

    fig, ax = plt.subplots(nplots, 1, figsize=(14 * 0.6, (len(M) + 1) * 1.4))
    pn = 0
    if wf != 'None':
        ax[pn].plot(wf[0], wf[1], 'k', linewidth=0.2)
        ax[pn].set_ylabel('Pressure [Pa]')
        ax[pn].set_ylim(wflim)
        ax[pn].set_xlim([wf[0][0], wf[0][-1]])
        ax[pn].set_xticklabels([])
        pn = 1
    # Plot Spectrogram
    ax[pn].set_yscale('log')
    im = ax[pn].pcolormesh(tmid, fpsd, PSD, cmap='magma', vmin=powerlim[0], vmax=powerlim[1])
    ax[pn].axis('tight')
    ax[pn].set_xticklabels([])
    # ax[0].set_ylabel('Frequency [Hz]')
    # ax[0].grid()
    # cax = plt.axes([0.92, 0.72, 0.02, 0.16])
    hc = plt.colorbar(im, ax=ax[pn], aspect=6)  # ,cax=cax
    hc.set_label('Power [dB]')
    ax[pn].set_xlim([tmid[0], tmid[-1]])
    # Plot Misfit Spectrogram for LST&FST
    for i in range(len(M)):
        x, y = np.meshgrid(tmid, FREQ_vec_prob)
        # current_cmap = matplotlib.cm.get_cmap()
        plt.get_cmap(CMP[i]).set_bad(color='grey')
        im = ax[pn + i + 1].pcolormesh(x, y, M[i], cmap=CMP[i], vmin=vmin_vec[i], vmax=vmax_vec[i])
        # current_cmap = matplotlib.cm.get_cmap()
        # current_cmap.set_bad(color='k')
        ax[pn + i + 1].set_yscale('log')
        ax[pn + i + 1].axis('tight')
        # ax[i+1].set_ylabel('Frequency [Hz]')
        ax[pn + i + 1].set_xticklabels([])
        # fig.colorbar(pcm, ax=ax)
        # cax = plt.axes([0.92, 1 - (i*0.8+1)/len(M), 0.02, 0.16])
        hc = plt.colorbar(im, ax=ax[pn + i + 1], aspect=6, ticks=ticks_vec[i])  # ,cax=cax
        hc.set_label(labels[i])
        ax[pn + i + 1].set_ylim([FREQ_vec_prob[0], FREQ_vec_prob[-1]])
        ax[pn + i + 1].set_xlim([tmid[0], tmid[-1]])
        # ax[i+1].grid()
    fig.text(0.07, (1-1/nplots)/2, 'Frequency [Hz]', va='center', rotation='vertical')
    if show_models == True:
        fig.text(.88, (1-2/nplots)/2, 'Misfit', va='center', rotation='vertical')
    if wf != 'default':
        pos = ax[1].get_position()
        pos2 = ax[0].get_position()
        ax[0].set_position([pos.x0, pos2.y0, pos.width, pos2.height])
    # Time Ticks
    duration = (matplotlib.dates.num2date(tmid[-1]) - matplotlib.dates.num2date(tmid[0]))
    dt_sec = duration.total_seconds() / 6
    for i in range(nplots):
        if dt_sec / (60 * 60) >= 1:
            ax[i].xaxis.set_major_locator(
                matplotlib.dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
        elif np.abs(dt_sec / (60 * 60) - 1) < 0.5:
            ax[i].xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        elif dt_sec / (5 * 60) < 1:
            ax[i].xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))  # tick location
        elif dt_sec / (10 * 60) < 1:
            ax[i].xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        elif dt_sec / (60 * 60) < 1:
            ax[i].xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
    ax[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d-%H:%M"))  # tick formats
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(10)
    return fig, ax


def welch_man(waveform, sampling_rate, pref=20e-6):
    fs = sampling_rate
    window = 'hann'
    win, nperseg = scipy.signal.spectral._triage_segments(window, len(waveform), input_length=waveform.shape[-1])
    scale = 1.0 / (fs * (win * win).sum())
    S = scipy.fft(win * waveform, n=len(waveform))  # [0]
    S = np.conjugate(S) * S  # magnitude of complex number (same as abs()**2)
    S *= scale
    S[..., 1:] *= 2
    S = S[:int(len(S) / 2)]
    f = np.linspace(0, fs / 2, len(S))

    # PSD_W_man_dB = 10 * np.log10(PSD_W_man[1:len(f)] / pref**2)
    return f, S


def welch_wave(spectrum, frequency, duration, sampling_rate, phase_distribution='uniform'):
    tck = scipy.interpolate.splrep(frequency, spectrum, s=0)
    fnew = np.arange(1 / duration, sampling_rate / 2 + 1 / duration, 1 / duration)
    Snew = scipy.interpolate.splev(fnew, tck, der=0)
    Snew = np.append(0, Snew)
    Snew = np.abs(Snew)

    window = 'hann'
    win, nperseg = scipy.signal.spectral._triage_segments(window, len(Snew) * 2 - 2,
                                                          input_length=Snew.shape[-1] * 2 - 2)
    scale = 1.0 / (fnew[-1] * 2 * (win * win).sum())

    Snew[..., 1:] /= 2
    Snew /= scale
    Snew = Snew ** (1 / 2)

    if phase_distribution == 'uniform':
        phases = np.random.uniform(low=0, high=2 * np.pi, size=len(Snew) - 2) * 1j
    phase_dist = np.hstack([[0], phases, [0], np.conj(phases[:: -1])])
    T = np.real(np.fft.ifft(np.hstack([Snew, Snew[-2: 0: -1]]) * phase_dist))
    t = np.linspace(0, duration, len(T))

    T = np.append([0], T[1:])
    return t, T


def simil_wave(spectrum, frequency, duration, sampling_rate, phase_distribution='uniform'):
    '''
    :param spectrum: not in dB
    :param frequency: corresponding to spectrum
    :param duration: target duration of waveform
    :param sampling_rate: target sampling rate of waveform
    :return: t: time arrray
            T: waveform array
    '''
    # interpolate for specific duration and sampling rate for waveform
    tck = scipy.interpolate.splrep(frequency, spectrum, s=0)
    fnew = np.arange(1 / duration, sampling_rate / 2 + 1 / duration, 1 / duration)
    Snew = scipy.interpolate.splev(fnew, tck, der=0)
    Snew = np.append(0, Snew)
    if phase_distribution == 'uniform':
        phases = np.random.uniform(low=0, high=2 * np.pi, size=len(Snew) - 2) * 1j
    phase_dist = np.hstack([[0], phases, [0], np.conj(phases[:: -1])])
    T = np.real(np.fft.ifft(np.hstack([Snew, Snew[-2: 0: -1]]) * phase_dist))
    t = np.linspace(0, duration, len(T))
    return t, T


def simil_wave_reverse(waveform, time):
    S = np.fft.fft(waveform)
    S = (S * np.conj(S)) ** (1 / 2)
    S = S[:int(len(S) / 2)]
    f = np.linspace(0, 1 / (2 * np.diff(time)[0]), len(S))
    return f, S


def poly2fun(m, f):
    # a1, a2, f_p = m
    # a1 = - 10**(a1)
    # a2 = - 10**(a2)
    # x1 = lambda f_p: np.log10(f[f < f_p] / f_p)
    # x2 = lambda f_p: np.log10(f[f >= f_p] / f_p)
    # F1 = lambda f_p: a1 * x1(f_p) ** 2
    # F2 = lambda f_p: a2 * x2(f_p) ** 2
    # Flog = np.concatenate([F1(f_p), F2(f_p)])
    # F = 10 ** (Flog / 10)

    a2, a3, fw, f_p = m
    fwidth = 10 ** (fw)
    f1_b = f_p / 10 ** fwidth
    f2_b = f_p * 10 ** fwidth
    a2 = - 10 ** (a2)
    a3 = - 10 ** (a3)

    a1 = 2 * a2 * np.log10(f1_b / f_p)
    a4 = 2 * a3 * np.log10(f2_b / f_p)
    b1 = a2 * np.log10(f1_b / f_p) ** 2 - a1 * np.log10(f1_b / f_p)

    b4 = a3 * np.log10(f2_b / f_p) ** 2 - a4 * np.log10(f2_b / f_p)

    x1 = lambda f_p: np.log10(f[f < f1_b] / f_p)
    x2 = lambda f_p: np.log10(f[np.all([f >= f1_b, f < f_p], axis=0)] / f_p)
    x3 = lambda f_p: np.log10(f[np.all([f >= f_p, f < f2_b], axis=0)] / f_p)
    x4 = lambda f_p: np.log10(f[f >= f2_b] / f_p)

    F1 = lambda f_p: a1 * x1(f_p) + b1
    F2 = lambda f_p: a2 * x2(f_p) ** 2
    F3 = lambda f_p: a3 * x3(f_p) ** 2
    F4 = lambda f_p: a4 * x4(f_p) + b4

    Flog = np.concatenate([F1(f_p), F2(f_p), F3(f_p), F4(f_p)])
    F = 10 ** (Flog / 10)

    return F
