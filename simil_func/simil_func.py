"""
Functions to fit similarity spectra to data
"""

import numpy as np
import scipy
import matplotlib
import obspy as obs


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
        (for LST and FST)
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (2)), 'orig' spectrum (as in eq. (1)), default: 'dB'
    model: [string] specifies the model used: 'LSTFST' (default, calculates combined LST&FST spectrum), 'LST', 'FST'
        (calculates spectrum for 3rd order polynomial)
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


def misfit(m, f, d, model='LSTFST', **kwargs):
    '''
    Function to calculate misfit (std) between data and synthetic similarity spectrum
    INPUT:
    m: [array shape (4,1)] model vector with m[0]=ln(A*C**(-2)), m[1]=ln(B*C**(-2))), m[2]=fL, m[3]=fF with C=ln(r/Dj)
        (for LST and FST)
    f: [array like] frequency
    d: [array like, same shape as f] data vector
    model: [string] 'LSTFST' (default), 'LST', 'FST'
    **kwargs: keyword arguments for simil_func
    OUTPUT:
    M: [array like, equal length to f and d] std between data and similarity spectrum
    '''
    S = simil_func(m, f, model=model, **kwargs)
    S = np.reshape(S, np.shape(d))
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
# M0 = np.array([np.log(400), np.log(300), 10 ** (np.linspace(np.log10(FREQMIN), np.log10(FREQMAX), 3))[1],
#                10 ** (np.linspace(np.log10(FREQMIN), np.log10(FREQMAX), 3))[1]])
B1 = np.array([1e-5, 1e-5])
B2 = np.array([100, 100])


def simil_fit(stream=None, PSD_f=None, model='LSTFST', freqmin=FREQMIN, freqmax=FREQMAX, baz=F8BAZ, m0=None, PSD_win='1min',
              peaks='variable', bounds=(B1, B2), reject_band=None, response=None):
    '''
    Tool for automated fitting of similarity spectra to a given spectrum using non-linear least squares fitting and root-mean-square error aas misfit function.
    INPUT:
    stream: [Obspy stream object] One trace per array element
    PSD_f: [tuple]
    Optional:
    model: [string] defines model for calculateion, options: 'LSTFST' (defualt), 'LST', 'FST'
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
    if np.all([m0 == None]):
        m0 = np.array([np.log(400), np.log(300), 10 ** (np.linspace(np.log10(freqmin), np.log10(freqmax), 3))[1],
               10 ** (np.linspace(np.log10(freqmin), np.log10(freqmax), 3))[1]])
    ##### Defaults: ##############
    if np.any([bounds == None]):
        bounds = (B1,B2)

    if len(bounds[0]) == 2:
        # b1, b2 = bounds
        b1 = np.append(np.array([-np.inf, -np.inf]), bounds[0])
        b2 = np.append(np.array([np.inf, np.inf]), bounds[1])

    elif len(bounds[0]) == 1:
        b1 = np.append(np.array([-np.inf, -np.inf]), np.array([bounds[0].squeeze(),bounds[0].squeeze()]))
        b2 = np.append(np.array([np.inf, np.inf]), np.array([bounds[1].squeeze(),bounds[1].squeeze()]))
            # b1, b2 = bounds

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

        ## Calculate PSD
        fpsd_o, PSD_o = scipy.signal.welch(beam, sampling_rate,
                                           nperseg=PSD_win)  # calculate PSD with Welch's method
        PSDdB_o = 10 * np.log10(abs(PSD_o) / pref ** 2)  # converting to decibel
        fpsd, PSD = myinterp(fpsd_o[1:], PSDdB_o[
                                         1:])  # interpolateing to have equal spacing in log10 frequency space (important for equal fitting of low and high frequencies)
        mnum = np.argmin(np.abs((np.log10(fpsd) - np.log10(fpsd[0])) - 0.125))
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

    m = m0.copy()
    ARGS = [f, d]
    ARGS2 = ARGS.copy()
    M_func = misfit
    a0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))




    if peaks == 'constant':
        fc = np.array([b1[-2], b1[-1]])
        m = m0[:2]
        b2 = b2[:2]
        b1 = b1[:2]
        ARGS2 = [f, d, fc]
        M_func = misfit_peak
        ARGS = [f, d, fc[0]]
        m_try = np.array([a0])



    for i in range(len(model)):

        if model[i] == 'LST':
            m_try = np.array([a0, m0[2]])
            if peaks == 'bound':
                bound_try = (b1[[0, 2]], b2[[0, 2]])
            elif peaks == 'constant':
                bound_try = (b1[0], b2[0])
            elif peaks == 'variable':
                bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        elif model[i] == 'FST':
            m_try = np.array([a0, m0[3]])
            if peaks == 'bound':
                bound_try = (b1[[1, 3]], b2[[1, 3]])
            elif peaks == 'constant':
                bound_try = (b1[1], b2[1])
            elif peaks == 'variable':
                bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        elif model[i] == 'LSTFST':
            m_try = m0.copy()
            if peaks == 'bound':
                bound_try = (b1, b2)
            elif peaks == 'constant':
                bound_try = (b1[:2], b2[:2])
            elif peaks == 'variable':
                bound_try = (np.array([-np.inf, -np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf, np.inf]))

        out_ind = np.where(np.any([m_try < bound_try[0], m_try > bound_try[1]], axis=0))[0]
        if len(out_ind) > 0:
            m_try[out_ind] = (bound_try[0][out_ind]+bound_try[1][out_ind]) / 2


        ### finally the fitting ###
        sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, bounds=bound_try, method='trf',
                                                 kwargs={'model': model[i]})
        ###########################
        norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
        sol_temp = np.zeros(4)

        if model[i] == 'LST':
            sol_temp[0] = sol_temp0.x[0]
            if peaks == 'constant':
                sol_temp[2] = fc[0]
            else:
                sol_temp[2] = sol_temp0.x[1]

        elif model[i] == 'FST':
            sol_temp[1] = sol_temp0.x[0]
            if peaks == 'constant':
                sol_temp[3] = fc[1]
            else:
                sol_temp[3] = sol_temp0.x[1]

        else:
            sol_temp[[0,1]] = sol_temp0.x[[0,1]]
            if peaks == 'constant':
                sol_temp[[2,3]] = fc
            else:
                sol_temp[[2,3]] = sol_temp0.x[[2,3]]

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

    #
    #
    #
    # else:
    #     for i in range(len(model)):
    #         if model[i] == 'poly3d':
    #             z = np.polyfit(np.log10(f.squeeze()), d.squeeze(), 3)
    #             sol_temp = np.asarray(np.poly1d(z))
    #             norm_temp = np.asarray([np.dot(misfit(sol_temp, f, d, model='poly3d'), misfit(sol_temp, f, d, model='poly3d')) ** (
    #                              1 / 2)])
    #         elif model[i] == 'LST':
    #
    #             a0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
    #             m_try = np.array([a0, m0[2]])
    #             if peaks == 'variable':
    #                 bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
    #             else:
    #                 bound_try = (b1[[0,2]], b2[[0, 2]])
    #             sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, bounds=bound_try, method='trf',
    #                                                 kwargs={'model': model[i]})
    #             norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
    #             sol_temp = np.zeros(4)
    #             sol_temp[[0,2]] = sol_temp0.x
    #
    #         elif model[i] == 'FST':
    #             b0 = np.log(10 ** ((np.mean(d) + 20 * np.log10(20e-6)) / 10))
    #             m_try = np.array([b0, m0[3]])
    #             if peaks == 'variable':
    #                 bound_try = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
    #             else:
    #                 bound_try = (b1[[1, 3]], b2[[1, 3]])
    #             sol_temp0 = scipy.optimize.least_squares(M_func, m_try, args=ARGS, bounds=bound_try, method='trf',
    #                                                      kwargs={'model': model[i]})
    #             norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
    #             sol_temp = np.zeros(4)
    #             sol_temp[[1, 3]] = sol_temp0.x
    #
    #         else:
    #             if peaks == 'variable':
    #                 bound_try = (np.array([-np.inf, -np.inf,-np.inf, -np.inf]), np.array([np.inf, np.inf,np.inf, np.inf]))
    #             else:
    #                 bound_try = (b1, b2)
    #             sol_temp0 = scipy.optimize.least_squares(M_func, m, args=ARGS, bounds=bound_try, method='trf',
    #                                                 kwargs={'model': model[i]})
    #             norm_temp = np.array([np.dot(sol_temp0.fun, sol_temp0.fun)]) ** (1 / 2)
    #             sol_temp = sol_temp0.x
    #         if i == 0:
    #             sol_all = np.array([sol_temp])
    #             norm_all = np.array([norm_temp])
    #         else:
    #             sol_all = np.concatenate((sol_all, np.array([sol_temp])), axis=0)
    #             norm_all = np.concatenate((norm_all, np.array([norm_temp])), axis=0)
    #     if stream == None:
    #         return sol_all, norm_all.squeeze()
    #     else:
    #         return beam, tvec, PSD, fpsd, sol_all, norm_all.squeeze()  # , sol_all_trf, sol_LST_trf, sol_FST_trf, ARGS, m, b1, b2


def misfit_spectrum(stream=None, PSD_f=None, FREQ_vec=None, FREQ_vec_prob=None, baz=None, peaks='bound', bounds=None, fwidth=1, fpwidth=1/2, wwidth=10 * 60,
                    overlap=0.7, model='LSTFST', response=None, **kwargs):
    '''
    :param stream: Obspy steam object of data (one trace for each array element)
    :param PSD_f: Tuple with PSD_f[0] = time, PSD_f[1] = frequency, PSD_f[2]= PSD; np.shape(PSD_f[2]) = (np.shape(PSD_f[1]), np.shape(PSD_f[0]))
    :param FREQ_vec: lower frequency bounds of first set of overlapping frequency bands (f_max = 10**fwidth * f_min)
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
                    b1 = np.array([f_p, f_p])
                    b2 = np.array([f_p, f_p])
                else:
                    b1 = bounds[0]
                    b2 = bounds[1]
                m0 = np.array([f_p, f_p])
                BOUNDS = None
            elif peaks == 'bound':
                PEAKS = 'bound'
                f_p = freqmin * 10 ** (fwidth / 2)
                f_p_min = f_p / 10 ** (fpwidth / 2)
                f_p_max = f_p * 10 ** (fpwidth / 2)
                if np.any([bounds==None], axis=0):
                    b1 = np.array([f_p_min, f_p_min])
                    b2 = np.array([f_p_max, f_p_max])
                else:
                    b1 = bounds[0]
                    b2 = bounds[1]
                m0 = np.array([0, 0, f_p, f_p])
                BOUNDS = (b1, b2)
            if np.all(PSD_f == None):
                st = stream.copy()
                st.trim(starttime=tstart, endtime=tend)
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

    return M_diff