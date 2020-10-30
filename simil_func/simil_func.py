import numpy as np
import scipy
from matplotlib import dates
import obspy as obs

def poly2fun(m,f):
    a2, a3, fwidth, f_p = m
    f1_b = f_p / 10 ** np.abs(fwidth)
    f2_b = f_p * 10 ** np.abs(fwidth)
    a2 = - np.abs(a2)
    a3 = - np.abs(a3)

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

def GF(f,fL=None,fF=None, output='GF'):
    '''
    Function to calculate the parameters F and G from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (6)
    INPUT:
    f: [array like] frequency
    fL: [float] peak frequency of large scale turbulence spectrum
    fF: [float] peak frequency of fine scale turbulence spectrum
    output: [string] 'G' returns only the parameter for fine scale turbulence, 'F' returns parameter for large scale turbulence, 'GF' returns both parameters, default: 'GF'
    OUTPUT:
    G or F (if output='G' or output='F'): [array like, equal length to f] spectrum
    G, F (if output='GF'): [array like, both arrays equal length to f] spectrum
    Note: the output is not in dB scale as in equation (5) and (6) but just G and F!
    '''
    g1 = lambda fF: f[f >= 30*fF]/fF
    g2 = lambda fF: f[np.all([f < 30*fF,f >= 10*fF],axis=0)]/fF
    g3 = lambda fF: f[np.all([f < 10*fF,f >= 1*fF],axis=0)]/fF
    g4 = lambda fF: f[np.all([f < 1*fF,f >= 0.15*fF],axis=0)]/fF
    g5 = lambda fF: f[np.all([f < 0.15*fF,f >= 0.05*fF],axis=0)]/fF
    g6 = lambda fF: f[f < 0.05*fF]/fF

    G1 = lambda fF: 29.77786 - (38.16739 * np.log10(g1(fF)))
    G2 = lambda fF: -11.8 - ( 27.2523 + 0.8091863 *(np.log10(g2(fF))-1) \
        + 14.851964*(np.log10(g2(fF))-1)**2 )*(np.log10(g2(fF))-1)
    G2 = lambda fF: -11.8 - ( 27.2523 + 0.8091863 *np.log10(.1*g2(fF)) \
        + 14.851964*(np.log10(.1*g2(fF)))**2 )*np.log10(.1*g2(fF))
    G3 = lambda fF: -(8.1476823 + 3.6523177*np.log10(g3(fF)))*((np.log10(g3(fF)))**2)
    G4 = lambda fF: (-1.0550362 + 4.9774046*np.log10(g4(fF)))*((np.log10(g4(fF)))**2)
    G5 = lambda fF: -3.5 + (11.874876 + 2.1202444*np.log10( (20/3)*g5(fF) ) \
        + 7.5211814*((np.log10( (20./3)*g5(fF) ))**2) )*np.log10( (20/3)*g5(fF) )
    G6 = lambda fF: 9.9 + 14.91126*np.log10(g6(fF))

    f1 = lambda fL: f[f >= 2.5*fL]/fL
    f2 = lambda fL: f[np.all([f < 2.5*fL,f >= 1*fL],axis=0)]/fL
    f3 = lambda fL: f[np.all([f < 1*fL,f >= 0.5*fL],axis=0)]/fL
    f4 = lambda fL: f[f < 0.5*fL]/fL

    F1 = lambda fL: 5.64174 - (27.7472 * np.log10(f1(fL)))
    F2 = lambda fL: (1.06617 - (45.29940 * np.log10(f2(fL))) + (21.40972 * (np.log10(f2(fL)))**2) )*(np.log10(f2(fL)))
    F3 = lambda fL: -38.19338*((np.log10(f3(fL)))**2) - 16.91175*((np.log10(f3(fL)))**3)
    F4 = lambda fL: 2.53895 + 18.4*np.log10(f4(fL))

    if output == 'G':
        Glog = np.concatenate([G6(fF),G5(fF), G4(fF), G3(fF), G2(fF), G1(fF)])
        G = 10**(Glog/10)
        return G

    if output == 'F':
        Flog = np.concatenate([F4(fL), F3(fL), F2(fL), F1(fL)])
        F = 10**(Flog/10)
        return F

    if output == 'GF':
        Glog = np.concatenate([G6(fF),G5(fF), G4(fF), G3(fF), G2(fF), G1(fF)])
        Flog = np.concatenate([F4(fL), F3(fL), F2(fL), F1(fL)])
        G = 10**(Glog/10)
        F = 10**(Flog/10)
        return G, F

def simil_func(m, f, p=20e-6, output='dB', model='LSTFSTcombined', **kwargs):
    '''
    Function to calculate the similarity spectrum for combined contributions of fine scale turbulence (FST) and large scale turbulence (LST) from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (2)
    INPUT:
    m: [array shape (5,1)] model vector with m[0]=ln(A*C**(-2)), m[1]=ln(B*C**(-2))), m[2]=fL, m[3]=fF with C=ln(r/Dj
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (2)), 'orig' spectrum (as in eq. (1)), default: 'dB'
    OUTPUT:
    SdB (if output='dB') [array like, equal length to f] spectrum
    S (if output='orig') [array like, equal length to f] spectrum
    '''
    if model == 'LSTFSTcombined':
        a = m[0]
        b = m[1]
        fL = m[2]
        fF = m[3]
        G,F = GF(f,fL=fL,fF=fF)
        if output == 'dB':
            SdB = 10 * np.log10(np.exp(a) * F + np.exp(b) * G) - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S =  (np.exp(a)*F + np.exp(b)*G)
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

    if model == 'poly2fun':
        a = m[0]
        m0 = m[1:]
        F = poly2fun(m0,  f)
        if output == 'dB':
            SdB = 10 * np.log10(np.exp(a) * F) - 20 * np.log10(p)
            return SdB
        elif output == 'orig':
            S = np.exp(a) * F
            return S


def misfit(m, f, d, **kwargs):
    '''
    Function to calculate misfit (difference) between data and synthetic similarity spectrum
    INPUT:
    m: [array shape (5,1) or (3,1)] model vector. if shape=(5,1): m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF. if shape=(3,1): m[0]=ln(A) (or ln(B)), m[1]=ln(r/Dj), m[2]=fL (or fF)
    f: [array like] frequency
    d: [array like, same shape as f] data vector
    input: [string] 'LST' calculates misfit between data and LST spectrum (`simil_LST_func`), 'FST' calculates mifit between data and FST spectrum (`simil_FST_func`), 'total' calculates misfit between data and combined similarity spectrum (`simil_func`)
    **kwargs: [tuple and dict] keyword arguments for simil_func, simil_LST_func or simil_FST_func
    OUTPUT:
    M: [array like, equal length to f and d] absolute difference between data and similarity spectrum
    '''
    # if input == 'total':
    S = simil_func(m, f, **kwargs)
    S = np.reshape(S,np.shape(d))
    M = ((1/len(d))**(1/2) * np.abs(d - S)).squeeze()
    return M

def misfit_peak(m0, f, d, fc, model='LSTFSTcombined', **kwargs):
    '''
    Function to calculate misfit (difference) between data and synthetic similarity spectrum
    INPUT:
    m: [array shape (5,1) or (3,1)] model vector. if shape=(5,1): m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF. if shape=(3,1): m[0]=ln(A) (or ln(B)), m[1]=ln(r/Dj), m[2]=fL (or fF)
    f: [array like] frequency
    d: [array like, same shape as f] data vector
    input: [string] 'LST' calculates misfit between data and LST spectrum (`simil_LST_func`), 'FST' calculates mifit between data and FST spectrum (`simil_FST_func`), 'total' calculates misfit between data and combined similarity spectrum (`simil_func`)
    **kwargs: [tuple and dict] keyword arguments for simil_func, simil_LST_func or simil_FST_func
    OUTPUT:
    M: [array like, equal length to f and d] absolute difference between data and similarity spectrum
    '''
    m = np.append(m0,fc)
    S = simil_func(m, f, **kwargs)
    S = np.reshape(S,np.shape(d))
    M = ((1/len(d))**(1/2) * np.abs(d - S)).squeeze()
    return M


def myinterp(x, y, der = 0, s = 0):
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
    tck = scipy.interpolate.splrep(x, y, s = s)
    xnew = 10**(np.arange(np.log10(x[0]), np.log10(x[-1]), np.log10(x[-1]/x[0])/x.size))
    ynew = scipy.interpolate.splev(xnew, tck, der = der)
    return xnew, ynew


FREQMAX = 10
FREQMIN = 0.3
F8BAZ = 121
M0 = np.array([np.log(400),np.log(300),10**(np.linspace(np.log10(FREQMIN),np.log10(FREQMAX),3))[1],10**(np.linspace(np.log10(FREQMIN),np.log10(FREQMAX),3))[1]])

def simil_fit(st, freqmin=FREQMIN, freqmax=FREQMAX, baz=F8BAZ, method='lm', m0=None, model='LSTFSTcombined', PSD_win='1min', peaks='variable', fL=None, fF=None, fwidth=1):
    from array_processing.tools import beamForm
    from array_processing.algorithms.helpers import getrij
    '''
        Tool for automated fitting of similarity spectra to a given spectrum using non-linear least squares fitting and root-mean-square error aas misfit function. 
    INPUT:
    f: [array like] frequency array
    st: [Obspy stream object] Stream with at least 4 traces (array elements)
    Optional:
    freqmin: [float] lower bound of frequency range for fitting the similarity spectrum (waveform will not be filtered), default: FREQMIN (defined above)
    freqmax: [float] upper bound of frequency range for fitting the similarity spectrum (waveform will not be filtered), default: FREQMAX (defined above)
    baz: [float] backazimuth, used for beamforming, default: F8BAZ=121 (Kilauea fissure 8)
    method: [string] optimization algorithm used (see scipy.optimize.least_squares for details), options: ['lm, 'trf, 'lm&trf'], default: 'lm' (Levenberg Marquard)
    m0: [array, len(m0)=5] initial guess of model parameters with m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF, default: M0 (defined above)
    PSD_win: [float] number of points per psd window (they overlap by 50% and will be averaged), (equals nperseg for scipy.signal.welch), default: 1 minute windows (60 * sampling rate)
    '''
    ##### Defaults: ##############
    if PSD_win == '1min':
        PSD_win = int(60 * st[0].stats.sampling_rate)

    if model != 'poly2fun':
        b1 = np.array([np.log(1e-10), np.log(1e-10), 1e-5, 1e-5])
        b2 = np.array([np.log(1e10), np.log(1e10), 20, 30])
        if np.all(m0) == None:
            m0 = np.array([np.log(400),np.log(300),10**(np.linspace(np.log10(FREQMIN),np.log10(FREQMAX),3))[1],10**(np.linspace(np.log10(FREQMIN),np.log10(FREQMAX),3))[1]])

    if model == 'poly2fun':
        b1 = np.array([-1e10, -1e3, -1e3, 0.1, freqmin])
        b2 = np.array([1e10, 1e3, 1e3, 3, freqmax])
        if np.all(m0) == None:
            m0 = np.array([1, -1, -1, 1, 1])
    pref = 20e-6
    ############################

    tvec=dates.date2num(st[0].stats.starttime.datetime)+st[0].times()/86400   #datenum time vector

    stf = st.copy() ## filter for sound pressure level
    stf.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)
    stf.taper(max_percentage=.01)

    if len(st) == 1:
        beam = st[0].data
        beamf = stf[0].data
    else:
        latlist = []
        lonlist = []
        [latlist.append(st[i].stats.latitude) for i in range(len(st))]
        [lonlist.append(st[i].stats.longitude) for i in range(len(st))]
        rij=getrij(latlist,lonlist)

        data = np.zeros((len(st[0].data),len(st)))
        for i in range(len(st)):
            data[:,i] = st[i].data
        beam = beamForm(data, rij, st[0].stats.sampling_rate, baz, M=len(tvec)) #unfiltered beamformed data

        dataf = np.zeros((len(stf[0].data), len(stf)))
        for i in range(len(stf)):
            dataf[:, i] = stf[i].data
        beamf = beamForm(dataf, rij, stf[0].stats.sampling_rate, baz, M=len(tvec)) #filtered beamformed data (used for SPL)

    p_rms = np.sqrt(np.nanmean(beamf[beamf!=0]**2))
    SPL = 10 * np.log10(p_rms**2/pref**2) #sound pressure level

    ## Calculate PSD
    fpsd_o, PSD_o = scipy.signal.welch(beam, st[0].stats.sampling_rate, nperseg=PSD_win) #calculate PSD with Welch's method
    PSDdB_o = 10 * np.log10(abs(PSD_o) /pref**2) #converting to decibel
    fpsd, PSD = myinterp(fpsd_o[1:],PSDdB_o[1:]) #interpolateing to have equal spacing in log10 frequency space (important for equal fitting of low and high frequencies)

    #choose frequencies and PSD between frequency bounds for fitting
    f = fpsd[np.all([fpsd>freqmin,fpsd<freqmax],axis=0)]
    d = np.reshape(PSD[np.all([fpsd>freqmin,fpsd<freqmax],axis=0)],(len(f),1))

    if peaks == 'variable':
        m = m0
        ARGS = [f,d]
        M_func = misfit
    elif peaks == 'bound':
        m = m0
        if model != 'poly2fun':
            b1[2:] = m0[2]#10**(np.linspace(np.log10(fL / 10**fwidth),np.log10(fL),5))[3]
            b2[2:] = m0[3]#10**(np.linspace(np.log10(fL),np.log10(fL * 10**fwidth),5))[1]
        if model == 'poly2fun':
            b1[-1] = m0[3]
            b2[-1] = m0[4]
            m[3] = 1
            m[4] = 10**(np.linspace(np.log10(b1[-1]),np.log10(b2[-1]),3))[1]
        ARGS = [f, d]
        M_func = misfit
    elif peaks == 'constant':
        if model != 'poly2fun':
            if fL == None:
                fL = m0[2]
            if fF == None:
                fF = m0[3]
            m = m0[:2]
            b2 = b2[:2]
            b1 = b1[:2]
            fc = np.array([fL,fF])
            ARGS = [f,d,fc]
        if model == 'poly2fun':
            fw = m0[3]
            fp = m0[4]
            fc = np.array([fw, fp])
            m = m0[:3]
            b2 = b2[:3]
            b1 = b1[:3]
            ARGS = [f, d, fc]
        M_func = misfit_peak

    if method == 'lm':
        sol_all = scipy.optimize.least_squares(M_func,m,method='lm',args=ARGS, kwargs={'model':model})
        norm_m = np.transpose([np.array([np.dot(sol_all.fun, sol_all.fun)])]) ** (1 / 2)
        return beam, beamf, tvec, SPL, PSD, fpsd, sol_all.x, norm_m
    elif method == 'trf':
        sol_all = scipy.optimize.least_squares(M_func, m, args=ARGS, bounds=(b1,b2),method='trf', kwargs={'model':model})
        norm_m = np.transpose([np.array([np.dot(sol_all.fun, sol_all.fun)])]) ** (1 / 2)
        return beam, beamf, tvec, SPL, PSD, fpsd, sol_all.x, norm_m  # , sol_all_trf, sol_LST_trf, sol_FST_trf, ARGS, m, b1, b2

def misfit_spectrum(st_day,FREQ_vec, FREQ_vec_prob, baz, peaks='bound', fwidth=1, wwidth=10*60, method='trf', model='LSTFSTcombined'):
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

    OVERLAP = 0.7
    METHOD = method
    tstart = st_day[0].stats.starttime
    tend_abs = st_day[0].stats.endtime
    tend = tstart + wwidth
    n=0

    while tend <= tend_abs:
        st = st_day.copy()
        st.trim(starttime=tstart, endtime=tend)
        for iif in range(len(FREQ_vec)):
            freqmin = FREQ_vec[iif]
            freqmax = freqmin * 10**fwidth
            if model != 'poly2fun':
                m0 = np.array([np.log(400),np.log(300),10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1],10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]])
            elif model == 'poly2fun':
                m0 = np.array([1,-1,-1,10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1],10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]])

            if peaks == 'variable':
                PEAKS = 'variable'
            elif peaks == 'constant':
                PEAKS = 'constant'
                if model != 'poly2fun':
                    m0[-2] = 10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]
                    m0[-1] = 10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]
                if model == 'poly2fun':
                    m0[-2] = fwidth / 4
                    m0[-1] = 10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]
            elif peaks == 'bound':
                PEAKS = 'bound'
                fL = 10**(np.linspace(np.log10(freqmin),np.log10(freqmax),3))[1]
                m0[-2] = 10**(np.linspace(np.log10(fL / 10**fwidth),np.log10(fL),5))[3]
                m0[-1] = 10**(np.linspace(np.log10(fL), np.log10(fL * 10**fwidth), 5))[1]
            beam, beamf, tvec, SPL, PSD, fpsd, sol_m_temp, norm_m_temp  = simil_fit(st, freqmin=freqmin, freqmax=freqmax, m0=m0,baz=baz, peaks=PEAKS, fwidth=fwidth, method=METHOD, model=model)

            if iif==0:
                norm_m_f = norm_m_temp
                sol_m_f = np.array([sol_m_temp])
            else:
                norm_m_f = np.append(norm_m_f, norm_m_temp)
                sol_m_f = np.concatenate((sol_m_f, np.array([sol_m_temp])), axis=0)

        if n == 0:
            beam_all = beam
            tvec_all = tvec
            P_mat = np.array([PSD]).T
            sol_m = np.array([sol_m_f])
            norm_m = np.array([norm_m_f])
            tmid = dates.date2num((tstart + wwidth/2).datetime)
        else:
            beam_all = np.append(beam_all, [beam])
            tvec_all = np.append(tvec_all, [tvec])
            P_mat = np.concatenate((P_mat, np.array([PSD]).T), axis=1)
            sol_m = np.concatenate((sol_m, np.array([sol_m_f])), axis=0)
            norm_m = np.concatenate((norm_m, np.array([norm_m_f])), axis=0)
            tmid = np.append(tmid, dates.date2num((tstart + wwidth/2).datetime))
        tstart = tstart + wwidth * (1-OVERLAP)
        tend = tstart + wwidth
        n = n+1

    FREQ_prob_d = np.diff(np.log10(FREQ_vec_prob))[0]
    FREQ_vec_max = FREQ_vec * 10**fwidth
    nf = len(FREQ_vec_prob)
    M = np.zeros((nf - 1, len(tmid)))

    for iit in range(len(tmid)):
        for i in range(nf - 1):
            mask = np.arange(0, len(FREQ_vec))[
                np.all([FREQ_vec_prob[i + 1] - FREQ_vec > 0, FREQ_vec_max - FREQ_vec_prob[i] > 0], axis=0)]
            l = 0
            for s in range(len(mask)):
                j = mask[s]
                if FREQ_vec_max[j] < FREQ_vec_prob[i + 1]:  # when frequency band ends before
                    factor = (np.log10(FREQ_vec_max[j]) - np.log10(FREQ_vec_prob[i])) / FREQ_prob_d
                if FREQ_vec[j] > FREQ_vec_prob[i]:  # when frequency band starts after
                    factor = (np.log10(FREQ_vec_prob[i + 1]) - np.log10(FREQ_vec[j])) / FREQ_prob_d
                else:
                    factor = 1
                l = l + factor
                M[i, iit] = M[i, iit] + (factor * norm_m[iit,j])
            M[i, iit] = M[i, iit] / l

    print('Calculations are done.')
    return beam_all, tvec_all, P_mat, fpsd, norm_m, tmid, M, sol_m

COLOR_SEQUENCE=np.array(['green','red','blue','magenta','cyan','orange'])

def simil_plot(beam, tvec, SPL, PSD, fpsd, tmid, norm_M, sol_lm=None, freqmin=FREQMIN, freqmax=FREQMAX, colorstring=np.array(['green','red','blue','magenta','cyan','orange'])):
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
    method: ['lm' or 'trf' or 'lm&trf'] method for Gauss Newton inversion used to generate norm and inversion solution
    norm_lm, norm_trf: [array (3,m)] misfit norm calculated with simil_fit, only use *_lm or *_trf when method includes them
    sol_lm, sol_trf [array (5,3,m)] solution model parameters calculated with simil_fit, only use *_lm or *_trf when method includes them
    freqmin, freqmax [float] frequency min and max for fitting (defaults to 0.3-10Hz)
    
    OUTPUT:
    fig: [matplotlib figure object]
    ax: [array of matplotlib acis objects] 
   
    '''
    if len(np.shape(norm_M))==1:
        norm_M = np.array([norm_M])
    if len(np.shape(sol_lm))==2:
        sol_lm = np.array([sol_lm])

    Pmax = np.max(np.max(PSD[np.all([fpsd<freqmax,fpsd>freqmin],axis=0),:]))
    Pmin_avg = np.median(np.min(PSD[np.all([fpsd<freqmax,fpsd>freqmin],axis=0),:]))

    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    if sol_lm != None:
        gs = fig.add_gridspec(5, 2)
    else:
        gs = fig.add_gridspec(4, 2)
    ######## Spectrogram #####################
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_yscale('log')
    im = ax0.pcolormesh(tmid, fpsd, PSD, cmap='magma')
    im.set_clim([Pmin_avg+20, Pmax])
    ax0.axis('tight')
    ax0.set_xticklabels([])
    ax0.set_yticks(np.logspace(-1,2,4))
    ax0.set_ylim([0.1,50])
    ax0.plot([tmid[0],tmid[-1]],[freqmin,freqmin],'w--',linewidth=0.8)
    ax0.plot([tmid[0],tmid[-1]],[freqmax,freqmax],'w--',linewidth=0.8)
    Pmax_ind = np.asarray([np.where(PSD[np.all([fpsd<freqmax,fpsd>freqmin],axis=0), i] == np.max(PSD[np.all([fpsd<freqmax,fpsd>freqmin],axis=0),:],axis=0)[i])[0] for i in range(len(tmid))]).squeeze()
    freqmin_ind = np.where(np.abs(fpsd-freqmin) == np.min(np.abs(fpsd-freqmin)))
    ax0.plot(tmid,fpsd[Pmax_ind+freqmin_ind].squeeze(),'w.',markersize=2)
    ax0.set_ylabel('Frequency [Hz]')
    ax0.grid()
    cax = plt.axes([0.84, 0.76, 0.025, 0.14])
    hc =plt.colorbar(im,cax)#,cax=cax
    hc.set_label('Power [dB]')
    ######## Waveform #######################
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(tvec, beam,'k',linewidth=0.5)
    ax1.set_ylabel('Pa')
    ######## SPL (same subplot as waveform) ###########
    axn = ax1.twinx()
    axn.plot(tmid,SPL,'yellowgreen')
    YLIMn = axn.get_ylim()
    axn.set_xticklabels([])
    axn.set_xlim([tmid[0],tmid[-1]])
    axn.grid()
    axn.set_ylabel('SPL [dB]',color='yellowgreen')
    axn.tick_params(axis='y', colors='yellowgreen')
    ######## Peak Frequency ##########################
    if sol_lm != None:
        axf = fig.add_subplot(gs[2, :])
        freqmin_ind = np.where(np.abs(fpsd - freqmin) == np.min(np.abs(fpsd - freqmin)))
        for i in range(len(sol_lm)):
            Pmax_lm_ind = np.asarray([np.where(simil_func(sol_lm[i][j, :], fpsd)[np.all([fpsd < freqmax, fpsd > freqmin], axis=0)] ==
                          np.max(simil_func(sol_lm[i][j, :], fpsd)[np.all([fpsd < freqmax, fpsd > freqmin], axis=0)],
                                 axis=0))[0] for j in range(len(tmid))]).squeeze()
            fmax_lm = fpsd[freqmin_ind + Pmax_lm_ind].squeeze()
            axf.plot(tmid, fmax_lm, '.-', color=colorstring[i])
        axf.plot(tmid, fpsd[Pmax_ind+freqmin_ind].squeeze(), 'k.', label='f$_{peak}$(data)')
        axf.set_ylabel('Peak Frequency')
        axf.legend(loc='upper left', bbox_to_anchor=(1.1,1), borderaxespad=0.)
        axf.set_xlim([tmid[0], tmid[-1]])
        axf.set_xticklabels([])
    ######## Misfit ##################################
    if sol_lm != None:
        ax2 = fig.add_subplot(gs[3:, :])
    else:
        ax2 = fig.add_subplot(gs[2:, :])
    for i in range(len(norm_M)):
        ax2.plot(tmid, norm_M[i],'-',color=colorstring[i],linewidth=1)
    YLIM2 = ax2.get_ylim()
    ax2.set_ylim(YLIM2)
    ax2.tick_params(axis='x', which='major', labelsize=10,rotation=45,left=True)
    ax2.set_ylabel('Misfit Norm')
    ax2.grid()
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1,1), borderaxespad=0.)
    ax2.set_xlim([tmid[0],tmid[-1]])
    ##### time ticks ####################
    duration = (dates.num2date(tmid[-1])-dates.num2date(tmid[0]))
    dt_sec = duration.total_seconds()/6
    if dt_sec/(60*60) >= 1:
        ax0.xaxis.set_major_locator(dates.HourLocator(byhour=range(0, 24, round(dt_sec/(60*60))))) #tick location
        axn.xaxis.set_major_locator(dates.HourLocator(byhour=range(0, 24, round(dt_sec/(60*60))))) #tick location
        ax2.xaxis.set_major_locator(dates.HourLocator(byhour=range(0, 24, round(dt_sec/(60*60))))) #tick location
        if sol_lm != None:
            axf.xaxis.set_major_locator(dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
    elif np.abs(dt_sec / (60 * 60) - 1) < 0.5:
        ax0.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        axn.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        ax2.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        if sol_lm != None:
            axf.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
    elif dt_sec / (10 * 60) < 1:
        ax0.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        axn.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        ax2.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        if sol_lm != None:
            axf.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
    elif dt_sec/(60*60) < 1:
        ax0.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 30))) #tick location
        axn.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 30))) #tick location
        ax2.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 30))) #tick location
        if sol_lm != None:
            axf.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
    ax2.xaxis.set_major_formatter(dates.DateFormatter("%m/%d-%H:%M")) # tick formats
    ######################################
    if sol_lm != None:
        ax2.get_shared_x_axes().join( ax2, ax0, axn, axf)
    else:
        ax2.get_shared_x_axes().join(ax2, ax0, axn)
    fig.suptitle(str(dates.num2date(tmid[0])).replace(':','_')[:10],y=0.95)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.autofmt_xdate()
    if sol_lm != None:
        ax = [ax0, ax1, axn, axf, ax2]
    else:
        ax = [ax0, ax1, axn, ax2]
    return fig, ax



def misfit_spec_plot(P_mat, fpsd, tmid, M, FREQ_vec_prob, mid_point = 'default'):
    '''
    Function to plot the misfit spectra produced with misfit_spectrum
    :param P_mat: Spectrogram
    :param fpsd: frequency for spectrogram
    :param tmid: time for spectrogram
    :param M: Misfit spectrogram for LST&FST
    :param M_LST: Misfit spectrogram for LST
    :param M_FST: Misfit spectrogram for FST
    :param FREQ_vec_prob: frequency array for misfit spectrograms
    :param mid_point: Misfit value to center the colorbar on
    :return:
    '''

    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if mid_point == 'default':
        mid_point = np.round(np.mean([M]))-np.round(np.std([M]))

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

    newcmp0 = cm.get_cmap('viridis', 128)
    # newcmp1 = cm.get_cmap('gist_earth', 128)
    # newcmp2 = cm.get_cmap('gist_earth', 128)
    allcmp = ListedColormap(newcmp0(np.linspace(1,0,128)), name='GreyGreen')
    # LSTcmp = ListedColormap(newcmp1(np.linspace(1,0,128)), name='GreyRed')
    # FSTcmp = ListedColormap(newcmp2(np.linspace(1,0,128)), name='GreyBlue')

    fig, ax = plt.subplots(len(M)+1,1,figsize=(10,(len(M) + 1) * 2))

    #Plot Spectrogram
    ax[0].set_yscale('log')
    im = ax[0].pcolormesh(tmid, fpsd, P_mat, cmap='magma')
    ax[0].axis('tight')
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Frequency [Hz]')
    ax[0].grid()
    cax = plt.axes([0.92, 0.72, 0.02, 0.16])
    hc = plt.colorbar(im, cax)  # ,cax=cax
    hc.set_label('Power [dB]')
    ax[0].set_xlim([tmid[0],tmid[-1]])

    #Plot Misfit Spectrogram for LST&FST
    for i in range(len(M)):
        x, y = np.meshgrid(tmid, FREQ_vec_prob)
        im = ax[i+1].pcolormesh(x, y, M[i], cmap=allcmp, vmin=np.nanmax([mid_point - 2 * np.round(np.nanstd([M])), 0]) , vmax=mid_point + 2 * np.round(np.nanstd([M])))
        ax[i+1].set_yscale('log')
        ax[i+1].axis('tight')
        ax[i+1].set_ylabel('Frequency [Hz]')
        ax[i+1].set_xticklabels([])
        cax = plt.axes([0.92, 0.52, 0.02, 0.16])
        hc = plt.colorbar(im, cax, ticks=[mid_point - 1 * np.round(np.nanstd([M])), mid_point, mid_point + 1 * np.round(np.nanstd([M]))])  # ,cax=cax
        # hc.set_label('LST & FST Misfit')
        ax[i+1].set_ylim([FREQ_vec_prob[0], FREQ_vec_prob[-1]])
        ax[i+1].set_xlim([tmid[0],tmid[-1]])
        ax[i+1].grid()


    # Time Ticks
    duration = (dates.num2date(tmid[-1]) - dates.num2date(tmid[0]))
    dt_sec = duration.total_seconds() / 6
    for i in range(len(M)+1):
        if dt_sec / (60 * 60) >= 1:
            ax[i].xaxis.set_major_locator(dates.HourLocator(byhour=range(0, 24, round(dt_sec / (60 * 60)))))  # tick location
        elif np.abs(dt_sec / (60 * 60) - 1) < 0.5:
            ax[i].xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 60)))  # tick location
        elif dt_sec / (5 * 60) < 1:
            ax[i].xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 5)))  # tick location
        elif dt_sec / (10 * 60) < 1:
            ax[i].xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 10)))  # tick location
        elif dt_sec / (60 * 60) < 1:
            ax[i].xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, 30)))  # tick location
    ax[-1].xaxis.set_major_formatter(dates.DateFormatter("%m/%d-%H:%M"))  # tick formats
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)

    return fig, ax

def welch_man(waveform,sampling_rate, pref=20e-6):
    fs = sampling_rate
    window = 'hann'
    win, nperseg = scipy.signal.spectral._triage_segments(window, len(waveform), input_length=waveform.shape[-1])
    scale = 1.0 / (fs * (win * win).sum())
    S = scipy.fft(win * waveform, n=len(waveform))#[0]
    S = np.conjugate(S) * S #magnitude of complex number (same as abs()**2)
    S *= scale
    S[..., 1:] *= 2
    S = S[:int(len(S) / 2)]
    f = np.linspace(0, fs /2,len(S))

    # PSD_W_man_dB = 10 * np.log10(PSD_W_man[1:len(f)] / pref**2)
    return f, S

def welch_wave(spectrum, frequency, duration, sampling_rate, phase_distribution='uniform'):

    tck = scipy.interpolate.splrep(frequency, spectrum, s=0)
    fnew = np.arange(1 / duration, sampling_rate / 2 + 1 / duration, 1 / duration)
    Snew = scipy.interpolate.splev(fnew, tck, der=0)
    Snew = np.append(0, Snew)
    Snew = np.abs(Snew)

    window = 'hann'
    win, nperseg = scipy.signal.spectral._triage_segments(window, len(Snew) * 2 - 2, input_length=Snew.shape[-1] * 2 - 2)
    scale = 1.0 / (fnew[-1] * 2 * (win * win).sum())

    Snew[..., 1:] /= 2
    Snew /= scale
    Snew = Snew ** (1 / 2)

    if phase_distribution == 'uniform':
        phases = np.random.uniform(low=0, high=2 * np.pi, size=len(Snew) - 2) * 1j
    phase_dist = np.hstack([[0], phases, [0], np.conj(phases[:: -1])])
    T = np.real(np.fft.ifft(np.hstack([Snew, Snew[-2: 0: -1]]) * phase_dist))
    t = np.linspace(0,duration,len(T))

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
    t = np.linspace(0,duration,len(T))
    return t, T


def simil_wave_reverse(waveform, time):
    S = np.fft.fft(waveform)
    S = (S * np.conj(S)) ** (1 / 2)
    S = S[:int(len(S)/2)]
    f = np.linspace(0,1/(2*np.diff(time)[0]),len(S))
    return f, S