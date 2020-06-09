import numpy as np
import scipy
from matplotlib import dates
import obspy as obs


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



def simil_func(m, f, p=20e-6, output='dB'):
    '''
    Function to calculate the similarity spectrum for combined contributions of fine scale turbulence (FST) and large scale turbulence (LST) from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (2)
    INPUT:
    m: [array shape (5,1)] model vector with m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (2)), 'orig' spectrum (as in eq. (1)), default: 'dB'
    OUTPUT:
    SdB (if output='dB') [array like, equal length to f] spectrum
    S (if output='orig') [array like, equal length to f] spectrum
    '''
    a = m[0]
    b = m[1]
    c = m[2]
    fL = m[3]
    fF = m[4]

    G,F = GF(f,fL=fL,fF=fF)

    if output == 'dB':
        SdB = 10 * np.log10(np.exp(a) * F + np.exp(b) * G) - 20 * np.log10(np.exp(c)) - 20 * np.log10(p)
        return SdB
    elif output == 'orig':
        S =  (np.exp(a)*F + np.exp(b)*G) * np.exp(c)**(-2)
        return S



def simil_FST_func(m, f, p=20e-6, output='dB'):
    '''
    Function to calculate the similarity spectrum for fine scale turbulence (FST) from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (4)
    INPUT:
    m: [array shape (5,1) or (3,1)] model vector. if shape=(5,1): m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF. if shape=(3,1): m[0]=ln(B), m[1]=ln(r/Dj), m[2]=fF
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (4)), 'orig' spectrum (as in eq. (1) with A~0), default: 'dB'
    OUTPUT:
    SdB (if output='dB') [array like, equal length to f] spectrum
    S (if output='orig') [array like, equal length to f] spectrum
    '''
    if len(m) == 5:
        a = m[0]
        b = m[1]
        c = m[2]
        fL = m[3]
        fF = m[4]
    elif len(m) == 3:
        b = m[0]
        c = m[1]
        fF = m[2]

    G = GF(f, fF=fF, output='G')

    if output == 'dB':
        SdB = 10 * np.log10(np.exp(b) * G) - 20 * np.log10(np.exp(c)) - 20 * np.log10(p)
        return SdB
    elif output == 'orig':
        S =  np.exp(b) * G * np.exp(c)**(-2)
        return S



def simil_LST_func(m, f, p=20e-6, output='dB'):
    '''
    Function to calculate the similarity spectrum for large scale turbulence (FST) from Tam et al., 1996: "On the Two Components of Turbulent Mixing Noise from Supersonic Jets" equation (3)
    INPUT:
    m: [array shape (5,1) or (3,1)] model vector. if shape=(5,1): m[0]=ln(A), m[1]=ln(B), m[2]=ln(r/Dj), m[3]=fL, m[4]=fF. if shape=(3,1): m[0]=ln(A), m[1]=ln(r/Dj), m[2]=fL
    f: [array like] frequency
    p: [float] reference pressure, default: p=20e-6 Pa
    output: [string] 'dB' spectrum in dB scale (as in eq. (3)), 'orig' spectrum (as in eq. (1), with B~0), default: 'dB'
    OUTPUT:
    SdB (if output='dB') [array like, equal length to f] spectrum
    S (if output='orig') [array like, equal length to f] spectrum
    '''
    if len(m) == 5:
        a = m[0]
        b = m[1]
        c = m[2]
        fL = m[3]
        fF = m[4]
    elif len(m) == 3:
        a = m[0]
        c = m[1]
        fL = m[2]
    else:
        print('model vector must have length of 3 or 5')

    F = GF(f, fL=fL, output='F')

    if output == 'dB':
        SdB = 10 * np.log10(np.exp(a) * F) - 20 * np.log10(np.exp(c)) - 20 * np.log10(p)
        return SdB
    elif output == 'orig':
        S =  np.exp(a) * F * np.exp(c)**(-2)
        return S

def misfit(m, f, d, input='total', **kwargs):
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
    if input == 'total':
        S = simil_func(m,f, **kwargs, output='orig')
        S = 10 * np.log10(S/(20e-6)**(2))
    if input == 'FST':
        S = simil_FST_func(m,f, **kwargs)
    if input == 'LST':
        S = simil_LST_func(m,f, **kwargs)
    # if len(d) != 0:
    #     return S
    S = np.reshape(S,np.shape(d))

    M = ((2/len(d))**(1/2) * np.abs(d - S)).squeeze()
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
M0 = np.array([np.log(400),np.log(300),np.log(20),1,1])

def simil_fit(st, freqmin=FREQMIN, freqmax=FREQMAX, baz=F8BAZ, method='lm', m0=M0, PSD_win=None):
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
    if PSD_win is None:
        PSD_win = int(60 * st[0].stats.sampling_rate)
    b1 = np.array([np.log(1e-5),np.log(1e-5),np.log(1e-5),1e-5,1e-5])
    b2 = np.array([np.log(1000),np.log(1000),np.log(1000),20,30])
    pref = 20e-6
    ############################

    tvec=dates.date2num(st[0].stats.starttime.datetime)+st[0].times()/86400   #datenum time vector

    stf = st.copy() ## for sound pressure level
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

        data = np.zeros((len(st[0].data),4))
        data[:,0] = st[0].data
        data[:,1] = st[1].data
        data[:,2] = st[2].data
        data[:,3] = st[3].data
        beam = beamForm(data, rij, st[0].stats.sampling_rate, baz, M=len(tvec))

        dataf = np.zeros((len(st[0].data),4))
        dataf[:,0] = stf[0].data
        dataf[:,1] = stf[1].data
        dataf[:,2] = stf[2].data
        dataf[:,3] = stf[3].data
        beamf = beamForm(dataf, rij, stf[0].stats.sampling_rate, baz, M=len(tvec))

    p_rms = np.sqrt(np.nanmean(beamf[beamf!=0]**2))
    SPL = 10 * np.log10(p_rms**2/pref**2)

    ## Calculate PSD
    fpsd_o, Pxx = scipy.signal.welch(beam, st[0].stats.sampling_rate, nperseg=PSD_win)
    PxxdBpsd_o = 10 * np.log10(abs(Pxx) /pref**2)
    fpsd, PSD = myinterp(fpsd_o[1:],PxxdBpsd_o[1:])

    f = fpsd[np.all([fpsd>freqmin,fpsd<freqmax],axis=0)]
    d = np.reshape(PSD[np.all([fpsd>freqmin,fpsd<freqmax],axis=0)],(len(f),1))
    fL = f[np.where(d==np.max(d))[0]]
    fF = f[np.where(d==np.max(d))[0]]

    if method == 'lm':
        sol_all = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d])
        sol_LST = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d],kwargs={'input':'LST'})
        sol_FST = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d],kwargs={'input':'FST'})
        norm_m = (np.transpose([np.array([np.dot(misfit(sol_all.x, f, d),misfit(sol_all.x, f, d)), np.dot(misfit(sol_LST.x, f, d,input='LST'),misfit(sol_LST.x, f, d,input='LST')),np.dot(misfit(sol_FST.x, f, d,input='FST'),misfit(sol_FST.x, f, d,input='FST'))])]))**(1/2)
        sol_mat = np.array([np.array([sol_all.x,sol_LST.x,sol_FST.x])]).T
        return beam, beamf, SPL, PSD, sol_mat, norm_m
    elif method == 'trf':
        sol_all_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],bounds=(b1,b2),method='trf')
        sol_LST_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],kwargs={'input':'LST'},bounds=(b1,b2),method='trf')
        sol_FST_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],kwargs={'input':'FST'},bounds=(b1,b2),method='trf')
        norm_trf = (np.transpose([np.array([np.dot(misfit(sol_all_trf.x, f, d),misfit(sol_all_trf.x, f, d)), np.dot(misfit(sol_LST_trf.x, f, d,input='LST'),misfit(sol_LST_trf.x, f, d,input='LST')),np.dot(misfit(sol_FST_trf.x, f, d,input='FST'),misfit(sol_FST_trf.x, f, d,input='FST'))])]))**(1/2)
        sol_trf_mat = np.array([np.array([sol_all_trf.x,sol_LST_trf.x,sol_FST_trf.x])]).T
        return beam, beamf, SPL, PSD, sol_trf_mat, norm_trf
    elif method =='lm&trf':
        sol_all = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d])
        sol_LST = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d],kwargs={'input':'LST'})
        sol_FST = scipy.optimize.least_squares(misfit,m0,method='lm',args=[f,d],kwargs={'input':'FST'})
        norm_m = (np.transpose([np.array([np.dot(misfit(sol_all.x, f, d),misfit(sol_all.x, f, d)), np.dot(misfit(sol_LST.x, f, d,input='LST'),misfit(sol_LST.x, f, d,input='LST')),np.dot(misfit(sol_FST.x, f, d,input='FST'),misfit(sol_FST.x, f, d,input='FST'))])]))**(1/2)
        sol_mat = np.array([np.array([sol_all.x,sol_LST.x,sol_FST.x])]).T
        sol_all_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],bounds=(b1,b2),method='trf')
        sol_LST_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],kwargs={'input':'LST'},bounds=(b1,b2),method='trf')
        sol_FST_trf = scipy.optimize.least_squares(misfit,m0,args=[f,d],kwargs={'input':'FST'},bounds=(b1,b2),method='trf')
        norm_trf = (np.transpose([np.array([np.dot(misfit(sol_all_trf.x, f, d),misfit(sol_all_trf.x, f, d)), np.dot(misfit(sol_LST_trf.x, f, d,input='LST'),misfit(sol_LST_trf.x, f, d,input='LST')),np.dot(misfit(sol_FST_trf.x, f, d,input='FST'),misfit(sol_FST_trf.x, f, d,input='FST'))])]))**(1/2)
        sol_trf_mat = np.array([np.array([sol_all_trf.x,sol_LST_trf.x,sol_FST_trf.x])]).T
        return beam, beamf, tvec, SPL, PSD, fpsd, sol_mat, sol_trf_mat, norm_m, norm_trf

def simil_plot(beam, tvec, SPL, PSD, fpsd, tmid, norm_m=None, norm_trf=None, freqmin=FREQMIN, freqmax=FREQMAX, method='lm'):
    import matplotlib.pyplot as plt
    import numpy.matlib
    '''
    INPUT
    '''

    Pmax = np.max(np.max(PSD[np.all([fpsd<freqmax,fpsd>freqmin],axis=0),:]))
    err = 0.05
    ######## find outlier where LST&FST has larger misfir then LST or FST separately
    outlier_all = None
    outlier_trf = None
    if method == 'lm':
        if np.any([np.any([norm_m[0,:]>norm_m[1,:]+err,norm_m[0,:]>norm_m[2,:]+err],axis=0)]):
            outlier_all = np.where(np.any([norm_m[0,:]>norm_m[1,:]+err,norm_m[0,:]>norm_m[2,:]+err],axis=0))
    elif method == 'trf':
        if np.any([np.any([norm_trf[0,:]>norm_trf[1,:]+err,norm_trf[0,:]>norm_trf[2,:]+err],axis=0)]):
            outlier_trf = np.where(np.any([norm_trf[0,:]>norm_m[1,:]+err,norm_trf[0,:]>norm_trf[2,:]+err],axis=0))
    elif method == 'lm&trf':
        if np.any([np.any([norm_m[0,:]>norm_m[1,:]+err,norm_m[0,:]>norm_m[2,:]+err],axis=0)]):
            outlier_all = np.where(np.any([norm_m[0,:]>norm_m[1,:]+err,norm_m[0,:]>norm_m[2,:]+err],axis=0))
        if np.any([np.any([norm_trf[0,:]>norm_trf[1,:]+err,norm_trf[0,:]>norm_trf[2,:]+err],axis=0)]):
            outlier_trf = np.where(np.any([norm_trf[0,:]>norm_m[1,:]+err,norm_trf[0,:]>norm_trf[2,:]+err],axis=0))

    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    gs = fig.add_gridspec(4, 2)
    ######## Spectrogram #####################
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_yscale('log')
    im = ax0.pcolormesh(tmid, fpsd, PSD, cmap='magma')
    im.set_clim([80, Pmax])
    ax0.axis('tight')
    ax0.set_xticklabels([])
    ax0.set_yticks(np.logspace(-1,2,4))
    ax0.set_ylim([0.1,50])
    ax0.plot([tmid[0],tmid[-1]],[freqmin,freqmin],'w--',linewidth=0.8)
    ax0.plot([tmid[0],tmid[-1]],[freqmax,freqmax],'w--',linewidth=0.8)
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
    ######## Misfit ##################################
    ax2 = fig.add_subplot(gs[2:, :])
    if method == 'lm':
        ax2.plot(tmid, norm_m[1,:],'-',color='r',label='LST',linewidth=1)
        ax2.plot(tmid, norm_m[2,:],'-',color='b',label='FST',linewidth=1)
        ax2.plot(tmid, norm_m[0,:],'-',color='g',label='LST & FST',linewidth=1)
    elif method == 'trf':
        ax2.plot(tmid, norm_trf[1,:],'-',color='r',label='LST',linewidth=1)
        ax2.plot(tmid, norm_trf[2,:],'-',color='b',label='FST',linewidth=1)
        ax2.plot(tmid, norm_trf[0,:],'-',color='g',label='LST & FST',linewidth=1)
    elif method == 'lm&trf':
        ax2.plot(tmid, norm_m[1,:],'-',color='r',label='LST (lm)',linewidth=1)
        ax2.plot(tmid, norm_m[2,:],'-',color='b',label='FST (lm)',linewidth=1)
        ax2.plot(tmid, norm_m[0,:],'-',color='g',label='LST & FST (lm)',linewidth=1)
        ax2.plot(tmid, norm_trf[1,:],'--',color='r',label='LST (trf)',linewidth=1)
        ax2.plot(tmid, norm_trf[2,:],'--',color='b',label='FST (trf)',linewidth=1)
        ax2.plot(tmid, norm_trf[0,:],'--',color='g',label='LST & FST (trf)',linewidth=1)
    YLIM2 = ax2.get_ylim()
    if outlier_all is not None:
        ax2.scatter(tmid[outlier_all[0]],numpy.matlib.repmat(np.max(np.max(norm_m)),1,len(outlier_all[0])),c='g',s=30,label='Misfit Anomaly',marker='v')
    if outlier_trf is not None:
        ax2.scatter(tmid[outlier_trf[0]],numpy.matlib.repmat(np.max(np.max(norm_trf)),1,len(outlier_trf[0])),c='k',s=30,label='Misfit Anomaly',marker='v')
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
    elif dt_sec/(60*60) < 1:
        ax0.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, round(dt_sec/60)))) #tick location
        axn.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, round(dt_sec/60)))) #tick location
        ax2.xaxis.set_major_locator(dates.MinuteLocator(byminute=range(0, 60, round(dt_sec/60)))) #tick location
    ax2.xaxis.set_major_formatter(dates.DateFormatter("%m/%d-%H:%M")) # tick formats
    ######################################
    ax2.get_shared_x_axes().join( ax2, ax0, axn)
    fig.suptitle(str(dates.num2date(tmid[0])).replace(':','_')[:10],y=0.95)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.autofmt_xdate()
    ax = [ax0, ax1, axn, ax2]
    return fig, ax
