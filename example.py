
import numpy as np
import obspy as obs
import simil_func as sf
from waveform_collection import read_local
from array_processing.tools import beamForm
from array_processing.algorithms.helpers import getrij
from matplotlib import dates
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client

#%% starting

STARTTIME = obs.UTCDateTime('2018-6-16T17:00')
ENDTIME   = STARTTIME + 1 * 60 *60

fdsn_client = Client(
    'IRIS')  # Fetch waveform from IRIS FDSN web service into a ObsPy stream object# and automatically attach correct response
st_day = fdsn_client.get_waveforms(network='5L', station='FIS8', location='*',channel='*', starttime=STARTTIME, endtime=ENDTIME,attach_response=True)# define a filter band to prevent amplifying noise during the deconvolution

SENSITIVITY = 8210.1
for tr in st_day:
    tr.data = tr.data / SENSITIVITY

#%%
tr_try = st_day[0].copy()
tr_try.remove_response(inventory=inv, output="DISP",
                   water_level=60, plot=True)
#%%
local_coords = dict({
     '01': [19.4638, -154.91287, 254.0],
     '02': [19.4638, -154.91315, 251.0],
     '03': [19.46406, -154.91307, 255.0],
     '04': [19.46388, -154.91308, 259.0]})
for tr in st_day:
    tr.stats.latitude, tr.stats.longitude, \
    tr.stats.elevation = local_coords[tr.stats.location]


#%%
BEAM_WINDOW = 10*60
OVERLAP = 0.7
tstart = STARTTIME
tend = tstart + BEAM_WINDOW
FREQMIN = 0.05
FREQMAX = 10
n=0

while tend <= ENDTIME:
    st = st_day.copy()
    st.trim(starttime=tstart, endtime=tend)
    beam, tvec, PSD, fpsd, sol_all, norm_all = sf.simil_fit(stream=st, model=['LSTFST', 'LST', 'FST'], freqmin=FREQMIN, freqmax=FREQMAX)

    if n == 0:
        beam_all = beam
        tvec_all = tvec
        P_mat = np.array([PSD]).T
        sol_vec = np.array([sol_all])
        norm_vec = np.array([norm_all])
        tmid = dates.date2num((tstart + BEAM_WINDOW/2).datetime)
    else:
        beam_all = np.append(beam_all, [beam[tvec_all[-1]<=tvec]])
        tvec_all = np.append(tvec_all, [tvec[tvec_all[-1]<=tvec]])
        P_mat = np.concatenate((P_mat, np.array([PSD]).T), axis=1)
        sol_vec = np.concatenate((sol_vec,np.array([sol_all])), axis=0)
        norm_vec = np.append(norm_vec, np.array([norm_all]), axis=0)
        tmid = np.append(tmid, dates.date2num((tstart + BEAM_WINDOW/2).datetime))
    tstart = tstart + BEAM_WINDOW * (1-OVERLAP)
    tend = tstart + BEAM_WINDOW
    n = n+1

print('Calculations are done.')
#%% calculate misfit spectrum
nf1 = 20
FREQ_vec = 10**(np.linspace(np.log10(0.0166),np.log10(25/10),nf1))#np.logspace(-2,2,7)np.log10(st_day[0].stats.sampling_rate / 2)
nf = 40
FREQ_vec_prob = 10 ** (np.linspace(np.log10(0.0166), np.log10(25), nf))
b1 = np.array([-np.inf, -np.inf, 0.15, 0.15])
b2 = np.array([np.inf, np.inf, FREQMAX, FREQMAX])
beam_all, tvec_all, P_mat, fpsd, norm_trf, tmid, M, sol_trf = sf.misfit_spectrum(stream=st_day, FREQ_vec=FREQ_vec, FREQ_vec_prob=FREQ_vec_prob, baz=121, peaks='bound', bounds=(b1,b2), fwidth=1, model=['LSTFST', 'LST', 'FST'])

#%% calculate misfit spectrum
threshold = 2
M_diff = sf.misfit_diff(M[1:3,:,:], threshold)
M_all = np.concatenate((M,np.array([M_diff])),axis=0)#[M[0,:,:],M_LST,M_FST,M_diff


#%% plot similarity misfit etc.
fig, ax = plt.subplots(4,1, figsize=(6,6), sharex=True)

# plot waveform
ax[0].plot(dates.num2date(tvec_all), beam_all, 'k', linewidth=0.5)
ax[0].set_ylabel('Pressure [Pa]')

#plot spectrogram
im = ax[1].pcolormesh(dates.num2date(tmid), fpsd, P_mat, cmap='magma', vmin=70, vmax=100)
ax[1].set_yscale('log')
ax[1].set_ylabel('Freq. [Hz]')
cb_ax = fig.add_axes([0.93, 0.51,0.03, 0.17])
hc = plt.colorbar(im, ax=ax[1], cax=cb_ax, aspect=8)

#plot standard deviation for large frequency band
ax[2].plot(dates.num2date(tmid), norm_vec[:, 0], '.-', color='g', label='LST & FST')
ax[2].plot(dates.num2date(tmid), norm_vec[:, 1], '.-', color='r', label='LST')
ax[2].plot(dates.num2date(tmid), norm_vec[:, 2], '.-', color='b', label='FST')
ax[2].set_ylabel('SD [dB] \n (0.05 Hz-10 Hz)') #SD = standard deviation
ax[2].grid()

#plot misfit difference spectrogram
im = ax[3].pcolormesh(dates.num2date(tmid), FREQ_vec_prob, M_diff, cmap='seismic', vmin=-3, vmax=3)
cb_ax = fig.add_axes([0.93, 0.12, 0.03, 0.17])
hc = plt.colorbar(im, ax=ax[3], cax=cb_ax, aspect=8)
ax[3].set_yscale('log')
ax[3].set_ylabel('Freq. [Hz]')

for axi in ax:
    axi.set_xlim(dates.num2date(tmid)[0], dates.num2date(tmid)[-1])
for tick in ax[-1].get_xticklabels():
    tick.set_rotation(20)
plt.savefig('/Users/julia/Documents/UAF_Research/Kilauea_Project/figures/example.png',bbox_inches='tight')

plt.show()
