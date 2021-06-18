
import numpy as np
import obspy as obs
import simil_func as sf
from waveform_collection import read_local
from array_processing.tools import beamForm
from array_processing.algorithms.helpers import getrij
from matplotlib import dates
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client

#%% reading in data from IRIS 

NETWORK = '5L'
STATION = 'FIS8'
LOCATION = '*'
CHANNEL = '*'

STARTTIME = obs.UTCDateTime('2018-6-16T17:00')
ENDTIME   = STARTTIME + 1 * 60 *60

fdsn_client = Client('IRIS')  # Fetch waveform from IRIS FDSN web service into a ObsPy stream object# and automatically attach correct response
st_day = fdsn_client.get_waveforms(network=NETWORK, station=STATION, 
                                   location='*',channel='*', 
                                   starttime=STARTTIME, endtime=ENDTIME,
                                   attach_response=True)

#%% resample and remove response

#resample to 100 Hz to save time
st_day.interpolate(sampling_rate=100, method="lanczos", a=15)

# print('Removing response...') 
# Fs =st_day[0].stats.sampling_rate
# pre_filt = [0.001, 0.005, Fs/2-2, Fs/2] #pre-filt for response removal
# st_day.remove_response(pre_filt=pre_filt, output='VEL', water_level=None) 

SENSITIVITY = 8210.1
for tr in st_day:
    tr.data = tr.data / SENSITIVITY

#%% Attaching array coordinates
inv = fdsn_client.get_stations(network='5L', station='FIS8',
                                  location='*', channel='*',
                                  starttime=STARTTIME,
                                  endtime=ENDTIME,
                                  level='channel')

for tr in st_day:
    coords = inv.get_coordinates(tr.id)
    tr.stats.longitude = coords['longitude']
    tr.stats.latitude = coords['latitude']
    tr.stats.elevation = coords['elevation']

#%% Fitting the similarity spectra to data
FREQMIN = 0.05
FREQMAX = 10
beam, tvec, PSD, fpsd, sol, norm = sf.simil_fit(stream=st_day, freqmin=FREQMIN, freqmax=FREQMAX, model=['LST', 'FST'])

#%% Plotting the result
fig, ax = plt.subplots()
ax.plot(fpsd, PSD, 'k', 'Data')
ax.plot(fpsd, sf.simil_func(sol[0,:], fpsd, model='LST'), 'r', label='LST: {:3.2f} dB'.format(norm[0]))
ax.plot(fpsd, sf.simil_func(sol[1,:], fpsd, model='FST'), 'b', label='FST: {:3.2f} dB'.format(norm[1]))
ax.axvline(FREQMIN, color='grey', linestyle='--'); ax.axvline(FREQMAX, color='grey', linestyle='--')
ax.set_xscale('log')
ax.set_ylabel('Power [dB]'); ax.set_xlabel('Frequency [Hz]')
ax.grid()
ax.legend()
ax.set_ylim(40,105)
plt.show()

#%% Fitting the similarity spectra for smaller time windows
BEAM_WINDOW = 10*60
OVERLAP = 0.7
tstart = STARTTIME
tend = tstart + BEAM_WINDOW
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
#%% Calculate misfit spectrum
nf1 = 20
FREQ_vec = 10**(np.linspace(np.log10(0.0166),np.log10(25/10),nf1))#np.logspace(-2,2,7)np.log10(st_day[0].stats.sampling_rate / 2)
nf = 40
FREQ_vec_prob = 10 ** (np.linspace(np.log10(0.0166), np.log10(25), nf))
b1 = np.array([0.15, 0.15])
b2 = np.array([FREQMAX, FREQMAX])
beam_all, tvec_all, P_mat, fpsd, norm_trf, tmid, M, sol_trf = sf.misfit_spectrum(stream=st_day, FREQ_vec=FREQ_vec, FREQ_vec_prob=FREQ_vec_prob, baz=121, peaks='bound', bounds=(b1,b2), fwidth=1, model=['LSTFST', 'LST', 'FST'])

#%% calculate misfit difference spectrum
threshold = 2
M_diff = sf.misfit_diff(M[1:3,:,:], threshold)
M_all = np.concatenate((M,np.array([M_diff])),axis=0)#[M[0,:,:],M_LST,M_FST,M_diff


#%% plot similarity misfit etc.
plt.get_cmap('seismic').set_bad(color='grey')

fig, ax = plt.subplots(4,1, figsize=(6,6), sharex=True)

# plot waveform
ax[0].plot(dates.num2date(tvec_all), beam_all, 'k', linewidth=0.5)
ax[0].set_ylabel('Pressure [Pa]')

#plot spectrogram
im = ax[1].pcolormesh(dates.num2date(tmid), fpsd, P_mat, cmap='magma', vmin=70, vmax=100, shading='auto')
ax[1].set_yscale('log')
ax[1].set_ylabel('Freq. [Hz]')
cb_ax = fig.add_axes([0.92, 0.51, 0.03, 0.17])
hc = plt.colorbar(im, ax=ax[1], cax=cb_ax, aspect=8)

#plot standard deviation for large frequency band
ax[2].plot(dates.num2date(tmid), norm_vec[:, 0], '.-', color='g', label='LST & FST')
ax[2].plot(dates.num2date(tmid), norm_vec[:, 1], '.-', color='r', label='LST')
ax[2].plot(dates.num2date(tmid), norm_vec[:, 2], '.-', color='b', label='FST')
ax[2].set_ylabel('SD [dB] \n (0.05 Hz-10 Hz)') #SD = standard deviation
ax[2].grid()

#plot misfit difference spectrogram
im = ax[3].pcolormesh(dates.num2date(tmid), FREQ_vec_prob, M_diff, cmap='seismic', vmin=-3, vmax=3, shading='auto')
cb_ax = fig.add_axes([0.92, 0.11, 0.03, 0.17])
hc = plt.colorbar(im, ax=ax[3], cax=cb_ax, aspect=8)
cb_ax.set_ylabel('FST - LST')
ax[3].set_yscale('log')
ax[3].set_ylabel('Freq. [Hz]')

for axi in ax:
    axi.set_xlim(dates.num2date(tmid)[0], dates.num2date(tmid)[-1])
for tick in ax[-1].get_xticklabels():
    tick.set_rotation(20)
plt.show()