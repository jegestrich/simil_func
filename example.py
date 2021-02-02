
import numpy as np
import obspy as obs
import simil_func as sf
from waveform_collection import read_local
from array_processing.tools import beamForm
from array_processing.algorithms.helpers import getrij
from matplotlib import dates
import matplotlib.pyplot as plt

filepath = 'data_example/mseed_infra_June16-17/'
coord_file = 'data_example/local_infra_coords.json'
network = 'HV'
fheight_path = 'data_example/2018LERZeruption_data.xlsx'

#%% starting
tstart = obs.UTCDateTime('2018-6-16T17:00')
tstart_abs = tstart
tend = tstart + 5*60*60
tend_abs = tend

st_day = read_local(filepath, coord_file, network, 'AF0','*','*', tstart, tend)
st_day += read_local(filepath, coord_file, network, 'AF2','*','*', tstart, tend)
st_day += read_local(filepath, coord_file, network, 'AEY','*','*', tstart, tend)
st_day += read_local(filepath, coord_file, network, 'AS0','*','*', tstart, tend)

latlist = []
lonlist = []
[latlist.append(st_day[i].stats.latitude) for i in range(len(st_day))]
[lonlist.append(st_day[i].stats.longitude) for i in range(len(st_day))]

rij=getrij(latlist,lonlist)

#%%
BEAM_WINDOW = 10*60
OVERLAP = 0.7
tstart = tstart_abs
tend = tstart + BEAM_WINDOW
FREQMIN = 0.3
FREQMAX = 10
n=0

while tend <= tend_abs:
    st = st_day.copy()
    st.trim(starttime=tstart, endtime=tend)
    beam, beamf, tvec, SPL, PSD, fpsd, sol_all, norm_all = sf.simil_fit(st, model=['LSTFST', 'LST', 'FST'], freqmin=FREQMIN, freqmax=FREQMAX)
    if n == 0:
        beam_all = beam
        tvec_all = tvec
        SPL_all = SPL
        P_mat = np.array([PSD]).T
        sol_vec = np.array([sol_all])
        norm_vec = np.array([norm_all])
        tmid = dates.date2num((tstart + BEAM_WINDOW/2).datetime)
    else:
        beam_all = np.append(beam_all, [beam])
        tvec_all = np.append(tvec_all, [tvec])
        SPL_all = np.append(SPL_all,SPL)
        P_mat = np.concatenate((P_mat, np.array([PSD]).T), axis=1)
        sol_vec = np.concatenate((sol_vec,np.array([sol_all])), axis=0)
        norm_vec = np.append(norm_vec, np.array([norm_all]), axis=0)
        tmid = np.append(tmid, dates.date2num((tstart + BEAM_WINDOW/2).datetime))
    tstart = tstart + BEAM_WINDOW * (1-OVERLAP)
    tend = tstart + BEAM_WINDOW
    n = n+1

print('Calculations are done.')


#%% plot similarity misfit etc.
norm_M = norm_vec.T

fig,ax = sf.simil_plot(beam_all, tvec_all, SPL_all, P_mat, fpsd, tmid, norm_M, freqmin=FREQMIN, freqmax=FREQMAX)#insert: ", sol_lm=sol_vec.T" if you want to show peak frequencies of solutions

#optional plotting of fountain height and cone height
# import pandas as pd
# df_fheight = pd.read_excel(fheight_path, engine='openpyxl')
# df_manual = df_fheight[['date & time (manual)', 'max fountain height (m) (manual)']].sort_values('date & time (manual)')
# fheight_manual = df_manual['max fountain height (m) (manual)']
# manmask = np.isfinite(fheight_manual)
# fheight_manual = fheight_manual.loc[manmask].reset_index(drop=True)
# dtime_manual = df_manual['date & time (manual)']
# dtime_manual = (dtime_manual.loc[manmask] + np.timedelta64(10, 'h')).reset_index(drop=True)
# fheight_cam = df_fheight['fountain height (m) (cam)'].to_numpy()
# cammask = np.where(fheight_cam > 0.)
# fheight_cam = fheight_cam[cammask]
# dtime_cam = (df_fheight['date (cam)'].loc[cammask] + np.timedelta64(10,'h')).reset_index(drop=True)
# conemax = df_fheight['max cone height (m)'].to_numpy()
# conemask = np.isfinite(conemax)
# conemax = conemax[conemask]
# conemin = df_fheight['min cone height (m)'].to_numpy()
# conemin = conemin[conemask]
# dtime_cone = df_fheight['date/time (cone)'] + np.timedelta64(10, 'h')
# dtime_cone = dtime_cone[conemask].reset_index(drop=True)
#
# if tmid[-1] < dates.date2num(dtime_cam.iloc[-1]):
#     axx = ax[3].twinx()
#     axx.plot(dates.date2num(dtime_cam), fheight_cam.squeeze(), 'orange', linestyle='', marker='o', zorder=1, alpha=0.4, label='F8 height (camera)', markersize=4)
#     axx.plot(dates.date2num(dtime_manual), fheight_manual.squeeze(), 'orange', linestyle='', marker='*', zorder=2, alpha=0.8, label='F8 height (manual)', markeredgecolor='k')
#     axx.plot(dates.date2num(dtime_cone), conemax, color='orangered', label='F8 cone max')
#     axx.plot(dates.date2num(dtime_cone), conemin, linestyle='--', color='orangered', label='F8 cone min')
#     axx.set_ylim([0, 80])
#     axx.set_ylabel('Height [m]', color='orange')
#     axx.tick_params(axis='y', colors='orange')
#     axx.legend(loc='upper left', bbox_to_anchor=(1.1,0.4), borderaxespad=0.)

plt.show()

#%% calculate misfit spectrum
nf1 = 20
FREQ_vec = 10**(np.linspace(np.log10(0.0166),np.log10(25/10),nf1))#np.logspace(-2,2,7)np.log10(st_day[0].stats.sampling_rate / 2)
nf = 40
FREQ_vec_prob = 10 ** (np.linspace(np.log10(0.0166), np.log10(25), nf))
beam_all, tvec_all, P_mat, fpsd, norm_trf, tmid, M, sol_trf = sf.misfit_spectrum(st_day, FREQ_vec, FREQ_vec_prob, 121, peaks='bound', fwidth=1, model=['LSTFST', 'LST', 'FST'])

#%% plot misfit spectrum
M_diff = M[2,:,:] - M[1,:,:]
theshold = 2
M_diff = np.zeros(np.shape(M[2,:,:]))
M_diff[:] = np.NaN
M_diff[np.any([M[2,:,:] < theshold,M[1,:,:] < theshold], axis=0)] = M[2,:,:][np.any([M[2,:,:] < theshold,M[1,:,:] < theshold], axis=0)] - M[1,:,:][np.any([M[2,:,:] < theshold,M[1,:,:] < theshold], axis=0)]
M_all = np.concatenate((M,np.array([M_diff])),axis=0)#[M[0,:,:],M_LST,M_FST,M_diff

fig, ax = sf.misfit_spec_plot(P_mat, fpsd, tmid, M_all, FREQ_vec_prob, mid_point = 2, extend=2)
plt.show()
