
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
tstart = obs.UTCDateTime('2018-6-16T15:00')
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
OVERLAP = 0.5
tstart = tstart_abs
tend = tstart + BEAM_WINDOW
n=0

while tend <= tend_abs:
    st = st_day.copy()
    st.trim(starttime=tstart, endtime=tend)
    beam, beamf, tvec, SPL, PSD, fpsd, sol_m, norm_m = sf.simil_fit(st, method='trf', model='LSTFSTcombined')
    beam, beamf, tvec, SPL, PSD, fpsd, sol_LST_m, norm_LST_m = sf.simil_fit(st, method='trf', model='LST')
    beam, beamf, tvec, SPL, PSD, fpsd, sol_FST_m, norm_FST_m = sf.simil_fit(st, method='trf', model='FST')
    if n == 0:
        beam_all = beam
        tvec_all = tvec
        SPL_all = SPL
        P_mat = np.array([PSD]).T
        sol_vec = np.array([sol_m])
        sol_LST_vec = np.array([sol_LST_m])
        sol_FST_vec = np.array([sol_FST_m])
        norm_vec = norm_m
        norm_LST_vec = norm_LST_m
        norm_FST_vec = norm_FST_m
        tmid = dates.date2num((tstart + BEAM_WINDOW/2).datetime)
    else:
        beam_all = np.append(beam_all, [beam])
        tvec_all = np.append(tvec_all, [tvec])
        SPL_all = np.append(SPL_all,SPL)
        P_mat = np.concatenate((P_mat, np.array([PSD]).T), axis=1)
        sol_vec = np.concatenate((sol_vec,np.array([sol_m])), axis=0)
        sol_LST_vec = np.concatenate((sol_LST_vec, np.array([sol_LST_m])), axis=0)
        sol_FST_vec = np.concatenate((sol_FST_vec, np.array([sol_FST_m])), axis=0)
        norm_vec = np.append(norm_vec, norm_m)
        norm_LST_vec = np.append(norm_LST_vec, norm_LST_m)
        norm_FST_vec = np.append(norm_FST_vec, norm_FST_m)
        tmid = np.append(tmid, dates.date2num((tstart + BEAM_WINDOW/2).datetime))
    tstart = tstart + BEAM_WINDOW * (1-OVERLAP)
    tend = tstart + BEAM_WINDOW
    n = n+1

print('Calculations are done.')


#%% plot similarity misfit etc.
norm_M = [norm_vec, norm_LST_vec, norm_FST_vec]
sol_M = [sol_vec, sol_LST_vec, sol_FST_vec]

fig,ax = sf.simil_plot(beam_all, tvec_all, SPL_all, P_mat, fpsd, tmid, norm_M, sol_lm=sol_M)

# #optional plotting of fountain height and cone height
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
nf = 150
FREQ_vec_prob = 10 ** (np.linspace(np.log10(0.0166), np.log10(25), nf))
beam_all, tvec_all, P_mat, fpsd, norm_trf, tmid, M, sol_trf = sf.misfit_spectrum(st_day, FREQ_vec, FREQ_vec_prob, 121, peaks='bound', fwidth=1)
beam_all, tvec_all, P_mat, fpsd, norm_LST_trf, tmid, M_LST, sol_LST_trf = sf.misfit_spectrum(st_day, FREQ_vec, FREQ_vec_prob, 121, peaks='bound', fwidth=1, model='LST')
beam_all, tvec_all, P_mat, fpsd, norm_FST_trf, tmid, M_FST, sol_FST_trf = sf.misfit_spectrum(st_day, FREQ_vec, FREQ_vec_prob, 121, peaks='bound', fwidth=1, model='FST')

#%% plot misfit spectrum
M_all = [M,M_LST,M_FST]

fig, ax = sf.misfit_spec_plot(P_mat, fpsd, tmid, M_all, FREQ_vec_prob, mid_point = 4)
plt.show()
#%% calculate a representative spectrum
st_short = st_day.copy()
st_short.trim(starttime=tstart_abs, endtime=tstart_abs + 10*60)
beam_rep, beamf_rep, tvec_rep, SPL_rep, PSD_rep, fpsd_rep, sol_m_rep, norm_m_rep = sf.simil_fit(st_short, freqmin=0.1, freqmax=1, baz=121, method='trf', model='poly2fun', peaks='variable')

#%% Plot
fig, ax = plt.subplots()
ax.plot(fpsd_rep, PSD_rep, 'k', linewidth=1)
ax.plot(fpsd, sf.simil_func(sol_trf[0,np.argmin(norm_trf[0,:]),:], fpsd, model='LSTFSTcombined'), color='g', label='LST&FST')
ax.plot(fpsd, sf.simil_func(sol_LST_trf[0,np.argmin(norm_LST_trf[0,:]),:], fpsd, model='LST'), color='r', label='LST')
ax.plot(fpsd, sf.simil_func(sol_FST_trf[0,np.argmin(norm_FST_trf[0,:]),:], fpsd, model='FST'), color='b', label='FST')
ax.plot(fpsd_rep, sf.simil_func(sol_m_rep, fpsd_rep, model='poly2fun'), color='orange', label='new Function')
ax.set_xscale('log')
ax.set_ylabel('PSD')
ax.set_xlabel('Frequency')
ax.legend()
ax.grid()
plt.show()
