
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import obspy as obs
import simil_func as sf
import scipy
from waveform_collection import read_local
from array_processing.tools import beamForm
from array_processing.algorithms.helpers import getrij
from matplotlib import dates
import matplotlib.pyplot as plt
from datetime import datetime
from array_processing.tools.plotting import array_plot
import pandas as pd
import numpy.matlib

filepath = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/data/mseed_usb/'
coord_file = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/programs/local_infra_coords.json'
network = 'HV'
fheight_path = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/data/2018LERZeruption_data.xlsx'

#%%
df_fheight = pd.read_excel(fheight_path, engine='openpyxl')
df_manual = df_fheight[['date & time (manual)','max fountain height (m) (manual)']].sort_values('date & time (manual)')
fheight_manual = df_manual['max fountain height (m) (manual)']
manmask = np.isfinite(fheight_manual)
fheight_manual = fheight_manual.loc[manmask].reset_index(drop=True)
dtime_manual = df_manual['date & time (manual)']
dtime_manual = (dtime_manual.loc[manmask] + np.timedelta64(10,'h')).reset_index(drop=True)
fheight_cam = df_fheight['fountain height (m) (cam)'].to_numpy()
cammask = np.where(fheight_cam > 0.)
fheight_cam = fheight_cam[cammask]
dtime_cam = (df_fheight['date (cam)'].loc[cammask] + np.timedelta64(10,'h')).reset_index(drop=True)
conemax = df_fheight['max cone height (m)'].to_numpy()
conemask = np.isfinite(conemax)
conemax = conemax[conemask]
conemin = df_fheight['min cone height (m)'].to_numpy()
conemin = conemin[conemask]
dtime_cone = df_fheight['date/time (cone)'] + np.timedelta64(10,'h')
dtime_cone = dtime_cone[conemask].reset_index(drop=True)

#%% starting
tstart = obs.UTCDateTime('2018-6-16T16:00')
tstart_abs = tstart
tend = tstart + 24*60*60
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
n=0

while tend <= tend_abs:
    st = st_day.copy()
    st.trim(starttime=tstart, endtime=tend)
    beam_temp, beamf_temp, tvec_temp, SPL_temp, PSD_temp, fpsd, sol_mat_temp, sol_trf_mat_temp, norm_m_temp, norm_trf_temp = sf.simil_fit(st, method='lm&trf')
    if n == 0:
        beam_all = beam_temp
        tvec_all = tvec_temp
        SPL = SPL_temp
        P_mat = np.array([PSD_temp]).T
        sol_vec = sol_mat_temp
        sol_trf = sol_trf_mat_temp
        norm_m = norm_m_temp
        norm_trf = norm_trf_temp
        tmid = dates.date2num((tstart + BEAM_WINDOW/2).datetime)
    else:
        beam_all = np.append(beam_all, [beam_temp])
        tvec_all = np.append(tvec_all, [tvec_temp])
        SPL = np.append(SPL,SPL_temp)
        P_mat = np.concatenate((P_mat, np.array([PSD_temp]).T), axis=1)
        sol_vec = np.concatenate((sol_vec,sol_mat_temp), axis=2)
        sol_trf = np.concatenate((sol_trf,sol_trf_mat_temp), axis=2)
        norm_m = np.concatenate((norm_m, norm_m_temp), axis=1)
        norm_trf = np.concatenate((norm_trf, norm_trf_temp), axis=1)
        tmid = np.append(tmid, dates.date2num((tstart + BEAM_WINDOW/2).datetime))
    tstart = tstart + BEAM_WINDOW * (1-OVERLAP)
    tend = tstart + BEAM_WINDOW
    n = n+1

print('Calculations are done.')
#%%
fig,ax = sf.simil_plot(beam_all[:10000], tvec_all[:10000], SPL, P_mat, fpsd, tmid, norm_m=norm_m, norm_trf=norm_trf, method='lm&trf')

if tmid[-1] < dates.date2num(dtime_cam.iloc[-1]):
    axx = ax[3].twinx()
    axx.plot(dates.date2num(dtime_cam),fheight_cam.squeeze(),'orange',linestyle='',marker='o',zorder=1,alpha=0.4,label='F8 height (camera)',markersize=4)
    axx.plot(dates.date2num(dtime_manual),fheight_manual.squeeze(),'orange',linestyle='',marker='*',zorder=2,alpha=0.8,label='F8 height (manual)',markeredgecolor='k')
    axx.plot(dates.date2num(dtime_cone),conemax,color='orangered',label='F8 cone max')
    axx.plot(dates.date2num(dtime_cone),conemin,linestyle='--',color='orangered',label='F8 cone min')
    axx.set_ylim([0,80])
    axx.set_ylabel('Height [m]',color='orange')
    axx.tick_params(axis='y', colors='orange')
    axx.legend(loc='upper left', bbox_to_anchor=(1.1,0.4), borderaxespad=0.)
#fig.savefig('/Users/julia/Documents/UAF_Research/Jet_Noise_Project/figures/LST_FST_misfit' + str(tstart_abs).replace(':','_')[:10] + '.png', bbox_inches='tight')

#%%
freqmin = 0.3
freqmax = 10
Pmax = np.max(np.max(P_mat[np.all([fpsd<freqmax,fpsd>freqmin],axis=0),:]))
fig,ax = plt.subplots(1,1)
for iP in range(len(tmid)):
    pD = ax.plot(fpsd,P_mat[:,iP],'k',alpha=0.02)
for iP in range(len(tmid)):
    pL = ax.plot(fpsd,sf.simil_LST_func(sol_vec[:,1,iP],fpsd),'r',alpha=0.02)
for iP in range(len(tmid)):
    pF = ax.plot(fpsd,sf.simil_FST_func(sol_vec[:,2,iP],fpsd),'b',alpha=0.02)
for iP in range(len(tmid)):
    pA = ax.plot(fpsd,sf.simil_func(sol_vec[:,0,iP],fpsd),'g',alpha=0.02)
ax.set_xlabel('Frequency [Hz]')
ax.set_xscale('log')
ax.set_xlim([0.03,20])
YLIM = [Pmax-50,Pmax]
ax.set_ylim(YLIM)
ax.plot([freqmin,freqmin],YLIM,'k--')
ax.plot([freqmax,freqmax],YLIM,'k--')
#plt.legend([pD,pA,pL,pF],('Data','LST & FST', 'LST','FST'))
ax.set_ylabel('Spectrum [dB]')
ax.grid()
ax.set_title(str(tstart_abs)[:10])
fig.savefig('/Users/julia/Documents/UAF_Research/Jet_Noise_Project/figures/spectrum_density_' + str(tstart_abs).replace(':','_')[:10] + '.png', bbox_inches='tight')


