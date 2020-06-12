
import numpy as np
import obspy as obs
import simil_func as sf
from waveform_collection import read_local
from array_processing.tools import beamForm
from array_processing.algorithms.helpers import getrij
from matplotlib import dates
import matplotlib.pyplot as plt

filepath = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/data/mseed_usb/'
coord_file = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/programs/local_infra_coords.json'
network = 'HV'
fheight_path = '/Users/julia/Documents/UAF_Research/Jet_Noise_Project/data/2018LERZeruption_data.xlsx'

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
fig,ax = sf.simil_plot(beam_all, tvec_all, SPL, P_mat, fpsd, tmid, norm_trf=norm_trf, sol_trf=sol_trf, method='trf')