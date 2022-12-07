import pandas as pd
import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.optimize import leastsq
from numpy import exp, cos

def get_volume(i):
    VOL = pd.read_csv('./data/'+i+'/'+i+'_vol.csv')
    vol_hr = np.round(1000/VOL.iloc[-1,0]*60)
    return VOL, vol_hr

def pressure_gradient_condition(i):
    working_table = pd.read_csv('./data/'+i+'/pressure_'+i+'.csv',header=None)

    grad1LVP = signal.savgol_filter(working_table.iloc[:,1],35,4,deriv =1)
    grad2LVP = signal.savgol_filter(working_table.iloc[:,1],25,4,deriv =2)
    grad3LVP = signal.savgol_filter(working_table.iloc[:,1],35,4,deriv =3)

    grad1_peak_idx1,_=find_peaks(grad1LVP,height=.8*max(grad1LVP)) # max
    grad1_peak_idx2,_=find_peaks(-grad1LVP,height=.8*max(grad1LVP)) # min

    grad2_peak_idx1,_=find_peaks(grad2LVP,height=.08*max(grad2LVP),distance=35) # max
    grad2_peak_idx2,_=find_peaks(-grad2LVP,height=.1*max(grad2LVP),distance=51) # min

    grad3_peak_idx1,_=find_peaks(grad3LVP,height=.1*max(grad3LVP),distance=35) # max
    grad3_peak_idx2,_=find_peaks(-grad3LVP,height=.07*max(grad3LVP),distance=25) # min
        
        
    return (working_table,grad1LVP,grad2LVP,grad3LVP,grad1_peak_idx1,grad1_peak_idx2,grad2_peak_idx1,grad2_peak_idx2,
            grad3_peak_idx1,grad3_peak_idx2)



def calculate_valve_timing(i,working_table,grad1_peak_idx1,grad2_peak_idx1,grad2_peak_idx2):
    grad_data = pd.DataFrame({
    'ID':[i],
    'grad_AO_open':[np.nan],
    'grad_AO_close':[np.nan],
    'grad_MI_open':[np.nan],
    'grad_preA':[np.nan],
    'grad_MI_close':[np.nan],
    })

    # ao open prediction
    grad_data.loc[0,grad_data.columns[1]] = working_table.iloc[grad2_peak_idx2[0],0]
    # ao close prediction
    grad_data.loc[0,grad_data.columns[2]] = working_table.iloc[grad2_peak_idx2[1],0]
    
    # mi open prediction
    for zz in range(len(grad2_peak_idx1)):
        if working_table.iloc[grad2_peak_idx1[zz],0] > working_table.iloc[grad2_peak_idx2[1],0]:
            grad_data.loc[0,grad_data.columns[3]] = working_table.iloc[grad2_peak_idx1[zz],0]
            break
    
    # mi close prediction
    # using peak curvature
    end_curve_idx = grad1_peak_idx1[1]
    start_curve_idx = (end_curve_idx-25) #100 ms before next dpdtmax
    
    x_t = np.gradient(working_table.iloc[start_curve_idx:end_curve_idx,0])
    y_t = np.gradient(working_table.iloc[start_curve_idx:end_curve_idx,1])
    vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])
    speed = np.sqrt(x_t*x_t + y_t*y_t)
    tangent = np.array([1/speed] * 2).transpose() * vel
    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)
    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    idx_curve_max = np.argmax(curvature_val)
    
    grad_data.loc[0,grad_data.columns[5]] = working_table.iloc[start_curve_idx+idx_curve_max,0]
    
    # try to fix edp error temporarily
    low_p_val_thres = np.where(working_table.iloc[grad2_peak_idx2[1]:(start_curve_idx+idx_curve_max),1] <= working_table.iloc[0,1])
    index_limit_edp = grad2_peak_idx2[1]+low_p_val_thres[-1]
    if abs(working_table.iloc[(start_curve_idx+idx_curve_max),1] - working_table.iloc[(index_limit_edp[-1]),1]) < 20:
        print('Please fix this!')
        edp_index = index_limit_edp[-1]
    else:
        edp_index = start_curve_idx+idx_curve_max

    return grad_data,edp_index

def sti_correction(hr_p,hr_v):
    STI_predict = -1.46*hr_v + 432.7
    STI_delta_correction = STI_predict - (-1.46*hr_p + 432.7)

    return STI_delta_correction

def cardiac_timing_rescale(grad_data, STI_delta_correction,hr_v,working_table,grad2_peak_idx2,edp_index):
    sync_sti = grad_data['grad_AO_close'] + STI_delta_correction
    sync_dti = 60/hr_v*1000 - sync_sti

    sti_rescale_factor = sync_sti/working_table.iloc[grad2_peak_idx2[1],0]
    dti_rescale_factor = sync_dti/(working_table.iloc[(edp_index),0]-working_table.iloc[grad2_peak_idx2[1],0])

    return sti_rescale_factor, dti_rescale_factor

def rebuild_p_waveform(working_table,grad2_peak_idx2,edp_index,sti_rescale_factor, dti_rescale_factor):
    # rescale STI
    new_sti_time = working_table.iloc[:grad2_peak_idx2[1],0]*sti_rescale_factor.values
    new_sti_val = working_table.iloc[:grad2_peak_idx2[1],1]
    
    # rescale DTI
    new_dti_time = (working_table.iloc[grad2_peak_idx2[1]:(edp_index),0]*dti_rescale_factor.values - 
                    (working_table.iloc[grad2_peak_idx2[1],0]*dti_rescale_factor.values - new_sti_time.iloc[-1]))
    new_dti_time = new_dti_time.iloc[1:] # remove duplicate initial point
    
    new_dti_val = working_table.iloc[grad2_peak_idx2[1]:(edp_index),1]
    new_dti_val = new_dti_val.iloc[1:] # remove duplicate initial point
    
    # append new timing to one array
    new_lvp_time = np.array(pd.concat([new_sti_time,new_dti_time]))
    new_lvp_val = np.array(pd.concat([new_sti_val,new_dti_val]))
    print(sti_rescale_factor.values)

    return new_lvp_time, new_lvp_val, np.array(new_sti_time), np.array(new_sti_val), working_table.iloc[:edp_index,:]


def match_res_pv(VOL,new_lvp_time,new_lvp_val):
    correction_factor = -1
    VOL_val_shift = np.array(VOL.iloc[correction_factor:,1])
    VOL_val_shift = np.concatenate([VOL_val_shift,VOL.iloc[:correction_factor,1]])
    VOL['vol'] = VOL_val_shift
        
    if VOL.iloc[0,0] == 0:
        VOL.iloc[-1,1] = VOL.iloc[0,1]
        print('ok')
    elif VOL.iloc[0,1]<VOL.iloc[-1,1]:
        temp_vol = pd.DataFrame({
            'time':[0],
            'vol':VOL.iloc[-1,1]
        })
        VOL = pd.concat([temp_vol,VOL])
    else:
        VOL.iloc[-1,1] = VOL.iloc[0,1]
        temp_vol = pd.DataFrame({
            'time':[0],
            'vol':VOL.iloc[0,1]
        })
        VOL = pd.concat([temp_vol,VOL])
        
    VOL = VOL.reset_index(drop=True)

    
    
    # interpolate volume to be smoother
    x = VOL['time']
    y = VOL['vol']
    tck = interpolate.splrep(x, y, s=2)
    xnew = np.arange(VOL.iloc[0,0],VOL.iloc[-1,0],30)
    ynew = interpolate.splev(xnew, tck, der=0)
    
    
    #interpolate pressure data
    x1 = new_lvp_time
    y1 = new_lvp_val
    tck = interpolate.splrep(x1, y1, s=0)
    ynew1 = interpolate.splev(xnew, tck, der=0)

    ynew = np.concatenate([ynew,[ynew[0]]])
    ynew1 = np.concatenate([ynew1,[ynew1[0]]])

    return xnew, ynew, ynew1