from flask import Flask, render_template
import plotly.graph_objects as go
import pandas as pd
import json
import plotly
import plotly.express as px
from pvsync_sample import *
from plotly.subplots import make_subplots


app = Flask(__name__)

@app.route('/')
def main():
    (working_table,grad1LVP,grad2LVP,grad3LVP,grad1_peak_idx1,grad1_peak_idx2,grad2_peak_idx1,grad2_peak_idx2,
            grad3_peak_idx1,grad3_peak_idx2) = pressure_gradient_condition("HELIX20121102")

    (grad_data,edp_index) = calculate_valve_timing("HELIX20121102",working_table,grad1_peak_idx1,grad2_peak_idx1,grad2_peak_idx2)

    VOL, vol_hr = get_volume("HELIX20121102")
    ####################
    hr_p = 66
    hr_v = 78 # ovveride for now

    fig = go.Figure(layout_xaxis_range=[0,1500])
    # fig = make_subplots(rows=1, cols=2)
    for step in np.arange(40, 140, 1):
        STI_delta_correction = sti_correction(hr_p,step) #hr_p -> hr_v
        (sti_rescale_factor, dti_rescale_factor) = cardiac_timing_rescale(grad_data, STI_delta_correction,step,working_table,grad2_peak_idx2,edp_index)
        (new_lvp_time, new_lvp_val, new_sti_time, new_sti_val, ori_lvp) = rebuild_p_waveform(working_table,grad2_peak_idx2,edp_index,sti_rescale_factor, dti_rescale_factor)

        fig.add_trace(go.Scatter(x=new_lvp_time,y=new_lvp_val,visible=False))
        fig.add_trace(go.Scatter(x=[new_sti_time[-1]],y=[new_sti_val[-1]],marker=dict(size=16),visible=False))
        fig.add_trace(go.Scatter(x=[new_lvp_time[-1]],y=[new_lvp_val[-1]],marker=dict(size=16),visible=False))
        fig.add_trace(go.Scatter(x=VOL['time'],y=VOL['vol'],visible=True))


    fig.data[28].visible = True

    # Create and add slider
    steps = []
    # for i in range(len(fig.data)):
    for i in np.arange(0,len(fig.data),4):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "HR LVP: " + str(i/4+40) + "\nHR VOL: 78"}],  # layout attribute
            label=""
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i+1] = True
        step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i+3] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
    active=26,
    currentvalue={"prefix": "HR slider"},
    pad={"t": 60},
    steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        showlegend=False
        )

    fig.update_layout(
        xaxis_title="time (ms)",
        width=1000,
        height=1000,
        minreducedwidth=250,
        minreducedheight=250,
        font=dict(
            size=20
        )
    )

    
    graphJSON = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
    ####################



    


    hr_p = 66
    hr_v = 78 # ovveride for now

    STI_delta_correction = sti_correction(hr_p,hr_v) #hr_p -> hr_v
    (sti_rescale_factor, dti_rescale_factor) = cardiac_timing_rescale(grad_data, STI_delta_correction,hr_v,working_table,grad2_peak_idx2,edp_index)
    (new_lvp_time, new_lvp_val, new_sti_time, new_sti_val, ori_lvp) = rebuild_p_waveform(working_table,grad2_peak_idx2,edp_index,sti_rescale_factor, dti_rescale_factor)
    (sync_time, v_val, p_val) = match_res_pv(VOL,new_lvp_time,new_lvp_val)
    
    fig1 = go.Figure(layout_xaxis_range=[0,200])
    fig1.add_trace(
            go.Scatter(
                x=v_val,y=p_val,visible=False
                )
            )

    fig1.update_layout(
        xaxis_title="Volume (mL)",
        yaxis_title="Pressure (mmHg)",
        width=1000,
        height=1000,
        minreducedwidth=250,
        minreducedheight=250,
        updatemenus=[
            dict(
                buttons=list([
                dict(
                    args=["visible", True],
                    label="Build PV loop",
                    method="restyle"
                    ),
                    ]),
                type = "buttons",
                direction="right",
                pad={"r": 10, "t": 10,"b":10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.25,
                yanchor="top"
            )              
        ],
        font=dict(
            size=20
        )
    )




    graphJSON1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
  
    return render_template('main.html',
    graphJSON1=graphJSON1,
    graphJSON=graphJSON)



@app.route('/name')
def print_name():
    return "<p>This is pvsync</p>"

@app.route('/home/')
@app.route('/home/<name>')
def show_hello(name=None):
    return render_template('home.html',name=name)