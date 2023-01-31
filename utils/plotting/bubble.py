import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def rescale(vals, new_min, new_max, num_points):
    if vals is None:
        vals = [new_min] * num_points
    else:
        arg_vals_range = np.max(vals) - np.min(vals)
        if arg_vals_range == 0.0:
            vals = [new_min] * num_points
        elif new_max - new_min == 0.0:
            vals = [new_min] * num_points
        else:
            vals = vals - np.min(vals)
            vals = vals / arg_vals_range
            vals = vals * (new_max - new_min)
            vals = vals + new_min
    return vals


def save_bubble(x_vals, y_vals, labels=None, sizes=None, logx=False,
                 logy=False, dir_path='./', file_name='bubble', ylabel='y-axis', title=None,
                 size_min=50, size_max=550, axis_fontsize=30, xtick_size=30, ytick_size=30, point_label_size=15, legend_size=30):

    sizes = rescale(sizes, size_min, size_max, len(x_vals))

    fig = px.scatter(
        #df.query("year==2007"),
        x=x_vals,
        y=y_vals,
        size=sizes,
        log_x=True,
        #size_max=size_max,
        #labels=labels,
        text=labels,
        opacity=[0.5]*len(x_vals),
        color=labels,

        #leg
    )
    fig.add_scatter(
        x=[np.max(x_vals)]*3,
        #x=np.max(x_vals),
        #y0=np.min(y_vals),
        #d0=
        y=[1,2,3],
        #size=[50, 100, 150]
        mode='markers+text',
        marker=dict(size=size_max, color='White'),
        text=['aaaa']*3,
        #size_max=size_max,
    )
    # fig = go.Figure()
    # fig.add_scatter(
    #     x=x_vals,
    #     y=y_vals,
    #     text=labels,
    #     textposition='top center',
    #     # log_x=True
    #     # go.Scatter(
    #     #     x=x_vals,
    #     #     y=y_vals,
    #     #     text=labels,
    #     #     textposition='top center',
    #     #     # log_x=True
    #     # )
    #)

    # fig.add_trace(
    #     go.Scatter(
    #         x=x_vals,
    #         y=y_vals,
    #         text=labels,
    #         textposition='top center',
    #         #log_x=True
    #     )
    #  )

    fig.update_traces(
        textfont=dict(size=30),
        textposition='middle center'
    )
    fig.update_xaxes(
        title=dict(
            font=dict(size=axis_fontsize),
            text='Floating point operations'
        )
    )
    fig.update_yaxes(
        title=dict(
            font=dict(size=axis_fontsize),
            text=ylabel
        )
    )
    fig.update_layout(
        yaxis=dict(
            tickfont=dict(size=ytick_size)
        ),
        xaxis=dict(
            tickfont=dict(size=xtick_size)
        ),
        showlegend=False,
        legend=dict(
            font=dict(size=legend_size),
            title=dict(
                font=dict(size=legend_size),
                text='Parameters'
            ),
            borderwidth=2,
            #xanchor='right',
            #x=0.5
        ),
        scattermode='overlay'
    )
    fig.show()

if __name__ == '__main__':
    x = np.array([0.5, 1, 1.1, 2, 20])*1e6
    y = [0.5, 1, 1.1, 2, 3]
    s = np.array([1, 2, 2.2, 3, 4])*100
    l = ['a', 'b', 'c', 'd' ,'e']
    save_bubble(x, y, labels=l, sizes=s, xtick_size=30, ytick_size=30, legend_size=30, axis_fontsize=30, size_max=300)
