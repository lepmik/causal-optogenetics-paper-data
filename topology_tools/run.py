import click
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from plot import Plot
from animate import Animate
from data import Data
from simulator import Simulator, load_params
from tools import set_style

plt.rcParams.update({'figure.figsize': (16, 9)})

plots_with_function_name = [
    'pop-rate-spectrum', 'single-rate-spectrum', 'pop-rate', 'voltage',
    'psp', 'scatter-rate-2D', 'hist-rate-2D', 'mean-rate-1D'
]
plots = [
    'single-state', 'connectivity'
] + plots_with_function_name

animations = [
    'scatter-rate-2D', 'hist-rate-2D', 'voltage-2D'
]


@click.command()
@click.argument('params', type=click.Path(exists=True))
@click.option('--data', '-d', is_flag=True)
@click.option('--save-connections', is_flag=True)
@click.option('--start', type=click.FLOAT)
@click.option('--stop', type=click.FLOAT)
@click.option('--sampling_period', type=click.FLOAT, default=1.)
@click.option('--dt', type=click.FLOAT, default=2.)
@click.option('--no-state', is_flag=True)
@click.option('--show', is_flag=True)
@click.option('--yes', '-y', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--nosave', is_flag=True)
@click.option('--plot', '-p', multiple=True, type=click.Choice(plots))
@click.option('--animate', '-a', multiple=True, type=click.Choice(animations))
@click.option('--pop', multiple=True, type=click.Choice(['ex', 'in']),
              default=('ex',))
def run(**kw):
    save = not kw['nosave']
    if save:
        set_style('article')
    job_name, data_path, parameters = load_params(kw['params'])
    data_path = Path(data_path)
    parameters.update({
        'save_connections': kw['save_connections'],
        'run_keys': kw,
        'verbose': kw['verbose'],
        'sampling_period': kw['sampling_period'],
        'data_path': data_path,
        'fname': job_name})
    if not kw['data']:
        sim = Simulator(parameters)
        simulate = True
        if sim.data_fname.exists():
            if not kw['yes']:
                answer = input('Data exist, overwrite? ([Y]/n) ')
                if answer.lower() in ['n', 'no']:
                    simulate = False
        if simulate:
            for setup in parameters['setup']:
                getattr(sim, setup)()
            sim.set_spike_rec()
            if not kw['no_state']:
                sim.set_state_rec()
            else:
                need_state = ['denisty', 'marginals', 'top-state', 'top']
                if any(a in kw['animate'] + kw['plot'] for a in need_state):
                    raise ValueError(
                        'Do not use "--no-state" to plot any of ' +
                        '{}'.format(need_state))
            sim.simulate()
            sim.save_data(overwrite=True)

    data = Data(data_path=data_path, job_name=job_name)
    assert data['params']['sampling_period'] <= kw['dt']
    t_start = kw['start'] or 0.
    t_stop = kw['stop'] or float(data['params']['simtime'])
    duration = t_stop - t_start
    assert duration > 0
    pops = ['ex', 'in']
    plot = Plot(
        data_path=data_path, job_name=job_name, data=data,
        t_start=t_start, t_stop=t_stop)
    animate = Animate(
        data_path=data_path, job_name=job_name, data=data, dt=kw['dt'],
        t_start=t_start, t_stop=t_stop)
    path_name = data_path / job_name

    for key in kw['plot']:
        if key not in plots_with_function_name:
            continue
        fig, (ax1, ax2) = plt.subplots(1,2)
        for ax, pop in zip([ax1, ax2], pops):
            getattr(plot, key.replace('-','_'))(
                pop=pop, ax=ax, save=False, show=kw['show'])
            ax.set_title(pop)
        if save:
            fig.savefig(
                str(data_path / (key + '.png')), dpi=300, bbox_inches='tight')


    if 'connectivity' in kw['plot']:
        plot.connectivity(show=kw['show'], save=save)

    ###########################################################################
    ######                          ANIMATE                             #######
    ###########################################################################

    for key in kw['animate']:
        key = key.replace('-', '_')
        if not hasattr(animate, key):
            continue
        _, fname = animate.generate_path(key, save=save)
        fig, axs = plt.subplots(1, 2)
        updates = []
        for ax, pop in zip(axs, ('ex', 'in')):
            updates.append(getattr(animate, key)(
                pop=pop, ax=ax, multiple=True))
            ax.set_title(pop)
        animate.updatefig(
            fname=fname, save=save, show=kw['show'],
            updates=updates)

if __name__ == '__main__':
    run()
