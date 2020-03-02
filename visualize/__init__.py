from __future__ import print_function

import copy
import warnings
from functools import reduce
from math import inf

import graphviz
import joblib
import matplotlib.pyplot as plt
import numpy as np


def message_sort_key(message):
    if type(message) == str:
        return inf

    number = reduce(lambda a, b: (a << 1) + int(b), message)
    return number


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def plot_spectrum(spectra, cmap='rainbow', view=False, vmin=None, vmax=None,
                  filename='spectrum.svg', title="Use of the communication spectrum by generation"):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    spectra = np.array(spectra).T
    fig, ax = plt.subplots()
    p = ax.pcolormesh(spectra[:, :-1], cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(p, ax=ax)

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Spectrum")
    plt.grid()
    # plt.legend(loc="best")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_message_spectrum(spectra: dict, view: bool = False, vmin: float = None, vmax: float = None,
                          filename: str = 'spectrum.svg'):
    '''Plots the population's average and best fitness.

    :param spectra:
    :param view:
    :param vmin: Minimum value for the colourbar range
    :param vmax: Maximum value for the colourbar range
    :param filename:
    :return:
    '''

    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    if vmin is None:
        vmin = 0

    if vmax is None:
        vmax = max([np.max(spectra[m]) for m in spectra])

    fig, axarr = plt.subplots(4, 2, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)

    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Generation', fontsize='small')
    plt.ylabel('Spectrum', fontsize='small')

    messages = sorted(spectra.keys(), key=message_sort_key)

    for i, (ax, message) in enumerate(zip(axarr.flat, messages)):
        spectrum = np.array(spectra[message]).T
        p = ax.pcolormesh(spectrum[:, :-1], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax.set_title(message, fontsize='x-small')

        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.tick_params(labelsize='xx-small')
        ax.label_outer()

    fig.subplots_adjust(hspace=0.4)
    cb = fig.colorbar(p, ax=axarr.flat)
    cb.ax.tick_params(labelsize='xx-small')

    fig.suptitle("Use of the communication spectrum by generation and message")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_cohesion(cohesion_avg, cohesion_std, loudness_avg, loudness_std, view=False, filename='message_cohesion.svg'):
    """ Plots the average distance between the same message for each generation. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(cohesion_avg))
    cohesion_array = np.array(cohesion_avg)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(generation, cohesion_array, 'b-', label="overall")
    ax1.fill_between(generation, cohesion_avg - cohesion_std, cohesion_avg + cohesion_std, facecolor='blue', alpha=0.25)
    ax2.plot(generation, loudness_avg, 'g-', label="loudness")
    ax2.fill_between(generation, loudness_avg - loudness_std, loudness_avg + loudness_std, facecolor='green',
                     alpha=0.25)

    plt.title("Message Difference and Loudness")
    plt.xlabel("Generations")
    ax1.set_ylabel("Distance")
    ax2.set_ylabel("Loudness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_clustering(ch, silhouette, archive, view=False, filename='clustering_stats'):
    """ Plots the silhouette and calinski-harabaz scores for the messages within and between each species. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    for species in archive:
        n_plots = len(archive[species])
        for i, arch in enumerate(archive[species]):
            ax = plt.subplot2grid((2, n_plots + 1), (0, i))

            # TODO This doesn't work for ALL and labels is still broken.
            indices = [np.array(arch.labels) == b for b in set(arch.labels)]
            # colours = np.sum(np.array(archive[0][0].labels) << np.arange(2,-1,-1), axis=1)
            # labels = np.array2string(colours, formatter={'int':lambda x: '{:03b}'.format(x)})

            # TODO Try the legand((LABELS),(STUFF)... format
            axes = []

            for b in set(arch.numeric_labels):
                index = np.array(arch.numeric_labels) == b
                # print(index.shape, arch.two_dimensional[index,:].shape)
                # b_int = b if type(b) == int else sum([b[j] << j for j in range(3)])
                axes.append(ax.scatter(arch.two_dimensional[index, 0], arch.two_dimensional[index, 1], s=10, alpha=0.75,
                                       edgecolors='none'))

            ax.set_title(arch.generation)
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_major_formatter(plt.NullFormatter())

            if i == len(archive[species]) - 1:
                ax.legend(axes, ['{:03b}'.format(b) for b in set(arch.numeric_labels)], bbox_to_anchor=(1.05, 1),
                          borderaxespad=0., loc='upper left', fontsize='small', fancybox=True)

        ax = plt.subplot2grid((2, n_plots + 1), (1, 0), colspan=n_plots + 1)
        generation = range(len(ch[species]))

        if type(species) == int:
            ax.plot(generation, silhouette[species], label='Whole', c='k')
            for i in range(3):
                ax.plot(generation, silhouette['{}.{}'.format(species, i)], label=i)
        else:
            ax.plot(generation, silhouette[species], label=species, c='k')

        # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        # cycler = plt.cycler(c=['b','g','r'])
        #
        # for species, kwargs in zip(ch,cycler):
        #     generation = range(len(ch[species]))
        #     ax1.plot(generation, ch[species], label=species, **kwargs)
        #     ax2.plot(generation, silhouette[species], label=species, **kwargs)

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle("Clustering of Messages in {}".format(species))
        plt.xlabel("Generations")
        # ax1.set_ylabel("CH Score")
        ax.set_ylabel("Silhouette Score")
        # ax1.grid()
        ax.grid()

        if type(species) == int:
            plt.legend(loc="best")

        plt.savefig('{}-{}.svg'.format(filename, species))
        if view:
            plt.show()

        plt.close()


def plot_scores(species_avg, species_std, bits_avg, bits_std, total_avg, total_std, view=False, filename='scores.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(species_avg))

    plt.plot(generation, species_avg, 'b-', label="species")
    # plt.plot(generation, species_avg - species_std, 'g-.', label="-1 sd")
    # plt.plot(generation, species_avg + species_std, 'g-.', label="+1 sd")
    plt.fill_between(generation, species_avg - species_std, species_avg + species_std, facecolor='blue', alpha=0.5)

    plt.plot(generation, bits_avg, 'r-', label="bits")
    # plt.plot(generation, bits_avg - bits_std, 'g-.', label="-1 sd")
    # plt.plot(generation, bits_avg + bits_std, 'g-.', label="+1 sd")
    plt.fill_between(generation, bits_avg - bits_std, bits_avg + bits_std, facecolor='red', alpha=0.5)

    plt.plot(generation, total_avg, 'g-', label="total")
    # plt.plot(generation, total_avg - total_std, 'g-.', label="-1 sd")
    # plt.plot(generation, total_avg + total_std, 'g-.', label="+1 sd")
    plt.fill_between(generation, total_avg - total_std, total_avg + total_std, facecolor='green', alpha=0.5)

    plt.title("Scores by generation")
    plt.xlabel("Generations")
    plt.ylabel("Scores")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_n_channels(channels, view=False, filename='n_channels.pdf'):
    channels = channels[:-1]

    generations = range(channels.shape[0])
    n_channels = []

    for threshold in range(10, 40):
        n_channels.append(np.count_nonzero(channels > threshold, axis=1))

    n_channels = np.array(n_channels)
    avg = np.average(n_channels, axis=0)
    std = np.std(n_channels, axis=0)

    plt.plot(generations, avg)
    plt.fill_between(generations, avg - std, avg + std, alpha=0.2)

    plt.title("Number of channels used")
    plt.xlabel("Generations")
    plt.ylabel("Channels")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             prune_disconnected=False, node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    hidden_colours = {
        'sigmoid': 'lightblue',
        'relu': 'lightcoral',
        'softplus': 'khaki'
    }

    out_colours = {
        'sigmoid': 'lightskyblue',
        'relu': 'lightsalmon',
        'softplus': 'palegoldenrod'
    }

    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    ######### INPUTS #########
    inputs = set()
    last = None

    input_attrs = {'style': 'filled',
                   'shape': 'box'}
    with dot.subgraph(name='inputs', node_attr=input_attrs) as in_graph:
        in_graph.attr(rank='source')
        for k in config.genome_config.input_keys:
            name = node_names.get(k, str(k))
            inputs.add(name)
            attrs = {'fillcolor': node_colors.get(k, 'lightgray')}
            in_graph.node(name, _attributes=attrs)

            if last is not None:
                in_graph.edge(str(last), str(name), _attributes={'style': 'invis'})

            last = name

    ######### OUTPUTS #########
    outputs = set()
    last = None

    out_attrs = {'style': 'filled'}
    with dot.subgraph(name='outputs', node_attr=out_attrs) as out_graph:
        out_graph.attr(rank='sink')
        for k in config.genome_config.output_keys:
            name = node_names.get(k, str(k))
            outputs.add(name)
            # node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
            node_attrs['fillcolor'] = out_colours.get(genome.nodes[k].activation, 'whitesmoke')

            out_graph.node(name, _attributes=node_attrs)

            if last is not None:
                out_graph.edge(str(last), str(name), _attributes={'style': 'invis'})

            last = name

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input, output = [node_names[k] if k in node_names else k for k in cg.key]
                connections.add((input, output))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    ######### HIDDEN #########
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        fillcolor = hidden_colours.get(genome.nodes[n].activation, 'white')
        attrs = {'style': 'filled',
                 'fillcolor': fillcolor}
        dot.node(str(n), label='{}\n{}'.format(n, genome.nodes[n].activation[:4]), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = [node_names[k] if k in node_names else k for k in cg.key]
            if input not in used_nodes or output not in used_nodes:
                continue
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(str(input), str(output), _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test graphing functions.')
    parser.add_argument('datafile', help='File containing joblibed run data.')

    subparsers = parser.add_subparsers(help='sub-command help')
    clustering = subparsers.add_parser('cluster', help='Test Cluster plotting')
    clustering.set_defaults(func=plot_clustering)
    clustering.set_defaults(params=['ch', 'silhouette', 'archive'], kwargs={})

    spectrum = subparsers.add_parser('spectrum', help='Test spectral plotting')
    spectrum.set_defaults(func=plot_spectrum)
    get_spectrum = lambda d: next(iter(d['message_spectra'].values()))['total']
    spectrum.set_defaults(params=[get_spectrum], kwargs={'cmap': 'RdPu', 'filename': 'spectrum.pdf'})

    message_spectra = subparsers.add_parser('mspec', help='Test message spectra plotting')
    message_spectra.set_defaults(func=plot_message_spectrum)
    get_message_spectra = lambda d: d['message_spectra'][0]
    message_spectra.set_defaults(params=[get_message_spectra], kwargs={})

    message_channels = subparsers.add_parser('n_channels', help='Plot N channels')
    message_channels.set_defaults(func=plot_n_channels)
    get_message_channels = lambda d: next(iter(d['message_spectra'].values()))['total']
    message_channels.set_defaults(params=[get_message_channels], kwargs={})

    arguments = parser.parse_args()

    data = joblib.load(arguments.datafile)
    args = []
    for a in arguments.params:
        if callable(a):
            args.append(a(data))
        else:
            args.append(data[a])

    kwargs = arguments.kwargs

    arguments.func(*args, **kwargs)
