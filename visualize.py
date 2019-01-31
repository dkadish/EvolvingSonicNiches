from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


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

def plot_spectrum(spectra, view=False, vmin=None, vmax=None, filename='spectrum.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    spectra = np.array(spectra).T
    fig, ax = plt.subplots()
    p = ax.pcolormesh(spectra, cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(p,ax=ax)

    plt.title("Use of the communication spectrum by generation")
    plt.xlabel("Generations")
    plt.ylabel("Spectrum")
    plt.grid()
    # plt.legend(loc="best")

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
    ax2.fill_between(generation, loudness_avg - loudness_std, loudness_avg + loudness_std, facecolor='green', alpha=0.25)

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

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             prune_disconnected=False, node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
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

    inputs = set()
    last = None

    input_attrs = {'style': 'filled',
                   'shape': 'box'}
    with dot.subgraph(name='inputs', node_attr=input_attrs) as in_graph:
        in_graph.attr(rank='source')
        for k in config.genome_config.input_keys:
            name = node_names.get(k, str(k))
            inputs.add(k)
            attrs = {'fillcolor': node_colors.get(k, 'lightgray')}
            in_graph.node(name, _attributes=attrs)

            if last is not None:
                in_graph.edge(str(last), str(k), _attributes={'style': 'invis'})

            last = k

    outputs = set()
    last = None

    out_attrs = {'style': 'filled'}
    with dot.subgraph(name='outputs', node_attr=out_attrs) as out_graph:
        out_graph.attr(rank='sink')
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

            out_graph.node(name, _attributes=node_attrs)

            if last is not None:
                out_graph.edge(str(last), str(k), _attributes={'style': 'invis'})

            last = k

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input, output = cg.key
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

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
