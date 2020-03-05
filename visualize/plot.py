from datetime import datetime
from subprocess import CalledProcessError

import numpy as np

import stats
from species import Species
from . import draw_net, plot_cohesion as pc, plot_spectrum as pspec, plot_message_spectrum as pms, plot_scores as ps, \
    plot_stats as pst, plot_species as psp

VIEW = False


def plot_scores(now, dirname, species_id, species, view=VIEW):
    # Visualize the scores
    decoding_scores = species.decoding_scores.get()
    species_gen_avg = []
    bits_gen_avg = []
    total_gen_avg = []
    species_gen_std = []
    bits_gen_std = []
    total_gen_std = []
    while decoding_scores is not False:
        # The decoding scores for an individual over an entire generation
        dc_species = decoding_scores['species']
        dc_bit = decoding_scores['bit']
        dc_total = decoding_scores['total']
        species_avg = []
        bits_avg = []
        total_avg = []

        for g in dc_species:
            species_avg.append(np.average(dc_species[g]))
            bits_avg.append(np.average(dc_bit[g]))
            total_avg.append(np.average(dc_total[g]))

        species_gen_avg.append(np.nanmean(species_avg))
        species_gen_std.append(np.nanstd(species_avg))
        bits_gen_avg.append(np.nanmean(bits_avg))
        bits_gen_std.append(np.nanstd(bits_avg))
        total_gen_avg.append(np.nanmean(total_avg))
        total_gen_std.append(np.nanstd(total_avg))

        decoding_scores = species.decoding_scores.get()

    ps(np.array(species_gen_avg), np.array(species_gen_std),
       np.array(bits_gen_avg), np.array(bits_gen_std),
       np.array(total_gen_avg), np.array(total_gen_std),
       view=view,
       filename='data/%s/%s-%i-scores.svg' % (
           dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))

    return {
        'species': np.array(species_gen_avg),
        'bits': np.array(bits_gen_avg),
        'total': np.array(total_gen_avg)
    }


def plot_cohesion(cohesion_stats, now, dirname, species_id, loudness_stats, species, view=VIEW):
    # Visualize the cohesion
    cohesion = [np.array(cohesion_stats.avg[species.species_id]), np.array(cohesion_stats.std[species.species_id])]
    loudness = [np.array(loudness_stats.avg[species.species_id]), np.array(loudness_stats.std[species.species_id])]
    pc(cohesion[0], cohesion[1], loudness[0], loudness[1], view=VIEW,
       filename='data/%s/%s-%i-message_cohesion.svg' % (
           dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))


def plot_message_spectrum(now: datetime, dirname: str, species_id: int, message_spectrum_stats: stats.MessageSpectrum,
                          species: Species, spectrum_stats: stats.Spectrum, vmin: float, view: bool = VIEW):
    '''

    :param now: Current timestamp, used for saving files.
    :param dirname:
    :param species_id:
    :param message_spectrum_stats:
    :param species:
    :param spectrum_stats:
    :param vmin:
    :param view:
    :return:
    '''
    # Visualize the spectra
    max_spectrum = max([np.max(np.array(spectrum_stats.spectra[s])) for s in spectrum_stats.spectra])
    spectra = spectrum_stats.spectra[species.species_id]
    pspec(spectra, view=view, vmin=vmin, vmax=max_spectrum,
          filename='data/%s/%s-%i-spectrum.svg' % (
              dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    message_spectra = message_spectrum_stats.spectra[species.species_id]
    message_spectra['total'] = np.average([message_spectra[message] for message in message_spectra], axis=0)
    pms(message_spectra, view=view, filename='data/%s/%s-%i-message_spectrum.svg' % (
        dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    return message_spectra


def plot_received_message_spectrum(now: datetime, dirname: str, species_id: int, message_spectrum_stats: stats.MessageSpectrum,
                          species: Species, spectrum_stats: stats.Spectrum, vmin: float, view: bool = VIEW):
    '''

    :param now: Current timestamp, used for saving files.
    :param dirname:
    :param species_id:
    :param message_spectrum_stats:
    :param species:
    :param spectrum_stats:
    :param vmin:
    :param view:
    :return:
    '''
    # Visualize the spectra
    max_spectrum = max([np.max(np.array(spectrum_stats.received_spectra[s])) for s in spectrum_stats.received_spectra])
    spectra = spectrum_stats.received_spectra[species.species_id]
    pspec(spectra, view=view, vmin=vmin, vmax=max_spectrum,
          filename='data/%s/%s-%i-received_spectrum.svg' % (
              dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id),
          title="Use of the communication spectrum (received) by generation")
    message_spectra = message_spectrum_stats.received_spectra[species.species_id]
    message_spectra['total'] = np.average([message_spectra[message] for message in message_spectra], axis=0)
    pms(message_spectra, view=view, filename='data/%s/%s-%i-received_message_spectrum.svg' % (
        dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id),
        title="Use of the communication spectrum (received) by generation and message")
    return message_spectra


def plot_networks(config_dec, config_enc, now, dirname, species_id, node_names_dec, node_names_enc, species, view=VIEW):
    try:
        draw_net(config_dec, species.decoder.population.best_genome, view=False,
                 prune_unused=True, show_disabled=False,
                 filename='data/%s/%s-%i-digraph_dec_pruned.gv' % (dirname,
                                                                   now.strftime('%y-%m-%d_%H-%M-%S'),
                                                                   species_id),
                 node_names=node_names_dec)
        draw_net(config_enc, species.encoder.population.best_genome, view=False,
                 prune_unused=True, show_disabled=False,
                 filename='data/%s/%s-%i-digraph_enc_pruned.gv' % (dirname,
                                                                   now.strftime('%y-%m-%d_%H-%M-%S'),
                                                                   species_id),
                 node_names=node_names_enc)
    except CalledProcessError as e:
        print(e)


def plot_stats(now, dirname, species_id, species, view=VIEW):
    pst(species.encoder.stats, view=view,
        filename='data/%s/%s-%i-avg_fitness_enc.svg' % (
            dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    psp(species.encoder.stats, view=view,
        filename='data/%s/%s-%i-speciation_enc.svg' % (
            dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    pst(species.decoder.stats, view=view,
        filename='data/%s/%s-%i-avg_fitness_dec.svg' % (
            dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
    psp(species.decoder.stats, view=view,
        filename='data/%s/%s-%i-speciation_dec.svg' % (
            dirname, now.strftime('%y-%m-%d_%H-%M-%S'), species_id))
