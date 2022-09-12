import os
import argparse
from benchmark_plot import BenchmarkPlotter
from methods.nn_bo_synth import NeuralNetworkBOSynth

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='data path')
args = parser.parse_args()

data_path = args.data_path
results_path = "results/"
output_path = "plots/"

experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP", "NBO"]
n_trials = 100


name = "nbo_vs_gp"
experiments = ["NBO", "GP"]
benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                     name=name,
                                     n_trials=n_trials,
                                     results_path=results_path,
                                     output_path=output_path,
                                     data_path=data_path)
benchmark_plotter.plot()


name = "nbo_vs_classic"
experiments = ["NBO", "Random", "BOHAMIANN", "DNGO"]
benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                     name=name,
                                     n_trials=n_trials,
                                     results_path=results_path,
                                     output_path=output_path,
                                     data_path=data_path)
benchmark_plotter.plot()


name = "nbo_vs_sota"
# add hebo here
experiments = ["NBO", "Random"]
benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                     name=name,
                                     n_trials=n_trials,
                                     results_path=results_path,
                                     output_path=output_path,
                                     data_path=data_path)
benchmark_plotter.plot()


name = "nbo_vs_transfer"
# add hebo here
experiments = ["NBO", "FSBO2", "TST", "RGPE"]
benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                     name=name,
                                     n_trials=n_trials,
                                     results_path=results_path,
                                     output_path=output_path,
                                     data_path=data_path)
benchmark_plotter.plot()