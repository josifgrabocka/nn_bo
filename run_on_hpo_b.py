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
name = "nn_bo_benchmark"
surrogate = "nn_bo"
#experiments = ["Random", "FSBO", "TST", "DGP", "RGPE" , "BOHAMIANN", "DNGO", "TAF", "GP"]
experiments = ["Random"]

n_trials = 100

method = NeuralNetworkBOSynth

benchmark_plotter = BenchmarkPlotter(experiments=experiments,
                                     name=name,
                                     n_trials=n_trials,
                                     results_path=results_path,
                                     output_path=output_path,
                                     data_path=data_path)

benchmark_plotter.generate_results(method,
                                   n_trials,
                                   results_file=results_path+os.sep+surrogate+".json")
benchmark_plotter.plot()
