from poc.visualize_smbo import visualize_smbo
from poc.generate_sample_tasks import generate_sample_tasks
from methods.nn_bo_synth import NeuralNetworkBOSynth
import random

gst = generate_sample_tasks()
X, y = gst.create_task()

num_evals = len(X)
print("Num evals in meta-dataset", num_evals)

# initial design indices
#num_initial_observations = 3
#initial_design_idxs = random.sample(range(num_evals), num_initial_observations)

initial_design_idxs = [250, 550, 850]

# create an smbo method
config = {'is_rank_version': True, 'eta': 0.01, 'optim_iters': 300, 'train_batch_size': 30,
          'acquisition_batch_size': 1000, 'log_iters': 300, 'hidden_layers_units': [16, 16, 16, 16],
          'use_batch_norm': False, 'use_dropout': False, 'dropout_rate': 0.0, 'beta': 7.0}

nn_bo = NeuralNetworkBOSynth(config=config)

# visualize SMBO
vs = visualize_smbo(smbo_method=nn_bo, X=X, y=y, init_idxs=initial_design_idxs, task_name="test")
vs.evaluate(n_trials=4)
