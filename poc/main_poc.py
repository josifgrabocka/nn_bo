from HPO_B.poc.visualize_smbo import visualize_smbo
from HPO_B.poc.generate_sample_tasks import generate_sample_tasks
from HPO_B.methods.nn_bo_synth import NeuralNetworkBOSynth
import random

gst = generate_sample_tasks()
X, y = gst.create_task()

num_evals = len(X)
print("Num evals in meta-dataset", num_evals)

# initial design indices
#num_initial_observations = 3
#initial_design_idxs = random.sample(range(num_evals), num_initial_observations)

initial_design_idxs = [250, 550, 850]

is_rank_version = True

# create an smbo method
config = {'is_rank_version': is_rank_version, 'eta': 0.01, 'optim_iters': 2000, 'train_batch_size': 100,
          'acquisition_batch_size': 1000, 'log_iters': 1000, 'hidden_layers_units': [16, 16, 16, 16],
          'use_batch_norm': True, 'use_dropout': True, 'dropout_rate': 0.2, 'alpha': 0.5, 'beta': 3.0, 'gamma': 5.0}
nn_bo = NeuralNetworkBOSynth(config=config)

# visualize SMBO
vs = visualize_smbo(smbo_method=nn_bo, X=X, y=y, init_idxs=initial_design_idxs, task_name="test")
vs.evaluate(n_trials=4)
