from lib import *
import os 

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

dirname = os.path.dirname(__file__)

experiment_dir = os.path.join(dirname, 'runs')

ex = Experiment("ceng796")
ex.observers.append(FileStorageObserver(experiment_dir))
ex.add_config(configs)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    
    os.makedirs(os.path.join(experiment_dir, _run._id, "checkpoints"))
    os.makedirs(os.path.join(experiment_dir, _run._id, "results"))
    
    train(_config['model_params'], _config['hparams'], _run)
