import argparse
import random
from experiment import run_experiment
from plot_exps import plot
import utils
import torch

def main(config):
    torch.set_default_dtype(torch.float32)
    
    seed_sampler = random.Random()
    
    nexp = config.S
    del config.S

    logsfile = utils.logs_path(config)
    plotpath = utils.plot_path(config)

    for i in range(nexp):
        config.seed = seed_sampler.randint(0, 10000)
        run_experiment(config)
        plot(logsfile, plotpath)


########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    ######## Number of experiments ##########
    parser.add_argument('--S', type=int, help='Number of different experiments (different Seeds)')

    ######## EXPERIMENT'S PARAMETERS ########
    parser = utils.args_exp_parameters(parser)

    ######## NEW: Optional sample limit ########
    parser.add_argument('--samples', type=int, default=None,
                        help='Limit number of training samples used')
                        
    ######## NEW: Optional test set rotation ########
    parser.add_argument('--test_rotation_angle', type=float, default=None,
                        help='Rotate test set images by this fixed angle (e.g., 45, 90 degrees)')


    config = parser.parse_args()
    main(config)

