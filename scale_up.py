import sys
#sys.path.insert(0, '/home/ludeep/Desktop/PopGen/NanoFlow-master/image_density_experiments/')
import numpy as np
import torch
#from model_nanoflow import NanoFlow


# Create a fake dataset of effect sizes of N traits and M snps
N = 1000 # num traits
M = 1000000 # num SNPS

# Create true effects
true_effects = torch.distributions.LogNormal(0, 1).sample((N, M)) 
torch.save(true_effects,'true_effects.pkl')

# Half-Cauchy distribution for standard-errors 
# std_error = torch.distributions.half_cauchy.HalfCauchy(torch.tensor[1.0]).sample((N,M))

# Generate Correlation matrix via LJK prior (M x M)
