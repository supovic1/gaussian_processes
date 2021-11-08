import math

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

train_x = np.array([[2.42192029845479, -3.40625498677908], [4.15400042880726, -2.69654093304395],
                      [2.32815799195191, -2.00416583442628], [1.13983846996822, -5],
                      [-1.80785467832001, -3.20785404046540], [2.44673584474322, 3.36911835780936]])
train_y = np.array([221.055625327954,
                      223.194970815307,
                      224.376806565064,
                      225.305410124018,
                      244.024493427411,
                      272.919367965527])

test_x = np.array([[-5.0000, -4.7074],
          [-3.8736, -5.0000],
          [-2.7772, -3.8903],
          [-0.7109, -5.0000],
          [-0.3060, -4.5989],
          [-2.9019, -1.7429],
          [-0.1797, -3.1396],
          [-1.5168, 0.6738],
          [0.6472, -1.1862],
          [5.0000, -3.1685],
          [3.2366, -1.7481],
          [1.3077, 0.9684]])

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=3, ard_num_dims=2)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

training_iter = 1500

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 100 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(test_x))

    # # Initialize plot
    # f, ax = plt.subplots(1, 1, figsize=(4, 3), projection='3d')
    #
    # # Get upper and lower confidence bounds
    # lower, upper = observed_pred.confidence_region()
    # # Plot training data as black stars
    # ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # # Plot predictive means as blue line
    # ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'bo')
    # # Shade between the lower and upper confidence bounds
    # # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # # ax.set_ylim([-3, 3])
    # # ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(train_x.numpy()[:,0], train_x.numpy()[:,1], train_y.numpy())
    # Plot predictive means as blue line
    ax.scatter(test_x.numpy()[:,0], test_x.numpy()[:,1], observed_pred.mean.numpy())
    plt.show()

