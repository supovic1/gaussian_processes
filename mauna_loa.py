# Imports
import os
from itertools import islice
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm.notebook import tqdm
import bokeh
import bokeh.io
import bokeh.plotting
import bokeh.models
from IPython.display import display, HTML

bokeh.io.output_notebook(hide_banner=True)

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

np.random.seed(42)
tf.random.set_seed(42)


def load_data():
    # Load the data
    # Load the data from the Scripps CO2 program website.
    co2_df = pd.read_csv(
        # Source: https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv
        './monthly_in_situ_co2_mlo.csv',
        header=3,  # Data starts here
        skiprows=[4, 5],  # Headers consist of multiple rows
        usecols=[3, 4],  # Only keep the 'Date' and 'CO2' columns
        na_values='-99.99',  # NaNs are denoted as '-99.99'
        dtype=np.float64
    )

    # Drop missing values
    co2_df.dropna(inplace=True)
    # Remove whitespace from column names
    return co2_df.rename(columns=lambda x: x.strip(), inplace=True)

co2_df = load_data()

# Split the data into observed and to predict
date_split_predict = 2010
df_observed = co2_df[co2_df.Date < date_split_predict]
print('{} measurements in the observed set'.format(len(df_observed)))
df_predict = co2_df[co2_df.Date >= date_split_predict]
print('{} measurements in the test set'.format(len(df_predict)))
#



# Define mean function which is the means of observations
observations_mean = tf.constant(
    [np.mean(df_observed.CO2.values)], dtype=tf.float64)
mean_fn = lambda _: observations_mean
#


# Define the kernel with trainable parameters.
# Note we transform some of the trainable variables to ensure
#  they stay positive.

# Use float64 because this means that the kernel matrix will have
#  less numerical issues when computing the Cholesky decomposition

# Constrain to make sure certain parameters are strictly positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

# Smooth kernel hyperparameters
smooth_amplitude = tfp.util.TransformedVariable(
    initial_value=10., bijector=constrain_positive, dtype=np.float64,
    name='smooth_amplitude')
smooth_length_scale = tfp.util.TransformedVariable(
    initial_value=10., bijector=constrain_positive, dtype=np.float64,
    name='smooth_length_scale')
# Smooth kernel
smooth_kernel = tfk.ExponentiatedQuadratic(
    amplitude=smooth_amplitude,
    length_scale=smooth_length_scale)

# Local periodic kernel hyperparameters
periodic_amplitude = tfp.util.TransformedVariable(
    initial_value=5.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_amplitude')
periodic_length_scale = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_length_scale')
periodic_period = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_period')
periodic_local_length_scale = tfp.util.TransformedVariable(
    initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
    name='periodic_local_length_scale')
# Local periodic kernel
local_periodic_kernel = (
    tfk.ExpSinSquared(
        amplitude=periodic_amplitude,
        length_scale=periodic_length_scale,
        period=periodic_period) *
    tfk.ExponentiatedQuadratic(
        length_scale=periodic_local_length_scale))

# Short-medium term irregularities kernel hyperparameters
irregular_amplitude = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_amplitude')
irregular_length_scale = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_length_scale')
irregular_scale_mixture = tfp.util.TransformedVariable(
    initial_value=1., bijector=constrain_positive, dtype=np.float64,
    name='irregular_scale_mixture')
# Short-medium term irregularities kernel
irregular_kernel = tfk.RationalQuadratic(
    amplitude=irregular_amplitude,
    length_scale=irregular_length_scale,
    scale_mixture_rate=irregular_scale_mixture)

# Noise variance of observations
# Start out with a medium-to high noise
observation_noise_variance = tfp.util.TransformedVariable(
    initial_value=1, bijector=constrain_positive, dtype=np.float64,
    name='observation_noise_variance')

trainable_variables = [v.variables[0] for v in [
    smooth_amplitude,
    smooth_length_scale,
    periodic_amplitude,
    periodic_length_scale,
    periodic_period,
    periodic_local_length_scale,
    irregular_amplitude,
    irregular_length_scale,
    irregular_scale_mixture,
    observation_noise_variance
]]

#


kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)

# Define mini-batch data iterator
batch_size = 128

batched_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (df_observed.Date.values.reshape(-1, 1), df_observed.CO2.values))
    .shuffle(buffer_size=len(df_observed))
    .repeat(count=None)
    .batch(batch_size)
)
#


# Use tf.function for more efficient function evaluation
@tf.function(autograph=False, experimental_compile=False)
def gp_loss_fn(index_points, observations):
    """Gaussian process negative-log-likelihood loss function."""
    gp = tfd.GaussianProcess(
        mean_fn=mean_fn,
        kernel=kernel,
        index_points=index_points,
        observation_noise_variance=observation_noise_variance
    )

    negative_log_likelihood = -gp.log_prob(observations)
    return negative_log_likelihood


# Fit hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
batch_nlls = []  # Batch NLL for plotting
full_ll = []  # Full data NLL for plotting
nb_iterations = 10001
for i, (index_points_batch, observations_batch) in tqdm(
        enumerate(islice(batched_dataset, nb_iterations)), total=nb_iterations):
    # Run optimization for single batch
    with tf.GradientTape() as tape:
        loss = gp_loss_fn(index_points_batch, observations_batch)
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    batch_nlls.append((i, loss.numpy()))
    # Evaluate on all observations
    if i % 100 == 0:
        # Evaluate on all observed data
        ll = gp_loss_fn(
            index_points=df_observed.Date.values.reshape(-1, 1),
            observations=df_observed.CO2.values)
        full_ll.append((i, ll.numpy()))
#


# Plot NLL over iterations
fig = bokeh.plotting.figure(
    width=600, height=350,
    x_range=(0, nb_iterations), y_range=(50, 200))
fig.add_layout(bokeh.models.Title(
    text='Negative Log-Likelihood (NLL) during training',
    text_font_size="14pt"), 'above')
fig.xaxis.axis_label = 'iteration'
fig.yaxis.axis_label = 'NLL batch'
# First plot
fig.line(
    *zip(*batch_nlls), legend_label='Batch data',
    line_width=2, line_color='midnightblue')
# Seoncd plot
# Setting the second y axis range name and range
fig.extra_y_ranges = {
    'fig1ax2': bokeh.models.Range1d(start=130, end=250)}
fig.line(
    *zip(*full_ll), legend_label='All observed data',
    line_width=2, line_color='red', y_range_name='fig1ax2')
# Adding the second axis to the plot.
fig.add_layout(bokeh.models.LinearAxis(
    y_range_name='fig1ax2', axis_label='NLL all'), 'right')

fig.legend.location = 'top_right'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)
#




# Show values of parameters found
variables = [
    smooth_amplitude,
    smooth_length_scale,
    periodic_amplitude,
    periodic_length_scale,
    periodic_period,
    periodic_local_length_scale,
    irregular_amplitude,
    irregular_length_scale,
    irregular_scale_mixture,
    observation_noise_variance
]

data = list([(var.variables[0].name[:-2], var.numpy()) for var in variables])
df_variables = pd.DataFrame(
    data, columns=['Hyperparameters', 'Value'])
display(HTML(df_variables.to_html(
    index=False, float_format=lambda x: f'{x:.4f}')))
#

# Posterior GP using fitted kernel and observed data
gp_posterior_predict = tfd.GaussianProcessRegressionModel(
    mean_fn=mean_fn,
    kernel=kernel,
    index_points=df_predict.Date.values.reshape(-1, 1),
    observation_index_points=df_observed.Date.values.reshape(-1, 1),
    observations=df_observed.CO2.values,
    observation_noise_variance=observation_noise_variance)

# Posterior mean and standard deviation
posterior_mean_predict = gp_posterior_predict.mean()
posterior_std_predict = gp_posterior_predict.stddev()


# Plot posterior predictions

# Get posterior predictions
μ = posterior_mean_predict.numpy()
σ = posterior_std_predict.numpy()

# Plot
fig = bokeh.plotting.figure(
    width=600, height=400,
    x_range=(2010, 2021.3), y_range=(384, 418))
fig.xaxis.axis_label = 'Date'
fig.yaxis.axis_label = 'CO₂ (ppm)'
fig.add_layout(bokeh.models.Title(
    text='Posterior predictions conditioned on observations before 2010.',
    text_font_style="italic"), 'above')
fig.add_layout(bokeh.models.Title(
    text='Atmospheric CO₂ concentrations',
    text_font_size="14pt"), 'above')
fig.circle(
    co2_df.Date, co2_df.CO2, legend_label='True data',
    size=2, line_color='midnightblue')
fig.line(
    df_predict.Date.values, μ, legend_label='μ (predictions)',
    line_width=2, line_color='firebrick')
# Prediction interval
band_x = np.append(
    df_predict.Date.values, df_predict.Date.values[::-1])
band_y = np.append(
    (μ + 2*σ), (μ - 2*σ)[::-1])
fig.patch(
    band_x, band_y, color='firebrick', alpha=0.4,
    line_color='firebrick', legend_label='2σ')

fig.legend.location = 'top_left'
fig.toolbar.autohide = True
bokeh.plotting.show(fig)
#

