# README

## Project Structure

### Task 1: "pca/"
- model.py, energy.py - helpers.py implement pca
- showcase_PCA.ipynb - visualization and tests for the task 1 part 1
- picture_PCA.ipynb - visualization and tests for the task 1 part 2
- vadere_PCA.ipynb - visualization and tests for the task 1 part 3

### Task 2: "diffusion_maps/"
- model.py implements - diffusion maps
- fourier_DMAP.ipynb - visualization and tests for the task 2 part 1
- swissroll_DMAP.ipynb - visualization and tests for the task 2 part 2
- vadere_DMAP.ipynb - visualization and tests for the task 2 part 3
- datafold_DMAP.ipynb - visualization and tests for the task 2 part bonus
- test_DMAP.ipynb - test of inefficient and sparse diffusion map algorithm

### Task 3: "vae/"
- model.py - implements the VAE model
- engine.py - train and test the VAE model with the given hyperparameters
- data_setup.py - creates the dataloaders to be passed to the model while training and testing
- model_utils.py - implements some functions to be able to save a model and to save the best model of the whole training session
- visualization.py - generates all plots for task 3 i.e. reconstruction, generation, latent space and loss evolution
- vae.ipynb - notebook to train, test and evaluate the model with the plots. The user should only modify this file.

### Task 4: "fire_evac/"

* add_pedestrian_with_distribution.ipynb - add multiple pedestrians with the distribution p(x) to the Vedere scenario
* critical_number_estimation.ipynb - calculate the critical number and visualization
* fire_evacuation_planning.ipynb - task 4 implementation with a vae from task 3
* model_fire_evacuation.py - vae model for task 4
* visualise_FireEvac_dataset.ipynb - visualise all data points in the FireEvac_dataset.
