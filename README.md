# forex_rl

## Directory Structure
- config: default YAML config containing all the parameters that can be tuned
- models: details of the model runs contained in the following files:
	- &lt;model time stamp&gt;\_config.yaml: contains the configuration of the model run
	- &lt;model time stamp&gt;\_metrics.yaml: contains the test set metrics
- reference: some random reference pdfs
- scripts: random scripts e.g. run analysis on all the model/*metrics.yaml files
- src: main code, this contains these important files:
	- fx_env.py: defines the trading environment using a subclass of gym.ENV
	- fx_model.py: defines the deep learning model and memory classes
	- helpers.py: defines helper functions
	- indicators.py: defines functions for calculating technical indicators for trading
	- main.py: main script where the env is created, model is trained, etc.
	- optimze_params.py: bayesian optimization script of all the variables found in config/config.yaml
