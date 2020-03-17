import os

import yaml

model_dir = '/home/nabs/Projects/forex_rl/models'

files = os.listdir(model_dir)
model_details = {}
for file in files:
	model_name = file[0:file.rfind('_')]
	try:
		if '.yaml' in file and model_name in model_details:
			with open(os.path.join(model_dir, file), 'r') as f:
				model_details[model_name].update(yaml.load(f, Loader = yaml.FullLoader))
		elif '.yaml' in file and model_name not in model_details:
			with open(os.path.join(model_dir, file), 'r') as f:
				model_details[model_name] = yaml.load(f, Loader = yaml.FullLoader)
	except:
		print(f'skipping {file}')
		continue
