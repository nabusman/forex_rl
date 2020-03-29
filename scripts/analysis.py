import os
from datetime import datetime

import yaml
import pandas as pd

model_dir = '/home/nabs/Projects/forex_rl/models'
analysis_dir = '/home/nabs/Projects/forex_rl/analysis'

# Load model details
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

# Convert to pandas
col_names = list({x for v in model_details.values() for x in v.keys()})
col_names.append('model_name')
data = {c : [] for c in col_names}
for model_name in model_details:
	for col in data.keys():
		if col == 'model_name':
			data[col].append(model_name)
		elif col in model_details[model_name]:
			data[col].append(str(model_details[model_name][col]))
		else:
			data[col].append('NA')

df = pd.DataFrame(data)

# Output as CSV
now_str = str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]

df.to_csv(os.path.join(analysis_dir, f'{now_str}_analysis.csv'), index = False)