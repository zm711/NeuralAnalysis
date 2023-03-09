import yaml
from pathlib import Path

config_file = Path('./na_settings.yaml')

if config_file.is_file():
    pass
else:
    defaults =[
    {'zscore': {'inhib': [-2, 3], 'sustained': [3.3, 5], 'onset': [4, 3], 'offset': [2.5, 3]}},
    {'raw': {'sustained': [75]}},
    {'sorter_dict': {'Sustained': [50, 100], 'Onset': [50, 65], 'Onset-Offset': [50, 65, 90, 110], 'Relief': [100, 150], ' Inhib': [50, 67]}}
    ]

    with open('na_settings.yaml', 'w') as f:
        yaml.dump(defaults, f)
