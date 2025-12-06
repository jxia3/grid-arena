from arena import render
from config import CONFIGS

for config in CONFIGS:
    arguments = ["all", "--path", str(config.output_directory)]
    render.main(arguments=arguments)