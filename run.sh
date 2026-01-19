#!/bin/bash
source path_to_conda_environment/bin/activate name_of_conda_environment
experiment_name="file_name_of_config_yaml_without_yaml_extension"

# Start a new screen session named after the experiment

screen -S $experiment_name bash -c "
    python main.py \"$experiment_name\" 
    exec bash
"

echo "Started main.py in a new screen session named $experiment_name"
echo "To attach to the session, use: screen -r $experiment_name"
