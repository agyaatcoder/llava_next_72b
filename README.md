## llava_next_72b

This repository contains code for downloading model weights, running a Jupyter server inside Modal, and performing inference using the llava_next_72b model.
### Setup Instructions

Download the model weights to a Modal volume using seed_model_repo.py.
Launch a Jupyter server inside Modal with all required libraries using launch_jupyter_llava_next.py.
Open the llava_next_72b.ipynb notebook inside the Jupyter server to perform inference.

### Repository Contents

README.md: This file, providing an overview of the repository.
seed_model_repo.py: Script for downloading model weights to a Modal volume.
launch_jupyter_llava_next.py: Script for launching a Jupyter server inside Modal with required libraries.
llava_next_72b.ipynb: Jupyter notebook for performing inference using the llava_next_72b model.

### Requirements

Modal account and appropriate configuration
Required libraries as specified in launch_jupyter_llava_next.py

### Usage Notes
Ensure you have a Modal account properly configured before running the scripts. Follow the setup instructions in order and use the Jupyter notebook for inference. Consult the Modal documentation for additional guidance on using Modal volumes and servers
