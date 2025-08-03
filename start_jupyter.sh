#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export JUPYTER_PREFER_ENV_PATH=1
export JUPYTER_PATH=$VIRTUAL_ENV/share/jupyter

# Start Jupyter with specific kernel
jupyter notebook --NotebookApp.kernel_spec_manager_class='jupyter_client.kernelspec.KernelSpecManager' toxic_comment_moderation.ipynb