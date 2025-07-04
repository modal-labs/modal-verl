# Project structure

- `grpo.py` -> has code for launching grpo, and for preprocessing the data
- `reward.py` -> has custom reward function. the reward function needs to follow this [spec](https://verl.readthedocs.io/en/latest/preparation/reward_function.html#customized)

# Instructions

- To run the local entrypoint: `modal run --detach grpo.py` (currently set to invoke `train`)
- To run the data processing function: `modal run --detach grpo.py::prep_dataset`
- To run the train function: `modal run --detach grpo.py::train`

This example uses WANDB in the code by default.

