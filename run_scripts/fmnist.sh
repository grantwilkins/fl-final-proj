#!/bin/bash
# Add your code for the job manager

poetry run python -m project.main --config-name=fmnist-sgd

poetry run python -m project.main --config-name=fmnist-adam

poetry run python -m project.main --config-name=fmnist-dgnmc

# poetry run python -m project.main --config-name=fmnist-dgnexact

poetry run python -m project.main --config-name=fmnist-bdgnmc

poetry run python -m project.main --config-name=fmnist-bdgnexact

poetry run python -m project.main --config-name=fmnist-lbfgs