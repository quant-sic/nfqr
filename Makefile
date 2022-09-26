.PHONY: help env env-update env-remove init install format lint test \
	docs-sphinx requirements


ifeq ($(shell uname -s),Darwin)
PLATFORM = osx
else ifneq ($(shell (command -v nvidia-smi && nvidia-smi --list-gpus)),)
PLATFORM = gpu
else
PLATFORM = cpu
endif


PROJECTNAME=nfqr

help:
	@echo "Available commands:"
	@echo "conda-env              create the conda environment '$(PROJECTNAME)-env'."
	@echo "env-update       update '$(PROJECTNAME)-env'."
	@echo "env-remove       remove '$(PROJECTNAME)-env'."
	@echo "install          install package in editable mode."
	@echo "requirements	    compiles requirements from .in files"

env:
	conda create -n nfqr-env python=3.10 && \
	conda activate nfqr-env && \
	pip install --upgrade pip wheel pip-tools

env-update:
	pip install --upgrade -r requirements/requirements.txt

env-remove:
	rm -rf nfqr-env

# This is not unsafe, and will become the default in a future version
# of pip-compile.
PIP_COMPILE := pip-compile -q --allow-unsafe

requirements:
	pip install -q pip-tools
	CONSTRAINTS=/dev/null $(PIP_COMPILE) \
	  requirements/*.in -o requirements/constraints.txt
	CONSTRAINTS=constraints.txt $(PIP_COMPILE) requirements/base-$(PLATFORM).in

	
install:
	pip-sync requirements/base-$(PLATFORM).txt
	pip install -e .


# pip-sync requirements/base-$(PLATFORM).txt