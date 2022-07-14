.PHONY: help env env-update env-remove init install format lint test \
	docs-sphinx requirements

PROJECTNAME=nfqr

help:
	@echo "Available commands:"
	@echo "conda-env              create the conda environment '$(PROJECTNAME)-env'."
	@echo "env-update       update '$(PROJECTNAME)-env'."
	@echo "env-remove       remove '$(PROJECTNAME)-env'."
	@echo "install          install package in editable mode."
	@echo "requirements	    compiles requirements from .in files"

env:
	python -m venv nfqr-env && \
		nfqr-env/bin/pip install --upgrade pip
conda-env:
	conda create -n nfqr-env python=3.9

env-update:
	pip install --upgrade -r requirements/requirements.txt

env-remove:
	rm -rf nfqr-env

requirements:
	pip install --upgrade pip wheel pip-tools
	pip-compile --output-file requirements/requirements.txt \
	requirements/*.in
	
install:
	pip install --upgrade pip wheel pip-tools &&\
	pip-sync requirements/requirements.txt
	pip install -e .