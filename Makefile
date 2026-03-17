SHELL := /bin/bash

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help test test-data-cleaning test-api test-dashboard test-training

help:
	@echo "Available targets:"
	@echo "  make test                 Run all tests"
	@echo "  make test-data-cleaning   Run data cleaning tests"
	@echo "  make test-api             Run API tests"
	@echo "  make test-dashboard       Run dashboard tests"
	@echo "  make test-training        Run training tests"

test:
	$(PYTEST) tests -q

test-data-cleaning:
	$(PYTEST) tests/data_cleaning -q

test-api:
	$(PYTEST) tests/api -q

test-dashboard:
	$(PYTEST) tests/dashboard -q

test-training:
	$(PYTEST) tests/training -q
