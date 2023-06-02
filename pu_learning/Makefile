.PHONY: install-tools
install-tools:
	@pip install black==22.3.0 \
	flake8==4.0.1 \
	flake8-bugbear==22.3.23 \
	flake8-comprehensions==3.8.0 \
	flake8-mutable==1.2.0 \
	isort==5.10.1

.PHONY: run-tools
run-tools:
	@isort * /puLearning
	@black * puLearning
	@flake8 * puLearning

.PHONY: lint
lint:
	@isort --check-only --diff * puLearning
	@black --check --diff * puLearning
	@flake8 * puLearning

.PHONY: pre
pre: run-tools