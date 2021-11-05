DOCKER_IMAGE := "mastak/car-plate-recognizer"

.PHONY: install
install:
	@git submodule update --init --recursive
	@python3 -m pip install wheel
	@python3 -m pip install -r ./requirements.txt

.PHONY: install-dev
install-dev:
	@python3 -m pip install -r ./requirements.txt -r ./requirements.dev.txt

.PHONY: build
build:
	@echo "Build $(DOCKER_IMAGE)"
	@docker build -t $(DOCKER_IMAGE) ./

.PHONY: docker-bash
docker-bash:
	@echo "Start bash session at $(DOCKER_IMAGE)"
	@docker run -it -v `pwd`:/work_dir $(DOCKER_IMAGE) bash

.PHONY: test
test:
	@echo "TODO: black,flake8,mypy,tests"
	python3 -m flake8 ./car_plate_recognizer
	python3 -m black --check ./car_plate_recognizer
	python3 -m isort --check ./car_plate_recognizer

.PHONY: black
black:
	python3 -m black ./car_plate_recognizer
	python3 -m isort --check ./car_plate_recognizer

.PHONY: isort
isort:
	python3 -m isort --check ./car_plate_recognizer

.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -r {} \+
