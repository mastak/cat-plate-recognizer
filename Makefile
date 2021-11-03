DOCKER_IMAGE := "mastak/car-plate-recognizer"

.PHONY: install
install:
	@pip install -r ./requirements.txt -r ./requirements.dev.txt

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
	python -m flake8 ./car_plate_recognizer
	python -m black --check ./car_plate_recognizer
	python -m isort --check ./car_plate_recognizer

.PHONY: black
black:
	python -m black ./car_plate_recognizer
	python -m isort --check ./car_plate_recognizer

.PHONY: isort
isort:
	python -m isort --check ./car_plate_recognizer
