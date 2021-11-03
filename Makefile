DOCKER_IMAGE := "mastak/car-plate-recognizer"

.PHONY: install
install:
	@pip install -r ./requirements.txt


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
	@echo "TODO: tests"
