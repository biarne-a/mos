# -*- mode: makefile -*-
install:
	mamba env create -f env.yaml


run:
	python3 src/main.py


clean:
	./scripts/clean.sh


push:
	docker build . -t hsm-adasm
	docker tag hsm-adasm europe-west9-docker.pkg.dev/concise-haven-277809/biarnes/hsm-adasm
	docker push europe-west9-docker.pkg.dev/concise-haven-277809/biarnes/hsm-adasm
