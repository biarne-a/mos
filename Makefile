# -*- mode: makefile -*-
install_mac_m2:
    conda create --name=mos-m2 python=3.9
    conda activate mos-m2
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos tensorflow-metal
	conda install -c conda-forge --file requirement.in


install:
    conda create --name=mos python=3.9
    conda activate mos
    conda install -c conda-forge tensorflow
    conda install -c conda-forge --file requirement.in


run:
	python3 src/main.py


clean:
	./scripts/clean.sh


push:
	docker build . -t hsm-adasm
	docker tag hsm-adasm europe-west9-docker.pkg.dev/concise-haven-277809/biarnes/hsm-adasm
	docker push europe-west9-docker.pkg.dev/concise-haven-277809/biarnes/hsm-adasm
