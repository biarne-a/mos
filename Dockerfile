FROM apache/beam_python3.9_sdk:2.48.0

ARG WORKDIR=/usr/src/app

ENV PYTHONPATH "${PYTHONPATH}:${WORKDIR}/src"

#RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg && apt-get clean
#
## Install google-cloud-sdk
#RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
#    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
#    && apt-get update -y \
#    && apt-get install google-cloud-sdk -y \
#    && apt-get clean

RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

COPY requirements.in ${WORKDIR}/requirements.txt
RUN pip install -U --no-cache-dir -r requirements.txt

# Copy source code to the container
COPY . ${WORKDIR}/
