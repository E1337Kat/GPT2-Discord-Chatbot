FROM nvidia/cuda:10.2-devel-ubuntu18.04
# CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install gcc
RUN apt-get -y install unzip zlib1g-dev libjpeg8-dev
RUN apt-get -y install python3 python3-pip
# RUN apt-get -y install 

# WORKDIR /app
ADD gpt2bot/requirements.txt ./requirements.txt

# RUN pip3 install torch
RUN pip3 install torch==1.4.0+cu92 --find-links https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install transformers~=4.6.1

COPY gpt2bot .
COPY .env .

ENV NUM_EPOCHS=10
ENV MODEL_TYPE='EfficientDet'
ENV DATASET_LINK='HIDDEN'
ENV TRAIN_TIME_SEC=100

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY run.sh run.sh
RUN chmod u+x run.sh

ARG PYTHON_ENV=development
ENV PYTHON_ENV=${PYTHON_ENV}
CMD ["./run.sh"]