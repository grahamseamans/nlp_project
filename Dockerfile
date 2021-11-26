FROM nvcr.io/nvidia/pytorch:21.10-py3
# FROM  nvcr.io/nvidia/tensorflow:21.11-tf2-py3

COPY ./requirements.txt ./install_packages/
RUN pip install -r ./install_packages/requirements.txt

RUN curl -sSL https://sdk.cloud.google.com | bash
RUN gsutil cp gs://trax-ml/reformer/cp.320.* ./home/

