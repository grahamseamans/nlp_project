FROM nvcr.io/nvidia/pytorch:21.10-py3

# RUN apt-get update
# RUN apt install graphviz -y

COPY ./requirements.txt ./install_packages/
RUN pip install -r ./install_packages/requirements.txt
