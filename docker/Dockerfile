FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git
RUN pip install --upgrade pip

WORKDIR /

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME


