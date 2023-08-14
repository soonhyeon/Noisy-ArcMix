FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 as base

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install python3-pip -y

RUN ln -s /usr/bin/python3.8 /usr/bin/python

