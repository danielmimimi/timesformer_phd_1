
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools

# BELONGS TO TORCH
RUN pip install torchvision==0.9.1
RUN pip install torchaudio==0.8.1


# SIDE PACKAGES
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install imgaug
RUN pip install typing-extensions --upgrade
RUN pip install tqdm
RUN pip install pandas
RUN pip install einops
RUN pip install scikit-learn
RUN pip install simplejson

RUN pip install timm
RUN pip install fvcore

RUN pip install av
RUN pip install tensorboard
RUN pip install torch-tb-profiler
RUN pip install yacs

# MAKE OPENCV RUNNABLE
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN  apt-get update \
  && apt-get install -y wget
CMD [ "/bin/bash" ]