FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# https://stackoverflow.com/a/66473309
# https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv tzdata && apt-get clean

# From continuumio/miniconda:4.7.12
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Set timezone
# https://stackoverflow.com/questions/40234847/docker-timezone-in-ubuntu-16-04-image
ENV TIMEZONE="Asia/Singapore"
RUN echo $TIMEZONE >/etc/timezone
RUN rm /etc/localtime
RUN ln -snf /usr/share/zoneinfo/$TIMEZONE /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

COPY ./ /simswap
WORKDIR /simswap

RUN conda env create --file environment.yaml

RUN chmod +x entrypoint.sh

EXPOSE 8000

# https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45#77e9
ENTRYPOINT [ "./entrypoint.sh" ]
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "flask_app:create_app()"]
