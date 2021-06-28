FROM python:3.7
LABEL student1="Adam Lewandowski" student2="Wojciech Lulek" student3="Tomasz Kiljańczyk"

COPY ./src ./app/src
COPY ./requirements.txt ./app/requirements.txt

# Download updates and necessary libs
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Install all dependencies
RUN pip install -r ./app/requirements.txt

# Set work dir to app directory
WORKDIR "./app/"

# Set python root
ENV PYTHONPATH "${PYTHONPATH}:/app/"

# Entrypoint for docker run
ENTRYPOINT ["python", "./src/docker_main.py"]