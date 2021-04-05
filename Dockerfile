FROM python:3.7.9
MAINTAINER Priyash Bhugra <priyashbhugra@gmail.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 future -r requirements.txt
RUN pip uninstall numpy --yes
RUN pip install numpy

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

CMD echo Environment installed Properly

CMD ["python3" "api.py"]