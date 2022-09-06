FROM python:3.7-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

# WORKDIR /app
ADD gpt2bot/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY gpt2bot .
COPY .env .

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY run.sh run.sh
RUN chmod u+x run.sh

ARG PYTHON_ENV=development
ENV PYTHON_ENV=${PYTHON_ENV}
CMD ["./run.sh"]
