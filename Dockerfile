FROM python:3.7-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

COPY requirements.txt .
COPY .env .

RUN pip install --no-cache-dir -r requirements.txt

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY gpt2bot .

CMD ["python", "discord_bot.py"]