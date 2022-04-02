#!/bin/bash

if [ ${PYTHON_ENV} == "production" ]; then 
    python3 discord_bot.py --production_mode
else
    python3 discord_bot.py
fi