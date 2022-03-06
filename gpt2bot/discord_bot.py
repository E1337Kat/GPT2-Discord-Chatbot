#  Licensed under the MIT license.

import configparser
import argparse
import logging
import random
import asyncio

from decouple import config
import discord
from discord.errors import *
from discord.ext import commands
import time
import os
import sys
import re
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import traceback
from discord.message import Message
from model import download_model_folder, download_reverse_model_folder, load_model
from decoder import generate_response
from textblob import TextBlob
from googletrans import Translator
import datetime



# Enable logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

client = commands.Bot(command_prefix="BOT_NAME")

global conversations
global debug_command
global debug_modes
global translator

global num_samples
global max_turns_history
global model
global tokenizer
global mmi_model
global config_parser
global mmi_tokenizer
global start_time
global history_dict

@client.event
async def on_ready():
    global translator

    global conversations
    global debug_command
    global debug_modes

    global num_samples
    global max_turns_history
    global model
    global tokenizer
    global mmi_model
    global mmi_tokenizer
    global config_parser
    global history_dict

    if(conversations is None):
        conversations = {}
    if(debug_command is None):
        debug_command = {}
    if(debug_modes is None):
        debug_modes = {}
    if(history_dict is None):
        history_dict = {}
    logger.info('Logged in as '+client.user.name+' (ID:'+str(client.user.id)+') | '+str(len(client.guilds))+' servers | ' + getAllUsersCount())
    await client.change_presence(activity=discord.Game(name='remind me to fix bugs'))
    #schedule.every().day.at("00:00").do(client.loop.call_soon_threadsafe, restart_script())
    #client.loop.create_task(run_schedule())

from table2ascii import table2ascii as t2a, PresetStyle

# # In your command:
# output = t2a(
#     header=["Rank", "Team", "Kills", "Position Pts", "Total"],
#     body=[[1, 'Team A', 2, 4, 6], [2, 'Team B', 3, 3, 6], [3, 'Team C', 4, 2, 6]],
#     style=PresetStyle.thin_compact
# )
#Called when a message is received
@client.listen()
async def on_message(message: Message):
    global conversations
    global debug_command
    global debug_modes

    global from_index
    global history_dict
    global turn
    global turn2
    global translator
    debug_modes[message.id] = False
    if not (message.author == client.user): #Check to ensure the bot does not respond to its own messages
        if(message.mention_everyone == False):
            if(should_respond):
                responseHandler(message)
            elif(debug_command[message.id] == "state"):
                response = t2a(
                    header=["User Message", "Bot Message"],
                    body=[[1, 'Team A', 2, 4, 6], [2, 'Team B', 3, 3, 6], [3, 'Team C', 4, 2, 6]],
                    style=PresetStyle.thin_compact
                )
                async with message.channel.typing(): #Show that the bot is typing
                    try:
                        await message.channel.send(response) #Fire away!
                    except HTTPException as e:
                        logger.exception("ERROR occured:")
                        formatted_ex = traceback.format_exc() #Fetch error
                        response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
                        await message.channel.send(response)  # Fire away!
                    finally:
                        debug_modes.pop(message.id)
                        logger.debug("Toggling debug mode off")
                        logger.setLevel(logging.INFO)

async def responseHandler(message):
    txtinput = parse_commands(message)
    txtinput = filter_self(txtinput)  #Filter out the mention so the bot does not get confused
    response = ""
    blob = TextBlob(txtinput)
    lang = translator.detect(txtinput).lang
    #lang = "en"
    debug_mode = debug_modes[message.id]
    try:
        if(lang != "en"):
            txtinput = str(translator.translate(txtinput, dest="en", src=lang).text)
            #_context.append(txtinput)
    except:
        print("A translation error occured")
        e = sys.exc_info()[0]
        logger.exception("Error occured with translation service:")

    async with message.channel.typing(): #Show that the bot is typing
        if(isinstance(message.channel, discord.abc.PrivateChannel)):
            maybe_response = get_response(txtinput, message.author.id, False, debug_mode) #Get a response!
            if(isinstance(maybe_response, str)):
                response =  maybe_response
        else:
            maybe_response = get_response(txtinput, message.guild.id, False, debug_mode) #Get a response!
            if(isinstance(maybe_response, str)):
                response =  maybe_response
        response_blob = TextBlob(response)

        try:
            await message.channel.send(response) #Fire away!
        except HTTPException as e:
            logger.exception("ERROR occured:")
            formatted_ex = traceback.format_exc() #Fetch error
            if debug_mode:
                response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
            else:
                response = "I tried to send nothing to discord.. I am very sorry goshujin-sama"
            await message.channel.send(response)  # Fire away!
            history_dict = {} #Clear history
        except:
            logger.exception("ERROR occured:")
            formatted_ex = traceback.format_exc() #Fetch error
            if debug_mode:
                response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
            else:
                response = "I'm a dumb bot and couldn't understand that. Sorry!"
            await message.channel.send(response)  # Fire away!
            history_dict = {} #Clear history
        finally:
            debug_modes.pop(message.id)
            if debug_mode:
                logger.debug("Toggling debug mode off")
            logger.setLevel(logging.INFO)


def getAllUsersCount():
    guilds = client.guilds
    user_count = 0
    for g in guilds:
         user_count += len(g.members)
    return("Current user count: " + str(user_count))

def run_chat():
    # Parse parameters
    global translator
    
    global num_samples
    global max_turns_history
    global model
    global tokenizer
    global mmi_model
    global mmi_tokenizer
    global config_parser
    global history_dict
    global token
    
    num_samples = config_parser.getint('decoder', 'num_samples')
    max_turns_history = config_parser.getint('decoder', 'max_turns_history')

    logger.info("Running the chatbot...")
    turns = []
    loop = asyncio.get_event_loop()
    task1 = loop.create_task(client.start(token))
    gathered = asyncio.gather(task1, loop=loop)
    loop.run_until_complete(gathered)


def parse_commands(msg: Message) -> str:
    """
    Parses the message content to determine if we need to set any variables
    """
    global conversations
    global debug_modes
    global debug_command

    content = msg.content
    message_id = msg.id
    channel_id = msg.channel.id
    if not message_id in debug_modes:
        debug_modes[message_id] = []
    if not message_id in debug_command:
        debug_command[message_id] = []
    if not channel_id in conversations:
        conversations[channel_id] = False

    if("!debug" in content):
        logger.setLevel(logging.DEBUG)
        logger.info("Toggling debug mode on")
        debug_modes[message_id] = True
        content = content.replace("!debug", "")
    else:
        debug_modes[message_id] = False

    if("!convo" in content):
        logger.info("Toggling conversation mode")
        conversations[channel_id] = not conversations[channel_id]
        content = content.replace("!convo", "")

    if("current state?" in content and debug_modes[message_id]):
        logger.info("Replying with state instead")
        debug_command[message_id] = "state"

    return content


def should_respond(message: Message) -> bool:
    """
    Checks if the bot is mentioned or if the message is in DMs or in a conversation in the current channel.
    """
    return client.user.mentioned_in(message) or isinstance(message.channel, discord.abc.PrivateChannel) or conversations[message.channel.id]


def filter_self(content: str) -> str:
    """
    Filter out the mention so the bot does not get confused
    """
    return content.replace("<@" + str(client.user.id) + ">", "").replace("<@!" + str(client.user.id) + ">", "")


def get_prescripted_lines(filepath):
    lines = []
    with open(filepath, "r") as f:
        for line in f:
            lines.append(line)
    return lines
global static_history
static_history = get_prescripted_lines("./constant_thoughts.txt")

def get_response(prompt: str, channel_id: str, do_infinite: bool, debug_mode: bool) -> str:
    global translator
    global turn
    global turn2
    global num_samples
    global max_turns_history
    global model
    global tokenizer
    global mmi_model
    global mmi_tokenizer
    global config_parser
    global history_dict
    global from_index
    if max_turns_history == 0:
        # If you still get different responses then set seed
        turns = []

    # A single turn is a group of user messages and bot responses right after
    turn = {
        'user_messages': [],
        'bot_messages': []
    }
    str_channel_id = str(channel_id)    
    #turns.append(turn)
    turn['user_messages'].append(prompt)
    if not channel_id in history_dict:
        history_dict[channel_id] = []
    
    
    history_dict[channel_id].append(turn)
    # Merge turns into a single history (don't forget EOS token)
    history = ""
    from_index = max(len(history_dict[channel_id])-max_turns_history-1, 0) if max_turns_history >= 0 else 0
    for message in static_history:
        history += message + tokenizer.eos_token
    for i in range(len(history_dict[channel_id])):
        if(i >= from_index):
            turn2 = history_dict[channel_id][i]
        else:
            continue
        # Each turn begings with user messages
        for message in turn2['user_messages']:
            history += message + tokenizer.eos_token
        for message in turn2['bot_messages']:
            history += message + tokenizer.eos_token

    try:
        # Generate bot messages
        bot_messages = generate_response(
            model, 
            tokenizer, 
            history, 
            config_parser, 
            mmi_model=mmi_model, 
            mmi_tokenizer=mmi_tokenizer
        )
    except:
        logger.exception("ERROR occured:")
        formatted_ex = traceback.format_exc() #Fetch error
        if debug_mode:
            response = "An error has occurred. Please try again:\n```" + formatted_ex + "```"
        else:
            response = "I'm a dumb bot and couldn't understand that. Sorry!"
        history_dict = {} #Clear history
        return response 

    logger.debug("Found responses: " + str(bot_messages))
    logger.debug("history: " + str(history_dict))
    if debug_mode:
        bot_message = str(bot_messages) if bot_messages != [''] else ''
    elif num_samples == 1:
        bot_message = bot_messages[0]
        turn['bot_messages'].append(bot_message)
    else:
        # TODO: Select a message that is the most appropriate given the context
        # This way you can avoid loops
        bot_message = random.choice(bot_messages)
        turn['bot_messages'].append(bot_message)
    return bot_message

def main():
    global translator

    global conversations
    global debug_command
    global debug_modes
    global num_samples
    global max_turns_history
    global model
    global tokenizer
    global mmi_model
    global mmi_tokenizer
    global config_parser
    global history_dict
    global token

    conversations = {}
    debug_command = {}
    debug_modes = {}
    history_dict = {}
    translator = Translator()
    # Script arguments can include path of the config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default="chatbot.cfg")
    arg_parser.add_argument('--production_mode', nargs='?', const=True, default=False, type=bool)
    args = arg_parser.parse_args()

    logger.info("starting with arguments: " + str(args))

    if args.production_mode:
        logger.info("Starting bot in PRODUCTION mode")
        token = config('DISCORD_TOKEN') # Replace TOKEN_GOES_HERE with your discord API bot token!
    else:
        logger.info("Starting bot in DEVELOPMENT mode")
        token = config('DEVELOPMENT_TOKEN')

    logger.info("using token: " + token)

    # Read the config
    config_parser = configparser.ConfigParser(allow_no_value=True)
    with open(args.config) as f:
        config_parser.read_file(f)

    # Download and load main model
    target_folder_name = download_model_folder(config_parser)
    model, tokenizer = load_model(target_folder_name, config_parser)

    # Download and load reverse model
    use_mmi = config_parser.getboolean('model', 'use_mmi')
    if use_mmi:
        mmi_target_folder_name = download_reverse_model_folder(config_parser)
        mmi_model, mmi_tokenizer = load_model(mmi_target_folder_name, config_parser)
    else:
        mmi_model = None
        mmi_tokenizer = None
    
    # Run chatbot with GPT-2
    run_chat()

if __name__ == '__main__':
    main()
    

