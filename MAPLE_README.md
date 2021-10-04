# Maxwell - A DialoGPT variant for discord.py - Original Readme

Maxwell is my experiment with Microsoft's DialoGPT model and OpenAI's GPT-2 language model, for use on any discord server.  I am planning on fine-tuning much more in the future but for now DialoGPT's model performs admirably.
As of 3-2-2020 I am running this model on my old bot, Maple, which you can invite here:
<a href="https://discordbotlist.com/bots/571924469661302814">
    <img
        width="380"
        height="140"
        src="https://discordbotlist.com/bots/571924469661302814/widget"
        alt="Lithium stats on Discord Bot List">
</a>

## Setup

The recommended python version is 3.6.8.  

### Requirements
You can install all requirements with "pip install -r requirements.txt".

python 3.6.8
numpy 1.16.4
torch 1.2.0
transformers 2.3.0
python-telegram-bot 12.3.0 (Only if you are using the telegram bot)
discord.py 1.2.5
goolgetrans 2.4.0 (For automatic translation with non-english users)
textblob 0.15.3 (Used in some text processing cases)
matplotlib 2.0.2 (Used for modeling statistics)

In discord_bot.py, at line 117, replace "TOKEN_GOES_HERE" with your discord bot's API token.
The model will automatically download and set up upon the first run of the program; you should be good to go!