# Discord GPT2 Chatbot

#### Based off of: https://github.com/DoktorHolmes/Maxwell and https://github.com/polakowo/gpt2bot    

![The AI uprising is coming](/images/hal.gif)  

#### Changelog:

- Fixed issues with long messages causing the original Maxwell bot to freeze (clears history if an error occurs and prints traceback in Discord)
- Fixed issues with translation causing no response from bot due to misread language
- Removed statistics reporting which had errors that prevented the original program from running

## Installation

Please ignore instructions from original readme, I updated packages to the latest working versions:    
1. Clone this repository  
2. Install Python 3.7.9 if not installed  
3. Install required libraries (I reccomend using a virtual environment or an IDE like PyCharm). You can install all requirements with `pip install -r requirements.txt`:  

  ```
    requests~=2.24.0  
    torch==1.4.0+cu92  
    tqdm~=4.48.2  
    transformers~=2.3.0  
    python-telegram-bot~=12.8  
    numpy~=1.19.1  
    discord~=1.0.1  
    textblob~=0.15.3  
    googletrans~=4.0.0-rc1
    matplotlib~=3.3.1
    python-decouple==3.5
  ```

4. Copy `.env.conf` to `.env` (`cp .env.conf .env`)
5. In `.env``, replace "YOUR_TOKEN_GOES_HERE" with your discord bot's API token and save the file
6. Open the folder "gpt2bot" 
7. Run discord_bot.py (`python discord_bot.py`). The model will download automatically.
8. @ the bot or DM to get a response!

#### Alternatively

After step 5, run the dockerfile and run in a container.. I think should work... maybe.

# Maxwell - A DialoGPT variant for discord.py - Original Readme

To reduce confusion, see the [MAPLE_README](/MAPLE_README.md). Install info contained therein is outdated.

# gpt2bot - Original Readme

To reduce confusion, see the [TELEGRAM_README](/TELEGRAM_README.md). Install info contained therein is outdated.

## Updates

#### 2021/10/04

  - Update readme
  - Fix deps
  - pull token code out into a dotenv file

#### 2020/01/18

- EOS token is being checked during generation -> gpt2bot is now fast enough to be run on CPU.
- Add support for maximum mutual information (MMI) -> more quality, but slower.

## References

- [Official DialoGPT implementation](https://github.com/microsoft/DialoGPT) and [DialoGPT paper](https://arxiv.org/abs/1911.00536)
- [Thread on current decoding scripts](https://github.com/microsoft/DialoGPT/issues/3)

You can wait for a full DialoGPT release and then replace the decoder.
