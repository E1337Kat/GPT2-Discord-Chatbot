from configparser import ConfigParser
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.nn import functional as F

from decoder import top_k_top_p_filtering
from discord_bot import get_prescripted_lines
from model import load_model, download_model_folder

# Enable logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

config_parser = ConfigParser(allow_no_value=True)
with open("chatbot.cfg") as f:
    config_parser.read_file(f)
target_folder_name = download_model_folder(config_parser)
model, tokenizer = load_model(target_folder_name, config_parser)


global static_history
static_history = get_prescripted_lines("./constant_thoughts.txt")
history = ""
for message in static_history:
  history += message  
# Tokenize input phrase
history += f'Starting to think that sometimes it is my testing input that could be a problem. Same with the numbers or the long winded sentences. This is a conversational model, so feeding her say a stacktrace of course would be a problem for her. I sleep in a bed that is poorly '
context_ids = tokenizer.encode(history, return_tensors='pt')
# num_samples = config_parser.getint('decoder', 'num_samples')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
generated = context_tensor
with torch.no_grad():
  while True:
      inputs = {'input_ids': context_ids}
      # Get logits from last layer
      last_layer_logits = model(**inputs)[0][:, -1, :] / .6474

      # Keep top 30 logits at max; stop if cumulative probability >= 1.0.
      top_logits = top_k_top_p_filtering(last_layer_logits, top_k=40, top_p=0.0)

      next_token = torch.argmax(top_logits, dim=-1).unsqueeze(-1)
      logger.debug("argmax_next_token: " + str(next_token))
      
      # Softmax the logits into probabilities
      probabilities = F.softmax(top_logits, dim=-1)

      # Generate next token
      generated_next_token = torch.multinomial(probabilities, num_samples=3)
      logger.debug("mult_next_token: " + str(generated_next_token))

      generated = torch.cat([generated, generated_next_token], dim=-1)

      # generated = torch.cat((generated, next_token), dim=1)
      
      # logger.debug("length: " + str(len(context_ids)))
      # logger.debug("original_tokening: " + str(generated[:, len(context_ids):]))
      # logger.debug("meow_tokening: " + str(generated[:len(context_ids):]))
      # logger.debug("alt_meow_tokening: " + str(generated[:len(context_ids),:]))
      # logger.debug("alt_tokening: " + str(generated[:, :len(context_ids)]))
      # if (generated[:, len(context_ids):] == tokenizer.eos_token_id).any(dim=1).all():
      #     # EOS token id found in each sample
      #     logger.debug("EOS token id found in each sample")
      #     break
      if generated.shape[1] - len(context_ids) >= 25:
          # Maximum length reached
          logger.debug("Maximum length reached")
          break
# # Get logits from last layer
# last_layer_logits = model(inputs)[0][:, -1, :]

# # Keep top 30 logits at max; stop if cumulative probability >= 1.0.
# top_logits = top_k_top_p_filtering(last_layer_logits, top_k=30, top_p=0.0)

# # Softmax the logits into probabilities
# probabilities = F.softmax(top_logits, dim=-1)

# # Generate next token
# generated_next_token = torch.multinomial(probabilities, num_samples=1)
# generated = torch.cat([inputs, generated_next_token], dim=-1)

# Get result
samples = generated[:, len(context_ids):].tolist()
# result_string = tokenizer.decode(samples[0])
texts = []
for sample in samples:
    text = tokenizer.decode(sample)
    # text = text[: text.find(tokenizer.eos_token)]
    texts.append(text)

# Print string
print(texts)