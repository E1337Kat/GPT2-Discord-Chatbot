#  Copyright (c) polakowo
#  Licensed under the MIT license.

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import ipdb


# Enable logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # ipdb.set_trace()
    logger.debug("logits dimension: " + str(logits.dim())) # batch size 1 for now - could be updated for more but the code would be less clear
    logger.debug("logits: " + str(logits)) # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, largest = False)[0][..., -1, None]
        logger.debug("indices_to_remove: " + str(indices_to_remove))
        logits[indices_to_remove] = filter_value
        logger.debug("logits after topk filter: " + str(logits))
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tokenizer, context_ids, config):
    # Parse parameters
    no_cuda = config.getboolean('model', 'no_cuda')
    num_samples = config.getint('decoder', 'num_samples')
    max_length = config.getint('decoder', 'max_length')
    temperature = config.getfloat('decoder', 'temperature')
    top_k = config.getint('decoder', 'top_k')
    top_p = config.getfloat('decoder', 'top_p')

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
    logger.debug("tensor built: " + str(context_tensor))
    logger.debug("tensor shape (before): " + str(context_tensor.shape))
    context_tensor = context_tensor.unsqueeze(0).repeat(num_samples, 1)
    logger.debug("tensor unsqueezed: " + str(context_tensor))
    generated = context_tensor
    logger.debug("Generated shape (before): " + str(generated.shape))
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            logits = outputs[0][:, -1, :]
            next_token_logits = logits / (temperature if temperature > 0 else 1.)
            logger.debug("next_token_logits: " + str(next_token_logits))
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            logger.debug("filtered logits: " + str(filtered_logits))
            if temperature == 0.0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                logger.debug("next_token: " + str(next_token))
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                logger.debug("next_token: " + str(next_token))
            generated = torch.cat((generated, next_token), dim=1)
            logger.debug("next_token added: " + str(next_token))
            logger.debug("Generated shape (after): " + str(generated.shape))
            if (generated[:, len(context_ids):] == tokenizer.eos_token_id).any(dim=1).all():
                # EOS token id found in each sample
                break
            if generated.shape[1] - len(context_ids) >= max_length:
                # Maximum length reached
                break
    return generated

def select_using_mmi(mmi_model, mmi_tokenizer, candidates, config):
    # Parse parameters
    no_cuda = config.getboolean('model', 'no_cuda')

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    scores = []
    for i, candidate in enumerate(candidates):
        context = []
        for response in reversed(candidate):
            context.extend(response)
            context.append(mmi_tokenizer.eos_token_id)
        context_ids = mmi_tokenizer.encode(context)
        context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
        loss, _, _ = mmi_model(input_ids=context_tensor, labels=context_tensor)
        scores.append(-loss.float()) 

    scores = torch.stack(scores, dim=0)
    # The smaller the loss, the higher must be the probability of selecting it
    winner = torch.multinomial(F.softmax(scores, dim=0), num_samples=1).item()
    return winner

def generate_response(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, context: str, config, mmi_model=None, mmi_tokenizer=None):
    # Parse parameters
    use_mmi = config.getboolean('model', 'use_mmi')
    num_samples = config.getint('decoder', 'num_samples')
    max_length = config.getint('decoder', 'max_length')
    seed = config.get('decoder', 'seed')
    seed = int(seed) if seed is not None else None

    # Make answers reproducible only if wanted
    if seed is not None:
        set_seed(seed)

    # Generate response
    context_ids = tokenizer.encode(context, max_length=max_length, pad_to_max_length=False, add_prefix_space=True)
    logger.debug("found context_ids: " + str(context_ids))
    samples = sample_sequence(model, tokenizer, context_ids, config)
    samples = samples[:, len(context_ids):].tolist()
    logger.debug("found samples: " + str(samples))
    texts = []
    for sample in samples:
        text = tokenizer.decode(sample, clean_up_tokenization_spaces=True)
        text = text[: text.find(tokenizer.eos_token)]
        texts.append(text)
    
    if use_mmi:
        assert(num_samples > 1, "MMI requires num_samples > 1")
        candidates = [context + text for text in texts]
        best_i = select_using_mmi(mmi_model, mmi_tokenizer, candidates, config)
        return [texts[best_i]]
    return texts