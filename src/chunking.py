import warnings

import spacy
from transformers import GPT2Tokenizer, logging

logging.set_verbosity_error()

nlp = spacy.load("en_core_web_sm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def split_into_chunks_with_pmc(text, pmc_id, max_tokens=900, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) > 1024:
            print(f"Warning: chunk length {len(chunk_tokens)} exceeds 1024")
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)

        chunk = {
            "pmc": pmc_id,
            "text": chunk_text,
            "token_start": start,
            "token_end": end,
            "token_length": len(chunk_tokens),
        }

        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks
