"""Experimental data augmentation via back-translation. Not used in the main pipeline."""

from transformers import MarianMTModel, MarianTokenizer

# Load models
def load_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Translate text
def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Back translate
def back_translate(texts):
    fr_tok, fr_model = load_model("en", "es")
    en_tok, en_model = load_model("es", "en")
    
    # English → French
    fr_texts = translate(texts, fr_tok, fr_model)
    # French → English
    back_translated = translate(fr_texts, en_tok, en_model)
    return back_translated

# Example
ads = ["the best at what they do. The quality of their products and customer service standards are second to none. Go to manscape.com, use code trigger15 and get 15% off your first order. That's trigger 15 at manscape.com. White hot looks, smooth results. Leave the burns to the sun. That last bit doesn't flow, but do what we can. And what were the Roman moral sensibilities around sex, sexuality, women, etc.? Well, they were pagans. So I mean on the one hand they can be pretty conservative. They have nuclear families. They have clans as well. Uh on the other hand and and women were, you know, had to be chased or they couldn't be married. Uh but on the other hand, they had the double standard."]
augmented_ads = back_translate(ads)
for orig, aug in zip(ads, augmented_ads):
    print(f"Original: {orig}\n\nAugmented: {aug}\n")
