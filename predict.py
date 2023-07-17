from transformers import MarianMTModel, MarianTokenizer
  
def translate_text(input_text,model_name='Kovalev/opus-mt-ru-en-finetuned-ru-to-en-lett'):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)  
    # Tokenize the input text
    input_text = [input_text]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    # Generate translation
    translated = model.generate(**inputs)
    # Decode the translated text
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]