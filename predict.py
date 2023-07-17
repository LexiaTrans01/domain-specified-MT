from transformers import MarianMTModel, MarianTokenizer
import translate.storage.tmx as tmx

  
def translate_text(input_text,model_name='Kovalev/opus-mt-ru-en-finetuned-ru-to-en-lett',src_lang='ru', tgt_lang='en'):
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

def translate_tmx(input_file,model_name='Kovalev/opus-mt-ru-en-finetuned-ru-to-en-lett'):
    tmx_file = tmx.tmxfile(input_file)
    translated_tmx_file = tmx.tmxfile()

    for tu in tmx_file.units:
        source_segment = tu.source
        translation = translate_text(source_segment, src_lang='ru', tgt_lang='en')
        # Create a new translation unit with the translated text
        translated_tu = tmx.tmxunit.TMXUnit(tu.tuid, source_segment, translation)        
        translated_tmx_file.add_tu(translated_tu)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    translated_tmx_file.save(temp_file.name)

    return temp_file.name