from transformers import MarianMTModel, MarianTokenizer
import translate.storage.xliff as xliff
 
def translate_text(input_text,model_name='Kovalev/OPUS-ru-en-finetuned-letters'):
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

def translate_xliff(input_file_path, model_name='Kovalev/OPUS-ru-en-finetuned-letters'):
    with open(input_file_path, 'rb+') as fin:
        xliff_file = xliff.xlifffile(fin)
        for translate_unit in xliff_file.unit_iter():
            source_segment = translate_unit.source
            translation = translate_text(source_segment, model_name)
            translate_unit.target = translation
        xliff_file.save()
        return xliff_file