from transformers import MarianMTModel, MarianTokenizer
import translate.storage.xliff as xliff

BATCH_SIZE = 1
MODEL_NAME='Kovalev/OPUS-ru-en-finetuned-letters'

model = MarianMTModel.from_pretrained(MODEL_NAME)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)


def translate_text(input_text, batch_size = BATCH_SIZE):
    
    global model,tokenizer

    input_text = [input_text]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    
    # Split the inputs into smaller batches
    input_ids = inputs['input_ids']
    num_batches = (len(input_ids) + batch_size - 1) // batch_size
    
    translated_texts = []
    
    # Generate translation in batches
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_inputs = {'input_ids': batch_input_ids}
        
        # Generate translation for the batch
        batch_translated = model.generate(**batch_inputs)
        
        # Decode the translated text for the batch
        batch_translated_text = tokenizer.batch_decode(batch_translated, skip_special_tokens=True)
        translated_texts.extend(batch_translated_text)   
    return translated_texts[0]

def translate_xliff(input_file_path, MODEL_NAME='Kovalev/OPUS-ru-en-finetuned-letters'):
    with open(input_file_path, 'rb+') as fin:
        xliff_file = xliff.xlifffile(fin)
        for translate_unit in xliff_file.unit_iter():
            source_segment = translate_unit.source
            translation = translate_text(source_segment, BATCH_SIZE)
            translate_unit.target = translation
        xliff_file.save()
        return xliff_file