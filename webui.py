from io import BytesIO
import streamlit as st
import tempfile
from predict import translate_text,translate_tmx
def main():
    st.title("Domain specified MT")  # tmx translation
    st.subheader('Translate TMX file')
    st.write("Upload a .tmx file for translation")

    file = st.file_uploader("Upload File")

    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
            translated_file_path = translate_tmx(temp_file_path)
            st.download_button("Download Translated TMX File", translated_file_path, "translated.tmx")

            
   
    st.subheader('Translate text')  # text translation
    input_text = st.text_input('Please enter text to translate')
    st.write(translate_text(input_text))
    

if __name__ == "__main__":
    main()