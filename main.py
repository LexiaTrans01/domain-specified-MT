import tempfile
import streamlit as st
from predict import translate_text,translate_xliff

def save_uploaded_file(file):
    if file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False,suffix='.xliff')
        temp_file.write(file.read())
        temp_file.close()
        return temp_file.name

def read_file_as_bytes(file_path):
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
    return file_bytes

def main():
    st.title("Domain specified MT")  # xliff translation
    st.subheader('Translate xliff file')
    st.write("Upload a .xliff file for translation")

    file = st.file_uploader("Upload File")

    if file is not None:
        file_path = save_uploaded_file(file)
        translate_xliff(file_path)    
        st.download_button("Download Translated Xliff file",read_file_as_bytes(file_path) , "translated.xliff")
      
    st.subheader('Translate text')  # text translation
    input_text = st.text_input('Please enter text to translate')
    st.write(translate_text(input_text))
    

if __name__ == "__main__":
    main()
