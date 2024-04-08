from gtts import gTTS
import os

def text_to_speech(text, language='en', filename='output.mp3', slow=False):
    """
    Convert text to speech and save it as an audio file.

    Args:
        text (str): The text to be converted to speech.
        language (str, optional): The language of the text. Defaults to 'en'.
        filename (str, optional): The filename to save the audio as. Defaults to 'output.mp3'.
        slow (bool, optional): Whether to generate the audio slowly. Defaults to False.
    """
    # Check if the input text is not empty
    if not text:
        print("Error: No text to speak.")

    # Passing the text and language to the engine
    myobj = gTTS(text=text, lang=language, slow=slow)

    # Saving the converted audio
    myobj.save(filename)

    # Playing the converted file
    os.system(f"mpg321 {filename}")