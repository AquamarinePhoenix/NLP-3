from nlplib.tts import output_language_report, test_speech
from nlplib.convert import convert_to_wav
from nlplib.audio import analyze_audio
from nlplib.config import REFERENCE_FILE_NAME, SYNTHESIZED_FILE_NAME
from nlplib.write import clear_file

def main():
    clear_file()
    output_language_report()
    convert_to_wav(REFERENCE_FILE_NAME, "m4a")
    test_speech(REFERENCE_FILE_NAME, SYNTHESIZED_FILE_NAME)
    analyze_audio(REFERENCE_FILE_NAME, SYNTHESIZED_FILE_NAME)

if __name__ == "__main__":
    main()