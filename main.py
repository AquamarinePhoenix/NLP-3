from nlplib.tts import output_language_report, test_speech
from nlplib.convert import convert_to_wav
from nlplib.audio import analyze_audio
from nlplib.config import INPUT_EN_FILE_NAME, INPUT_LT_FILE_NAME, OUTPUT_EN_FILE_NAME, OUTPUT_LT_FILE_NAME
from nlplib.write import clear_file
from nlplib.dataset import prepare_dataset

def main():
    clear_file()
    output_language_report()
    convert_to_wav(INPUT_EN_FILE_NAME, "m4a")
    convert_to_wav(INPUT_LT_FILE_NAME, "m4a")

    test_speech(INPUT_EN_FILE_NAME, OUTPUT_EN_FILE_NAME)
    test_speech(INPUT_LT_FILE_NAME, OUTPUT_LT_FILE_NAME)

    analyze_audio(INPUT_EN_FILE_NAME, OUTPUT_EN_FILE_NAME, "english")
    analyze_audio(INPUT_LT_FILE_NAME, OUTPUT_LT_FILE_NAME, "lithuanian")

    prepare_dataset()

if __name__ == "__main__":
    main()