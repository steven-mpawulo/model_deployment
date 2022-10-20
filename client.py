import requests
import json

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "test/kasooli.wav"


if __name__ == "__main__":

    # open files
    audio_file = open(FILE_PATH, "rb")
    values = {"file": (FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    # package stuff to send and perform POST request
    # values = {"file": "hello"}
    # values = json.dumps(values)
    # response = requests.post(URL, values)
    print(data)
    # print(response.json())
    # data = response.json()

    # print(response)
    # print("Predicted keyword: {}".format(data["keyword"]))
