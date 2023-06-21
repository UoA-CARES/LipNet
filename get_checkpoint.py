import gdown

if __name__ =="__main__":
    # Download and extract checkpoint-96 in ./models folder
    url = 'https://drive.google.com/uc?id=1Od9JNHUnD6bzxayPEV0raVoceSbKJ9G1&export=download'
    output = 'models.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('models.zip')