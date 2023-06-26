import gdown

if __name__ =="__main__":
    # Download and extract the GRID dataset in ./data folder
    url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
    output = 'data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('data.zip')