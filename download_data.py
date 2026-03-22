import urllib.request
import os

os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
save_path = "data/heart.csv"

print("Downloading dataset...")
urllib.request.urlretrieve(url, save_path)
print("Done! File saved to data/heart.csv")