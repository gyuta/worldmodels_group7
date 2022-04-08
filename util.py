from datetime import datetime
import os
import pickle

def get_strdate():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_outputdir():
    path = f"data/{get_strdate()}"
    os.mkdir(path)
    return path

def save(obj, path, name):
    with open(f"{path}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def write(txt, path, name):
    with open(f"{path}/{name}.txt", "a") as f:
        f.write(f"{txt}\n")