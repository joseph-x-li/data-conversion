import youtube_dl
import json


path = "dataset/val_ids.json"

outpath = "data/val"

with open(path) as f:
    data = json.load(f)


options = {
    "quiet": True,
    "outtmpl": outpath + "/v_%(id)s.%(ext)s",
    "ignoreerrors": True,
}

data = [f"https://www.youtube.com/watch?v={v_id[2:]}" for v_id in data]

with youtube_dl.YoutubeDL(options) as ydl:
    ydl.download(data)

