import argparse
import json
import os

# ["train.json", "val_1.json", "val_2.json"]

parser = argparse.ArgumentParser(description="Script to slice videos between certain timestamps")
parser.add_argument('path', type=str, help='path of an ActivityNet json')
parser.add_argument('videos', type=str, help='path to 30 FPS videos') # has format v_#########_30.mp4
parser.add_argument('out', type=str, help='Name of output file. ')

ARGS = parser.parse_args()
path = ARGS.path
out = ARGS.out

with open(path, 'r') as f: 
    labels = json.load(f)

availabile_vids = set()
for root, dirs, files in os.walk("."):
    for filename in files:
        availabile_vids.add(filename)

convert_these = []

for key in labels:
    check_against = key + '_30.mp4'
    if check_against in availabile_vids:
        convert_these.append(key)

with open(out + '.sh', 'w') as f:
    for key in convert_these:
        f.write(f"ffmpeg -i {videos}/{key}_30.mp4 -ss {????} -to {????}")
        f.write('\n')
    
