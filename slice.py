import argparse
import json
import os

# /results/jxli/ActivityNet/data/{>test30<, >val30<, train30}

# ffmpeg -i asdf.mp4 -ss 00:00:05.99 -to 00:01:05  csdf.mp4

# ["train.json", "val_1.json", "val_2.json"]

# "v_pizl41xmw7k": {
# "duration": 172.07999999999998, "
# timestamps": [[0, 85.18], [27.53, 43.02], [86.04, 168.64]],
#  "sentences": ["A child mops the floor of a hallway in a house.", 
# " The child sets the mop down and plays with her family member.", 
# " The child walks into the bedroom area and continues to mop the floor."]}

# srun
# singularity shell --nv /projects/singularity/images/Multisense.img 


parser = argparse.ArgumentParser(description="Script to slice videos between certain timestamps")
parser.add_argument('path', type=str, help='path of an ActivityNet json')
parser.add_argument('indir', type=str, help='path to 30 FPS videos') # has format v_#########_30.mp4
parser.add_argument('outdir', type=str, help='path to cut 30 FPS videos') # has format v_#########_30.mp4
parser.add_argument('out', type=str, help='Name of output file. ')

ARGS = parser.parse_args()
path, indir, outdir, out = ARGS.path, ARGS.indir, ARGS.outdir, ARGS.out

with open(path, 'r') as f: 
    labels = json.load(f)

availabile_vids = set()
for root, dirs, files in os.walk(indir):
    for filename in files:
        availabile_vids.add(filename)

convert_these = []

for key in labels:
    check_against = key + '_30.mp4'
    if check_against in availabile_vids:
        convert_these.append(key)

with open(out + '.sh', 'w') as f:
    for key in convert_these:
        for idx, (l, r) in enumerate(labels[key]['timestamps']):
            f.write(f"ffmpeg -i {indir}/{key}_30.mp4 -ss {l} -to {r} {outdir}/{key}_30_{idx}.mp4\n")
