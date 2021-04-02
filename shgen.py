stuff = []

for pfx in ['val', 'test', 'train']:
    with open(pfx + ".txt", 'r') as f:
        HOLD = f.read().split('\n')
        stuff += [(pfx, x) for x in HOLD if x != '']
with open("to30.sh", 'w') as f:
    for (pfx, fname) in stuff:
        fnamepfx, fnamesfx = fname.split('.')
        f.write(f"ffmpeg -i {pfx}/{fname} -r 30 {pfx}30/{fnamepfx}_30.{fnamesfx}\n")
