from ultralytics import YOLO
model = YOLO('best.pt')

import requests, zipfile, shutil, os, stat, io

api = requests.get("https://api.github.com/repos/pnnx/pnnx/releases/latest").json()
print("Vesrion: ", api['tag_name'])
 
asset_url = next(a['browser_download_url'] for a in api['assets'] 
                 if a['name'] == 'pnnx-20260409-linux.zip')
print("URL:", asset_url)

r = requests.get(asset_url)
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    print("Files in directory:", z.namelist())
    z.extractall("pnnx_x86")

#pnnx_bin = next(f for f in os.listdir("pnnx_x86") if 'x86_64' in f or f == 'pnnx')
src = "pnnx_x86/pnnx-20260409-linux/pnnx"
dst = '/usr/local/lib/python3.8/dist-packages/ultralytics/pnnx'
shutil.copy2(src, dst)
os.chmod(dst, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

model.export(format='ncnn', imgsz=640, half=True)
