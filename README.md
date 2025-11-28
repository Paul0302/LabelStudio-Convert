# LabelStudio-To-YOLO-COCO-Format
This repo provide a qucikly convert from label-studio json to yolo or COCO dataset 2025

# 🔥 Video export from Label-studio
Label-Studio GUI export for video label hava some limited, including only export key frame label, if you are going to export all label, please run
```bash
python /Src/export_label_studio.py
```

# 🔥 FPS_Changing
For changing FPS, run 
```bash
python /Src/fps_changer.py
```


# 🔥 Video+Json to Img+Label
For export data to Img+Annotation, run
```bash
python /Src/make_dataset.py
```
