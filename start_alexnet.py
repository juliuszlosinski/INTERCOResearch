import json

from interco_yolov8m import INTERCOYolov8m

with open("configuration.json", "r") as f:
    config = json.load(f)

type = "alexnet"
run = int(config["run"])

pretrained = config["pretrained"]
yaml_file = config["yaml_file"]
epochs = config["epochs"]
imgsz = config["imgsz"]
batch_size = config["batch"]
workers = config["workers"]

interco_model = INTERCOYolov8m(type=type, pretrained=pretrained)
interco_model.print_model()
interco_model.print_params()

if run:
    interco_model.train(
        yaml_file=yaml_file,
        number_of_epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
    )
