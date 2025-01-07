from PIL import Image

from bmab import utils


def predict(image: Image, model, confidence):
	from ultralytics import YOLO
	yolo = utils.lazy_loader(model)
	boxes = []
	confs = []
	try:
		model = YOLO(yolo)
		pred = model(image, conf=confidence, device=utils.get_device())
		boxes = pred[0].boxes.xyxy.cpu().numpy()
		boxes = boxes.tolist()
		confs = pred[0].boxes.conf.tolist()
	except:
		pass
	del model
	utils.torch_gc()
	return boxes, confs

