import sys
import torch
from bmab import utils
import os
import folder_paths


def get_device():
	if sys.platform == 'darwin':
		# MPS is not good.
		return 'cpu'
	elif torch.cuda.is_available():
		return 'cuda'
	return 'cpu'


def predict(pilimg, prompt, box_threahold=0.35, text_threshold=0.25, device=get_device()):
	from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
	if os.path.exists("/stable-diffusion-cache/models/grounding-dino-base"):
		model_id = "/stable-diffusion-cache/models/grounding-dino-base"
	else:
		model_id = os.path.join(folder_paths.models_dir, "grounding-dino-base")
		print(f'pls download IDEA-Research/grounding-dino-base to {model_id}')

	processor = AutoProcessor.from_pretrained(model_id)
	model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

	inputs = processor(images=pilimg, text=prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	results = processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		box_threshold=box_threahold,
		text_threshold=text_threshold,
		target_sizes=[pilimg.size[::-1]]
	)
	del processor
	model.to('cpu')
	utils.torch_gc()

	result = results[0]
	return result["boxes"], result["scores"], result["labels"]
