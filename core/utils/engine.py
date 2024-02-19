import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from .utils import MetricLogger, SmoothedValue, reduce_dict
# from .coco_eval import CocoEvaluator
# from .coco_utils import get_coco_api_from_dataset


# TODO: scaler 추가
# classification 으로 다 뭉쳐놨음. 수정 필요
def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq, scaler=None):
	model.train()
	metric_logger = MetricLogger(delimiter="  ")
	metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
	header = f"Epoch: [{epoch}]"

	# TODO: warmup scheduler 정리
	# lr_scheduler = None
	# if epoch == 0:
	# 	warmup_factor = 1.0 / 1000
	# 	warmup_iters = min(1000, len(data_loader) - 1)
		
	# 	lr_scheduler = torch.optim.lr_scheduler.LinearLR(
	# 			optimizer, start_factor=warmup_factor, total_iters=warmup_iters
	# 	)

	for images, targets in metric_logger.log_every(data_loader, print_freq, header):
		images = list(image.to(device) for image in images)
		targets = [t.to(device) for t in targets]
		print(targets)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			preds = model(images, targets)
			loss = criterion(preds, targets)
			train_loss += loss.item()
			pred_classes = torch.argmax(torch.softmax(preds,dim=1),dim = 1)
			train_acc += torch.sum(pred_classes == targets) / len(targets)
		# reduce losses over all GPUs for logging purposes
		# loss_dict_reduced = reduce_dict(preds)
		# losses_reduced = sum(loss for loss in loss_dict_reduced.values())

		# loss_value = losses_reduced.item()

		# if not math.isfinite(loss_value):
		# 	print(f"Loss is {loss_value}, stopping training")
		# 	print(loss_dict_reduced)
		# 	sys.exit(1)

		optimizer.zero_grad()
		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		# TODO: warmup scheduler 정리
		# if lr_scheduler is not None:
		# 	lr_scheduler.step()

		metric_logger.update(loss=train_loss / len(data_loader), acc=train_acc / len(data_loader))
		metric_logger.update(lr=optimizer.param_groups[0]["lr"])

	return metric_logger


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device):
	n_threads = torch.get_num_threads()
	# FIXME remove this and make paste_masks_in_image run on the GPU
	torch.set_num_threads(1)
	cpu_device = torch.device("cpu")
	model.eval()
	metric_logger = MetricLogger(delimiter="  ")
	header = "Test:"

	# coco = get_coco_api_from_dataset(data_loader.dataset)
	# iou_types = _get_iou_types(model)
	# coco_evaluator = CocoEvaluator(coco, iou_types)

	for images, targets in metric_logger.log_every(data_loader, 100, header):
		images = list(img.to(device) for img in images)

		if torch.cuda.is_available():
				torch.cuda.synchronize()
		model_time = time.time()
		preds = model(images)

		preds = [v.to(cpu_device) for v in preds]
		model_time = time.time() - model_time

		# res = {target["image_id"]: output for target, output in zip(targets, preds)}
		evaluator_time = time.time()
		# coco_evaluator.update(res)
		loss =  criterion(preds , targets)
		val_loss += loss.item()
		pred_classes = torch.argmax(torch.softmax(preds , dim = 1),dim=1)
		val_acc += torch.sum(targets == pred_classes) / len(targets)
		evaluator_time = time.time() - evaluator_time
		metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
		# metric_logger.update(model_time=model_time)
		
	return val_acc / len(data_loader) , val_loss / len(data_loader)
	# # gather the stats from all processes
	# metric_logger.synchronize_between_processes()
	# print("Averaged stats:", metric_logger)
	# # coco_evaluator.synchronize_between_processes()

	# # # accumulate predictions from all images
	# # coco_evaluator.accumulate()
	# # coco_evaluator.summarize()
	# torch.set_num_threads(n_threads)
	# return coco_evaluator
