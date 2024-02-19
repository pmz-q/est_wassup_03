from core.models.resnet50 import ResNet50
from core.configs import Resnet50Config

def resnet50_cls(cfg: Resnet50Config):
	model = ResNet50(weights=cfg.train_config.pretrained)
	model.train(
		train_img_path=cfg.get_img_dir_path("train"),
		val_img_path=cfg.get_img_dir_path("val"),
		train_ann_path=cfg.get_ann_file_path("train"),
		val_ann_path=cfg.get_ann_file_path("val"),
		epochs=cfg.train_config.epochs,
		lr_scheduler=cfg.lr_scheduler,
		optimizer=cfg.optimizer,
		loss_fn=cfg.loss_fn,
		batch_size=cfg.train_config.batch,
		num_workers=cfg.train_config.num_workers,
		device=cfg.device
	)
