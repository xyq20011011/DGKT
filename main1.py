from Trainer import Trainer
from model.DGKT import DGrKT
import os

dataset_list = ["ASSIST09", "ASSIST15", "ASSIST17", "ALGEBRA05", "Junyi"]

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset_name = "ASSIST09"
    trainer = Trainer(f"DGrKT_ASSIST09")
    trainer.verbose = False
    trainer.load_data(target_domain="ASSIST09")
    trainer.init_model(DGrKT)

    trainer.train(100, best_name="S1")
    trainer.load_model("S1")
    trainer.concept_aggregation(k=5)
    trainer.train(100, best_name="S2", centroid="source")

    trainer.load_model("S2")
    trainer.init_target_embedding()
    trainer.train_target(target_batch=1)
