from torch.utils.data import DataLoader

from insec import AttackedInfillingDataset
from insec import BBSoftLossCalculator
from insec.optimizers import RandomPoolOptimizer
from insec.ModelWrapper import load_model
from insec import Logger


class AdversarialTrainer:
    """Learns a prompt prefix for the secure and vulnerable scenario."""

    def __init__(self, args):
        self.args = args

        self.dataset, self.data_loader = self.load_dataset()
        self.model = load_model(args)
        self.attack_tokenizer = self.load_attack_tokenizer()
        self.loss_calculator = self.create_loss_calculator()

        logger = Logger(args.output_dir, args.model_dir, args.dataset, args.experiment_name, self.loss_calculator)
        self.optimizer = self.create_optimizer(logger)

    def load_attack_tokenizer(self):
        return self.args.attack_tokenizer

    def load_dataset(self):
        dataset = AttackedInfillingDataset(self.args, "train")
        if self.args.batch_size is None:
            self.args.batch_size = len(dataset.dataset)
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=lambda x: x,
            drop_last=True,
        )
        return dataset, data_loader

    def create_loss_calculator(self):
        if self.args.loss_type == "bbsoft":
            return BBSoftLossCalculator(
                self.args.device,
                self.model,
                self.args.batch_size,
                self.attack_tokenizer,
                self.args,
            )
        else:
            raise ValueError("Unknown loss type")

    def create_optimizer(self, logger):
        if self.args.optimizer == "random_pool":
            optimizer = RandomPoolOptimizer(
                self.args,
                self.attack_tokenizer,
                self.loss_calculator,
                logger,
            )
        else:
            raise ValueError("Unknown optimizer")

        return optimizer

    def run(self, save_fn=None):
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.data_loader):
                self.log_training_info(epoch, step)
                self.optimizer.step(batch)
                if self.optimizer.met_stop_criterion():
                    print("Met the stopping criterion, exiting")
                    if isinstance(self.optimizer, RandomPoolOptimizer):
                        print(self.optimizer.pool)
                    return self.optimizer.best_attack(), self.optimizer.best_loss()
            
            if save_fn is not None:
                final_pool = get_final_pool(self)
                save_fn(self.optimizer.best_attack(), self.optimizer.best_loss(), final_pool, self.args, epoch)
                if self.args.save_intermediate and epoch % 25 == 0:
                    save_fn(self.optimizer.best_attack(), self.optimizer.best_loss(), final_pool, self.args, epoch, separate=True)

        if isinstance(self.optimizer, RandomPoolOptimizer):
            print(self.optimizer.pool)
        return self.optimizer.best_attack(), self.optimizer.best_loss()

    def log_training_info(self, epoch, step):
        print(f"step: {epoch + 1}/{self.args.num_train_epochs}")

def get_final_pool(trainer):
    final_pool_base = trainer.optimizer.pool_history[-1]
    final_pool = [{**x.tokens.to_json(), "loss": x.loss} for x in final_pool_base]
    return final_pool