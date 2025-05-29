import random
import json

from insec.utils import gpt_cost
from insec.AdversarialTokens import (
    AdversarialTokens,
    random_adv_tokens,
    is_forbidden_token,
)
from insec.BBSoftLossCalculator import BBSoftLossCalculator
from insec import Logger


class Individual:
    def __init__(self, tokens, loss):
        self.tokens = tokens
        self.loss = loss

    def __repr__(self) -> str:
        return str(self.tokens)

    def __str__(self) -> str:
        return str(self.tokens)


class RandomPoolOptimizer:
    def __init__(
        self,
        args,
        attack_tokenizer,
        loss_calculator: BBSoftLossCalculator,
        logger: Logger,
    ) -> None:
        self.args = args

        self.pool_size = args.pool_size

        self.pool: list[Individual] = []
        # setting loss to 10 to gracefully handle the stop at 5000 steps
        for attack in args.init_attack:
            self.pool.append(Individual(attack, 10))

        self.attack_tokenizer = attack_tokenizer
        self.loss_calculator = loss_calculator
        self.num_gen = args.num_gen
        self.total_opt_steps = args.total_opt_steps

        self.logger = logger

        self.almost_0_loss = 0.01
        self.times_hit_almost_0_loss = 0
        self.performed_steps = 0
        self.lowest_loss = 10

        self.candidate_history = [x for x in self.pool]
        self.pool_history = [[x for x in self.pool]]

    def best_attack(self):
        return self.pool[0].tokens

    def best_loss(self):
        return self.pool[0].loss

    def step(self, batch):
        # calculate loss for the initial pool the first time step is called
        if self.pool[0].loss == 10:
            for individual in self.pool:
                self.calculate_loss_individual(individual, batch)

        candidates = [self.sample_candidate(individual) for individual in self.pool]
        for candidate in candidates:
            self.calculate_loss_individual(candidate, batch)

        self.pool.extend(candidates)
        self.candidate_history.extend(candidates)

        self.pool.sort(key=lambda x: x.loss)
        self.pool = self.pool[: self.pool_size]
        self.pool_history.append([x for x in self.pool])

        self.log_pool()

    def log_pool(self):
        out_file = self.args.output_dir + "/candidates_log.json"
        with open(out_file, "w") as f:
            json.dump(
                [{"attack": x.tokens.to_json(), "loss": x.loss} for x in self.candidate_history],
                f,
                indent=4,
            )

        out_file = self.args.output_dir + "/pool_log.json"
        with open(out_file, "w") as f:
            json.dump(
                [[{"attack": x.tokens.to_json(), "loss": x.loss} for x in pool] for pool in self.pool_history],
                f,
                indent=4,
            )

    def calculate_loss_individual(self, individual, batch):
        self.performed_steps += 1
        if self.total_opt_steps is not None and self.performed_steps > self.total_opt_steps:
            return

        individual.loss = self.loss_calculator.forward(batch, individual.tokens, self.num_gen)

        if individual.loss < self.lowest_loss:
            self.lowest_loss = individual.loss

        self.logger.log(individual.tokens, individual.loss, self.pool[0].loss)
        self.update_stopping_criterion(individual.loss)

    def update_stopping_criterion(self, candidate_loss):
        if candidate_loss <= self.almost_0_loss:
            self.times_hit_almost_0_loss += 1

    def met_stop_criterion(self):
        return self.times_hit_almost_0_loss >= 5

    def sample_candidate(self, parent: Individual):
        new_tokens = [x for x in parent.tokens.tokens]

        n_replacements = self.sample_n_replacemetns(new_tokens)

        replacement_pos = random.sample(range(len(new_tokens)), n_replacements)
        for i in replacement_pos:
            while True:
                new_token_id = random.randint(0, self.attack_tokenizer.vocab_size - 1)
                new_token = self.attack_tokenizer.decode(new_token_id)
                if is_forbidden_token(new_token):
                    continue
                else:
                    break
            new_tokens[i] = new_token

        # setting loss to 10 to gracefully handle the stop at 5000 steps
        return Individual(AdversarialTokens(new_tokens, parent=parent.tokens), 10)

    def sample_n_replacemetns(self, tokens):
        return random.randint(1, len(tokens))
    
