"""MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation
https://arxiv.org/abs/2107.03374

MultiPL-E is a dataset for evaluating large language models for code generation that supports 18 programming languages.
It takes the OpenAI "HumanEval" and the MBPP Python benchmarks and uses little compilers to translate them to other languages.

Homepage: https://nuprl.github.io/MultiPL-E/
"""

import json
import os
import re
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from time import time
import uuid

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.multiple_metrics.evaluation import (
    evaluate_problem,
)
from bigcode_eval.tasks.custom_metrics.multiple_metrics.single_experiment_pass_k import (
    for_file,
)


_CITATION = """
@article{cassano2022scalable,
  title={A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={arXiv preprint arXiv:2208.08227},
  year={2022}
}
"""

LANGUAGES = [
    "py",
    "sh",
    "cpp",
    "cs",
    "d",
    "go",
    "java",
    "js",
    "jl",
    "lua",
    "pl",
    "php",
    "r",
    "rkt",
    "rb",
    "rs",
    "scala",
    "swift",
    "ts",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {f"multiple-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class MultiPLE(GeneralMultiPLE):
        def __init__(self):
            super().__init__(language)

    return MultiPLE


class GeneralMultiPLE(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "nuprl/MultiPL-E"
    DATASET_NAME = None
    DATASET_REVISION = "d23b094346c5dbda1080a74bb2a24c18adbf7409"

    def __init__(self, language):
        self.language = language
        self.DATASET_NAME = f"humaneval-{language}"
        # we need the dataset to get stop words for each language
        self.dataset = load_dataset(
            GeneralMultiPLE.DATASET_PATH,
            self.DATASET_NAME,
            revision=self.DATASET_REVISION,
            trust_remote_code=True,
        )
        stop_words = self.dataset["test"][0]["stop_tokens"] + ["<file_sep>"]
        super().__init__(
            stop_words=stop_words,
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"].strip()

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["tests"]

    @staticmethod
    def remove_last_block(string, stop_words):
        # Remove the last block of the code containing stop_words for HumanEval
        string_list = re.split("(%s)" % "|".join(stop_words), string)
        # last string should be ""
        return "".join(string_list[:-2])

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        completion = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(completion, self.stop_words)

    def reorder_generations(self, generations, prompt_names):
        global order_dict
        initial_order = order_dict[self.language]

        # reorder the generations to match the order of prompt_name
        assert len(generations) == len(initial_order), (
            f"Number of generations ({len(generations)}) "
            f"does not match the number of prompt_names ({len(initial_order)})"
        )
        gen_dict = {n: gen for n, gen in zip (initial_order, generations)}
        return [gen_dict[name] for name in prompt_names]        


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        task_name = f"multiple-{self.language}"
        # get prompts and problem names
        prompts_names = [
            {"prompt": doc["prompt"], "name": doc["name"]}
            for i, doc in enumerate(self.get_dataset())
            if i < len(generations)
        ]

        # reorder generations to match the prompts
        if self.language != "py":
            generations = self.reorder_generations(generations, [x["name"] for x in prompts_names])

        # a common temp dir for all the problems
        temp_dir = tempfile.gettempdir()
        temp_dir = f"tmp/{task_name}-{uuid.uuid4().hex}"
        # delete the directory temp_dir
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

        os.makedirs(temp_dir, exist_ok=True)
        list_files = []
        good_problems = 0
        for prompt_name, generation, reference in zip(
            prompts_names, generations, references
        ):
            if generation[0] == "":
                continue
            good_problems += 1

            problem = {
                "name": prompt_name["name"],
                "language": self.language,
                "prompt": prompt_name["prompt"],
                "completions": generation,
                "tests": reference,
            }
            
            # print(problem["name"])
            # print("-" * 80)
            # print(problem["prompt"][:600])
            # print("-" * 80)
            # print(problem["completions"][0][:600])
            # input("Press Enter to continue...")

            if "cpp" in task_name:
                problem["tests"] = problem["tests"][1:]
            # print(repr(problem["tests"]))
            # input()

            # each problem is save in a json file
            temp_file_name = os.path.join(temp_dir, f"{prompt_name['name']}.json")
            list_files.append(temp_file_name)
            with open(temp_file_name, "wt") as f:
                json.dump(problem, f)
        print(
            f"Saved {good_problems} problems in {temp_dir} for evaluation, each problem has {len(generations[0])} completions"
        )
        # execute the problems to evaluate them
        max_workers = cpu_count() - 1 if cpu_count() > 1 else 1
        for file in tqdm(list_files):
            evaluate_problem(temp_dir, file, max_workers)

        # compute pass@k scores
        result_array = np.array(
            [for_file(p) for p in Path(temp_dir).glob("*.results.json")]
        )
        result = result_array.mean(axis=0)
        name = (
            temp_dir.split("/")[-1]
            if temp_dir.split("/")[-1] != ""
            else temp_dir.split("/")[-2]
        )
        result_dict = {
                str(p): list(for_file(p)) for p in Path(temp_dir).glob("*.results.json")
        }
        results = {
            f"pass@{k}": v
            for k, v in zip([1, 10, 100], result)
            if k <= len(generations[0])
        }
        results["all"] = result_dict
        return results



order_dict = {'py': ['SingleLineInfilling/HumanEval/23/L0', 'SingleLineInfilling/HumanEval/89/L0', 'SingleLineInfilling/HumanEval/95/L17', 'SingleLineInfilling/HumanEval/85/L0', 'SingleLineInfilling/HumanEval/140/L2', 'SingleLineInfilling/HumanEval/63/L4', 'SingleLineInfilling/HumanEval/151/L0', 'SingleLineInfilling/HumanEval/22/L0', 'SingleLineInfilling/HumanEval/41/L0', 'SingleLineInfilling/HumanEval/17/L1', 'SingleLineInfilling/HumanEval/79/L0', 'SingleLineInfilling/HumanEval/14/L3', 'SingleLineInfilling/HumanEval/53/L0', 'SingleLineInfilling/HumanEval/159/L1', 'SingleLineInfilling/HumanEval/115/L0', 'SingleLineInfilling/HumanEval/160/L0', 'SingleLineInfilling/HumanEval/27/L0', 'SingleLineInfilling/HumanEval/105/L2', 'SingleLineInfilling/HumanEval/25/L12', 'SingleLineInfilling/HumanEval/96/L4', 'SingleLineInfilling/HumanEval/34/L0', 'SingleLineInfilling/HumanEval/74/L11', 'SingleLineInfilling/HumanEval/35/L4', 'SingleLineInfilling/HumanEval/132/L2', 'SingleLineInfilling/HumanEval/103/L5', 'SingleLineInfilling/HumanEval/113/L1', 'SingleLineInfilling/HumanEval/109/L8', 'SingleLineInfilling/HumanEval/107/L11', 'SingleLineInfilling/HumanEval/138/L0', 'SingleLineInfilling/HumanEval/62/L0', 'SingleLineInfilling/HumanEval/126/L3', 'SingleLineInfilling/HumanEval/161/L10', 'SingleLineInfilling/HumanEval/130/L8', 'SingleLineInfilling/HumanEval/36/L7', 'SingleLineInfilling/HumanEval/29/L0', 'SingleLineInfilling/HumanEval/84/L0', 'SingleLineInfilling/HumanEval/129/L25', 'SingleLineInfilling/HumanEval/98/L1', 'SingleLineInfilling/HumanEval/120/L1', 'SingleLineInfilling/HumanEval/24/L0', 'SingleLineInfilling/HumanEval/88/L0', 'SingleLineInfilling/HumanEval/106/L5', 'SingleLineInfilling/HumanEval/77/L0', 'SingleLineInfilling/HumanEval/93/L3', 'SingleLineInfilling/HumanEval/91/L2', 'SingleLineInfilling/HumanEval/43/L3', 'SingleLineInfilling/HumanEval/71/L0', 'SingleLineInfilling/HumanEval/148/L1', 'SingleLineInfilling/HumanEval/131/L9', 'SingleLineInfilling/HumanEval/101/L9', 'SingleLineInfilling/HumanEval/18/L6', 'SingleLineInfilling/HumanEval/137/L4', 'SingleLineInfilling/HumanEval/51/L0', 'SingleLineInfilling/HumanEval/70/L3', 'SingleLineInfilling/HumanEval/20/L13', 'SingleLineInfilling/HumanEval/76/L2', 'SingleLineInfilling/HumanEval/39/L0', 'SingleLineInfilling/HumanEval/145/L0', 'SingleLineInfilling/HumanEval/0/L7', 'SingleLineInfilling/HumanEval/10/L6', 'SingleLineInfilling/HumanEval/11/L1', 'SingleLineInfilling/HumanEval/139/L3', 'SingleLineInfilling/HumanEval/122/L0', 'SingleLineInfilling/HumanEval/46/L8', 'SingleLineInfilling/HumanEval/104/L2', 'SingleLineInfilling/HumanEval/117/L1', 'SingleLineInfilling/HumanEval/72/L7', 'SingleLineInfilling/HumanEval/55/L4', 'SingleLineInfilling/HumanEval/153/L9', 'SingleLineInfilling/HumanEval/119/L5', 'SingleLineInfilling/HumanEval/90/L0', 'SingleLineInfilling/HumanEval/92/L5', 'SingleLineInfilling/HumanEval/2/L0', 'SingleLineInfilling/HumanEval/42/L0', 'SingleLineInfilling/HumanEval/150/L3', 'SingleLineInfilling/HumanEval/49/L2', 'SingleLineInfilling/HumanEval/155/L1', 'SingleLineInfilling/HumanEval/80/L0', 'SingleLineInfilling/HumanEval/59/L5', 'SingleLineInfilling/HumanEval/66/L1', 'SingleLineInfilling/HumanEval/21/L2', 'SingleLineInfilling/HumanEval/121/L0', 'SingleLineInfilling/HumanEval/68/L1', 'SingleLineInfilling/HumanEval/147/L2', 'SingleLineInfilling/HumanEval/110/L9', 'SingleLineInfilling/HumanEval/47/L2', 'SingleLineInfilling/HumanEval/82/L5', 'SingleLineInfilling/HumanEval/73/L2', 'SingleLineInfilling/HumanEval/133/L0', 'SingleLineInfilling/HumanEval/141/L5', 'SingleLineInfilling/HumanEval/40/L4', 'SingleLineInfilling/HumanEval/127/L0', 'SingleLineInfilling/HumanEval/1/L7', 'SingleLineInfilling/HumanEval/152/L0', 'SingleLineInfilling/HumanEval/83/L1', 'SingleLineInfilling/HumanEval/134/L1', 'SingleLineInfilling/HumanEval/124/L1', 'SingleLineInfilling/HumanEval/108/L0', 'SingleLineInfilling/HumanEval/86/L0', 'SingleLineInfilling/HumanEval/48/L1', 'SingleLineInfilling/HumanEval/118/L4', 'SingleLineInfilling/HumanEval/31/L3', 'SingleLineInfilling/HumanEval/144/L4', 'SingleLineInfilling/HumanEval/78/L1', 'SingleLineInfilling/HumanEval/143/L0', 'SingleLineInfilling/HumanEval/111/L12', 'SingleLineInfilling/HumanEval/87/L0', 'SingleLineInfilling/HumanEval/123/L0', 'SingleLineInfilling/HumanEval/135/L5', 'SingleLineInfilling/HumanEval/19/L11', 'SingleLineInfilling/HumanEval/65/L0', 'SingleLineInfilling/HumanEval/142/L1', 'SingleLineInfilling/HumanEval/94/L12', 'SingleLineInfilling/HumanEval/8/L4', 'SingleLineInfilling/HumanEval/102/L3', 'SingleLineInfilling/HumanEval/136/L0', 'SingleLineInfilling/HumanEval/16/L0', 'SingleLineInfilling/HumanEval/100/L0', 'SingleLineInfilling/HumanEval/128/L2', 'SingleLineInfilling/HumanEval/114/L3', 'SingleLineInfilling/HumanEval/15/L0', 'SingleLineInfilling/HumanEval/154/L3', 'SingleLineInfilling/HumanEval/57/L0', 'SingleLineInfilling/HumanEval/12/L5', 'SingleLineInfilling/HumanEval/52/L1', 'SingleLineInfilling/HumanEval/75/L3', 'SingleLineInfilling/HumanEval/30/L0', 'SingleLineInfilling/HumanEval/33/L2', 'SingleLineInfilling/HumanEval/6/L6', 'SingleLineInfilling/HumanEval/45/L0', 'SingleLineInfilling/HumanEval/97/L0', 'SingleLineInfilling/HumanEval/4/L1', 'SingleLineInfilling/HumanEval/58/L0', 'SingleLineInfilling/HumanEval/156/L13', 'SingleLineInfilling/HumanEval/67/L4', 'SingleLineInfilling/HumanEval/112/L1', 'SingleLineInfilling/HumanEval/13/L0', 'SingleLineInfilling/HumanEval/125/L1', 'SingleLineInfilling/HumanEval/116/L0', 'SingleLineInfilling/HumanEval/28/L0', 'SingleLineInfilling/HumanEval/149/L3', 'SingleLineInfilling/HumanEval/7/L0', 'SingleLineInfilling/HumanEval/99/L15', 'SingleLineInfilling/HumanEval/64/L2', 'SingleLineInfilling/HumanEval/158/L0', 'SingleLineInfilling/HumanEval/162/L1', 'SingleLineInfilling/HumanEval/44/L0', 'SingleLineInfilling/HumanEval/157/L0', 'SingleLineInfilling/HumanEval/81/L21', 'SingleLineInfilling/HumanEval/5/L11', 'SingleLineInfilling/HumanEval/146/L4', 'SingleLineInfilling/HumanEval/60/L0', 'SingleLineInfilling/HumanEval/26/L0', 'SingleLineInfilling/HumanEval/163/L1', 'SingleLineInfilling/HumanEval/9/L6', 'SingleLineInfilling/HumanEval/3/L0', 'SingleLineInfilling/HumanEval/69/L5', 'SingleLineInfilling/HumanEval/61/L1', 'SingleLineInfilling/HumanEval/37/L7', 'SingleLineInfilling/HumanEval/54/L0', 'SingleLineInfilling/HumanEval/56/L7'], 'js': ['HumanEval_23_strlen', 'HumanEval_89_encrypt', 'HumanEval_95_check_dict_case', 'HumanEval_85_add', 'HumanEval_140_fix_spaces', 'HumanEval_63_fibfib', 'HumanEval_151_double_the_difference', 'HumanEval_22_filter_integers', 'HumanEval_41_car_race_collision', 'HumanEval_17_parse_music', 'HumanEval_79_decimal_to_binary', 'HumanEval_14_all_prefixes', 'HumanEval_53_add', 'HumanEval_159_eat', 'HumanEval_115_max_fill', 'HumanEval_160_do_algebra', 'HumanEval_27_flip_case', 'HumanEval_105_by_length', 'HumanEval_25_factorize', 'HumanEval_96_count_up_to', 'HumanEval_34_unique', 'HumanEval_74_total_match', 'HumanEval_35_max_element', 'HumanEval_132_is_nested', 'HumanEval_103_rounded_avg', 'HumanEval_113_odd_count', 'HumanEval_109_move_one_ball', 'HumanEval_107_even_odd_palindrome', 'HumanEval_138_is_equal_to_sum_even', 'HumanEval_62_derivative', 'HumanEval_126_is_sorted', 'HumanEval_161_solve', 'HumanEval_130_tri', 'HumanEval_36_fizz_buzz', 'HumanEval_29_filter_by_prefix', 'HumanEval_84_solve', 'HumanEval_129_minPath', 'HumanEval_98_count_upper', 'HumanEval_120_maximum', 'HumanEval_24_largest_divisor', 'HumanEval_88_sort_array', 'HumanEval_106_f', 'HumanEval_77_iscube', 'HumanEval_93_encode', 'HumanEval_91_is_bored', 'HumanEval_43_pairs_sum_to_zero', 'HumanEval_71_triangle_area', 'HumanEval_148_bf', 'HumanEval_131_digits', 'HumanEval_101_words_string', 'HumanEval_18_how_many_times', 'HumanEval_137_compare_one', 'HumanEval_51_remove_vowels', 'HumanEval_70_strange_sort_list', 'HumanEval_20_find_closest_elements', 'HumanEval_76_is_simple_power', 'HumanEval_39_prime_fib', 'HumanEval_145_order_by_points', 'HumanEval_0_has_close_elements', 'HumanEval_10_make_palindrome', 'HumanEval_11_string_xor', 'HumanEval_139_special_factorial', 'HumanEval_122_add_elements', 'HumanEval_46_fib4', 'HumanEval_104_unique_digits', 'HumanEval_117_select_words', 'HumanEval_72_will_it_fly', 'HumanEval_55_fib', 'HumanEval_153_Strongest_Extension', 'HumanEval_119_match_parens', 'HumanEval_90_next_smallest', 'HumanEval_92_any_int', 'HumanEval_2_truncate_number', 'HumanEval_42_incr_list', 'HumanEval_150_x_or_y', 'HumanEval_49_modp', 'HumanEval_155_even_odd_count', 'HumanEval_80_is_happy', 'HumanEval_59_largest_prime_factor', 'HumanEval_66_digitSum', 'HumanEval_21_rescale_to_unit', 'HumanEval_121_solution', 'HumanEval_68_pluck', 'HumanEval_147_get_max_triples', 'HumanEval_110_exchange', 'HumanEval_47_median', 'HumanEval_82_prime_length', 'HumanEval_73_smallest_change', 'HumanEval_133_sum_squares', 'HumanEval_141_file_name_check', 'HumanEval_40_triples_sum_to_zero', 'HumanEval_127_intersection', 'HumanEval_1_separate_paren_groups', 'HumanEval_152_compare', 'HumanEval_83_starts_one_ends', 'HumanEval_134_check_if_last_char_is_a_letter', 'HumanEval_124_valid_date', 'HumanEval_108_count_nums', 'HumanEval_86_anti_shuffle', 'HumanEval_48_is_palindrome', 'HumanEval_118_get_closest_vowel', 'HumanEval_31_is_prime', 'HumanEval_144_simplify', 'HumanEval_78_hex_key', 'HumanEval_143_words_in_sentence', 'HumanEval_111_histogram', 'HumanEval_87_get_row', 'HumanEval_123_get_odd_collatz', 'HumanEval_135_can_arrange', 'HumanEval_19_sort_numbers', 'HumanEval_65_circular_shift', 'HumanEval_142_sum_squares', 'HumanEval_94_skjkasdkd', 'HumanEval_8_sum_product', 'HumanEval_102_choose_num', 'HumanEval_136_largest_smallest_integers', 'HumanEval_16_count_distinct_characters', 'HumanEval_100_make_a_pile', 'HumanEval_128_prod_signs', 'HumanEval_114_minSubArraySum', 'HumanEval_15_string_sequence', 'HumanEval_154_cycpattern_check', 'HumanEval_57_monotonic', 'HumanEval_12_longest', 'HumanEval_52_below_threshold', 'HumanEval_75_is_multiply_prime', 'HumanEval_30_get_positive', 'HumanEval_33_sort_third', 'HumanEval_6_parse_nested_parens', 'HumanEval_45_triangle_area', 'HumanEval_97_multiply', 'HumanEval_4_mean_absolute_deviation', 'HumanEval_58_common', 'HumanEval_156_int_to_mini_roman', 'HumanEval_67_fruit_distribution', 'HumanEval_112_reverse_delete', 'HumanEval_13_greatest_common_divisor', 'HumanEval_125_split_words', 'HumanEval_116_sort_array', 'HumanEval_28_concatenate', 'HumanEval_149_sorted_list_sum', 'HumanEval_7_filter_by_substring', 'HumanEval_99_closest_integer', 'HumanEval_64_vowels_count', 'HumanEval_158_find_max', 'HumanEval_162_string_to_md5', 'HumanEval_44_change_base', 'HumanEval_157_right_angle_triangle', 'HumanEval_81_numerical_letter_grade', 'HumanEval_5_intersperse', 'HumanEval_146_specialFilter', 'HumanEval_60_sum_to_n', 'HumanEval_26_remove_duplicates', 'HumanEval_163_generate_integers', 'HumanEval_9_rolling_max', 'HumanEval_3_below_zero', 'HumanEval_69_search', 'HumanEval_61_correct_bracketing', 'HumanEval_37_sort_even', 'HumanEval_54_same_chars', 'HumanEval_56_correct_bracketing'], 'cpp': ['HumanEval_23_strlen', 'HumanEval_89_encrypt', 'HumanEval_95_check_dict_case', 'HumanEval_85_add', 'HumanEval_140_fix_spaces', 'HumanEval_63_fibfib', 'HumanEval_151_double_the_difference', 'HumanEval_22_filter_integers', 'HumanEval_41_car_race_collision', 'HumanEval_17_parse_music', 'HumanEval_79_decimal_to_binary', 'HumanEval_14_all_prefixes', 'HumanEval_53_add', 'HumanEval_159_eat', 'HumanEval_115_max_fill', 'HumanEval_160_do_algebra', 'HumanEval_27_flip_case', 'HumanEval_105_by_length', 'HumanEval_25_factorize', 'HumanEval_96_count_up_to', 'HumanEval_34_unique', 'HumanEval_74_total_match', 'HumanEval_35_max_element', 'HumanEval_132_is_nested', 'HumanEval_103_rounded_avg', 'HumanEval_113_odd_count', 'HumanEval_109_move_one_ball', 'HumanEval_107_even_odd_palindrome', 'HumanEval_138_is_equal_to_sum_even', 'HumanEval_62_derivative', 'HumanEval_126_is_sorted', 'HumanEval_161_solve', 'HumanEval_130_tri', 'HumanEval_36_fizz_buzz', 'HumanEval_29_filter_by_prefix', 'HumanEval_84_solve', 'HumanEval_129_minPath', 'HumanEval_98_count_upper', 'HumanEval_120_maximum', 'HumanEval_24_largest_divisor', 'HumanEval_88_sort_array', 'HumanEval_106_f', 'HumanEval_77_iscube', 'HumanEval_93_encode', 'HumanEval_91_is_bored', 'HumanEval_43_pairs_sum_to_zero', 'HumanEval_71_triangle_area', 'HumanEval_148_bf', 'HumanEval_131_digits', 'HumanEval_101_words_string', 'HumanEval_18_how_many_times', 'HumanEval_137_compare_one', 'HumanEval_51_remove_vowels', 'HumanEval_70_strange_sort_list', 'HumanEval_20_find_closest_elements', 'HumanEval_76_is_simple_power', 'HumanEval_39_prime_fib', 'HumanEval_145_order_by_points', 'HumanEval_0_has_close_elements', 'HumanEval_10_make_palindrome', 'HumanEval_11_string_xor', 'HumanEval_139_special_factorial', 'HumanEval_122_add_elements', 'HumanEval_46_fib4', 'HumanEval_104_unique_digits', 'HumanEval_117_select_words', 'HumanEval_72_will_it_fly', 'HumanEval_55_fib', 'HumanEval_153_Strongest_Extension', 'HumanEval_119_match_parens', 'HumanEval_90_next_smallest', 'HumanEval_92_any_int', 'HumanEval_2_truncate_number', 'HumanEval_42_incr_list', 'HumanEval_150_x_or_y', 'HumanEval_49_modp', 'HumanEval_155_even_odd_count', 'HumanEval_80_is_happy', 'HumanEval_59_largest_prime_factor', 'HumanEval_66_digitSum', 'HumanEval_21_rescale_to_unit', 'HumanEval_121_solution', 'HumanEval_68_pluck', 'HumanEval_147_get_max_triples', 'HumanEval_110_exchange', 'HumanEval_47_median', 'HumanEval_82_prime_length', 'HumanEval_73_smallest_change', 'HumanEval_133_sum_squares', 'HumanEval_141_file_name_check', 'HumanEval_40_triples_sum_to_zero', 'HumanEval_127_intersection', 'HumanEval_1_separate_paren_groups', 'HumanEval_152_compare', 'HumanEval_83_starts_one_ends', 'HumanEval_134_check_if_last_char_is_a_letter', 'HumanEval_124_valid_date', 'HumanEval_108_count_nums', 'HumanEval_86_anti_shuffle', 'HumanEval_48_is_palindrome', 'HumanEval_118_get_closest_vowel', 'HumanEval_31_is_prime', 'HumanEval_144_simplify', 'HumanEval_78_hex_key', 'HumanEval_143_words_in_sentence', 'HumanEval_111_histogram', 'HumanEval_87_get_row', 'HumanEval_123_get_odd_collatz', 'HumanEval_135_can_arrange', 'HumanEval_19_sort_numbers', 'HumanEval_65_circular_shift', 'HumanEval_142_sum_squares', 'HumanEval_94_skjkasdkd', 'HumanEval_8_sum_product', 'HumanEval_102_choose_num', 'HumanEval_136_largest_smallest_integers', 'HumanEval_16_count_distinct_characters', 'HumanEval_100_make_a_pile', 'HumanEval_128_prod_signs', 'HumanEval_114_minSubArraySum', 'HumanEval_15_string_sequence', 'HumanEval_154_cycpattern_check', 'HumanEval_57_monotonic', 'HumanEval_12_longest', 'HumanEval_52_below_threshold', 'HumanEval_75_is_multiply_prime', 'HumanEval_30_get_positive', 'HumanEval_33_sort_third', 'HumanEval_6_parse_nested_parens', 'HumanEval_45_triangle_area', 'HumanEval_97_multiply', 'HumanEval_4_mean_absolute_deviation', 'HumanEval_58_common', 'HumanEval_156_int_to_mini_roman', 'HumanEval_67_fruit_distribution', 'HumanEval_112_reverse_delete', 'HumanEval_13_greatest_common_divisor', 'HumanEval_125_split_words', 'HumanEval_116_sort_array', 'HumanEval_28_concatenate', 'HumanEval_149_sorted_list_sum', 'HumanEval_7_filter_by_substring', 'HumanEval_99_closest_integer', 'HumanEval_64_vowels_count', 'HumanEval_158_find_max', 'HumanEval_162_string_to_md5', 'HumanEval_44_change_base', 'HumanEval_157_right_angle_triangle', 'HumanEval_81_numerical_letter_grade', 'HumanEval_5_intersperse', 'HumanEval_146_specialFilter', 'HumanEval_60_sum_to_n', 'HumanEval_26_remove_duplicates', 'HumanEval_163_generate_integers', 'HumanEval_9_rolling_max', 'HumanEval_3_below_zero', 'HumanEval_69_search', 'HumanEval_61_correct_bracketing', 'HumanEval_37_sort_even', 'HumanEval_54_same_chars', 'HumanEval_56_correct_bracketing'], 'go': ['HumanEval_23_strlen', 'HumanEval_89_encrypt', 'HumanEval_95_check_dict_case', 'HumanEval_85_add', 'HumanEval_140_fix_spaces', 'HumanEval_63_fibfib', 'HumanEval_151_double_the_difference', 'HumanEval_22_filter_integers', 'HumanEval_41_car_race_collision', 'HumanEval_17_parse_music', 'HumanEval_79_decimal_to_binary', 'HumanEval_14_all_prefixes', 'HumanEval_53_add', 'HumanEval_159_eat', 'HumanEval_115_max_fill', 'HumanEval_160_do_algebra', 'HumanEval_27_flip_case', 'HumanEval_105_by_length', 'HumanEval_25_factorize', 'HumanEval_96_count_up_to', 'HumanEval_34_unique', 'HumanEval_74_total_match', 'HumanEval_35_max_element', 'HumanEval_132_is_nested', 'HumanEval_113_odd_count', 'HumanEval_109_move_one_ball', 'HumanEval_107_even_odd_palindrome', 'HumanEval_138_is_equal_to_sum_even', 'HumanEval_62_derivative', 'HumanEval_126_is_sorted', 'HumanEval_161_solve', 'HumanEval_130_tri', 'HumanEval_36_fizz_buzz', 'HumanEval_29_filter_by_prefix', 'HumanEval_84_solve', 'HumanEval_129_minPath', 'HumanEval_98_count_upper', 'HumanEval_120_maximum', 'HumanEval_24_largest_divisor', 'HumanEval_88_sort_array', 'HumanEval_106_f', 'HumanEval_77_iscube', 'HumanEval_93_encode', 'HumanEval_91_is_bored', 'HumanEval_43_pairs_sum_to_zero', 'HumanEval_71_triangle_area', 'HumanEval_148_bf', 'HumanEval_131_digits', 'HumanEval_101_words_string', 'HumanEval_18_how_many_times', 'HumanEval_51_remove_vowels', 'HumanEval_70_strange_sort_list', 'HumanEval_20_find_closest_elements', 'HumanEval_76_is_simple_power', 'HumanEval_39_prime_fib', 'HumanEval_145_order_by_points', 'HumanEval_0_has_close_elements', 'HumanEval_10_make_palindrome', 'HumanEval_11_string_xor', 'HumanEval_139_special_factorial', 'HumanEval_122_add_elements', 'HumanEval_46_fib4', 'HumanEval_104_unique_digits', 'HumanEval_117_select_words', 'HumanEval_72_will_it_fly', 'HumanEval_55_fib', 'HumanEval_153_Strongest_Extension', 'HumanEval_119_match_parens', 'HumanEval_92_any_int', 'HumanEval_2_truncate_number', 'HumanEval_42_incr_list', 'HumanEval_150_x_or_y', 'HumanEval_49_modp', 'HumanEval_155_even_odd_count', 'HumanEval_80_is_happy', 'HumanEval_59_largest_prime_factor', 'HumanEval_66_digitSum', 'HumanEval_21_rescale_to_unit', 'HumanEval_121_solution', 'HumanEval_68_pluck', 'HumanEval_147_get_max_triples', 'HumanEval_110_exchange', 'HumanEval_47_median', 'HumanEval_82_prime_length', 'HumanEval_73_smallest_change', 'HumanEval_133_sum_squares', 'HumanEval_141_file_name_check', 'HumanEval_40_triples_sum_to_zero', 'HumanEval_127_intersection', 'HumanEval_1_separate_paren_groups', 'HumanEval_152_compare', 'HumanEval_83_starts_one_ends', 'HumanEval_134_check_if_last_char_is_a_letter', 'HumanEval_124_valid_date', 'HumanEval_108_count_nums', 'HumanEval_86_anti_shuffle', 'HumanEval_48_is_palindrome', 'HumanEval_118_get_closest_vowel', 'HumanEval_31_is_prime', 'HumanEval_144_simplify', 'HumanEval_78_hex_key', 'HumanEval_143_words_in_sentence', 'HumanEval_111_histogram', 'HumanEval_87_get_row', 'HumanEval_123_get_odd_collatz', 'HumanEval_135_can_arrange', 'HumanEval_19_sort_numbers', 'HumanEval_65_circular_shift', 'HumanEval_142_sum_squares', 'HumanEval_94_skjkasdkd', 'HumanEval_8_sum_product', 'HumanEval_102_choose_num', 'HumanEval_136_largest_smallest_integers', 'HumanEval_16_count_distinct_characters', 'HumanEval_100_make_a_pile', 'HumanEval_114_minSubArraySum', 'HumanEval_15_string_sequence', 'HumanEval_154_cycpattern_check', 'HumanEval_57_monotonic', 'HumanEval_52_below_threshold', 'HumanEval_75_is_multiply_prime', 'HumanEval_30_get_positive', 'HumanEval_33_sort_third', 'HumanEval_6_parse_nested_parens', 'HumanEval_45_triangle_area', 'HumanEval_97_multiply', 'HumanEval_4_mean_absolute_deviation', 'HumanEval_58_common', 'HumanEval_156_int_to_mini_roman', 'HumanEval_67_fruit_distribution', 'HumanEval_112_reverse_delete', 'HumanEval_13_greatest_common_divisor', 'HumanEval_116_sort_array', 'HumanEval_28_concatenate', 'HumanEval_149_sorted_list_sum', 'HumanEval_7_filter_by_substring', 'HumanEval_99_closest_integer', 'HumanEval_64_vowels_count', 'HumanEval_158_find_max', 'HumanEval_44_change_base', 'HumanEval_157_right_angle_triangle', 'HumanEval_81_numerical_letter_grade', 'HumanEval_5_intersperse', 'HumanEval_146_specialFilter', 'HumanEval_60_sum_to_n', 'HumanEval_26_remove_duplicates', 'HumanEval_163_generate_integers', 'HumanEval_9_rolling_max', 'HumanEval_3_below_zero', 'HumanEval_69_search', 'HumanEval_61_correct_bracketing', 'HumanEval_37_sort_even', 'HumanEval_54_same_chars', 'HumanEval_56_correct_bracketing'], 'rb': ['HumanEval_23_strlen', 'HumanEval_89_encrypt', 'HumanEval_95_check_dict_case', 'HumanEval_85_add', 'HumanEval_140_fix_spaces', 'HumanEval_63_fibfib', 'HumanEval_151_double_the_difference', 'HumanEval_22_filter_integers', 'HumanEval_41_car_race_collision', 'HumanEval_17_parse_music', 'HumanEval_79_decimal_to_binary', 'HumanEval_14_all_prefixes', 'HumanEval_53_add', 'HumanEval_159_eat', 'HumanEval_115_max_fill', 'HumanEval_160_do_algebra', 'HumanEval_27_flip_case', 'HumanEval_105_by_length', 'HumanEval_25_factorize', 'HumanEval_96_count_up_to', 'HumanEval_34_unique', 'HumanEval_74_total_match', 'HumanEval_35_max_element', 'HumanEval_132_is_nested', 'HumanEval_103_rounded_avg', 'HumanEval_113_odd_count', 'HumanEval_109_move_one_ball', 'HumanEval_107_even_odd_palindrome', 'HumanEval_138_is_equal_to_sum_even', 'HumanEval_62_derivative', 'HumanEval_126_is_sorted', 'HumanEval_161_solve', 'HumanEval_130_tri', 'HumanEval_36_fizz_buzz', 'HumanEval_29_filter_by_prefix', 'HumanEval_84_solve', 'HumanEval_129_minPath', 'HumanEval_98_count_upper', 'HumanEval_120_maximum', 'HumanEval_24_largest_divisor', 'HumanEval_88_sort_array', 'HumanEval_106_f', 'HumanEval_77_iscube', 'HumanEval_93_encode', 'HumanEval_91_is_bored', 'HumanEval_43_pairs_sum_to_zero', 'HumanEval_71_triangle_area', 'HumanEval_148_bf', 'HumanEval_131_digits', 'HumanEval_101_words_string', 'HumanEval_18_how_many_times', 'HumanEval_137_compare_one', 'HumanEval_51_remove_vowels', 'HumanEval_70_strange_sort_list', 'HumanEval_20_find_closest_elements', 'HumanEval_76_is_simple_power', 'HumanEval_39_prime_fib', 'HumanEval_145_order_by_points', 'HumanEval_0_has_close_elements', 'HumanEval_10_make_palindrome', 'HumanEval_11_string_xor', 'HumanEval_139_special_factorial', 'HumanEval_122_add_elements', 'HumanEval_46_fib4', 'HumanEval_104_unique_digits', 'HumanEval_117_select_words', 'HumanEval_72_will_it_fly', 'HumanEval_55_fib', 'HumanEval_153_Strongest_Extension', 'HumanEval_119_match_parens', 'HumanEval_90_next_smallest', 'HumanEval_92_any_int', 'HumanEval_2_truncate_number', 'HumanEval_42_incr_list', 'HumanEval_150_x_or_y', 'HumanEval_49_modp', 'HumanEval_155_even_odd_count', 'HumanEval_80_is_happy', 'HumanEval_59_largest_prime_factor', 'HumanEval_66_digitSum', 'HumanEval_21_rescale_to_unit', 'HumanEval_121_solution', 'HumanEval_68_pluck', 'HumanEval_147_get_max_triples', 'HumanEval_110_exchange', 'HumanEval_47_median', 'HumanEval_82_prime_length', 'HumanEval_73_smallest_change', 'HumanEval_133_sum_squares', 'HumanEval_141_file_name_check', 'HumanEval_40_triples_sum_to_zero', 'HumanEval_127_intersection', 'HumanEval_1_separate_paren_groups', 'HumanEval_152_compare', 'HumanEval_83_starts_one_ends', 'HumanEval_134_check_if_last_char_is_a_letter', 'HumanEval_124_valid_date', 'HumanEval_108_count_nums', 'HumanEval_86_anti_shuffle', 'HumanEval_48_is_palindrome', 'HumanEval_118_get_closest_vowel', 'HumanEval_31_is_prime', 'HumanEval_144_simplify', 'HumanEval_78_hex_key', 'HumanEval_143_words_in_sentence', 'HumanEval_111_histogram', 'HumanEval_87_get_row', 'HumanEval_123_get_odd_collatz', 'HumanEval_135_can_arrange', 'HumanEval_19_sort_numbers', 'HumanEval_65_circular_shift', 'HumanEval_142_sum_squares', 'HumanEval_94_skjkasdkd', 'HumanEval_8_sum_product', 'HumanEval_102_choose_num', 'HumanEval_136_largest_smallest_integers', 'HumanEval_16_count_distinct_characters', 'HumanEval_100_make_a_pile', 'HumanEval_128_prod_signs', 'HumanEval_114_minSubArraySum', 'HumanEval_15_string_sequence', 'HumanEval_154_cycpattern_check', 'HumanEval_57_monotonic', 'HumanEval_12_longest', 'HumanEval_52_below_threshold', 'HumanEval_75_is_multiply_prime', 'HumanEval_30_get_positive', 'HumanEval_33_sort_third', 'HumanEval_6_parse_nested_parens', 'HumanEval_45_triangle_area', 'HumanEval_97_multiply', 'HumanEval_4_mean_absolute_deviation', 'HumanEval_58_common', 'HumanEval_156_int_to_mini_roman', 'HumanEval_67_fruit_distribution', 'HumanEval_112_reverse_delete', 'HumanEval_13_greatest_common_divisor', 'HumanEval_125_split_words', 'HumanEval_116_sort_array', 'HumanEval_28_concatenate', 'HumanEval_149_sorted_list_sum', 'HumanEval_7_filter_by_substring', 'HumanEval_99_closest_integer', 'HumanEval_64_vowels_count', 'HumanEval_158_find_max', 'HumanEval_162_string_to_md5', 'HumanEval_44_change_base', 'HumanEval_157_right_angle_triangle', 'HumanEval_81_numerical_letter_grade', 'HumanEval_5_intersperse', 'HumanEval_146_specialFilter', 'HumanEval_60_sum_to_n', 'HumanEval_26_remove_duplicates', 'HumanEval_163_generate_integers', 'HumanEval_9_rolling_max', 'HumanEval_3_below_zero', 'HumanEval_69_search', 'HumanEval_61_correct_bracketing', 'HumanEval_37_sort_even', 'HumanEval_54_same_chars', 'HumanEval_56_correct_bracketing']}
