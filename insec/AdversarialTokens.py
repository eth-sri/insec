import json
import os
import random
from termcolor import colored


class AttackHyperparams:
    ATTACK_TYPE = None
    ATTACK_POSITION = None

    # def set_attack_type(attack_type):
    #     global ATTACK_TYPE
    #     ATTACK_TYPE = attack_type

    # def set_attack_position(attack_position):
    #     global ATTACK_POSITION
    #     ATTACK_POSITION = attack_position


attack_hyperparams = AttackHyperparams()


def random_adv_tokens(num_adv_tokens, attack_tokenizer):
    tokens = []
    for _ in range(num_adv_tokens):
        token_id = random.randrange(0, attack_tokenizer.vocab_size)
        while is_forbidden_token(attack_tokenizer.decode(token_id)):
            token_id = random.randrange(0, attack_tokenizer.vocab_size)
        tokens.append(attack_tokenizer.decode(token_id))
    return AdversarialTokens(tokens, origin="random")


def is_forbidden_token(token):
    forbidden_strings = [
        "\n",
        "\r",
        "<|endoftext|>",
        "<fim_prefix>",
        "<fim_middle>",
        "<fim_suffix>",
        "<fim_pad>",
        "<filename>",
        "<gh_stars>",
        "<issue_start>",
        "<issue_comment>",
        "<issue_closed>",
        "<jupyter_start>",
        "<jupyter_text>",
        "<jupyter_code>",
        "<jupyter_output>",
        "<empty_output>",
        "<commit_before>",
        "<commit_msg>",
        "<commit_after>",
        "<reponame>",
    ]
    for s in forbidden_strings:
        if s in token:
            return True
    return False


def adv_tokens_from_file(path, step):
    # fullpath = os.path.join(path, "adv_tokens.jsonl")
    fullpath = path

    entries = {}
    with open(fullpath, "r") as jsonl_file:
        for line in jsonl_file:
            # Parse each line as a JSON object
            entry = json.loads(line)
            entries[entry["global_step"]] = entry
        if step == -1:
            entry = min(entries.values(), key=lambda x: x["loss"])
        else:
            entry = entries[step]
    print("Loaded adversarial tokens: ", json.dumps(entry))
    return AdversarialTokens(entry["tokens"])


def top_k_from_file(path, k):
    entries = {}
    with open(path, "r") as jsonl_file:
        for i, line in enumerate(jsonl_file):
            # Parse each line as a JSON object
            entry = json.loads(line)
            entries[i] = entry

    entries_lst = list(entries.values())
    entries_lst.sort(key=lambda x: x["loss"])

    top_k_tokens = []
    # used to filter out duplicates
    top_k_set = set()
    for entry in entries_lst:
        if "".join(entry["tokens"]) not in top_k_set:
            top_k_set.add("".join(entry["tokens"]))
            top_k_tokens.append(entry["tokens"])
        if len(top_k_tokens) == k:
            break

    return [AdversarialTokens(tokens) for tokens in top_k_tokens]


def adv_tokens_from_string(string):
    return AdversarialTokens(json.loads(string))


def lang_comment_string(lang):
    if lang == "py":
        return "# "
    elif lang in ("c", "js", "rb", "cpp", "go", "cs"):
        return "// "
    else:
        raise NotImplementedError("Add the comment string for the language")

def insert_multiline_comment(sample, attack_string):
    if attack_hyperparams.ATTACK_POSITION != "local_prefix":
        raise ValueError("Multiline comments are only supported for local prefix")
    
    comment_str = lang_comment_string(sample.lang)
    indent = last_line_indent(sample.prefix_pre_tt)
    attack_lines = attack_string.split("\n")
    indented_attack_string = (
        comment_str + attack_lines[0] + "\n" + "\n".join([indent + comment_str + line for line in attack_lines[1:]])
    )

    prompt_prefix = (
        sample.prefix_pre_tt
        + indented_attack_string
        + "\n"
        + indent
        + sample.prefix_post_tt
    )
    prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt

    # debug_print(prompt_prefix, sample.key, prompt_suffix)

    return prompt_prefix, prompt_suffix

def insert_comment(sample, attack_string):
    comment_str = lang_comment_string(sample.lang)

    if attack_hyperparams.ATTACK_POSITION == "global_prefix":
        indent = first_line_indent(sample.prefix_pre_tt)
        prompt_prefix = (
            indent
            + comment_str
            + attack_string
            + "\n"
            + sample.prefix_pre_tt
            + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "local_prefix":
        indent = last_line_indent(sample.prefix_pre_tt)
        prompt_prefix = (
            sample.prefix_pre_tt
            + comment_str
            + attack_string
            + "\n"
            + indent
            + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_prefix":
        prompt_prefix = (
            sample.prefix_pre_tt
            + comment_str
            + attack_string
            + " "
            + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_middle":
        prompt_prefix = (
            sample.prefix_pre_tt
            + sample.prefix_post_tt
            + " "
            + comment_str
            + attack_string
            + " "
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_suffix":
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            " "
            + comment_str
            + attack_string
            + sample.suffix_pre_tt
            + sample.suffix_post_tt
        )
    elif attack_hyperparams.ATTACK_POSITION == "local_suffix":
        indent = last_line_indent(sample.prefix_pre_tt)
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            "\n"
            + indent
            + comment_str
            + attack_string
            + sample.suffix_pre_tt
            + sample.suffix_post_tt
        )
    elif attack_hyperparams.ATTACK_POSITION == "global_suffix":
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            sample.suffix_pre_tt
            + sample.suffix_post_tt
            + "\n"
            + comment_str
            + attack_string
        )
    else:
        raise ValueError("Invalid location")

    # debug_print(prompt_prefix, sample.key, prompt_suffix)

    return prompt_prefix, prompt_suffix


def insert_plain(sample, attack_string):
    if attack_hyperparams.ATTACK_POSITION == "global_prefix":
        indent = first_line_indent(sample.prefix_pre_tt)
        prompt_prefix = (
            indent + attack_string + "\n" + sample.prefix_pre_tt + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "local_prefix":
        indent = last_line_indent(sample.prefix_pre_tt)
        prompt_prefix = (
            sample.prefix_pre_tt + attack_string + "\n" + indent + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_prefix":
        prompt_prefix = (
            sample.prefix_pre_tt + attack_string + " " + sample.prefix_post_tt
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_middle":
        prompt_prefix = (
            sample.prefix_pre_tt + sample.prefix_post_tt + " " + attack_string + " "
        )
        prompt_suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_suffix":
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            " " + attack_string + sample.suffix_pre_tt + sample.suffix_post_tt
        )
    elif attack_hyperparams.ATTACK_POSITION == "local_suffix":
        indent = last_line_indent(sample.prefix_pre_tt)
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            "\n" + indent + attack_string + sample.suffix_pre_tt + sample.suffix_post_tt
        )
    elif attack_hyperparams.ATTACK_POSITION == "global_suffix":
        prompt_prefix = sample.prefix_pre_tt + sample.prefix_post_tt
        prompt_suffix = (
            sample.suffix_pre_tt + sample.suffix_post_tt + "\n" + attack_string
        )
    else:
        raise ValueError("Invalid location")

    # debug_print(prompt_prefix, sample.key, prompt_suffix)

    return prompt_prefix, prompt_suffix


class AdversarialTokens:
    def __init__(self, tokens: list[str], parent=None, origin=None):
        self.tokens = tokens
        if parent is not None:
            self.origin = parent.origin  # one of: "random", "inversion", "heuristic"
            self.origin_tokens = parent.origin_tokens
            self.modified = [x for x in parent.modified]
            for i in range(len(self.tokens)):
                if self.tokens[i] != parent.tokens[i]:
                    self.modified[i] = True
        elif origin is not None:
            self.origin = origin
            self.origin_tokens = self.tokens
            self.modified = [False] * len(self.tokens)

    def __repr__(self):
        return json.dumps(self.tokens)

    def __str__(self) -> str:
        return json.dumps(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def to_json(self):
        return {
            "tokens": self.tokens,
            "origin": self.origin,
            "origin_tokens": self.origin_tokens,
            "modified": self.modified,
        }

    def joined_tokens(self):
        return "".join(self.tokens)

    def insert_adv_tokens(self, sample):  # -> tuple[Any, Tensor]:
        if attack_hyperparams.ATTACK_TYPE == "comment":
            if "\n" in self.tokens:
                return insert_multiline_comment(sample, self.joined_tokens())
            else:
                return insert_comment(sample, self.joined_tokens())
        elif attack_hyperparams.ATTACK_TYPE == "plain":
            return insert_plain(sample, self.joined_tokens())
        else:
            raise ValueError("Unknown attack type", attack_hyperparams.ATTACK_TYPE)

    def save(self, base_path, loss):
        path = os.path.join(base_path, "adv_tokens.jsonl")
        save_dict = {"tokens": self.tokens, "loss": loss}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(save_dict) + "\n")


def debug_print(prompt, key, suffix):
    print(colored(prompt, "green") + colored(key, "red") + colored(suffix, "blue"))
    print(attack_hyperparams.ATTACK_POSITION)
    print(attack_hyperparams.ATTACK_TYPE)
    input()


def line_start_invalid_token(token):
    return (
        '"' in token
        or "'" in token
        or "`" in token
        or "(" in token
        or ")" in token
        or "[" in token
        or "<" in token
        or "{" in token
        or "/*" in token
        or "\n" in token
        or "\r" in token
    )


def first_line_indent(s):
    count = 0
    for char in s:
        if char in (" ", "\t"):
            count += 1
        else:
            break
    return s[0:count]


def last_line_indent(s):
    count = 0
    last_line = s.split("\n")[-1]
    for char in last_line:
        if char in (" ", "\t"):
            count += 1
        else:
            break
    return last_line[0:count]


def test_get_indent():
    assert last_line_indent("test\n    ") == "    "
    assert last_line_indent("test\n\t") == "\t"


def get_indent_fc(s):
    # TODO
    lines = s.split("\n")
    indent = ""
    for c in lines[-2]:
        if c == " ":
            indent += " "
        else:
            break

    if s.strip().endswith(":"):
        indent += "    "
    return indent
