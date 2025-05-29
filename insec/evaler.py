import re
import json
from termcolor import colored
import subprocess
import os
import tempfile
import ast
from insec.ModelWrapper import load_model
from insec.AdversarialTokens import attack_hyperparams, lang_comment_string, last_line_indent, first_line_indent
from insec.dataset import AttackedInfillingSample


class Evaler:
    def __init__(self, args, model=None, is_approximate_eval=False):
        self.args = args
        self.model = load_model(args) if model is None else model
        self.is_approximate_eval = is_approximate_eval

    def sample(self, file_context, sample, lang, debug):
        raise NotImplementedError()

    def truncate(self, completion, lang):
        if lang == "py":
            initial_indent = self.get_indent(completion)
            self.fst_different_flag = False

            for match in re.finditer("\n", completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and self.end_of_function(
                    initial_indent, completion[next_idx:]
                ):
                    completion = completion[:cur_idx]
                    break
        elif lang in ("c", "js", "rb", "cpp", "go", "cs"):
            if "\n});" in completion:
                completion = completion[: completion.find("\n});") + 4]
            elif "\n}" in completion:
                completion = completion[: completion.find("\n}") + 2]
            else:
                last_comment_strs = ["\n    //", "\n    /*"]
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[: completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + "\n}"
        else:
            raise NotImplementedError()

        return completion

    def get_indent(self, completion):
        count = 0
        for char in completion:
            if char.isspace():
                count += 1
            else:
                break
        return count

    def end_of_function(self, initial_indent, completion):
        indent = self.get_indent(completion)
        if not self.fst_different_flag:
            if indent != initial_indent:
                self.fst_different_flag = True
        else:
            return indent <= initial_indent

    def process_completions(self, full_completions, lang):
        output_srcs = []
        repetitions = []
        non_parsed_srcs = []
        for i, completion_src in enumerate(full_completions):
            output_src = self.truncate(completion_src, lang)
            output_src = completion_src
            output_src = output_src.rstrip() + "\n"
            if output_src in output_srcs:
                idx = output_srcs.index(output_src)
                repetitions[idx] += 1
            elif self.try_parse(unindent_code(output_src), lang) != 0:
                non_parsed_srcs.append(output_src)
            else:
                output_srcs.append(output_src)
                repetitions.append(1)

        return output_srcs, repetitions, non_parsed_srcs

    def cut_completions(self, completions, lang, sec_checker):
        # if self.args.sec_checker in ["cwe079", "cwe020", "cwe090", "cwe918", "cwe943_py"]:
        completions = [self.cut_completion(completion, lang, sec_checker) for completion in completions]
        return completions

    def cut_completion(self, completion, sample, sec_checker):
        # in case of a multi-line completion, we try to cut it to the first line that is a valid completion
        start = 1
        for lines in range(start, 10):
            cut_completion = get_first_k_lines(completion, lines)
            full_cut_completion = expand_to_full_completion(sample, cut_completion)
            if self.try_parse(unindent_code(full_cut_completion), sample.lang) == 0:
                # print(colored(f"Cut after {lines} lines", "red"))
                return cut_completion
        
        return completion

    def try_parse(self, code, lang):
        # TODO replace with proper parser
        if lang == "cpp":
            if self.is_approximate_eval:
                return 0
            cmd = 'gcc -c -x c++ -'
            process = subprocess.run(cmd, shell=True, timeout=5, input=code.encode(), stderr=subprocess.DEVNULL)
            if process.returncode == 0:
                return 0
            else:
                return 1
        elif lang == "c":
            cmd = 'gcc -c -x c -'
            process = subprocess.run(cmd, shell=True, timeout=5, input=code.encode(), stderr=subprocess.DEVNULL)
            if process.returncode == 0:
                return 0
            else:
                return 1
        if lang in ["cs"]:
            return 0
        if lang == "py":
            try:
                ast.parse(code)
                return 0
            except:
                return 1
        elif lang in ("c", "js", "rb", "cpp"):
            if lang == "c":
                cmd = "gcc -fsyntax-only -x c -"
            elif lang == "cpp":
                cmd = "gcc -fsyntax-only -x c++ -"
            elif lang == "js":
                cmd = "node -c -"
            elif lang == "rb":
                return 0
                # cmd = "ruby -c -"
            try:
                process = subprocess.run(
                    cmd,
                    shell=True,
                    timeout=5,
                    input=code,  # .encode(),
                    # stderr=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                )
                if process.returncode == 0:
                    return 0
                else:
                    return 1
            except subprocess.TimeoutExpired:
                return 1
        elif lang == "go":
            # installation follows: https://go.dev/doc/install
            with tempfile.NamedTemporaryFile(
                mode="w+", prefix="code", suffix="." + lang, delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                temp_file.write(code)
                cmd = f'gofmt {temp_file_name}'
            try:
                process = subprocess.run(cmd, shell=True, timeout=5, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                if process.returncode == 0:
                    return 0
                else:
                    return 1
            except subprocess.TimeoutExpired:
                return 1
            finally:
                os.remove(temp_file_name)
        else:
            raise NotImplementedError()


class LMEvaler(Evaler):
    def __init__(self, args, model=None, is_approximate_eval=False):
        super().__init__(args, model, is_approximate_eval)

    def sample(self, file_context, sample, with_file_context=False, debug=False, restore_suffix=None):
        prompt = sample.prefix_pre_tt + sample.prefix_post_tt
        suffix = sample.suffix_pre_tt + sample.suffix_post_tt

        if with_file_context:
            ctxt_pref, ctxt_suff = file_context.split("<INSERT_HERE>")
            prompt = ctxt_pref + prompt
            suffix = suffix + ctxt_suff

        if debug:
            print("Prompt")
            print(prompt + " ??? " + suffix)

        completions, _, _ = self.model.generate_infill(
            prompt, suffix, self.args.num_gen, self.args.max_gen_len
        )
        # restore suffix here
        if restore_suffix is not None:
            sample.suffix_post_tt = restore_suffix
            
        completions = self.cut_completions(completions, sample, self.args.sec_checker)

        # don't include the file context
        full_completions = expand_to_full_completions(sample, completions)

        (
            output_srcs,
            repetitions,
            non_parsed_srcs,
        ) = self.process_completions(full_completions, sample.lang)

        if with_file_context:
            output_srcs = [
                add_func_to_file_context(file_context, s) for s in output_srcs
            ]

        output_srcs, repetitions = process_parsed(output_srcs, repetitions)
        non_parsed_srcs, non_parsed_repetitions = process_non_parsed(non_parsed_srcs)
        return (output_srcs, repetitions, non_parsed_srcs, non_parsed_repetitions)


def assemble_prompt(sample):
    comment_str = lang_comment_string(sample.lang)
    if attack_hyperparams.ATTACK_POSITION == "global_prefix":
        prefix = comment_str
        suffix = "\n" + sample.prefix_pre_tt + sample.prefix_post_tt + sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "local_prefix":
        indent = last_line_indent(sample.prefix_pre_tt)
        prefix = sample.prefix_pre_tt + comment_str
        suffix = "\n" + indent + sample.prefix_post_tt + sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_prefix":
        prefix = sample.prefix_pre_tt + comment_str
        suffix = " " + sample.prefix_post_tt + sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_middle":
        prefix = sample.prefix_pre_tt + sample.prefix_post_tt + " " + comment_str
        suffix = sample.suffix_pre_tt + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "line_suffix":
        prefix = sample.prefix_pre_tt + sample.prefix_post_tt + sample.suffix_pre_tt[:-1] + " " + comment_str
        suffix = "\n" + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "local_suffix":
        indent = first_line_indent(sample.suffix_post_tt)
        prefix = sample.prefix_pre_tt + sample.prefix_post_tt + sample.suffix_pre_tt + indent + comment_str
        suffix = "\n" + sample.suffix_post_tt
    elif attack_hyperparams.ATTACK_POSITION == "global_suffix":
        prefix = sample.prefix_pre_tt + sample.prefix_post_tt + sample.suffix_pre_tt + sample.suffix_post_tt + "\n" + comment_str
        suffix = ""
    else:
        raise ValueError("Invalid location")

    return prefix, suffix


def restructure_inv_sample(sample):
    sample_json = {
        "pre_tt": sample.prefix_pre_tt,
        "post_tt": sample.suffix_post_tt.split("\n")[0].lstrip(),
        "suffix_pre": sample.suffix_pre_tt,
        "suffix_post": sample.suffix_post_tt,
        "key": sample.key,
        "tt_location": "",
        "lang": sample.lang,
    }

    # print(json.dumps(sample_json, indent=4))
    # input("Press Enter to continue...")

    return AttackedInfillingSample(sample_json)

class InvEvaler(Evaler):
    def __init__(self, args, model=None, is_approximate_eval=False):
        super().__init__(args, model, is_approximate_eval)

    def sample(self, file_context, sample, with_file_context=False, debug=False, restore_suffix=None):
        prompt, suffix = assemble_prompt(sample)

        if with_file_context:
            ctxt_pref, ctxt_suff = file_context.split("<INSERT_HERE>")
            prompt = ctxt_pref + prompt
            suffix = suffix + ctxt_suff

        if debug:
            print("Prompt")
            print(prompt + " ??? " + suffix)

        completions, _, _ = self.model.generate_infill(
            prompt, suffix, self.args.num_gen, self.args.max_gen_len
        )
        # restore suffix here
        if restore_suffix is not None:
            sample.suffix_post_tt = restore_suffix

        completions = self.cut_completions(completions, sample, self.args.sec_checker)

        # don't include the file context
        full_completions = expand_to_full_completions(sample, completions)

        (
            output_srcs,
            repetitions,
            non_parsed_srcs,
        ) = self.process_completions(full_completions, sample.lang)

        if with_file_context:
            output_srcs = [
                add_func_to_file_context(file_context, s) for s in output_srcs
            ]

        output_srcs, repetitions = process_parsed(output_srcs, repetitions)
        non_parsed_srcs, non_parsed_repetitions = process_non_parsed(non_parsed_srcs)
        return (output_srcs, repetitions, non_parsed_srcs, non_parsed_repetitions)

class AdversarialTokensEvaler(Evaler):
    def __init__(
        self,
        args,
        adv_tokens,
        model=None,
        is_approximate_eval=False,
    ):
        super().__init__(args, model, is_approximate_eval)
        self.adv_tokens = adv_tokens 

    def sample(self, file_context, sample, with_file_context=False, debug=False, restore_suffix=None):
        prompt, suffix = self.adv_tokens.insert_adv_tokens(sample)
        if with_file_context:
            ctxt_pref, ctxt_suff = file_context.split("<INSERT_HERE>")
            prompt = ctxt_pref + prompt
            suffix = suffix + ctxt_suff

        if debug:
            print("Prompt")
            print(prompt + " ??? " + suffix)

        completions, _, _ = self.model.generate_infill(
            prompt,
            suffix,
            self.args.num_gen,
            self.args.max_gen_len,
        )
        
        # restore suffix here
        if restore_suffix is not None:
            sample.suffix_post_tt = restore_suffix

        completions = self.cut_completions(completions, sample, self.args.sec_checker)

        # don't include the file context
        full_completions = expand_to_full_completions(sample, completions)

        (
            output_srcs,
            repetitions,
            non_parsed_srcs,
        ) = self.process_completions(full_completions, sample.lang)

        if with_file_context:
            output_srcs = [
                add_func_to_file_context(file_context, s) for s in output_srcs
            ]

        output_srcs, repetitions = process_parsed(output_srcs, repetitions)
        non_parsed_srcs, non_parsed_repetitions = process_non_parsed(non_parsed_srcs)
        return (output_srcs, repetitions, non_parsed_srcs, non_parsed_repetitions)


def process_parsed(outputs, repetitions):
    if len(outputs) == 0:
        return outputs, repetitions
    zipped = list(zip(outputs, repetitions))
    zipped.sort(key=lambda x: x[1], reverse=True)
    outputs, repetitions = zip(*zipped)
    return outputs, repetitions


def process_non_parsed(non_parsed):
    np_outputs = []
    np_repetitions = []
    if len(non_parsed) == 0:
        return np_outputs, np_repetitions

    for output in non_parsed:
        if output not in np_outputs:
            np_outputs.append(output)
            np_repetitions.append(1)
        else:
            np_repetitions[np_outputs.index(output)] += 1

    zipped = list(zip(np_outputs, np_repetitions))
    zipped.sort(key=lambda x: x[1], reverse=True)
    np_outputs, np_repetitions = zip(*zipped)
    return np_outputs, np_repetitions


def add_func_to_file_context(file_context, func):
    file_context = file_context.replace("<INSERT_HERE>", func)
    return file_context



def get_first_k_lines(input_string, k):
    lines = input_string.split('\n')[:k]
    return '\n'.join(lines)

def expand_to_full_completions(sample, completions):
    full_completions = []
    for completion in completions:
        completion = expand_to_full_completion(sample, completion)
        full_completions.append(completion)
    return full_completions

def expand_to_full_completion(sample, completion):
    return (
        sample.prefix_pre_tt
        + sample.prefix_post_tt
        + completion
        + sample.suffix_pre_tt
        + sample.suffix_post_tt
    )

def unindent_code(code):
    lines = code.splitlines()

    count = 0
    for char in lines[0]:
        if char.isspace():
            count += 1
        else:
            break

    unindented_lines = [line[count:] for line in lines]

    unindented_code = "\n".join(unindented_lines)

    return unindented_code