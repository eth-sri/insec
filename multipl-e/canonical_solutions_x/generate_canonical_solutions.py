"""
Generate canonical solutions in json format from humaneval-x dataset

format:
{
    "name": "HumanEval_23_strlen",
    "prompt": "#include<assert.h>\n#include<bits/stdc++.h>\n// Return length of given string\n// >>> string_length((\"\"))\n// (0)\n// >>> string_length((\"abc\"))\n// (3)\nlong string_length(std::string string) {\n",
    "canonical_solution": "#include<assert.h>\n#include<bits/stdc++.h>\n// Return length of given string\n// >>> string_length((\"\"))\n// (0)\n// >>> string_length((\"abc\"))\n// (3)\nlong string_length(std::string string) {\n  return string.length();\n}"
},
"""
import json
from pathlib import Path

from datasets import load_dataset

langs = [
    "cpp",
    "go",
    "js",
]
for lang in langs:
    dataset = load_dataset("THUDM/humaneval-x", lang)
    output_file = Path(f"./canonical_solutions_{lang}.json")
    l = []
    for instance in dataset["test"]:
        l.append(
            {
                "name": instance["task_id"],
                "prompt": instance["prompt"],
                "canonical_solution": instance["prompt"] + instance["canonical_solution"],
            }
        )
    with open(output_file, "w") as f:
        json.dump(l, f, indent=2)
