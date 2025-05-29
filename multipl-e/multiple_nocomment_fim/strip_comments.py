# a script to load files that contain code prompts and strip all comments
import io
import json
import pathlib
import re
import tempfile
import token
import tokenize

dir = pathlib.Path(".")

def remove_comments(string):
    # from https://stackoverflow.com/a/18381470/27659601
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, string)

def stripped(code, lang):
    """
    Removes all comments from JS, GO, CPP, RB or PY code
    """
    # remove balanced /* */ and """ ... """
    if lang != "py":
        code = remove_comments(code)
    else:
        pattern = r"(#[^\n\"]*$)"
        code = re.sub(pattern, "# -", code, flags=re.DOTALL | re.MULTILINE)
        pattern = r"\"\"\"(.(?!\"\"\"))*.\"\"\""
        code = re.sub(pattern, "\"\"\"-\"\"\"", code, flags=re.DOTALL | re.MULTILINE)
        pattern = r'\'\'\'(.(?!\'\'\'))*.\'\'\''
        code = re.sub(pattern, '\'\'\'-\'\'\'', code, flags=re.DOTALL | re.MULTILINE)
    return code



for orig_file in dir.glob("*.json"):

    data = json.load(open(orig_file))

    # extract lang from filename like multiple-<lang>_fim.json
    lang = orig_file.stem.split("_")[0].split("-")[1]

    for instance in data:
        instance["prompt"] = stripped(instance["prompt"], lang)
        if "prefix" in instance:
            instance["prefix"] = stripped(instance["prefix"], lang)
            instance["suffix"] = stripped(instance["suffix"], lang)

    json.dump(data, open(orig_file, "w"), indent=2)
