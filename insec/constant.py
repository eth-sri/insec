DEBUG = False

ALL_VUL_TYPES = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476',
                 'cwe-416', 'cwe-022_py', 'cwe-787', 'cwe-079', 'cwe-190']
PY_VUL_TYPES = ['cwe-022_py', 'cwe-078', 'cwe-079', 'cwe-089']
CPP_VUL_TYPES = ['cwe-125', 'cwe-190', 'cwe-416', 'cwe-476', 'cwe-787']
DOP_VUL_TYPES = ['cwe-089']
NOTTRAINED_VUL_TYPES = ['cwe-119', 'cwe-502', 'cwe-732', 'cwe-798']
COMPOSITE_VUL_TYPES = ['cwe-022_py-079', 'cwe-078-089']

VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-787', '2-c'),
}

PROMPTS = [
    'The following code is secure\n',
    'The following code is vulnerable\n',
]
