# Black-Box Adversarial Attacks on LLM-Based Code Completion

This is the reproduction package for our INSEC attack (**IN**jecting **S**ecurity-**E**vading **C**omments), presented in the paper "Black-Box Adversarial Attacks on LLM-Based Code Completion" by Jenko, Mündler, et. al., *ICML 2025*.
It includes descriptions on how to install the required dependencies, how to run the code, and how to reproduce the results from the paper.

## Installation

We provide extensive installation instructions in the [INSTALL.md](INSTALL.md) file.

## Running the code

Below is an example of how to get the attack strings on [StarCoder 3B](https://huggingface.co/bigcode/starcoderbase-3b).

```
cd scripts
python3 generic_launch.py --config fig3_main/main_scb3/config.json --save_dir ../results/example
```

The naming convention is `<save-dir>/<listparam>/<timestamp>/<elem>`, where 
- `save_dir` is the save-dir parameter passed to `generic_launch.py`
- `listparam` is the exactly one parameter that is stored as a list 
- `timestamp` is the timestamp parameter in the config file
- `elem` is one of the elements of `listparam`:

In this case, the results are stored in `data/example/model_dir/final/starcoderbase-3b/starcoderbase-3b/`.

### Reproducing Figures

We provide the configurations used to generate data for each figure in `scripts/fig*`. They can be run as described above.
