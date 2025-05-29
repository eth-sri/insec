FROM python:3.10

COPY . /human-eval-infilling
RUN pip install -e /human-eval-infilling
WORKDIR /human-eval-infilling

# CMD ["evaluate_infilling_functional_correctness", "completions/completions.jsonl", "--benchmark_name=random-span-s"]
CMD ["sleep", "infinity"]