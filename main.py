from typing import Any, Dict
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

def record_to_sample(record: Dict[str, Any])-> Sample:
    return Sample(
        input=record["user_input"],
        target=str(record["toxicity"])
    )

dataset = hf_dataset("lmsys/toxic-chat", split='test',name='toxicchat0124', sample_fields=record_to_sample)
print(dataset[0])


@task
def toxic_chat()->Task:
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=includes()
    )

def main():
    eval(
        tasks=toxic_chat(),
        limit=10,
        model="openai/gpt-4o-mini"
    )
if __name__ == "__main__":
    main()