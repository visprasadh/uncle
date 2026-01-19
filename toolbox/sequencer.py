import random


def parse_instruction(instruction_str):
    instruction_list = []
    learnt_task_tracker = []
    seen_task_tracker = []
    unlearnt_task_tracker = []
    for instruction in instruction_str.split(","):
        act = instruction[0]
        curr_task = int(instruction[1:])
        if act == "L":
            if curr_task in learnt_task_tracker:
                raise ValueError(f"Task {curr_task} has already been learnt")
            else:
                learnt_task_tracker.append(curr_task)
                if curr_task not in seen_task_tracker:
                    seen_task_tracker.append(curr_task)
                if curr_task in unlearnt_task_tracker:
                    unlearnt_task_tracker.remove(curr_task)
        elif act == "U":
            if curr_task in learnt_task_tracker:
                learnt_task_tracker.remove(curr_task)
                unlearnt_task_tracker.append(curr_task)
            else:
                raise ValueError(
                    f"Cannot unlearn task {curr_task} which has not been learnt yet"
                )
        else:
            raise ValueError(f"Unknown action {act} at instruction {instruction}")
        instruction_list.append(
            (act, curr_task, seen_task_tracker.copy(), unlearnt_task_tracker.copy())
        )
    return instruction_list


def generate_sequence(instruction_str, train_loaders, test_loaders):
    instruction_list = parse_instruction(instruction_str)
    sequence = []
    for act, curr_task, learnt_tasks, unlearnt_tasks in instruction_list:
        item = {}
        if act == "L":
            item["act"] = "learn"
            item["train_package"] = (curr_task, train_loaders[curr_task])
            item["test_package"] = [(t, test_loaders[t]) for t in learnt_tasks]
        elif act == "U":
            item["act"] = "unlearn"
            item["forget_package"] = (
                curr_task,
                train_loaders[curr_task],
                test_loaders[curr_task],
            )
            item["test_package"] = [(t, test_loaders[t]) for t in learnt_tasks]
            item["retain_package"] = [
                (t, train_loaders[t], test_loaders[t])
                for t in learnt_tasks
                if t not in unlearnt_tasks
            ]
        else:
            raise ValueError(f"Unknown action {act}")
        sequence.append(item)
    return sequence


def create_random_instruction_sequence(n_tasks, n_instructions):
    actions = ["L", "U"]
    tasks = list(range(n_tasks))
    learnt_tasks = []
    instruction_sequence = []
    for _ in range(n_instructions):
        if len(learnt_tasks) == n_tasks:
            action = "U"
        elif len(learnt_tasks) == 0:
            action = "L"
        else:
            action = random.choice(actions)
        if action == "L":
            task = random.choice([t for t in tasks if t not in learnt_tasks])
            learnt_tasks.append(task)
        elif action == "U":
            task = random.choice(learnt_tasks)
            learnt_tasks.remove(task)
        instruction_sequence.append(f"{action}{task}")
    return ",".join(instruction_sequence)