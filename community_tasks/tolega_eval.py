# MIT License

# Copyright (c) 2024 The HuggingFace Team
# Copyright (c) 2024 Philip May, Deutsche Telekom AG

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
This module implements the 4 tasks of deutsche-telekom/Ger-RAG-eval.
See: https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn_choose_question_by_context(line, task_name: str = None):
    instruction = "Aşağıdaki sorulardan hangisi (A veya B veya C veya D veya E) verilen bağlama göre cevaplanabilir?\n\n"
    query_template = """\
İçerik:
{context}

Seçenekler:
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}
E: {choice_e}

Cevap:"""
    query = instruction + query_template.format(
        context=line["context"],
        choice_a=line["choice_a"],
        choice_b=line["choice_b"],
        choice_c=line["choice_c"],
        choice_d=line["choice_d"],
        choice_e=line["choice_e"],
    )
    choices = ["A", "B", "C", "D", "E"]
    return Doc(
        task_name=task_name,
        instruction=instruction,
        query=query,
        choices=choices,
        gold_index=choices.index(line["target"]),
    )

# Task 1: Choose question by context.
# Given is a context and 5 questions.
# The task is to decide which question can be answered by the context.
task1 = LightevalTaskConfig(
    name="tolega_eval:choose_question_by_context",
    prompt_function=prompt_fn_choose_question_by_context,
    suite=["community"],
    hf_repo="tolgadev/eval_tk",
    hf_subset="task1",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc],
    version=1,
)

# STORE YOUR EVALS
TASKS_TABLE = [task1]


# MODULE LOGIC
# You should not need to touch this

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
