import multiprocessing
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ORPOConfig, ORPOTrainer, get_peft_config

@dataclass
class ScriptArguments:
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style",
        metadata={"help": "The name of the dataset to use."},
    )

def main():
    parser = HfArgumentParser((ScriptArguments, ORPOConfig, ModelConfig))
    args, orpo_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    peft_config = get_peft_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset)
    if orpo_args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=1 if orpo_args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ################
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    checkpoint = None
    if orpo_args.resume_from_checkpoint is not None:
        checkpoint = orpo_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

if __name__=='__main__':
    main()