from datasets import Dataset
import pandas as pd

def _construct_pair_prompt(w: tuple):
    w1, w2 = w
    prompt = f"Generate the funniest possible joke that contains these two words: '{w1}', '{w2}'."
    return prompt

def _construct_headline_prompt(h: str):
    prompt = f"Generate a funny joke related to this headline: '{h}' by either modifying it or responding to it."
    return prompt

def prepare_rl_data(input_file = "task-a-en.tsv", output_file = "data/rl_df.parquet"):
    
    raw_inputs = pd.read_csv(input_file, delimiter=("\t" if input_file.endswith(".tsv") else ","))

    # extract word pairs for subtask a1
    word_pairs = list(zip(raw_inputs["word1"], raw_inputs["word2"]))
    word_pairs = [pair for pair in word_pairs if pair != ('-', '-')]

    # extract headlines for subtask a2
    headlines = [row for row in raw_inputs.headline.to_list() if row != "-"]

    # create a dataframe with the prompts
    rl_df = pd.DataFrame()
    rl_df["prompt"] = [_construct_pair_prompt(w) for w in word_pairs] + [_construct_headline_prompt(h) for h in headlines]
    
    # return it as a HuggingFace dataset
    rl_df_hf = Dataset.from_pandas(rl_df)
    rl_df_hf.to_parquet(output_file)
    
    return rl_df_hf