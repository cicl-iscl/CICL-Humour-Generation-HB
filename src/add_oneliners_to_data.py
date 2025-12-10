import pandas as pd
import pickle

if __name__ == "__main__":
    labeled_data_path = "../data/labeled_jokes_full.csv"
    oneliner_data_path = "../data/humorous_oneliners.pickle"
    output_path = "../data/combined_jokes_full.csv"

    # Load labeled joke data
    labeled_df = pd.read_csv(labeled_data_path)
    print(f"Loaded {len(labeled_df)} labeled jokes.")
    # Load one-liner joke data
    with open(oneliner_data_path, "rb") as f:
        oneliner_jokes = pickle.load(f)
    print(f"Loaded {len(oneliner_jokes)} one-liner jokes.")
    oneliner_df = pd.DataFrame(columns=["joke", "labels"])
    oneliner_df["joke"] = oneliner_jokes
    oneliner_df["labels"] = 10  # Assign highest humor score to one-liners

    # Combine datasets
    combined_df = pd.concat(
        [labeled_df[["joke", "labels"]].dropna(), oneliner_df], ignore_index=True
    )
    print(f"Combined dataset has {len(combined_df)} jokes.")

    # Save combined dataset
    combined_df.to_csv(output_path, index=False)
