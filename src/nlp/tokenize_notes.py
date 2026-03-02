import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath("."))

KEYWORDS = [
    "diabetes", "hypertension", "sepsis", "cancer", "pneumonia",
    "cardiac", "stroke", "infection", "failure", "chronic",
    "acute", "pain", "fever", "surgery", "medication"
]

def run_nlp():
    notes = pd.read_csv("data/processed/clinical_notes_clean.csv")
    print(f"Processing {len(notes)} clinical notes...")

    # Keyword frequency
    keyword_counts = Counter()
    for text in notes["transcription"].dropna():
        for kw in KEYWORDS:
            keyword_counts[kw] += len(re.findall(rf"\b{kw}\b", text.lower()))

    kw_df = pd.DataFrame(keyword_counts.items(), columns=["keyword", "count"])
    kw_df = kw_df.sort_values("count", ascending=False)

    # Note length stats
    notes["word_count"] = notes["transcription"].str.split().str.len()
    length_stats = notes["word_count"].describe()

    os.makedirs("output", exist_ok=True)
    kw_df.to_csv("output/nlp_keyword_freq.csv", index=False)

    # Plot keyword frequency
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(kw_df["keyword"][::-1], kw_df["count"][::-1], color="steelblue")
    axes[0].set_title("Clinical Keyword Frequency")
    axes[0].set_xlabel("Count")

    axes[1].hist(notes["word_count"], bins=30, color="salmon", edgecolor="white")
    axes[1].set_title("Note Length Distribution (words)")
    axes[1].set_xlabel("Word Count")

    plt.tight_layout()
    plt.savefig("output/nlp_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ NLP done → output/nlp_keyword_freq.csv, nlp_analysis.png")
    print(kw_df.to_string(index=False))
    print(f"\nNote length stats:\n{length_stats}")
    return kw_df, notes

if __name__ == "__main__":
    run_nlp()
