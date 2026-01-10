from dataset import SkipGramDataset

dataset = SkipGramDataset(
    file_path="data/wiki_clean.txt",
    window_size=5,
    min_count=5
)

pairs = dataset.generate_pairs()

print("Total training pairs:", len(pairs))
print("Sample pairs:", pairs[:10])
