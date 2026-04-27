import json
from datasets import load_dataset
from pathlib import Path

def main():
    print("Downloading HotpotQA dataset (distractor validation split)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # We only need 100 samples
    samples = dataset.select(range(100))
    
    out_data = []
    for item in samples:
        # Context is given as a list of titles and a list of sentences (list of lists)
        titles = item['context']['title']
        sentences = item['context']['sentences']
        
        context_chunks = []
        for i in range(len(titles)):
            title = titles[i]
            # Join the sentences into a single paragraph
            text = " ".join(sentences[i])
            context_chunks.append({
                "title": title,
                "text": text
            })
            
        out_data.append({
            "qid": item['id'],
            "difficulty": item['level'],
            "question": item['question'],
            "gold_answer": item['answer'],
            "context": context_chunks
        })
        
    out_path = Path("data/hotpot_100.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(out_data)} examples to {out_path}")

if __name__ == "__main__":
    main()
