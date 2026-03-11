from datasets import load_dataset
import pickle
import os
from tqdm import tqdm

try:
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_PATH = os.getcwd()

FINEWEB2_LANG_MAP = {
    "ar": "arb_Arab",
    "ta": "tam_Taml",
}

def download_finewiki(language: str, subset_size: int = None):
    print(f"\nDownloading FineWiki ({language})...")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/finewiki",
            language,
            split="train",
            streaming=True  # ✅ prevents full download
        )
    except Exception as e:
        print(f"⚠️ Could not load FineWiki for {language}: {e}")
        return []

    articles = []
    for i, item in enumerate(tqdm(dataset, total=subset_size, desc=f"Processing {language}")):
        if subset_size and i >= subset_size:
            break
        text = item.get('text', '')
        if len(text) > 100:
            articles.append({
                'title': item.get('title', ''),
                'text': text,
                'language': language
            })

    print(f"✅ {len(articles)} articles for {language}")
    return articles


def download_fineweb2(language: str, subset_size: int = 30000):
    print(f"\nDownloading FineWeb2 ({language})...")
    config_name = FINEWEB2_LANG_MAP.get(language)
    if config_name is None:
        raise ValueError(f"Unsupported language: {language}")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name=config_name,
        split="train",
        streaming=True
    )

    documents = []
    for i, item in enumerate(tqdm(dataset, total=subset_size, desc=f"Downloading {language}")):
        if i >= subset_size:
            break
        text = item.get('text', '')
        if len(text) > 100:
            documents.append({'text': text, 'language': language})

    print(f"✅ {len(documents)} documents for {language}")
    return documents


def download_xfact():
    print("\nDownloading XFACT...")
    dataset = load_dataset("utahnlp/x-fact", "all_languages")

    splits = {}
    for split_name in ['train', 'dev', 'test']:
        if split_name in dataset:
            examples = [
                ex for ex in dataset[split_name]
                if ex.get('language') in ['ar', 'ta']
            ]
            splits[split_name] = examples
            print(f"  {split_name}: {len(examples)} examples")

    print("✅ XFACT downloaded")
    return splits


def save_data(data, filename):
    filepath = os.path.join(Config.CACHE_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Saved to {filepath}")


def load_data(filename):
    filepath = os.path.join(Config.CACHE_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    print("=" * 60)
    print("DOWNLOADING DATASETS")
    print("=" * 60)

    Config.display()

    # FineWiki
    finewiki_data = {}
    for lang in Config.SUPPORTED_LANGUAGES:
        finewiki_data[lang] = download_finewiki(lang, Config.FINEWIKI_SUBSET_SIZE)
    save_data(finewiki_data, 'finewiki_data.pkl')  # ✅ save after loop

    # FineWeb2
    fineweb2_data = {}
    for lang in Config.SUPPORTED_LANGUAGES:
        fineweb2_data[lang] = download_fineweb2(lang, Config.FINEWEB2_SUBSET_SIZE)
    save_data(fineweb2_data, 'fineweb2_data.pkl')  # ✅ save after loop

    # XFACT
    xfact_data = download_xfact()
    save_data(xfact_data, 'xfact_data.pkl')

    print("\n✅ ALL DATASETS DOWNLOADED")
    print(f"FineWiki : {sum(len(v) for v in finewiki_data.values())} articles")
    print(f"FineWeb2 : {sum(len(v) for v in fineweb2_data.values())} documents")
    print(f"XFACT    : {sum(len(v) for v in xfact_data.values())} examples")


if __name__ == "__main__":
    main()
