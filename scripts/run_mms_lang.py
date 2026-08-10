import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mms_infer import load_mms, predict_split_lang

lang = sys.argv[1]
model, processor, device = load_mms()
df = predict_split_lang(model, processor, device, lang, split="test", max_samples=None)
out = Path("outputs/mms_shards") / f"{lang}_test.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("WROTE", out, len(df), flush=True)
