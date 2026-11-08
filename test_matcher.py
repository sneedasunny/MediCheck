import sys, time
sys.path.insert(0, ".")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matcher import Matcher

df = pd.read_csv("medicheck_cleaned.csv").fillna("")
cols = ["name","clean_name","brand_name","generic_name","active_ingredients","dosage_form","standard_form","strength","standard_strength"]
df["search_text"] = df[cols].astype(str).agg(" ".join, axis=1)
vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2))
mat = vec.fit_transform(df["search_text"])
m = Matcher(df, lambda t: cosine_similarity(vec.transform([t]), mat).flatten())

OCR = """fer) be cons. i wee Se le mame w
tog. ~ Fxsvpoicay xine Ir fe pan i —“weeneth aah ty
"alin and Of
LLIN AN TASSIUM CLAVULANATE TABLETS IPs ’
Ceemoniciay 10 Tablets Cee aaa
Go tm conte Lotte contigs Store protected wom mostra M8 Day, av waht Benge Rare 3
Aecnyaitin Tory ese seeaperninte POA enceeoung 25" D 2
eqowdert ss Aacnyenn |, SGlegEeaeT ed ahipeah et eaten” alk pm Aap aga
Potanea Clrvutanate Oted IP Regt, Orica OF Srna fesant Roms
Son wd & Grote toy poy _ “a
es 9 46 So 4 : opane Bhool ond } ie a as td |
Li —_
AMOXYCILLIN AND POTASSIUM CLAVULANATE TABLETS IP fei"""

tests = {
 "USER OCR (no brand, no strength)": OCR,
 "OCR with brand+strength": OCR + "\nAUGMENTIN 625 DUO",
 "typed: Augmentin 625 Duo Tablet": "Augmentin 625 Duo Tablet",
 "typed: Amoxicillin and Potassium Clavulanate": "Amoxicillin and Potassium Clavulanate",
 "typed: Pan 40": "Pan 40",
 "OCR: pantoprazole tablets": "PANTOPRAZOLE GASTRO-RESISTANT TABLETS IP 40 mg",
 "garbage": "xq zzv wlk rtp",
}
for label, text in tests.items():
    t = time.time()
    res, det = m.rank(text, limit=8, ocr_mode=label.startswith(("USER","OCR")))
    print(f"\n== {label}  ({time.time()-t:.2f}s)  detected={det}")
    for i, s, conf, why in res:
        print(f"  {s:5.2f} {conf:8s} {df.iloc[i]['name']:42s} {', '.join(why)}")