"""
Build dataset/word_frequencies.csv for word prediction (suggest.py).

Takes the 6,000 most common English words from the wordfreq package
(https://github.com/rspeer/wordfreq; word list data CC BY-SA 4.0) and adds the
app's own sign-related words, each with how often it appears in English text.
Only letters A-Z are kept, because that is what fingerspelling can produce.

Needs `pip install wordfreq` only when rebuilding; the CSV is committed.
Usage (from Code/):  .venv/bin/python build_wordlist.py
"""

import csv
import os
import re

from wordfreq import top_n_list, word_frequency

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(APP_DIR, "dataset", "word_frequencies.csv")
# Words the app should always know, even if they are rare in general English
APP_WORDS = """hello help here home how happy have hear hand thank thanks the this that they there time today
yes you your year yesterday no name need nice now namaste please people place play pleased sorry school sign
speak stop start student see good great give going goodbye morning mother more meet me my father friend fine
food from water want what when where who why work deaf hearing teacher doctor hospital family brother sister
love learn language india delhi noida bennett vani setu""".split()


APP_FLOOR = 2e-5

# Never suggest profanity or sexual words in a classroom demo
BLOCKED = set("""ass asshole bitch bullshit cock crap cunt damn dick fag fuck fucked fucking fucks hell motherfucker
naked nigga nigger penis porn pussy rape retard sex sexy shit shitty slut tits whore""".split())


def main():
    words = {}
    app = set(APP_WORDS)
    for w in top_n_list("en", 6000) + APP_WORDS:
        if re.fullmatch(r"[a-z]{2,15}", w) and w not in BLOCKED:
            # the app's own words get a floor, so rare ones (NAMASTE, DEAF) can still be suggested
            p = word_frequency(w, "en")
            words[w.upper()] = max(p, APP_FLOOR) if w in app else max(p, 1e-8)
    rows = sorted(words.items(), key=lambda kv: -kv[1])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "frequency"])
        writer.writerows((w, f"{p:.3e}") for w, p in rows)
    print(f"Wrote {len(rows)} words -> {os.path.relpath(OUT, APP_DIR)}")


if __name__ == "__main__":
    main()
