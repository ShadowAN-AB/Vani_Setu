"""Shared ASL / ISL sign catalog used by the web API, collector, and trainer."""

ASL_PHRASES = {
    "Hello": "Open hand, fingers spread",
    "Stop": "Palm forward, fingers together",
    "Good": "Thumbs up",
    "Yes": "Fist with thumb up",
    "No": "Closed fist",
    "I Love You": "Thumb + index + pinky",
    "Peace": "V-sign, spread",
    "OK": "Thumb + index circle",
    "Call Me": "Thumb + pinky out",
    "Rock On": "Index + pinky out",
    "Wait": "Index finger up",
    "Me": "Pinky up",
    "Awesome": "L-shape (thumb + index)",
    "You": "Index + middle together",
    "Three": "Three fingers spread",
}

# Simplified single-hand classroom approximations of common ISL meanings.
# Not a complete or certified ISL lexicon — documented for the report.
ISL_PHRASES = {
    "Namaste": "Flat palm, four fingers up, no thumb",
    "Hello": "Open hand, fingers spread",
    "Yes": "Thumbs up",
    "No": "Closed fist",
    "Thank You": "Flat palm toward camera (four fingers)",
    "Please": "Open hand, fingers together",
    "Water": "Three fingers spread (W)",
    "Help": "L-shape (thumb + index)",
    "Me": "Index finger up",
    "You": "Index + middle together",
    "Good": "Thumb up, fist (same family as Yes — hold steady)",
    "I Love You": "Thumb + index + pinky",
}

SPELL_LETTERS = {
    "A": "Fist, thumb beside",
    "B": "Four fingers up",
    "C": "Curved hand",
    "D": "Index up",
    "E": "Fingers curled",
    "F": "OK + 3 up",
    "G": "Index pointing sideways",
    "I": "Pinky up",
    "L": "L-shape",
    "O": "Circle",
    "S": "Fist, thumb across",
    "U": "Index + middle together",
    "V": "Peace sign",
    "W": "Three fingers spread",
    "Y": "Thumb + pinky",
    "5": "Open hand",
}

OFFLINE_HI = {
    "hello": "नमस्ते",
    "namaste": "नमस्ते",
    "stop": "रुकिए",
    "good": "अच्छा",
    "yes": "हाँ",
    "no": "नहीं",
    "i love you": "मैं तुमसे प्यार करता हूँ",
    "peace": "शांति",
    "ok": "ठीक है",
    "call me": "मुझे फोन करें",
    "rock on": "रॉक ऑन",
    "wait": "रुको",
    "me": "मैं",
    "awesome": "शानदार",
    "you": "तुम",
    "three": "तीन",
    "thank you": "धन्यवाद",
    "please": "कृपया",
    "water": "पानी",
    "help": "मदद",
}


def catalog():
    return {
        "ASL": {"phrases": ASL_PHRASES, "spell": SPELL_LETTERS},
        "ISL": {"phrases": ISL_PHRASES, "spell": SPELL_LETTERS},
    }


def all_train_labels():
    labels = set(ASL_PHRASES) | set(ISL_PHRASES) | set(SPELL_LETTERS)
    return sorted(labels)
