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


# ---------------------------------------------------------------------------
# Dictionary: how to make every sign, in English and Hindi (shown in the app's
# Dictionary). Letters are standard ASL fingerspelling; their drawings come from
# real dataset samples (dataset/sign_shapes.json, built by build_sign_shapes.py).
# Phrase and ISL entries describe the hand shape Vani-Setu's rules recognise,
# drawn from the "pose" below: which fingers are up (T I M R P), whether they
# are spread, and where the thumb is.
# ---------------------------------------------------------------------------

LETTER_GUIDE = [
    ("A", "Make a fist. Keep your thumb straight against the side of your index finger.",
          "मुट्ठी बनाएँ। अंगूठे को सीधा तर्जनी के बगल में रखें।"),
    ("B", "Hold four fingers straight up and together. Fold your thumb across your palm.",
          "चारों उंगलियाँ सीधी और सटाकर ऊपर रखें। अंगूठा हथेली पर मोड़ें।"),
    ("C", "Curve your fingers and thumb into the shape of a C.",
          "उंगलियों और अंगूठे को मोड़कर C का आकार बनाएँ।"),
    ("D", "Point your index finger up. Touch the other fingertips to your thumb to make a circle.",
          "तर्जनी ऊपर रखें। बाकी उंगलियों के सिरे अंगूठे से मिलाकर गोला बनाएँ।"),
    ("E", "Bend all fingers down so the fingertips rest on your thumb, folded across the palm.",
          "सभी उंगलियाँ नीचे मोड़ें ताकि उनके सिरे हथेली पर मुड़े अंगूठे पर टिकें।"),
    ("F", "Touch your thumb and index finger in a circle. Keep the other three fingers up and apart.",
          "अंगूठे और तर्जनी से गोला बनाएँ। बाकी तीन उंगलियाँ ऊपर और अलग रखें।"),
    ("G", "Point your index finger and thumb sideways, parallel to each other. Curl the other fingers.",
          "तर्जनी और अंगूठे को समानांतर रखकर बगल की ओर करें। बाकी उंगलियाँ मोड़ें।"),
    ("H", "Point your index and middle fingers sideways, together. Curl the other fingers.",
          "तर्जनी और मध्यमा को साथ में बगल की ओर करें। बाकी उंगलियाँ मोड़ें।"),
    ("I", "Raise your little finger. Close the other fingers with the thumb across them.",
          "छोटी उंगली ऊपर उठाएँ। बाकी उंगलियाँ बंद रखें और अंगूठा उन पर रखें।"),
    ("J", "Make the I shape, then draw a J in the air with your little finger.",
          "I का आकार बनाएँ, फिर छोटी उंगली से हवा में J बनाएँ।"),
    ("K", "Raise your index and middle fingers in a V. Touch your thumb to the middle finger, between them.",
          "तर्जनी और मध्यमा को V में उठाएँ। अंगूठे को दोनों के बीच मध्यमा से लगाएँ।"),
    ("L", "Raise your index finger and stretch your thumb out to make an L.",
          "तर्जनी ऊपर और अंगूठा बाहर फैलाकर L बनाएँ।"),
    ("M", "Fold your thumb under your index, middle and ring fingers.",
          "अंगूठे को तर्जनी, मध्यमा और अनामिका के नीचे मोड़ें।"),
    ("N", "Fold your thumb under your index and middle fingers.",
          "अंगूठे को तर्जनी और मध्यमा के नीचे मोड़ें।"),
    ("O", "Curve all your fingers to meet your thumb in a round O.",
          "सभी उंगलियाँ मोड़कर अंगूठे से मिलाएँ और गोल O बनाएँ।"),
    ("P", "Make the K shape, then point your hand downwards.",
          "K का आकार बनाकर हाथ को नीचे की ओर झुकाएँ।"),
    ("Q", "Make the G shape, then point your thumb and index finger downwards.",
          "G का आकार बनाकर अंगूठा और तर्जनी नीचे की ओर करें।"),
    ("R", "Cross your middle finger over your index finger, both pointing up.",
          "तर्जनी और मध्यमा ऊपर रखकर मध्यमा को तर्जनी के ऊपर क्रॉस करें।"),
    ("S", "Make a fist with your thumb across the front of your fingers.",
          "मुट्ठी बनाएँ और अंगूठा उंगलियों के आगे रखें।"),
    ("T", "Make a fist with your thumb tucked between your index and middle fingers.",
          "मुट्ठी बनाएँ और अंगूठे को तर्जनी और मध्यमा के बीच दबाएँ।"),
    ("U", "Raise your index and middle fingers together.",
          "तर्जनी और मध्यमा को सटाकर ऊपर रखें।"),
    ("V", "Raise your index and middle fingers apart, in a V.",
          "तर्जनी और मध्यमा को अलग करके V बनाएँ।"),
    ("W", "Raise your index, middle and ring fingers, spread apart.",
          "तर्जनी, मध्यमा और अनामिका को फैलाकर ऊपर रखें।"),
    ("X", "Raise your index finger and bend it into a hook. Close the other fingers.",
          "तर्जनी को हुक की तरह मोड़ें। बाकी उंगलियाँ बंद रखें।"),
    ("Y", "Stretch out your thumb and little finger. Fold the other three fingers.",
          "अंगूठा और छोटी उंगली बाहर फैलाएँ। बाकी तीन उंगलियाँ मोड़ें।"),
    ("Z", "Point your index finger and draw a Z in the air.",
          "तर्जनी से हवा में Z बनाएँ।"),
]
MOTION_LETTERS = {"J", "Z"}

# (sign, Hindi word, how in English, how in Hindi, pose)
# pose: up = fingers held up (T thumb, I index, M middle, R ring, P little), spread = fingers apart,
#       thumb = "out" (stretched sideways), "up" (thumbs-up), "top" (resting on the fist), "in" (folded),
#       circle = thumb and index fingertips touch
PHRASE_GUIDE = [
    ("Hello", "नमस्ते", "Open your hand with all five fingers spread, palm to the camera.",
     "हथेली कैमरे की ओर करके पाँचों उंगलियाँ फैलाएँ।", {"up": "TIMRP", "spread": True, "thumb": "out"}),
    ("Stop", "रुको", "Hold four fingers straight up and together, palm forward, thumb folded.",
     "चारों उंगलियाँ सटाकर सीधी ऊपर रखें, हथेली आगे, अंगूठा मोड़ें।", {"up": "IMRP", "spread": False, "thumb": "in"}),
    ("Good", "अच्छा", "Thumbs up: close your fingers and point your thumb up.",
     "अंगूठा ऊपर: उंगलियाँ बंद करें और अंगूठा ऊपर की ओर रखें।", {"up": "T", "thumb": "up"}),
    ("Yes", "हाँ", "Make a fist with your thumb resting on top of your fingers.",
     "मुट्ठी बनाएँ और अंगूठा उंगलियों के ऊपर टिकाएँ।", {"up": "", "thumb": "top"}),
    ("No", "नहीं", "Make a closed fist with the thumb tucked low.",
     "बंद मुट्ठी बनाएँ, अंगूठा नीचे दबा हो।", {"up": "", "thumb": "in"}),
    ("I Love You", "मैं तुमसे प्यार करता हूँ", "Raise your thumb, index finger and little finger. Fold the middle and ring fingers.",
     "अंगूठा, तर्जनी और छोटी उंगली उठाएँ। मध्यमा और अनामिका मोड़ें।", {"up": "TIP", "spread": True, "thumb": "out"}),
    ("Peace", "शांति", "Raise your index and middle fingers wide apart in a V.",
     "तर्जनी और मध्यमा को खूब फैलाकर V बनाएँ।", {"up": "IM", "spread": True, "thumb": "in"}),
    ("OK", "ठीक है", "Touch your thumb and index finger in a circle. Keep the other three fingers up.",
     "अंगूठे और तर्जनी से गोला बनाएँ। बाकी तीन उंगलियाँ ऊपर रखें।", {"up": "MRP", "spread": True, "circle": True}),
    ("Call Me", "मुझे फ़ोन करें", "Stretch out your thumb and little finger, like a phone.",
     "फ़ोन की तरह अंगूठा और छोटी उंगली फैलाएँ।", {"up": "TP", "spread": True, "thumb": "out"}),
    ("Rock On", "शानदार", "Raise your index and little fingers. Fold the others, thumb over them.",
     "तर्जनी और छोटी उंगली उठाएँ। बाकी मोड़ें, अंगूठा उन पर रखें।", {"up": "IP", "spread": True, "thumb": "in"}),
    ("Wait", "रुको", "Hold up only your index finger.",
     "सिर्फ़ तर्जनी ऊपर उठाएँ।", {"up": "I", "thumb": "in"}),
    ("Me", "मैं", "Hold up only your little finger.",
     "सिर्फ़ छोटी उंगली ऊपर उठाएँ।", {"up": "P", "thumb": "in"}),
    ("Awesome", "बहुत बढ़िया", "Make an L: index finger up and thumb stretched out.",
     "L बनाएँ: तर्जनी ऊपर और अंगूठा बाहर।", {"up": "TI", "spread": True, "thumb": "out"}),
    ("You", "तुम", "Raise your index and middle fingers held together.",
     "तर्जनी और मध्यमा को सटाकर ऊपर रखें।", {"up": "IM", "spread": False, "thumb": "in"}),
    ("Three", "तीन", "Raise your index, middle and ring fingers, spread apart.",
     "तर्जनी, मध्यमा और अनामिका को फैलाकर ऊपर रखें।", {"up": "IMR", "spread": True, "thumb": "in"}),
]

ISL_GUIDE = [
    ("Namaste", "नमस्ते", "Hold four fingers up close together with the thumb folded. (The full ISL sign joins both palms; record it to train the ISL network.)",
     "चार उंगलियाँ सटाकर ऊपर रखें, अंगूठा मोड़ें। (पूरे ISL चिह्न में दोनों हथेलियाँ जुड़ती हैं; ISL नेटवर्क के लिए उसे रिकॉर्ड करें।)",
     {"up": "IMRP", "spread": False, "thumb": "in"}),
    ("Hello", "नमस्ते", "Open your hand with all five fingers spread.",
     "पाँचों उंगलियाँ फैलाकर हाथ खोलें।", {"up": "TIMRP", "spread": True, "thumb": "out"}),
    ("Yes", "हाँ", "Thumbs up: close your fingers and point your thumb up.",
     "अंगूठा ऊपर: उंगलियाँ बंद करें और अंगूठा ऊपर रखें।", {"up": "T", "thumb": "up"}),
    ("No", "नहीं", "Make a closed fist.",
     "बंद मुट्ठी बनाएँ।", {"up": "", "thumb": "in"}),
    ("Thank You", "धन्यवाद", "Show a flat palm to the camera: four fingers up and slightly apart, thumb folded.",
     "कैमरे को सपाट हथेली दिखाएँ: चार उंगलियाँ ऊपर और थोड़ी अलग, अंगूठा मुड़ा हुआ।", {"up": "IMRP", "spread": True, "thumb": "in"}),
    ("Please", "कृपया", "Open your hand with all five fingers up and held together.",
     "पाँचों उंगलियाँ सटाकर ऊपर रखें।", {"up": "TIMRP", "spread": False, "thumb": "out"}),
    ("Water", "पानी", "Raise three fingers (index, middle, ring) spread apart, like a W.",
     "W की तरह तीन उंगलियाँ (तर्जनी, मध्यमा, अनामिका) फैलाकर उठाएँ।", {"up": "IMR", "spread": True, "thumb": "in"}),
    ("Help", "मदद", "Make an L: index finger up and thumb stretched out.",
     "L बनाएँ: तर्जनी ऊपर और अंगूठा बाहर।", {"up": "TI", "spread": True, "thumb": "out"}),
    ("Me", "मैं", "Hold up only your index finger.",
     "सिर्फ़ तर्जनी ऊपर उठाएँ।", {"up": "I", "thumb": "in"}),
    ("You", "आप", "Raise your index and middle fingers together.",
     "तर्जनी और मध्यमा को सटाकर ऊपर रखें।", {"up": "IM", "spread": False, "thumb": "in"}),
    ("Good", "अच्छा", "Thumbs up with a firm fist. The rules read this the same as Yes; the ISL network can learn the difference from recordings.",
     "मज़बूत मुट्ठी के साथ अंगूठा ऊपर। नियम इसे 'हाँ' जैसा पढ़ते हैं; रिकॉर्डिंग से ISL नेटवर्क अंतर सीख सकता है।", {"up": "T", "thumb": "up"}),
    ("I Love You", "मैं तुमसे प्यार करता हूँ", "Raise your thumb, index finger and little finger.",
     "अंगूठा, तर्जनी और छोटी उंगली उठाएँ।", {"up": "TIP", "spread": True, "thumb": "out"}),
]


def dictionary(shapes=None):
    """Everything the Dictionary screen shows, grouped by tab."""
    shapes = shapes or {}
    return {
        "letter": [{"sign": s, "how": en, "how_hi": hi, "motion": s in MOTION_LETTERS, "shape": shapes.get(s)}
                   for s, en, hi in LETTER_GUIDE],
        "phrase": [{"sign": s, "hindi": w, "how": en, "how_hi": hi, "pose": p} for s, w, en, hi, p in PHRASE_GUIDE],
        "isl": [{"sign": s, "hindi": w, "how": en, "how_hi": hi, "pose": p} for s, w, en, hi, p in ISL_GUIDE],
    }
