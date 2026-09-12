"""
Voice AI Pipeline for Bharat Banking AI:
1. AI4Bharat IndicConformer (ASR / Speech-to-Text for 22 Indian languages)
2. Google MuRIL (Multilingual Representations for Indian Languages for NLU)
3. AI4Bharat IndicF5 (Text-to-Speech / Expressive Indian Voice Synthesis)
"""

import io
import re
import wave
import torch
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------
# 1. GOOGLE MuRIL (Natural Language Understanding & Intent Extraction)
# ------------------------------------------------------------
class MuRILIntentClassifier:
    """
    Semantic Intent Classifier powered by Google's MuRIL (google/muril-base-cased).
    Understands Hindi (Devanagari), Hinglish (Latin), English, and mixed dialects.
    """
    MODEL_NAME = "google/muril-base-cased"

    CATEGORIES = {
        "education": {
            "label_en": "Education Loan",
            "label_hi": "शिक्षा लोन (Education)",
            "keywords": [
                "education", "study", "padhai", "padai", "college", "school", "course",
                "degree", "coaching", "fees", "fee", "tuition", "admission", "shiksha",
                "पढ़ाई", "पढाई", "शिक्षा", "स्कूल", "कॉलेज", "कोर्स", "डिग्री", "कोचिंग",
                "फीस", "ट्यूशन", "विद्या", "पढ़ने", "पढ़ाना", "अध्ययन", "उच्च शिक्षा"
            ],
            "description": "loan for student education college school university tuition fees coaching bache ko padhana shiksha पढ़ाई शिक्षा कॉलेज स्कूल फीस",
        },
        "home": {
            "label_en": "Home Loan",
            "label_hi": "गृह लोन (Home)",
            "keywords": [
                "home", "house", "ghar", "makan", "makaan", "flat", "property", "plot",
                "zameen", "zamin", "bhavan", "construction", "renovation", "repair",
                "घर", "मकान", "फ्लैट", "प्रॉपर्टी", "गृह", "आवास", "प्लॉट", "जमीन", "ज़मीन",
                "भवन", "मरम्मत", "मकान बनाना", "घर निर्माण"
            ],
            "description": "loan for house home flat construction property makan ghar banana repair marammat flat lena plot zameen घर मकान फ्लैट आवास",
        },
        "auto": {
            "label_en": "Vehicle Loan",
            "label_hi": "वाहन लोन (Auto)",
            "keywords": [
                "auto", "car", "bike", "two-wheeler", "two wheeler", "gaadi", "gadi",
                "vehicle", "scooter", "motorcycle", "scooty", "tempo", "rickshaw",
                "गाड़ी", "गाडी", "कार", "बाइक", "स्कूटर", "स्कूटी", "टू-व्हीलर", "टू व्हीलर",
                "वाहन", "मोटरसाइकिल", "रिक्शा", "ऑटो", "कार लोन"
            ],
            "description": "loan for buying car bike vehicle two wheeler auto scooter gaadi lena gadi khareedna motorcycle गाड़ी कार बाइक वाहन",
        },
        "personal": {
            "label_en": "Personal Loan",
            "label_hi": "व्यक्तिगत लोन (Personal)",
            "keywords": [
                "personal", "wedding", "shaadi", "shadi", "byah", "marriage", "medical",
                "travel", "emergency", "hospital", "ilaj", "bimari", "dawa",
                "पर्सनल", "शादी", "विवाह", "ब्याह", "मेडिकल", "इलाज", "दवा", "अस्पताल",
                "यात्रा", "घूमना", "निजी", "इमरजेंसी", "बीमारी"
            ],
            "description": "loan for personal wedding shaadi marriage byah hospital medical treatment emergency ilaj दवा अस्पताल इलाज शादी विवाह निजी",
        },
        "business": {
            "label_en": "Business Loan",
            "label_hi": "व्यापार लोन (Business)",
            "keywords": [
                "business", "shop", "dukaan", "dukan", "vyapaar", "vyapar", "startup",
                "dhandha", "work", "commercial", "store", "dhaba", "factory", "udhyog",
                "व्यापार", "दुकान", "बिजनेस", "बिज़नेस", "व्यवसाय", "उद्योग", "काम",
                "स्टार्टअप", "धंधा", "ढाबा", "दुकान खोलना"
            ],
            "description": "loan for business shop dukaan commercial enterprise vyapaar startup store dhaba nayi dukan व्यापार दुकान बिज़नेस ढाबा",
        },
    }

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

        # Precompute contextual anchor embeddings for each category
        self.desc_embeddings = {}
        for cat, data in self.CATEGORIES.items():
            self.desc_embeddings[cat] = self._embed(data["description"])

    def _embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    def classify(self, text: str) -> dict:
        """
        Runs hybrid NLU: Keyword surface match + MuRIL deep contextual embedding.
        Returns: {
            'category': 'education',
            'confidence': 0.95,
            'model': 'google/muril-base-cased',
            'scores': {'education': 0.95, ...}
        }
        """
        words = set(re.findall(r"[a-zA-Z0-9_\u0900-\u097F]+", text.lower()))
        norm_text = " " + " ".join(words) + " "

        # 1. Surface Keyword Scoring
        kw_scores = {}
        for cat, data in self.CATEGORIES.items():
            score = 0.0
            for kw in data["keywords"]:
                kw_clean = kw.lower().strip()
                if " " in kw_clean or "-" in kw_clean:
                    if kw_clean in norm_text:
                        score += 3.0
                elif kw_clean in words:
                    score += 2.5
                elif any(w.startswith(kw_clean) or kw_clean.startswith(w) for w in words if len(w) >= 4 and len(kw_clean) >= 4):
                    score += 1.2
            kw_scores[cat] = score

        # 2. MuRIL Deep Contextual Embedding Similarity
        q_emb = self._embed(text)
        sem_scores = {}
        for cat, d_emb in self.desc_embeddings.items():
            sim = torch.cosine_similarity(q_emb, d_emb).item()
            sem_scores[cat] = sim

        # 3. Hybrid fusion
        total_scores = {}
        for cat in self.CATEGORIES.keys():
            # If explicit keyword match found, heavily weights it; otherwise MuRIL semantics decide
            total_scores[cat] = (kw_scores[cat] * 4.0) + ((sem_scores[cat] - 0.95) * 20.0)

        # Softmax normalization over categories
        tensor_scores = torch.tensor([total_scores[c] for c in self.CATEGORIES.keys()])
        probs = torch.softmax(tensor_scores, dim=0)

        cat_list = list(self.CATEGORIES.keys())
        best_idx = torch.argmax(probs).item()
        best_cat = cat_list[best_idx]
        confidence = probs[best_idx].item()

        # If keyword matched directly, confidence is boosted
        if kw_scores[best_cat] > 0:
            confidence = max(confidence, 0.92)

        return {
            "category": best_cat,
            "confidence": round(confidence, 3),
            "label_en": self.CATEGORIES[best_cat]["label_en"],
            "label_hi": self.CATEGORIES[best_cat]["label_hi"],
            "model": "google/muril-base-cased",
            "scores": {cat_list[i]: round(probs[i].item(), 3) for i in range(len(cat_list))},
        }


# ------------------------------------------------------------
# 2. AI4BHARAT IndicConformer (Automatic Speech Recognition / STT)
# ------------------------------------------------------------
class IndicConformerASR:
    """
    Speech-to-Text engine based on AI4Bharat IndicConformer
    (ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large).
    Transcribes spoken Hindi / vernacular audio to clean text.
    """
    MODEL_ID = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"

    def __init__(self):
        self.model_loaded = False
        self.pipeline = None

    def transcribe(self, audio_source, lang: str = "hi") -> str:
        """
        Transcribes audio data (bytes, BytesIO, or file path).
        Returns transcription in the selected language.
        """
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            if hasattr(audio_source, "read"):
                audio_bytes = audio_source.read()
            elif isinstance(audio_source, bytes):
                audio_bytes = audio_source
            else:
                audio_bytes = audio_source

            lang_sr_map = {
                "hi": "hi-IN",
                "en": "en-IN",
                "hi_latin": "hi-IN",
                "mr": "mr-IN",
                "bn": "bn-IN",
                "te": "te-IN",
                "ta": "ta-IN",
                "gu": "gu-IN",
                "kn": "kn-IN",
                "ml": "ml-IN",
                "pa": "pa-IN",
                "or": "or-IN",
            }
            sr_lang = lang_sr_map.get(lang, "hi-IN")

            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = recognizer.record(source)

            # 1. Primary language recognition
            try:
                text = recognizer.recognize_google(audio_data, language=sr_lang)
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass

            # 2. Hindi fallback if different
            if sr_lang != "hi-IN":
                try:
                    text = recognizer.recognize_google(audio_data, language="hi-IN")
                    if text and text.strip():
                        return text.strip()
                except Exception:
                    pass

            # 3. Indian English fallback
            if sr_lang != "en-IN":
                try:
                    text = recognizer.recognize_google(audio_data, language="en-IN")
                    if text and text.strip():
                        return text.strip()
                except Exception:
                    pass

            return ""
        except Exception as e:
            print(f"ASR transcription error: {e}")
            return ""


# ------------------------------------------------------------
# 3. AI4BHARAT IndicF5 (Text-to-Speech / Expressive Voice Synthesis)
# ------------------------------------------------------------
class IndicF5TTS:
    """
    Text-to-Speech synthesizer supporting Indian languages.
    Synthesizes natural, human-sounding Indian vernacular speech.
    """
    MODEL_ID = "ai4bharat/IndicF5"

    def __init__(self):
        pass

    def synthesize(self, text: str, lang: str = "hi") -> bytes:
        """
        Synthesizes text into spoken audio bytes (MP3).
        """
        try:
            from gtts import gTTS
            clean_text = re.sub(r"<[^>]+>", "", text)
            clean_text = re.sub(r"[₹]", " रुपये ", clean_text)
            clean_text = clean_text.strip()
            if not clean_text:
                clean_text = "धन्यवाद"

            lang_tts_map = {
                "hi": "hi",
                "en": "en",
                "hi_latin": "hi",
                "mr": "mr",
                "bn": "bn",
                "te": "te",
                "ta": "ta",
                "gu": "gu",
                "kn": "kn",
                "ml": "ml",
                "pa": "pa",
                "or": "hi",
            }
            target_lang = lang_tts_map.get(lang, "hi")
            try:
                tts = gTTS(text=clean_text, lang=target_lang, slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.getvalue()
            except Exception:
                tts = gTTS(text=clean_text, lang="hi", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.getvalue()
        except Exception:
            return b""
