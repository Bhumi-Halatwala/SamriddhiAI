"""
src/chatbot/languages.py
Phase 6 - Language strings for the vernacular chatbot.

Supports: English, Hindi, Tamil, Marathi, Bengali.
"""

LANGUAGES = {
    "1": {"code": "en", "name": "English",   "native": "English"},
    "2": {"code": "hi", "name": "Hindi",     "native": "हिंदी"},
    "3": {"code": "ta", "name": "Tamil",     "native": "தமிழ்"},
    "4": {"code": "mr", "name": "Marathi",   "native": "मराठी"},
    "5": {"code": "bn", "name": "Bengali",   "native": "বাংলা"},
}


# Translation dictionary. Keys are message IDs; values per language.
# To extend, add a language and fill the strings.
STRINGS = {
    "greeting": {
        "en": "Namaste! Welcome to SamriddhiAI Banking.",
        "hi": "नमस्ते! समृद्धिAI बैंकिंग में आपका स्वागत है।",
        "ta": "வணக்கம்! சம்ரித்திAI வங்கிக்கு வரவேற்கிறோம்.",
        "mr": "नमस्कार! समृद्धीAI बँकिंगमध्ये आपले स्वागत आहे.",
        "bn": "নমস্কার! সমৃদ্ধিAI ব্যাঙ্কিং-এ আপনাকে স্বাগত।",
    },
    "ask_language": {
        "en": "Please choose your language:\n1) English  2) हिंदी  3) தமிழ்  4) मराठी  5) বাংলা",
        "hi": "कृपया अपनी भाषा चुनें:\n1) English  2) हिंदी  3) தமிழ்  4) मराठी  5) বাংলা",
        "ta": "உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்:\n1) English  2) हिंदी  3) தமிழ்  4) मराठी  5) বাংলা",
        "mr": "कृपया तुमची भाषा निवडा:\n1) English  2) हिंदी  3) தமிழ்  4) मराठी  5) বাংলা",
        "bn": "আপনার ভাষা নির্বাচন করুন:\n1) English  2) हिंदी  3) தமிழ்  4) मराठी  5) বাংলা",
    },
    "language_set": {
        "en": "Great, I'll continue in English.",
        "hi": "बहुत बढ़िया, मैं हिंदी में बात करूँगा।",
        "ta": "சரி, நான் தமிழில் தொடர்கிறேன்.",
        "mr": "छान, मी मराठीत बोलतो.",
        "bn": "দারুণ, আমি বাংলায় কথা বলব।",
    },
    "ask_intent": {
        "en": "How can I help you today? You can ask about: balance, loans, EMI, UPI, or just say 'help'.",
        "hi": "आज मैं आपकी कैसे मदद कर सकता हूँ? आप पूछ सकते हैं: बैलेंस, लोन, EMI, UPI, या 'help' कहें।",
        "ta": "இன்று நான் எப்படி உதவலாம்? நீங்கள் கேட்கலாம்: இருப்பு, கடன், EMI, UPI, அல்லது 'help' சொல்லுங்கள்.",
        "mr": "आज मी तुम्हाला कशी मदत करू शकतो? तुम्ही विचारू शकता: शिल्लक, कर्ज, EMI, UPI, किंवा 'help' म्हणा.",
        "bn": "আজ আমি কীভাবে সাহায্য করতে পারি? আপনি জিজ্ঞাসা করতে পারেন: ব্যালেন্স, ঋণ, EMI, UPI, অথবা 'help' বলুন।",
    },
    "balance_response": {
        "en": "Your monthly income is ₹{income:,.0f}. Based on your transactions, your average monthly savings rate is {savings:.1%}.",
        "hi": "आपकी मासिक आय ₹{income:,.0f} है। आपके लेन-देन के आधार पर, आपकी औसत मासिक बचत दर {savings:.1%} है।",
        "ta": "உங்கள் மாதாந்திர வருமானம் ₹{income:,.0f}. உங்கள் பரிவர்த்தனைகளின் அடிப்படையில், சராசரி மாத சேமிப்பு விகிதம் {savings:.1%}.",
        "mr": "तुमचे मासिक उत्पन्न ₹{income:,.0f} आहे. तुमच्या व्यवहारांवर आधारित, तुमचा सरासरी मासिक बचत दर {savings:.1%} आहे.",
        "bn": "আপনার মাসিক আয় ₹{income:,.0f}। আপনার লেনদেনের ভিত্তিতে, আপনার গড় মাসিক সঞ্চয় হার {savings:.1%}।",
    },
    "loan_offer_safe": {
        "en": "Based on your profile, a {product} loan could suit you. Estimated range: ₹{amount:,.0f}. Would you like to explore?",
        "hi": "आपकी प्रोफ़ाइल के आधार पर, {product} लोन आपके लिए उपयुक्त हो सकता है। अनुमानित सीमा: ₹{amount:,.0f}। क्या आप जानना चाहेंगे?",
        "ta": "உங்கள் சுயவிவரத்தின் அடிப்படையில், {product} கடன் உங்களுக்கு ஏற்றதாக இருக்கலாம். மதிப்பிடப்பட்ட வரம்பு: ₹{amount:,.0f}. நீங்கள் ஆராய விரும்புகிறீர்களா?",
        "mr": "तुमच्या प्रोफाइलवर आधारित, {product} कर्ज तुमच्यासाठी योग्य असू शकते. अंदाजे श्रेणी: ₹{amount:,.0f}. तुम्हाला शोधायचे आहे का?",
        "bn": "আপনার প্রোফাইলের ভিত্তিতে, {product} ঋণ আপনার জন্য উপযুক্ত হতে পারে। আনুমানিক পরিসীমা: ₹{amount:,.0f}। আপনি কি অন্বেষণ করতে চান?",
    },
    "loan_suppressed": {
        "en": "I understand you're asking about loans. Before we discuss new borrowing, let's look at your current commitments. Would you like to talk to a financial advisor?",
        "hi": "मैं समझता हूँ कि आप लोन के बारे में पूछ रहे हैं। नई उधारी पर चर्चा करने से पहले, आइए आपकी वर्तमान प्रतिबद्धताओं को देखें। क्या आप वित्तीय सलाहकार से बात करना चाहेंगे?",
        "ta": "நீங்கள் கடன் பற்றி கேட்கிறீர்கள் என்பதை புரிந்துகொள்கிறேன். புதிய கடன் பற்றி பேசுவதற்கு முன், உங்கள் தற்போதைய கடமைகளைப் பார்ப்போம். நிதி ஆலோசகரிடம் பேச விரும்புகிறீர்களா?",
        "mr": "मला समजते की तुम्ही कर्जाबद्दल विचारत आहात. नवीन कर्जाबद्दल बोलण्यापूर्वी, आपल्या सध्याच्या जबाबदाऱ्या पाहू. तुम्हाला आर्थिक सल्लागाराशी बोलायचे आहे का?",
        "bn": "আমি বুঝতে পারছি আপনি ঋণ সম্পর্কে জিজ্ঞাসা করছেন। নতুন ঋণ নিয়ে আলোচনার আগে, আপনার বর্তমান প্রতিশ্রুতিগুলি দেখা যাক। আপনি কি একজন আর্থিক উপদেষ্টার সাথে কথা বলতে চান?",
    },
    "emi_response": {
        "en": "Your average monthly EMI payment is ₹{emi:,.0f}, which is {ratio:.1%} of your monthly income.",
        "hi": "आपका औसत मासिक EMI भुगतान ₹{emi:,.0f} है, जो आपकी मासिक आय का {ratio:.1%} है।",
        "ta": "உங்கள் சராசரி மாதாந்திர EMI கட்டணம் ₹{emi:,.0f}, இது உங்கள் மாத வருமானத்தில் {ratio:.1%}.",
        "mr": "तुमचे सरासरी मासिक EMI पेमेंट ₹{emi:,.0f} आहे, जे तुमच्या मासिक उत्पन्नाच्या {ratio:.1%} आहे.",
        "bn": "আপনার গড় মাসিক EMI পেমেন্ট ₹{emi:,.0f}, যা আপনার মাসিক আয়ের {ratio:.1%}।",
    },
    "upi_response": {
        "en": "You use UPI for {share:.0%} of your transactions. Your UPI limit is typically ₹1,00,000 per day.",
        "hi": "आप अपने {share:.0%} लेन-देन के लिए UPI का उपयोग करते हैं। आपकी UPI सीमा आमतौर पर ₹1,00,000 प्रति दिन है।",
        "ta": "உங்கள் {share:.0%} பரிவர்த்தனைகளுக்கு UPI பயன்படுத்துகிறீர்கள். உங்கள் UPI வரம்பு பொதுவாக ₹1,00,000 ஒரு நாளைக்கு.",
        "mr": "तुम्ही तुमच्या {share:.0%} व्यवहारांसाठी UPI वापरता. तुमची UPI मर्यादा सामान्यतः ₹1,00,000 प्रतिदिन आहे.",
        "bn": "আপনি আপনার {share:.0%} লেনদেনের জন্য UPI ব্যবহার করেন। আপনার UPI সীমা সাধারণত ₹1,00,000 প্রতিদিন।",
    },
    "help_response": {
        "en": "I can help with: account balance, loan information, EMI details, UPI guidance, and connecting you with a human agent. Just ask naturally.",
        "hi": "मैं इनमें मदद कर सकता हूँ: खाता बैलेंस, लोन जानकारी, EMI विवरण, UPI मार्गदर्शन, और मानव एजेंट से जोड़ना। बस स्वाभाविक रूप से पूछें।",
        "ta": "நான் இதில் உதவ முடியும்: கணக்கு இருப்பு, கடன் தகவல், EMI விவரங்கள், UPI வழிகாட்டுதல், மனித முகவருடன் இணைப்பு. இயல்பாக கேளுங்கள்.",
        "mr": "मी यामध्ये मदत करू शकतो: खाते शिल्लक, कर्ज माहिती, EMI तपशील, UPI मार्गदर्शन, मानव एजंटशी जोडणे. फक्त नैसर्गिकपणे विचारा.",
        "bn": "আমি এই বিষয়ে সাহায্য করতে পারি: অ্যাকাউন্ট ব্যালেন্স, ঋণ তথ্য, EMI বিবরণ, UPI নির্দেশিকা, মানব এজেন্টের সাথে সংযোগ। শুধু স্বাভাবিকভাবে জিজ্ঞাসা করুন।",
    },
    "fallback": {
        "en": "I'm not sure I understood. Could you rephrase? Type 'help' to see what I can do.",
        "hi": "मुझे समझ नहीं आया। कृपया दोबारा कहें? 'help' टाइप करें।",
        "ta": "எனக்கு புரியவில்லை. மீண்டும் சொல்ல முடியுமா? 'help' தட்டச்சு செய்யுங்கள்.",
        "mr": "मला समजले नाही. पुन्हा सांगू शकता का? 'help' टाइप करा.",
        "bn": "আমি বুঝতে পারিনি। আবার বলতে পারেন? 'help' টাইপ করুন।",
    },
    "farewell": {
        "en": "Thank you for banking with us. Have a great day!",
        "hi": "हमारे साथ बैंकिंग के लिए धन्यवाद। आपका दिन शुभ हो!",
        "ta": "எங்களுடன் வங்கி சேவையை பயன்படுத்தியதற்கு நன்றி. நல்ல நாள்!",
        "mr": "आमच्यासोबत बँकिंग केल्याबद्दल धन्यवाद. तुमचा दिवस चांगला जावो!",
        "bn": "আমাদের সাথে ব্যাঙ্কিং করার জন্য ধন্যবাদ। আপনার দিন শুভ হোক!",
    },
}


def t(key: str, lang_code: str, **kwargs) -> str:
    """Fetch a translated string by key. Fallback to English if missing."""
    entry = STRINGS.get(key, {})
    template = entry.get(lang_code) or entry.get("en") or ""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template