import importlib
import streamlit as st

import data_manager
importlib.reload(data_manager)
from data_manager import DataManager
import chatbot_engine
importlib.reload(chatbot_engine)
from chatbot_engine import ChatbotEngine

import voice_ai
importlib.reload(voice_ai)
from voice_ai import IndicF5TTS, IndicConformerASR
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Bharat Banking AI", layout="centered")


# Load clean custom styling
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- Language Selection Map (All 12 Indian Languages) ----
LANGUAGES = {
    "हिंदी (Hindi)": "hi",
    "ગુજરાતી (Gujarati)": "gu",
    "English": "en",
    "मराठी (Marathi)": "mr",
    "বাংলা (Bengali)": "bn",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or",
    "Hinglish (Hindi - Latin)": "hi_latin",
}

SR_LANG_MAP = {
    "hi": "hi-IN",
    "gu": "gu-IN",
    "en": "en-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "or": "or-IN",
    "hi_latin": "hi-IN",
}

MIC_PROMPTS = {
    "hi": ("Speak / बोलें", "Click to Submit / भेजें"),
    "gu": ("Speak / બોલો", "Click to Submit / મોકલો"),
    "en": ("Speak to Answer", "Click to Submit"),
    "mr": ("Speak / बोला", "Click to Submit / पाठवा"),
    "bn": ("Speak / বলুন", "Click to Submit / পাঠান"),
    "te": ("Speak / మాట్లాడండి", "Click to Submit / పంపండి"),
    "ta": ("Speak / பேசுங்கள்", "Click to Submit / சமர்ப்பி"),
    "kn": ("Speak / ಮಾತನಾಡಿ", "Click to Submit / ಸಲ್ಲಿಸಿ"),
    "ml": ("Speak / സംസാരിക്കുക", "Click to Submit / സമർപ്പിക്കുക"),
    "pa": ("Speak / ਬੋਲੋ", "Click to Submit / ਭੇਜੋ"),
    "or": ("Speak / କୁହନ୍ତୁ", "Click to Submit / ପଠାନ୍ତୁ"),
    "hi_latin": ("Speak / Bolen", "Click to Submit / Bhejo"),
}

# General Banking Service Quick Chips across languages
SERVICE_CHIPS = {
    "hi": [
        ("बैलेंस (Balance)", "खाता बैलेंस चेक करें"),
        ("मिनी स्टेटमेंट (Statement)", "मिनी स्टेटमेंट दिखाएं"),
        ("केवाईसी स्थिति (KYC)", "मेरा KYC स्टेटस क्या है?"),
        ("सिबिल स्कोर (CIBIL)", "मेरा सिबिल स्कोर क्या है?"),
        ("मेरी लोन (Active Loans)", "मेरी मौजूदा लोन कौन सी हैं?"),
        ("नया लोन (Apply Loan)", "मुझे नया लोन चाहिए"),
        ("कार्ड ब्लॉक (Block Card)", "एटीएम कार्ड ब्लॉक करें"),
        ("शाखा समय / IFSC", "बैंक का समय और IFSC कोड क्या है?"),
    ],
    "gu": [
        ("બેલેન્સ (Balance)", "ખાતાનું બેલેન્સ કેટલું છે?"),
        ("મિની સ્ટેટમેન્ટ (Statement)", "છેલ્લા વ્યવહારો બતાવો"),
        ("કેવાયસી સ્થિતિ (KYC)", "કેવાયસી સ્થિતિ બતાવો"),
        ("સિબિલ સ્કોર (CIBIL)", "મારો સિબિલ સ્કોર શું છે?"),
        ("મારી લોન (Active Loans)", "મારી હાલની લોન કઈ છે?"),
        ("નવી લોન (Apply Loan)", "નવી લોન જોઈએ છે"),
        ("કાર્ડ બ્લોક (Block Card)", "એટીએમ કાર્ડ બ્લોક કરો"),
        ("બેંક સમય / IFSC", "શાખા સમય અને આઈએફએસસી કોડ શું છે?"),
    ],
    "en": [
        ("Check Balance", "What is my account balance?"),
        ("Mini Statement", "Show my recent transactions"),
        ("KYC Status", "What is my KYC status?"),
        ("CIBIL Score", "What is my credit score?"),
        ("My Active Loans", "Do I have any active loans?"),
        ("Apply for Loan", "I want to apply for a loan"),
        ("Block Debit Card", "Block my ATM card"),
        ("Branch & IFSC", "Branch working hours and IFSC code"),
    ],
    "mr": [
        ("खाते शिल्लक (Balance)", "माझे खाते शिल्लक किती आहे?"),
        ("मिनी स्टेटमेंट", "अलीकडील व्यवहार दाखवा"),
        ("केवायसी स्थिती", "केवायसी स्थिती काय आहे?"),
        ("क्रेडिट स्कोर", "माझा क्रेडिट स्कोर काय आहे?"),
        ("माझी कर्जे", "माझी सध्याची कर्जे कोणती आहेत?"),
        ("नवीन कर्ज", "मला नवीन कर्ज हवे आहे"),
        ("कार्ड ब्लॉक करा", "एटीएम कार्ड ब्लॉक करा"),
    ],
}

# Quick suggested answers for guided loan onboarding
QUICK_CHIPS = {
    "ask_income": {
        "hi": ["₹35,000", "₹50,000", "₹1,00,000"],
        "gu": ["₹35,000", "₹50,000", "₹1,00,000"],
        "en": ["₹35,000", "₹50,000", "₹1,00,000"],
    },
    "ask_purpose": {
        "hi": ["पढ़ाई (Education)", "गाड़ी (Auto)", "घर (Home)", "बिजनेस (Business)", "पर्सनल (Personal)"],
        "gu": ["શિક્ષણ (Education)", "વાહન (Auto)", "ઘર (Home)", "વેપાર (Business)", "વ્યક્તિગત (Personal)"],
        "en": ["Education", "Auto / Vehicle", "Home", "Business", "Personal"],
    },
    "ask_amount": {
        "hi": ["₹1,00,000", "₹2,00,000", "₹5,00,000"],
        "gu": ["₹1,00,000", "₹2,00,000", "₹5,00,000"],
        "en": ["₹1,00,000", "₹2,00,000", "₹5,00,000"],
    },
    "ask_consent": {
        "hi": ["हाँ (Yes)", "नहीं (No)"],
        "gu": ["હા (Yes)", "ના (No)"],
        "en": ["Yes", "No"],
    },
}

# ---- Initialize Pipeline & Data in Session State ----
dm = DataManager.get_instance()
if not dm.customers:
    dm.reload()
customer_list = dm.get_customer_ids(limit=5000)
if not customer_list:
    st.error("Customer dataset could not be loaded. Check data/customer_master.csv and restart the app.")
    st.stop()

if "tts" not in st.session_state:
    st.session_state.tts = IndicF5TTS()

if "asr" not in st.session_state:
    st.session_state.asr = IndicConformerASR()

if "current_lang_name" not in st.session_state:
    st.session_state.current_lang_name = "हिंदी (Hindi)"

# Read customer_id from URL query param (when embedded in the user panel)
_qp_cid = st.query_params.get("customer_id", None)
if _qp_cid and _qp_cid in customer_list:
    st.session_state.active_customer_id = _qp_cid
elif "active_customer_id" not in st.session_state:
    st.session_state.active_customer_id = customer_list[0] if customer_list else "CUST000001"

# ---- Sidebar Controls (Customer Profile & Info) ----
with st.sidebar:
    _embedded = st.query_params.get("customer_id", None) is not None
    if not _embedded:
        st.markdown("### Customer Profile / ग्राहक")
        selected_cid = st.selectbox(
            "Active Customer",
            customer_list,
            index=customer_list.index(st.session_state.active_customer_id) if st.session_state.active_customer_id in customer_list else 0,
            key="customer_selector",
        )
        if selected_cid != st.session_state.active_customer_id:
            st.session_state.active_customer_id = selected_cid
            if "bot" in st.session_state:
                st.session_state.bot.set_customer(selected_cid)
                st.rerun()

    cust_prof = dm.get_profile(st.session_state.active_customer_id)
    
    st.markdown(
        f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:0.75rem; font-size:0.83rem; line-height:1.5;">
            <b>ID:</b> {cust_prof['customer_id']}<br/>
            <b>Occupation:</b> {cust_prof['occupation']} (Age {cust_prof['age']})<br/>
            <b>Monthly Income:</b> ₹{cust_prof['monthly_income']:,}<br/>
            <b>Credit Score:</b> <span style="color:#0C447C; font-weight:700;">{cust_prof['bureau_score']}</span><br/>
            <b>Active Loans:</b> {cust_prof['existing_loan_count']} ({', '.join(cust_prof['existing_loan_types']) if cust_prof['existing_loan_types'] else 'None'})<br/>
            <b>KYC:</b> <span style="color:{'#27500A' if cust_prof['kyc_verified'] else '#854F0B'}; font-weight:600;">{'Verified' if cust_prof['kyc_verified'] else 'Re-KYC Due'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("Restart Chat / ફરી શરૂ કરો / दोबारा शुरू करें", use_container_width=True):
        st.session_state.pop("speech_warning", None)
        st.session_state.pop("_last_processed_audio_id", None)
        current_lang = LANGUAGES[st.session_state.current_lang_name]
        st.session_state.bot = ChatbotEngine(language=current_lang, customer_id=st.session_state.active_customer_id)
        first_msg = st.session_state.bot.start()
        first_audio = st.session_state.tts.synthesize(first_msg, lang=current_lang)
        st.session_state.chat_history = [("bot", first_msg, first_audio)]
        st.rerun()

# ---- Clean Popup Header with Prominent Language Selector ----
head_col, lang_col = st.columns([3, 2])
with head_col:
    st.markdown(
        """
        <div style="padding-top: 0.2rem;">
            <h1 style="font-size: 1.35rem; font-weight: 700; margin: 0; color: #1F2328;">Bharat Banking AI</h1>
            <span style="color: #6B7280; font-size: 0.82rem;">Multilingual Voice & Banking Assistant</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with lang_col:
    selected_lang_name = st.selectbox(
        "Select Language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.current_lang_name),
        label_visibility="collapsed",
        key="main_lang_selector",
    )
    new_lang = LANGUAGES[selected_lang_name]

# ---- Handle Language Change or Initial Bot Creation ----
if "bot" not in st.session_state or st.session_state.current_lang_name != selected_lang_name:
    st.session_state.current_lang_name = selected_lang_name
    st.session_state.pop("speech_warning", None)
    
    # Starting fresh: launch in selected language
    if "bot" not in st.session_state or len(st.session_state.get("chat_history", [])) <= 1:
        st.session_state.bot = ChatbotEngine(language=new_lang, customer_id=st.session_state.active_customer_id)
        welcome_msg = st.session_state.bot.start()
        welcome_audio = st.session_state.tts.synthesize(welcome_msg, lang=new_lang)
        st.session_state.chat_history = [("bot", welcome_msg, welcome_audio)]
    else:
        # Switched language mid-conversation
        st.session_state.bot.set_language(new_lang)


st.markdown("<hr style='margin: 0.6rem 0 1rem 0; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)

# ---- Render Conversation Messages ----
for idx, item in enumerate(st.session_state.chat_history):
    if len(item) == 3:
        speaker, text, audio_bytes = item
    else:
        speaker, text = item
        audio_bytes = None

    css_class = "chat-bot" if speaker == "bot" else "chat-user"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

    # Audio player for bot voice message
    if speaker == "bot" and audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

# ---- User Input Area (Voice + Dynamic Chips + Text) ----
sr_lang = SR_LANG_MAP.get(new_lang, "hi-IN")
start_prompt, stop_prompt = MIC_PROMPTS.get(new_lang, MIC_PROMPTS["hi"])

# 1. Voice Input using mic_recorder
col_mic, col_label = st.columns([1, 2])
with col_mic:
    audio_record = mic_recorder(
        start_prompt=start_prompt,
        stop_prompt=stop_prompt,
        just_once=True,
        use_container_width=True,
        format="wav",
        key=f"voice_rec_{st.session_state.bot.mode}_{st.session_state.bot.step_index}_{len(st.session_state.chat_history)}",
    )
with col_label:
    st.caption(f"Tap to speak in {selected_lang_name.split(' ')[0]}, then tap again to submit.")

# Process recorded voice
if audio_record and audio_record.get("bytes"):
    audio_id = audio_record.get("id")
    if audio_id != st.session_state.get("_last_processed_audio_id"):
        st.session_state["_last_processed_audio_id"] = audio_id
        
        with st.spinner("Processing speech / અવાજ રેકોર્ડિંગ પ્રોસેસ થઈ રહ્યું છે..."):
            transcribed_text = st.session_state.asr.transcribe(audio_record["bytes"], lang=new_lang)

        if transcribed_text:
            st.session_state.pop("speech_warning", None)
            st.session_state.chat_history.append(("user", f"{transcribed_text}", None))
            bot_reply = st.session_state.bot.step(transcribed_text)
            reply_audio = st.session_state.tts.synthesize(bot_reply, lang=new_lang)
            st.session_state.chat_history.append(("bot", bot_reply, reply_audio))
            st.rerun()
        else:
            st.session_state["speech_warning"] = "Could not understand the voice audio. Please try speaking closer to the mic, or choose a quick option below."
            st.rerun()

# Show warning if voice wasn't intelligible
if "speech_warning" in st.session_state:
    st.warning(st.session_state["speech_warning"])

# 2. Dynamic Suggestion Chips
if st.session_state.bot.mode == "loan_onboarding" and not st.session_state.bot.finished:
    # Loan onboarding specific chips
    current_flow_key = st.session_state.bot.LOAN_FLOW[st.session_state.bot.step_index][0]
    loan_chips = QUICK_CHIPS.get(current_flow_key, {}).get(new_lang, QUICK_CHIPS.get(current_flow_key, {}).get("hi", []))
    
    if loan_chips:
        chip_cols = st.columns(len(loan_chips) + 1)
        for c_idx, chip_text in enumerate(loan_chips):
            if chip_cols[c_idx].button(chip_text, key=f"chip_loan_{current_flow_key}_{c_idx}_{new_lang}", use_container_width=True):
                st.session_state.pop("speech_warning", None)
                clean_reply = chip_text.split(" ")[0].replace("₹", "").replace(",", "")
                st.session_state.chat_history.append(("user", chip_text, None))
                bot_reply = st.session_state.bot.step(clean_reply if clean_reply else chip_text)
                reply_audio = st.session_state.tts.synthesize(bot_reply, lang=new_lang)
                st.session_state.chat_history.append(("bot", bot_reply, reply_audio))
                st.rerun()
        
        # Exit loan onboarding button
        exit_label = "Main Menu" if new_lang == "en" else ("મુખ્ય સેવાઓ" if new_lang == "gu" else "मुख्य सेवाएँ")
        if chip_cols[-1].button(exit_label, key="btn_exit_loan", use_container_width=True):
            st.session_state.bot.mode = "general"
            menu_msg = st.session_state.bot.start()
            menu_audio = st.session_state.tts.synthesize(menu_msg, lang=new_lang)
            st.session_state.chat_history.append(("bot", menu_msg, menu_audio))
            st.rerun()

else:
    # General Multi-Service Banking Assistance Chips (2 rows)
    chips_list = SERVICE_CHIPS.get(new_lang, SERVICE_CHIPS.get("hi", SERVICE_CHIPS["en"]))
    
    # Row 1
    row1 = chips_list[:4]
    row1_cols = st.columns(len(row1))
    for c_idx, (label, query_text) in enumerate(row1):
        if row1_cols[c_idx].button(label, key=f"chip_srv_r1_{c_idx}_{new_lang}", use_container_width=True):
            st.session_state.pop("speech_warning", None)
            st.session_state.chat_history.append(("user", label, None))
            bot_reply = st.session_state.bot.step(query_text)
            reply_audio = st.session_state.tts.synthesize(bot_reply, lang=new_lang)
            st.session_state.chat_history.append(("bot", bot_reply, reply_audio))
            st.rerun()

    # Row 2
    row2 = chips_list[4:]
    if row2:
        row2_cols = st.columns(len(row2))
        for c_idx, (label, query_text) in enumerate(row2):
            if row2_cols[c_idx].button(label, key=f"chip_srv_r2_{c_idx}_{new_lang}", use_container_width=True):
                st.session_state.pop("speech_warning", None)
                st.session_state.chat_history.append(("user", label, None))
                bot_reply = st.session_state.bot.step(query_text)
                reply_audio = st.session_state.tts.synthesize(bot_reply, lang=new_lang)
                st.session_state.chat_history.append(("bot", bot_reply, reply_audio))
                st.rerun()

# 3. Main Text Input Box
user_input = st.chat_input("Ask anything: KYC, Balance, Loans, Block Card, FAQs... / પૂછો...")
if user_input:
    st.session_state.pop("speech_warning", None)
    st.session_state.chat_history.append(("user", user_input, None))
    bot_reply = st.session_state.bot.step(user_input)
    reply_audio = st.session_state.tts.synthesize(bot_reply, lang=new_lang)
    st.session_state.chat_history.append(("bot", bot_reply, reply_audio))
    st.rerun()
