import os
import re
import joblib
import pandas as pd
import streamlit as st
import speech_recognition as sr

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Call Classification",
    page_icon="📞",
    layout="wide"
)


# ============================================================
# NLTK RESOURCES
# ============================================================

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test")
except LookupError:
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "./call_classifier.pkl"
TFIDF_PATH = "./tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

except FileNotFoundError:

    st.error(
        "Model files not found. Please train the model in Jupyter "
        "and save the .pkl files inside the models folder."
    )

    st.stop()


# ============================================================
# SPEECH RECOGNITION
# ============================================================

recognizer = sr.Recognizer()


def transcribe_audio(file_path):

    try:

        with sr.AudioFile(file_path) as source:

            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        return text

    except sr.UnknownValueError:

        return ""

    except sr.RequestError:

        return ""

    except Exception:

        return ""


# ============================================================
# RULE-BASED KEYWORD CLASSIFICATION
# ============================================================

def keyword_rule(text):

    text = text.lower()

    # --------------------------------------------------------
    # URGENT
    # --------------------------------------------------------

    urgent_keywords = [

        "not working",
        "issue",
        "problem",
        "error",
        "trouble",
        "failed",
        "failure",
        "dissatisfied",
        "refund",
        "return",
        "complaint",
        "unhappy",
        "complain",
        "angry",
        "delay",
        "bad",
        "worst"

    ]

    # --------------------------------------------------------
    # CLOSED
    # --------------------------------------------------------

    closed_keywords = [

        "resolved",
        "fixed",
        "solved",
        "thank you",
        "thanks",
        "working fine",
        "working now",
        "happy",
        "satisfied"

    ]

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------

    open_keywords = [

        "need",
        "help",
        "support",
        "enquire",
        "inquiry",
        "question",
        "request",
        "assist",
        "information"

    ]

    # --------------------------------------------------------
    # PRIORITY
    # Urgent > Closed > Open
    # --------------------------------------------------------

    if any(word in text for word in urgent_keywords):

        return "Urgent"

    elif any(word in text for word in closed_keywords):

        return "Closed"

    elif any(word in text for word in open_keywords):

        return "Open"

    else:

        return None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        "",
        text
    )

    words = text.split()

    words = [

        lemmatizer.lemmatize(word)

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict(text):

    # --------------------------------------------------------
    # STEP 1: RULE BASED
    # --------------------------------------------------------

    rule_output = keyword_rule(text)

    if rule_output is not None:

        return rule_output, "Rule-Based"


    # --------------------------------------------------------
    # STEP 2: MACHINE LEARNING
    # --------------------------------------------------------

    text_clean = clean_text(text)

    vector = tfidf.transform([text_clean])

    prediction = model.predict(vector)[0]

    return prediction, "Machine Learning"


# ============================================================
# LOAD DATASET
# ============================================================

OUTPUT_FILE = "./Output.csv"

if os.path.exists(OUTPUT_FILE):

    df = pd.read_csv(OUTPUT_FILE)

else:

    df = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📞 CallSense AI")

st.sidebar.write(
    "AI-Powered Customer Call Classification"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Dashboard",
        "🎧 Call Prediction",
        "📊 Model Performance",
        "📋 Call Dataset"
    ]

)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.title("📞 Customer Call Classification")

    st.subheader(
        "AI-Powered Customer Support Call Analysis"
    )

    st.write(
        """
        This system combines Speech Recognition,
        NLP, TF-IDF, Machine Learning and
        rule-based classification to categorize
        customer calls into Open, Closed and Urgent.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if not df.empty and "category" in df.columns:

        total_calls = len(df)

        urgent_calls = (
            df["category"] == "Urgent"
        ).sum()

        closed_calls = (
            df["category"] == "Closed"
        ).sum()

        open_calls = (
            df["category"] == "Open"
        ).sum()

    else:

        total_calls = 0
        urgent_calls = 0
        closed_calls = 0
        open_calls = 0


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📞 Total Calls",
            total_calls
        )


    with col2:

        st.metric(
            "🔴 Urgent",
            urgent_calls
        )


    with col3:

        st.metric(
            "🟢 Closed",
            closed_calls
        )


    with col4:

        st.metric(
            "🔵 Open",
            open_calls
        )


    st.divider()


    # --------------------------------------------------------
    # CATEGORY CHARTS
    # --------------------------------------------------------

    if not df.empty and "category" in df.columns:

        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "📊 Call Category Distribution"
            )

            category_counts = (
                df["category"]
                .value_counts()
            )

            st.bar_chart(category_counts)


        with col2:

            st.subheader(
                "🥧 Category Percentage"
            )

            st.dataframe(

                category_counts.reset_index().rename(

                    columns={
                        "category": "Category",
                        "count": "Calls"
                    }

                ),

                use_container_width=True,

                hide_index=True

            )


        st.divider()


        # ----------------------------------------------------
        # RECENT CALLS
        # ----------------------------------------------------

        st.subheader("📋 Recent Calls")

        display_columns = [

            column

            for column in [

                "file",
                "transcription",
                "category"

            ]

            if column in df.columns

        ]

        if display_columns:

            st.dataframe(

                df[display_columns].head(10),

                use_container_width=True,

                hide_index=True

            )


# ============================================================
# CALL PREDICTION PAGE
# ============================================================

elif page == "🎧 Call Prediction":

    st.title("🎧 Customer Call Prediction")

    st.write(
        "Upload a WAV audio file to classify the customer call."
    )

    uploaded_file = st.file_uploader(

        "Upload WAV file",

        type=["wav"]

    )


    if uploaded_file is not None:

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        st.info("Audio uploaded successfully.")


        # ----------------------------------------------------
        # SAVE TEMP AUDIO
        # ----------------------------------------------------

        temp_file = "temp_uploaded_call.wav"

        with open(temp_file, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # TRANSCRIBE
        # ----------------------------------------------------

        if st.button(
            "🎤 Transcribe & Predict"
        ):

            with st.spinner(
                "Converting speech to text..."
            ):

                text = transcribe_audio(
                    temp_file
                )


            if text.strip() == "":

                st.error(
                    "Could not understand the audio."
                )

            else:

                st.subheader(
                    "📝 Transcription"
                )

                st.write(text)


                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                with st.spinner(
                    "Classifying customer call..."
                ):

                    prediction, method = predict(
                        text
                    )


                st.subheader(
                    "🤖 Prediction"
                )


                if prediction == "Urgent":

                    st.error(
                        f"🔴 URGENT"
                    )


                elif prediction == "Closed":

                    st.success(
                        f"🟢 CLOSED"
                    )


                else:

                    st.info(
                        f"🔵 OPEN"
                    )


                st.write(
                    f"Prediction Method: **{method}**"
                )


        # ----------------------------------------------------
        # REMOVE TEMP FILE
        # ----------------------------------------------------

        if os.path.exists(temp_file):

            os.remove(temp_file)


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        """
        The machine learning model is evaluated
        using accuracy, precision, recall and F1-score.
        """
    )

    st.info(
        "The metrics below are calculated from the "
        "test dataset used during model training."
    )


    # --------------------------------------------------------
    # NOTE
    # --------------------------------------------------------

    st.warning(
        """
        To display your exact training accuracy,
        precision, recall and F1-score here,
        we will save the evaluation results from
        your Jupyter notebook in the next step.
        """
    )


    st.subheader(
        "🤖 Machine Learning Model"
    )

    st.write(
        "Algorithm: Multinomial Naive Bayes"
    )

    st.write(
        "Feature Engineering: TF-IDF"
    )

    st.write(
        "N-grams: Unigrams + Bigrams"
    )

    st.write(
        "Maximum Features: 500"
    )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📋 Call Dataset":

    st.title("📋 Customer Call Dataset")

    if df.empty:

        st.warning(
            "Output.csv was not found."
        )

    else:

        st.write(
            f"Total records: **{len(df)}**"
        )

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True

        )
