# 📞 Customer Call Center Classification System

An AI-powered customer call analysis application that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify customer queries and support call center operations.

## 🌐 Live Demo

🚀 **Try the application:** [Customer Call Classification App](https://customer-call-classifier.streamlit.app/)

## 📌 Overview

Customer support teams handle a large volume of calls daily. Manually reviewing and categorizing every interaction can be time-consuming.

This project automates customer call analysis by processing call transcripts, extracting meaningful text features, and classifying customer queries into **Open, Urgent, and Closed** categories. It also provides an interactive Streamlit dashboard for exploring call classification results.

## 🎯 Objectives

- Automate customer query classification using Machine Learning.
- Process call transcripts using NLP techniques.
- Extract meaningful text features using TF-IDF.
- Analyze call categories through an interactive dashboard.
- Support faster identification of urgent and unresolved customer queries.

## ✨ Key Features

- **📊 Interactive Dashboard:** View call statistics and category-wise summaries.
- **🎙️ Speech Recognition:** Convert supported audio recordings into text.
- **🧹 Text Preprocessing:** Clean and prepare text for machine learning.
- **🔎 TF-IDF Feature Extraction:** Transform text into numerical features.
- **🤖 Machine Learning Classification:** Use a trained Naive Bayes model to predict call categories.
- **📈 Model Performance Analysis:** Explore classification evaluation results.
- **🗂️ Call Dataset View:** Review available call records and their predicted categories.
- **🌐 Streamlit Deployment:** Access the application through a web browser.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Natural Language Processing | Text Preprocessing, TF-IDF |
| Machine Learning | Multinomial Naive Bayes, Scikit-learn |
| Speech Recognition | SpeechRecognition |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Visualization | Matplotlib |

## ⚙️ Project Workflow

1. **Audio Input:** Provide a supported customer call recording when using the audio-processing feature.
2. **Speech-to-Text:** Convert audio into a text transcript.
3. **Text Preprocessing:** Clean and prepare the transcript for analysis.
4. **Feature Extraction:** Convert the text into numerical features using the trained TF-IDF vectorizer.
5. **Call Classification:** Predict the appropriate call category using the trained Machine Learning model and any configured classification rules.
6. **Dashboard Analytics:** Display call statistics and classification results in Streamlit.

## 🧠 Machine Learning Approach

- **Text Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **Classification Algorithm:** Multinomial Naive Bayes
- **Model Artifacts:** Trained classifier and TF-IDF vectorizer saved using Joblib
- **Prediction Categories:** Open, Urgent, Closed

## 📂 Project Structure

```text
customer-call-center-project/
│
├── app.py
├── SmartNode_POC.ipynb
├── call_classifier.pkl
├── tfidf_vectorizer.pkl
├── Output.csv
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The `calls_audio/` directory is not included in the repository because of file-size limitations. The deployed application's audio functionality depends on how audio input is configured.

## 🚀 Getting Started

### Prerequisites

- Python
- pip
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-call-center-project.git
cd customer-call-center-project
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

### 4. Explore the Jupyter Notebook

```bash
jupyter notebook
```

Open `SmartNode_POC.ipynb` to explore the project workflow, preprocessing, model training, and analysis.

## 📊 Dataset

The project uses customer call data and text transcripts for classification and analysis.

The audio recordings are not included in this repository due to file-size limitations. `Output.csv` contains the call records used by the dashboard, subject to the data available in the repository.

## 🔮 Future Improvements

- **Speaker Diarization:** Identify and distinguish multiple speakers within a call.
- **Advanced Sentiment Analysis:** Analyze customer sentiment and satisfaction from call transcripts.
- **Improved Classification:** Evaluate advanced NLP and transformer-based models to improve classification performance.
- **Automated Call Summarization:** Generate concise summaries of customer conversations and highlight key issues.
- **Enhanced Dashboard:** Add date filters, category trends, and more detailed analytics.
- **Real-Time Processing:** Explore real-time transcription and call classification.
- **Model Evaluation:** Improve and monitor precision, recall, F1-score, and other relevant metrics.
- **Cloud Audio Processing:** Improve audio upload and transcription reliability in the deployed application.

## 👩‍💻 Author

**Khushbu Prajapati**

AI/ML and Data Science Learner

🔗 [GitHub](https://github.com/YOUR-USERNAME)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live demo.
