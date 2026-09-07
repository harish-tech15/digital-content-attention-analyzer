import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digital Content Attention Analyzer",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #ddd;
    text-align: center;
    margin-top: 20px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# FILE LOADER
# =========================================================

def load_file(filename):

    file_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(file_path):

        st.error(f"❌ Required file not found: {filename}")

        st.info(
            "Please make sure the required files are uploaded "
            "to the same GitHub repository folder as app.py."
        )

        st.stop()

    return file_path


# =========================================================
# LOAD MODEL FILES
# =========================================================

MODEL_PATH = load_file("random_forest_model.pkl")
COLUMNS_PATH = load_file("model_columns.pkl")
KMEANS_PATH = load_file("kmeans_model.pkl")
DATA_PATH = load_file("digital_content_attention_data.csv")


# =========================================================
# LOAD MODELS
# =========================================================

try:

    model = joblib.load(MODEL_PATH)

    model_columns = joblib.load(COLUMNS_PATH)

    kmeans = joblib.load(KMEANS_PATH)

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error("❌ Error loading project files.")

    st.code(str(e))

    st.stop()


# =========================================================
# CREATE K-MEANS SCALER
# =========================================================

cluster_features = [
    "Watch_Percentage",
    "Replay_Count",
    "Quiz_Score",
    "Login_Frequency",
    "Session_Duration"
]


# ---------------------------------------------------------
# If cluster_scaler.pkl exists → use it
# Otherwise → recreate scaler from dataset
# ---------------------------------------------------------

CLUSTER_SCALER_PATH = os.path.join(
    BASE_DIR,
    "cluster_scaler.pkl"
)


if os.path.exists(CLUSTER_SCALER_PATH):

    cluster_scaler = joblib.load(
        CLUSTER_SCALER_PATH
    )

else:

    cluster_scaler = StandardScaler()

    cluster_scaler.fit(
        df[cluster_features]
    )


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">📊 Digital Content Attention Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze learner engagement and predict content completion'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ User Input")

st.sidebar.markdown(
    "Enter learner/content details below."
)


# =========================================================
# USER INPUTS
# =========================================================

content_type = st.sidebar.selectbox(
    "Content Type",
    [
        "Video",
        "Article",
        "Course",
        "Tutorial",
        "Quiz"
    ]
)


content_duration = st.sidebar.number_input(
    "Content Duration (minutes)",
    min_value=1,
    max_value=500,
    value=40
)


watch_time = st.sidebar.number_input(
    "Watch Time (minutes)",
    min_value=0,
    max_value=500,
    value=36
)


pause_count = st.sidebar.number_input(
    "Pause Count",
    min_value=0,
    max_value=50,
    value=2
)


replay_count = st.sidebar.number_input(
    "Replay Count",
    min_value=0,
    max_value=50,
    value=3
)


notes_count = st.sidebar.number_input(
    "Notes Count",
    min_value=0,
    max_value=50,
    value=3
)


quiz_attempts = st.sidebar.number_input(
    "Quiz Attempts",
    min_value=0,
    max_value=20,
    value=1
)


quiz_score = st.sidebar.slider(
    "Quiz Score",
    min_value=0,
    max_value=100,
    value=85
)


login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=1,
    max_value=100,
    value=10
)


session_duration = st.sidebar.number_input(
    "Session Duration (minutes)",
    min_value=1,
    max_value=500,
    value=60
)


previous_completion = st.sidebar.selectbox(
    "Previous Content Completion",
    [
        "Completed",
        "Not Completed"
    ]
)


device_type = st.sidebar.selectbox(
    "Device Type",
    [
        "Mobile",
        "Laptop",
        "Tablet"
    ]
)


time_of_day = st.sidebar.selectbox(
    "Time of Day",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.sidebar.button(
    "🔍 Analyze Content",
    use_container_width=True
)


# =========================================================
# DEFAULT VALUES
# =========================================================

if watch_time > content_duration:

    watch_time = content_duration


# =========================================================
# FEATURE ENGINEERING FUNCTION
# =========================================================

def create_features():

    watch_percentage = (
        watch_time / content_duration
    ) * 100

    watch_percentage = np.clip(
        watch_percentage,
        0,
        100
    )


    average_session_time = (
        session_duration /
        max(login_frequency, 1)
    )


    engagement_score = (

        (watch_percentage * 0.40)

        +

        (quiz_score * 0.25)

        +

        (notes_count * 5)

        +

        (replay_count * 3)

        +

        (login_frequency * 1)
    )


    # -----------------------------------------------------
    # Normalize engagement score using dataset range
    # -----------------------------------------------------

    dataset_engagement_min = df["Engagement_Score"].min()

    dataset_engagement_max = df["Engagement_Score"].max()


    if dataset_engagement_max != dataset_engagement_min:

        engagement_score = (

            (
                engagement_score
                -
                dataset_engagement_min
            )
            /
            (
                dataset_engagement_max
                -
                dataset_engagement_min
            )
        ) * 100

    else:

        engagement_score = 50


    replay_ratio = (
        replay_count /
        (watch_time + 1)
    )


    quiz_performance = (
        quiz_score /
        max(quiz_attempts, 1)
    )


    input_data = {

        "Content_Type": content_type,

        "Content_Duration": content_duration,

        "Watch_Time": watch_time,

        "Pause_Count": pause_count,

        "Replay_Count": replay_count,

        "Notes_Count": notes_count,

        "Quiz_Attempts": quiz_attempts,

        "Quiz_Score": quiz_score,

        "Login_Frequency": login_frequency,

        "Session_Duration": session_duration,

        "Previous_Content_Completion":
            previous_completion,

        "Device_Type":
            device_type,

        "Time_of_Day":
            time_of_day,

        "Watch_Percentage":
            watch_percentage,

        "Average_Session_Time":
            average_session_time,

        "Engagement_Score":
            engagement_score,

        "Replay_Ratio":
            replay_ratio,

        "Quiz_Performance":
            quiz_performance
    }


    return pd.DataFrame([input_data])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_completion(input_df):

    prediction_df = input_df.copy()


    categorical_columns = [

        "Content_Type",

        "Previous_Content_Completion",

        "Device_Type",

        "Time_of_Day"
    ]


    prediction_df = pd.get_dummies(
        prediction_df,
        columns=categorical_columns,
        drop_first=True
    )


    # -----------------------------------------------------
    # Match training columns
    # -----------------------------------------------------

    prediction_df = prediction_df.reindex(
        columns=model_columns,
        fill_value=0
    )


    prediction = model.predict(
        prediction_df
    )[0]


    # -----------------------------------------------------
    # Prediction probability
    # -----------------------------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            prediction_df
        )[0]

        completion_probability = probability[1] * 100

    else:

        completion_probability = (
            100 if prediction == 1
            else 0
        )


    return prediction, completion_probability


# =========================================================
# ENGAGEMENT LEVEL
# =========================================================

def get_engagement_level(score):

    if score >= 70:

        return "HIGH"

    elif score >= 45:

        return "MEDIUM"

    else:

        return "LOW"


# =========================================================
# USER SEGMENT FUNCTION
# =========================================================

def predict_user_segment(input_df):

    cluster_input = input_df[
        cluster_features
    ].copy()


    cluster_input_scaled = (
        cluster_scaler.transform(
            cluster_input
        )
    )


    cluster = kmeans.predict(
        cluster_input_scaled
    )[0]


    # -----------------------------------------------------
    # Find cluster characteristics
    # -----------------------------------------------------

    cluster_means = (
        df.groupby("Cluster")[cluster_features]
        .mean()
    )


    # -----------------------------------------------------
    # Calculate engagement ranking
    # -----------------------------------------------------

    cluster_means["Overall_Engagement"] = (

        cluster_means["Watch_Percentage"] * 0.40

        +

        cluster_means["Quiz_Score"] * 0.25

        +

        cluster_means["Login_Frequency"] * 2

        +

        cluster_means["Session_Duration"] * 0.10

        +

        cluster_means["Replay_Count"] * 2
    )


    sorted_clusters = (
        cluster_means[
            "Overall_Engagement"
        ]
        .sort_values()
        .index
        .tolist()
    )


    # -----------------------------------------------------
    # Automatically assign names
    # -----------------------------------------------------

    segment_names = {}


    if len(sorted_clusters) >= 3:

        segment_names[sorted_clusters[0]] = (
            "Passive Users"
        )

        segment_names[sorted_clusters[1]] = (
            "Regular Learners"
        )

        segment_names[sorted_clusters[2]] = (
            "Highly Engaged Learners"
        )

    else:

        segment_names = {
            cluster: f"Cluster {cluster}"
            for cluster in sorted_clusters
        }


    segment = segment_names.get(
        cluster,
        f"Cluster {cluster}"
    )


    return cluster, segment


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Create input features
    # -----------------------------------------------------

    input_df = create_features()


    # =====================================================
    # COMPLETION PREDICTION
    # =====================================================

    prediction, probability = (
        predict_completion(
            input_df
        )
    )


    # =====================================================
    # ENGAGEMENT
    # =====================================================

    engagement_score = (
        input_df[
            "Engagement_Score"
        ].iloc[0]
    )


    engagement_level = (
        get_engagement_level(
            engagement_score
        )
    )


    # =====================================================
    # USER SEGMENT
    # =====================================================

    cluster, segment = (
        predict_user_segment(
            input_df
        )
    )


    # =====================================================
    # MAIN DASHBOARD
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Analysis Result'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Watch Percentage",
            f"{input_df['Watch_Percentage'].iloc[0]:.1f}%"
        )


    with col2:

        st.metric(
            "Quiz Score",
            f"{quiz_score}%"
        )


    with col3:

        st.metric(
            "Engagement",
            engagement_level
        )


    with col4:

        st.metric(
            "Completion Probability",
            f"{probability:.1f}%"
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Completion Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.success(
            f"✅ HIGH LIKELIHOOD OF COMPLETION — "
            f"{probability:.1f}%"
        )

    else:

        st.warning(
            f"⚠️ LOW LIKELIHOOD OF COMPLETION — "
            f"{probability:.1f}%"
        )


    # =====================================================
    # ENGAGEMENT RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🔥 Engagement Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if engagement_level == "HIGH":

        st.success(
            f"🔥 HIGH ENGAGEMENT — Score: "
            f"{engagement_score:.2f}"
        )

    elif engagement_level == "MEDIUM":

        st.warning(
            f"🟡 MEDIUM ENGAGEMENT — Score: "
            f"{engagement_score:.2f}"
        )

    else:

        st.error(
            f"🔴 LOW ENGAGEMENT — Score: "
            f"{engagement_score:.2f}"
        )


    # =====================================================
    # USER SEGMENT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '👥 User Behavior Segmentation'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        f"Cluster: {cluster}  |  "
        f"User Segment: **{segment}**"
    )


    # =====================================================
    # FEATURE SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📋 User Feature Summary'
        '</div>',
        unsafe_allow_html=True
    )


    display_df = pd.DataFrame({

        "Feature": [

            "Content Type",

            "Content Duration",

            "Watch Time",

            "Watch Percentage",

            "Pause Count",

            "Replay Count",

            "Notes Count",

            "Quiz Score",

            "Login Frequency",

            "Session Duration",

            "Previous Completion",

            "Device Type",

            "Time of Day",

            "Average Session Time",

            "Engagement Score",

            "Replay Ratio",

            "Quiz Performance"
        ],


        "Value": [

            content_type,

            f"{content_duration} min",

            f"{watch_time} min",

            f"{input_df['Watch_Percentage'].iloc[0]:.2f}%",

            pause_count,

            replay_count,

            notes_count,

            f"{quiz_score}%",

            login_frequency,

            f"{session_duration} min",

            previous_completion,

            device_type,

            time_of_day,

            f"{input_df['Average_Session_Time'].iloc[0]:.2f}",

            f"{input_df['Engagement_Score'].iloc[0]:.2f}",

            f"{input_df['Replay_Ratio'].iloc[0]:.3f}",

            f"{input_df['Quiz_Performance'].iloc[0]:.2f}"
        ]

    })


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # VISUAL ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Dataset Insights'
        '</div>',
        unsafe_allow_html=True
    )


    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        st.subheader(
            "Completion Status"
        )


        completion_counts = (
            df["Completion_Status"]
            .value_counts()
        )


        st.bar_chart(
            completion_counts
        )


    with chart_col2:

        st.subheader(
            "Content Type Distribution"
        )


        content_counts = (
            df["Content_Type"]
            .value_counts()
        )


        st.bar_chart(
            content_counts
        )


    # =====================================================
    # WATCH TIME DISTRIBUTION
    # =====================================================

    st.subheader(
        "⏱️ Watch Time Distribution"
    )


    st.line_chart(
        df[
            [
                "Watch_Time"
            ]
        ].head(100)
    )


    # =====================================================
    # QUIZ SCORE DISTRIBUTION
    # =====================================================

    st.subheader(
        "📝 Quiz Score Distribution"
    )


    st.line_chart(
        df[
            [
                "Quiz_Score"
            ]
        ].head(100)
    )


    # =====================================================
    # USER SEGMENT DISTRIBUTION
    # =====================================================

    if "User_Segment" in df.columns:

        st.subheader(
            "👥 User Segment Distribution"
        )


        segment_counts = (
            df[
                "User_Segment"
            ]
            .value_counts()
        )


        st.bar_chart(
            segment_counts
        )


    # =====================================================
    # CORRELATION
    # =====================================================

    st.subheader(
        "🔗 Feature Correlation"
    )


    correlation_columns = [

        "Watch_Percentage",

        "Replay_Count",

        "Quiz_Score",

        "Login_Frequency",

        "Session_Duration",

        "Engagement_Score"
    ]


    correlation = (
        df[
            correlation_columns
        ]
        .corr()
    )


    st.dataframe(
        correlation.round(2),
        use_container_width=True
    )


# =========================================================
# INITIAL SCREEN
# =========================================================

else:

    st.markdown(
        """
        ### 👋 Welcome!

        Use the **sidebar** to enter learner behavior details.

        Then click **🔍 Analyze Content** to get:

        - 🎯 Completion Probability
        - 📊 Engagement Level
        - 👥 K-Means User Segment
        - 📈 Feature Analysis
        - 📊 Dataset Insights
        - 🔗 Feature Correlation

        """
    )


    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🚀 Project Workflow'
        '</div>',
        unsafe_allow_html=True
    )


    workflow = pd.DataFrame({

        "Stage": [

            "1",

            "2",

            "3",

            "4",

            "5",

            "6"
        ],


        "Process": [

            "SQL Database",

            "Python Data Processing",

            "Pandas + NumPy",

            "Machine Learning",

            "Completion Prediction",

            "K-Means Segmentation"
        ]

    })


    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Digital Content Attention Analyzer | "
    "Python • Pandas • NumPy • SQL • Machine Learning • K-Means • Streamlit"
)
