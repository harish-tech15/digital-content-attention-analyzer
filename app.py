
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_scaler = joblib.load("cluster_scaler.pkl")

df = pd.read_csv("digital_content_attention_data.csv")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Digital Content Attention Analyzer",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Digital Content Attention Analyzer")

st.write(
    "Analyze learner behavior, predict content completion "
    "and identify user engagement levels."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 User Behavior Input")

content_type = st.sidebar.selectbox(
    "Content Type",
    ["Video", "Article", "Course", "Tutorial", "Quiz"]
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

quiz_score = st.sidebar.number_input(
    "Quiz Score",
    min_value=0,
    max_value=100,
    value=85
)

login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=0,
    max_value=100,
    value=10
)

session_duration = st.sidebar.number_input(
    "Session Duration (minutes)",
    min_value=1,
    max_value=1000,
    value=60
)

previous_completion = st.sidebar.selectbox(
    "Previous Content Completion",
    ["Completed", "Not Completed"]
)

device_type = st.sidebar.selectbox(
    "Device Type",
    ["Mobile", "Laptop", "Tablet"]
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

watch_percentage = (
    watch_time / max(content_duration, 1)
) * 100

watch_percentage = min(watch_percentage, 100)

average_session_time = (
    session_duration /
    max(login_frequency, 1)
)

engagement_score = (
    (watch_percentage * 0.40) +
    (quiz_score * 0.25) +
    (notes_count * 5) +
    (replay_count * 3) +
    (login_frequency * 1)
)

replay_ratio = (
    replay_count /
    (watch_time + 1)
)

quiz_performance = (
    quiz_score /
    max(quiz_attempts, 1)
)


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_df = pd.DataFrame([{
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
    "Previous_Content_Completion": previous_completion,
    "Device_Type": device_type,
    "Time_of_Day": time_of_day,
    "Watch_Percentage": watch_percentage,
    "Average_Session_Time": average_session_time,
    "Engagement_Score": engagement_score,
    "Replay_Ratio": replay_ratio,
    "Quiz_Performance": quiz_performance
}])


# ============================================================
# ENCODING
# ============================================================

categorical_columns = [
    "Content_Type",
    "Previous_Content_Completion",
    "Device_Type",
    "Time_of_Day"
]

input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_columns,
    drop_first=True
)

input_encoded = input_encoded.reindex(
    columns=model_columns,
    fill_value=0
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(input_encoded)[0]

probabilities = model.predict_proba(
    input_encoded
)[0]

completion_probability = probabilities[1] * 100

completion_status = (
    "COMPLETED"
    if prediction == 1
    else "NOT COMPLETED"
)


# ============================================================
# ENGAGEMENT LEVEL
# ============================================================

if engagement_score >= 70:
    engagement_level = "HIGH"
elif engagement_score >= 45:
    engagement_level = "MEDIUM"
else:
    engagement_level = "LOW"


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Watch Percentage",
        f"{watch_percentage:.1f}%"
    )

with col2:
    st.metric(
        "Quiz Score",
        f"{quiz_score:.1f}"
    )

with col3:
    st.metric(
        "Engagement Score",
        f"{engagement_score:.1f}"
    )

with col4:
    st.metric(
        "Completion Probability",
        f"{completion_probability:.1f}%"
    )


st.divider()


# ============================================================
# PREDICTION RESULT
# ============================================================

st.subheader("🤖 Completion Prediction")

if prediction == 1:
    st.success(
        f"✅ {completion_status}"
    )
else:
    st.warning(
        f"⚠️ {completion_status}"
    )

st.progress(
    int(completion_probability)
)

st.write(
    f"**Completion Probability:** "
    f"{completion_probability:.2f}%"
)

st.write(
    f"**Engagement Level:** "
    f"{engagement_level}"
)


# ============================================================
# USER BEHAVIOR
# ============================================================

st.subheader("📈 User Behavior Analysis")

behavior_col1, behavior_col2 = st.columns(2)

with behavior_col1:

    st.write("### Learning Behavior")

    st.write(
        f"⏱️ Watch Time: **{watch_time} minutes**"
    )

    st.write(
        f"⏸️ Pause Count: **{pause_count}**"
    )

    st.write(
        f"🔁 Replay Count: **{replay_count}**"
    )

    st.write(
        f"📝 Notes Count: **{notes_count}**"
    )


with behavior_col2:

    st.write("### Learning Performance")

    st.write(
        f"📝 Quiz Score: **{quiz_score}%**"
    )

    st.write(
        f"🔐 Login Frequency: **{login_frequency}**"
    )

    st.write(
        f"💻 Device: **{device_type}**"
    )

    st.write(
        f"🕒 Time: **{time_of_day}**"
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.subheader("📊 Dataset Overview")

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

with dataset_col1:
    st.metric(
        "Total Users",
        df["User_ID"].nunique()
    )

with dataset_col2:
    completion_rate = (
        df["Completion_Status"]
        .eq("Completed")
        .mean() * 100
    )

    st.metric(
        "Completion Rate",
        f"{completion_rate:.1f}%"
    )

with dataset_col3:
    st.metric(
        "Average Quiz Score",
        f"{df['Quiz_Score'].mean():.1f}"
    )


# ============================================================
# USER SEGMENTATION
# ============================================================

st.divider()

st.subheader("👥 User Segmentation")

cluster_features = [
    "Watch_Percentage",
    "Replay_Count",
    "Quiz_Score",
    "Login_Frequency",
    "Session_Duration"
]

cluster_input = input_df[cluster_features]

cluster_input_scaled = cluster_scaler.transform(
    cluster_input
)

cluster = kmeans.predict(
    cluster_input_scaled
)[0]

cluster_means = df.groupby("Cluster")[
    cluster_features
].mean()

cluster_score = (
    cluster_means["Watch_Percentage"] * 0.40 +
    cluster_means["Quiz_Score"] * 0.25 +
    cluster_means["Login_Frequency"] * 2 +
    cluster_means["Session_Duration"] * 0.10 +
    cluster_means["Replay_Count"] * 2
)

sorted_clusters = cluster_score.sort_values().index

segment_names = {
    sorted_clusters[0]: "Passive Users",
    sorted_clusters[1]: "Regular Learners",
    sorted_clusters[2]: "Highly Engaged Learners"
}

segment = segment_names.get(
    cluster,
    "Regular Learners"
)

st.info(
    f"👤 Predicted User Segment: **{segment}**"
)


# ============================================================
# DATA VISUALIZATION
# ============================================================

st.divider()

st.subheader("📊 Dataset Visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    completion_data = (
        df["Completion_Status"]
        .value_counts()
    )

    st.bar_chart(completion_data)


with chart_col2:

    segment_data = (
        df["User_Segment"]
        .value_counts()
    )

    st.bar_chart(segment_data)


st.divider()

st.subheader("📋 Sample Dataset")

st.dataframe(
    df.head(20),
    use_container_width=True
)


st.divider()

st.caption(
    "Digital Content Attention Analyzer | "
    "Python + Pandas + NumPy + SQL + ML + K-Means + Streamlit"
)
