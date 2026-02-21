import streamlit as st
from recommender import CourseRecommender
import pandas as pd

# ---------------- Session State ----------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_to_recommender():
    st.session_state.page = 'recommender'

def go_to_thankyou():
    st.session_state.page = 'thankyou'

def go_to_home():
    st.session_state.page = 'home'

# ---------------- Load Recommender ----------------
@st.cache_resource
def load_recommender():
    rules_df = pd.read_csv('association_rules.csv')
    rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: frozenset(eval(x)))
    rules_df['consequents'] = rules_df['consequents'].apply(lambda x: frozenset(eval(x)))
    return CourseRecommender("Online_Courses.csv", rules_df)

recommender = load_recommender()


# ---------------- Home Page ----------------
# ---------------- Home Page ----------------
if st.session_state.page == 'home':
    st.markdown(
        """
        <style>
        /* Remove all default margin/padding and enforce full height */
        html, body, .stApp {
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden;
        }

        /* Full-screen background image */
        .stApp {
            background-image: url("https://i.pinimg.com/736x/6c/1b/7c/6c1b7c6dac22be46c02c60028d0c9361.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Centered white box */
        .centered-box {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            text-align: center;
        }

        .box {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 15px;
            max-width: 600px;
        }

        h1 {
            font-size: 32px;
            margin-bottom: 10px;
            color: black !important;
        }

        p {
            font-size: 18px;
            margin: 5px;
            color: black !important;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 28px;
            font-size: 18px;
            border-radius: 8px;
            margin-top: 20px;
            float: right;
        }
        </style>

        <div class="centered-box">
            <div class="box">
                <h1>Welcome to Online Course Recommender</h1>
                <p>Find the best courses tailored for you!</p>
                <p>Powered by Apriori rules</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Start Recommending"):
        go_to_recommender()


# ---------------- Recommender Page ----------------
elif st.session_state.page == 'recommender':
    st.title("Course Recommendation System")
    
    # Sidebar Filters
    with st.sidebar:
        st.header("Filters")
        # Only show categories that actually exist
        categories = ["All"] + recommender.df['Category'].dropna().unique().tolist()
        selected_category = st.selectbox("Select a Category", categories)
        num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

    # Course selection
    col1, col2 = st.columns([3, 1])
    if selected_category == "All":
        course_titles = recommender.df["Title"].tolist()
    else:
        df_filtered = recommender.df[recommender.df['Category'] == selected_category]
        if df_filtered.empty:
            st.warning("No courses available in this category yet.")
            course_titles = []
        else:
            course_titles = df_filtered["Title"].tolist()

    with col1:
        if course_titles:
            selected_course = st.selectbox("Select a Course", course_titles)
        else:
            selected_course = None
            st.info("Select another category or check back later.")

    with col2:
        st.write("")

    # Get recommendations
    if selected_course:
        recommended_courses = recommender.recommend_with_apriori(selected_course, num_recommendations, selected_category)

        if recommended_courses.empty:
            st.info("No Apriori recommendations found. Showing all courses in this category.")
            if selected_category != "All":
                recommended_courses = recommender.df[recommender.df['Category'] == selected_category]
            else:
                recommended_courses = recommender.df.copy()
            recommended_courses = recommended_courses.head(num_recommendations)

        # Display recommendations
        columns_to_show = ["Title", "Category", "Short Intro", "Duration", "Rating", "URL"]
        available_cols = [col for col in columns_to_show if col in recommended_courses.columns]

        display_df = recommended_courses[available_cols].copy()

        table_html = """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        </style>
        """

        table_html += "<table><thead><tr>"
        for col in display_df.columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr></thead><tbody>"

        for _, row in display_df.iterrows():
            table_html += "<tr>"
            for col in display_df.columns:
                value = row[col]
                if col == "URL" and pd.notna(value) and str(value).startswith("http"):
                    value = f'<a href="{value}" target="_blank">Link</a>'
                else:
                    value = str(value) if pd.notna(value) else ""
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"

        st.markdown("### Recommended Courses:")
        st.markdown(table_html, unsafe_allow_html=True)

    if st.button("Finish and Thank You"):
        go_to_thankyou()


# ---------------- Thank You Page ----------------
elif st.session_state.page == 'thankyou':
    st.markdown(
        """
        <style>
        /* Remove all default margin/padding and enforce full height */
        html, body, .stApp {
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden;
        }

        /* Full-screen background image */
        .stApp {
            background-image: url("https://i.pinimg.com/736x/79/d8/7f/79d87f65f8e7453d005c303340f32838.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }

        /* Centered white box */
        .centered-box {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            text-align: center;
        }

        .box {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 15px;
            max-width: 600px;
        }

        h1 {
            font-size: 32px;
            margin-bottom: 10px;
            color: black !important;
        }

        p {
            font-size: 18px;
            margin: 5px;
            color: black !important;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 28px;
            font-size: 18px;
            border-radius: 8px;
            margin-top: 20px;
        }
        </style>

        <div class="centered-box">
            <div class="box">
                <h1>Thank You!</h1>
                <p>Thank you for using the Online Course Recommender.</p>
                <p>Hope you found courses perfectly suited for your learning journey.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Streamlit button inside the box
    if st.button("Back to Home"):
        st.session_state.page = 'home'
