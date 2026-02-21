import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class CourseRecommender:
    def __init__(self, csv_file, rules_df=None):
        # -------------------------------
        # Step 1: Read CSV safely
        # -------------------------------
        try:
            chunks = pd.read_csv(csv_file, chunksize=5000, on_bad_lines='skip')
            self.df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            self.df = pd.DataFrame()  # fallback empty DataFrame

        # -------------------------------
        # Step 2: Ensure all needed columns exist
        # -------------------------------
        for col in ["Title", "Category", "Sub-Category", "Skills", "Short Intro", "Duration", "URL"]:
            if col not in self.df.columns:
                self.df[col] = ""

        # -------------------------------
        # Step 3: Combine text columns for TF-IDF
        # -------------------------------
        self.df["combined"] = (
            self.df["Title"].fillna("") + " " +
            self.df["Category"].fillna("") + " " +
            self.df["Sub-Category"].fillna("") + " " +
            self.df["Skills"].fillna("") + " " +
            self.df["Short Intro"].fillna("")
        )

        # -------------------------------
        # Step 4: TF-IDF vectorization
        # -------------------------------
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined"])

        # -------------------------------
        # Step 5: Map titles to indices
        # -------------------------------
        self.indices = pd.Series(self.df.index, index=self.df["Title"]).drop_duplicates(keep='first')

        # Store association rules DataFrame (optional)
        self.rules = rules_df

    def recommend(self, course_title, top_n=5, category=None):
        if course_title not in self.indices:
            return pd.DataFrame({"Error": ["Course not found!"]})

        if category and category != "All":
            df_filtered = self.df[self.df["Category"] == category].copy()
        else:
            df_filtered = self.df.copy()

        indices_filtered = pd.Series(df_filtered.index, index=df_filtered["Title"]).drop_duplicates(keep='first')

        if course_title not in indices_filtered:
            return pd.DataFrame({"Error": ["Course not found in this category!"]})

        idx = indices_filtered[course_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        tfidf_filtered = self.tfidf_matrix[df_filtered.index]
        sim_scores = linear_kernel(self.tfidf_matrix[idx], tfidf_filtered).flatten()

        sim_indices = sim_scores.argsort()[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:top_n]
        sim_indices = [i for i in sim_indices if i < len(df_filtered)]

        columns_to_show = ['Title', 'Category', 'Sub-Category', 'Skills', 'Short Intro', 'Duration', 'URL']
        available_cols = [col for col in columns_to_show if col in df_filtered.columns]

        return df_filtered.iloc[sim_indices][available_cols].reset_index(drop=True)

    def recommend_with_apriori(self, selected_course, top_n=5, category=None):
        if self.rules is None:
            return pd.DataFrame({"Error": ["Apriori rules not loaded!"]})

        course_skills = self.df.loc[self.df['Title'] == selected_course, 'Skills'].values
        if len(course_skills) == 0:
            return pd.DataFrame({"Error": ["Course not found!"]})
        course_skills = [s.strip() for s in course_skills[0].split(',')]

        assoc_skills = set()
        for _, row in self.rules.iterrows():
            antecedents = set(row['antecedents'])
            consequents = set(row['consequents'])
            if antecedents.intersection(course_skills):
                assoc_skills.update(consequents)

        def has_assoc_skills(skills):
            if pd.isna(skills):
                return False
            course_skill_list = [s.strip() for s in skills.split(',')]
            return bool(set(course_skill_list).intersection(assoc_skills))

        candidates = self.df[self.df['Skills'].apply(has_assoc_skills)]

        if category and category != "All":
            candidates = candidates[candidates['Category'] == category]

        return candidates.head(top_n).reset_index(drop=True)
