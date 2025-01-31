"""
Project: Reddit Rhetoric: Sentiment and Topic Insights from r/AskTrumpSupporters
Title: reddit_rhetoric_v1.py
Author: Jacob Collier
Class: CSCI 1051a - Deep Learning
"""

"""
IMPORTS
"""
import argparse # to handle arguments
import praw # 'Python Reddit API Wrapper'
import logging # for a pesky logging error
import pandas as pd # for dataframes
import datetime # for getting the date... and time
from datetime import datetime # ^
import re
import nltk
from nltk.corpus import stopwords # for removing common words (like 'an', 'a', 'the', etc)
from bertopic import BERTopic # BERTopic
from sentence_transformers import SentenceTransformer # leftover import (no longer used here)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline # for models
from collections import defaultdict # for dictionaries

"""
SPECIFYING AN OUTPUT PATH
"""
parser = argparse.ArgumentParser(description="Create some dataset from a zero-shot pipeline.")
parser.add_argument('--output_path', type=str, required=True, help='Path to save the data.')
args = parser.parse_args()
output_path = args.output_path

"""
DATA GATHERING
"""
# INITIALIZING PRAW (W/ SECRET API CREDENTIALS)
reddit = praw.Reddit(
    client_id="your_client_ID",
    client_secret="your_client_secret",
    user_agent="YourAgentName/0.1 (by u/YourUsername)"
)

# NIX THE ANNOYING LOG
logging.getLogger("praw").setLevel(logging.ERROR)

# INITIALIZE THE SUBREDDIT
subreddit_name = 'AskTrumpSupporters'
subreddit = reddit.subreddit(subreddit_name)

# INITIALIZE POST DATA
posts_data = []
limit_posts = 100  # adjust as needed

# ITERATE THROUGH SUBMISSIONS
for submission in subreddit.new(limit=limit_posts):
    # GRAB ALL COMMENTS
    submission.comments.replace_more(limit=None)
    all_comments = submission.comments.list()

    # GRAB SUBMISSION-LEVEL INFO
    post_info = {
        'submission_id': submission.id,
        'title': submission.title,
        'selftext': submission.selftext,
        'url': submission.url,
        'score': submission.score,
        'num_comments': submission.num_comments,
        'submission_flair': submission.link_flair_text if submission.link_flair_text else "N/A",
        'submission_timestamp': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
    }

    # GRAB COMMENT-LEVEL INFO FOR THIS SUBMISSION
    for comment in all_comments:
        comment_info = {
            'comment_id': comment.id,
            'comment_body': comment.body,
            'comment_score': comment.score,
            'comment_author': str(comment.author) if comment.author else "N/A",
            'submission_id': submission.id,
            'author_flair': comment.author_flair_text if comment.author_flair_text else "N/A",
            'parent_type': 'Comment' if isinstance(comment.parent(), praw.models.Comment) else 'Submission',
            'parent_id': comment.parent().id,
            # If parent is a comment, store parent body; if parent is submission, store submission selftext
            'parent_body': comment.parent().body if isinstance(comment.parent(), praw.models.Comment) else submission.selftext
        }
        combined_info = {**post_info, **comment_info}
        posts_data.append(combined_info)

# CONVERT TO DATAFRAME
df_posts = pd.DataFrame(posts_data)

# FILTER OUT AUTOMODERATOR, REMOVED, DELETED
df_posts = df_posts[df_posts['comment_author'] != "AutoModerator"].reset_index(drop=True)
df_posts = df_posts[df_posts['comment_body'] != "[removed]"].reset_index(drop=True)
df_posts = df_posts[df_posts['parent_body'] != "[removed]"].reset_index(drop=True)
df_posts = df_posts[df_posts['comment_body'] != "[deleted]"].reset_index(drop=True)
df_posts = df_posts[df_posts['parent_body'] != "[deleted]"].reset_index(drop=True)

print("Fetched", len(df_posts), "post-comment pairs.")
df_posts.head()

"""
TOPIC EXTRACTION
"""
stop_words = set(stopwords.words('english')) # grab stopwords

def clean_text(text):
    """
    REMOVES: 
        1) LOWERCASE 
        2) NON-STRINGS 
        3) URLS
        4) PUNCTUATION & NON-ALPHABETIC (emojis, etc) 
        5) SPLITS THE TEXT 
        6) REMOVES STOPWORDS
        7) REJOINS THE TEXT
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

def extract_topics_bertopic(df, max_topics=10):
    """
    TAKES IN A DATAFRAME OF COMMENTS AND RETURNS (bertopic_df, topic_model) WHERE:
        'bertopic_df' HAS [comment_id, bertopic_id, bertopic_name]
        topic_model IS THE BERTopic MODEL

    Disclaimer: Some of this code was written with the aid of LLMs.
    """
    # GATHER AND CLEAN COMMENTS FOR TOPIC ANALYSIS 
    comments = []
    comment_ids = []

    for _, row in df.iterrows():
        body = row['comment_body']
        if isinstance(body, str) and body.strip():
            cleaned = clean_text(body.strip())
            if cleaned:
                comments.append(cleaned)
                comment_ids.append(row['comment_id'])

    # INITIALIZING AND FITTING BERTopic MODEL
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topics, _ = topic_model.fit_transform(comments)

    # REDUCE THE NUMBER OF TOPICS (this is flexible, so you can experiment)
    topic_model.reduce_topics(comments, nr_topics=max_topics)

    # BUILD DATAFRAME MATCHING comment_id TO topic & topic_name
    topic_info = topic_model.get_topic_info()
    results = []
    for i, t in enumerate(topics):
        if t == -1:
            topic_name = "Miscellaneous/Outliers"
        else:
            # GRAB TOPIC NAME (rather than '0', '1', '2', etc, since we're looking for meaning)
            row_topic = topic_info[topic_info["Topic"] == t]
            if len(row_topic) > 0:
                topic_name = row_topic.iloc[0]["Name"]
            else:
                topic_name = f"Topic_{t}"

        results.append({
            "comment_id": comment_ids[i],
            "bertopic_id": t,
            "bertopic_name": topic_name
        })

    bertopic_df = pd.DataFrame(results)
    return bertopic_df, topic_model

"""
TOPIC SENTIMENT ANALYSIS (w/ a toggle for 'flair' topics or 'bertopic' topics)
"""
def analyze_topic_sentiments_toggle(df, topic_categories, toggle="flair", bertopic_df=None):
    """
    IF toggle == "flair":
        FOR EACH COMMENT, WE USE THE POST'S 'submission_flair' AS THE TOPIC AND CLASSIFY THE TEXT:
            "Topic: <flair>. Text: <comment_body>"
    IF toggle == "bertopic":
        FOR EACH COMMENT, WE USE THE 'bertopic_name' FROM bertopic_df AND CLASSIFY THE TEXT:
            "Topic: <bertopic_name>. Text: <comment_body>"
        FOR THIS TOGGLE, YOU MUST PASS IN bertopic_df FROM extract_topics_bertopic().
    THIS FUNCTION RETURNS A LIST OF DICTS W/ CLASSIFICATION RESULTS. LABELS WILL DIFFER IN ORDER, BUT ARE CONSISTENT WITHIN ROWS.

    Disclaimer: Some of this code was written with the aid of LLMs.
    """
    # LOAD 'bart-large-mnli'
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    results = []
    # USE FLAIR AS THE TOPIC
    if toggle == "flair":
        for _, row in df.iterrows():
            comment_body = row["comment_body"]
            if isinstance(comment_body, str) and comment_body.strip():
                prompt = f"Topic: {row['submission_flair']}. Text: {comment_body}"
                classification = classifier(
                    [prompt],
                    candidate_labels=topic_categories,
                    multi_label=True
                )[0]  # pipeline returns a list with one item
                results.append({
                    "submission_id": row["submission_id"],
                    "comment_id": row["comment_id"],
                    "topic": row["submission_flair"],
                    "labels": classification["labels"],
                    "scores": classification["scores"]
                })
    # USE BERTopic GENERATED TOPICS AS THE TOPIC
    elif toggle == "bertopic":
        # MAKE SURE bertopic_df IS PASSED 
        if bertopic_df is None:
            raise ValueError("You must provide `bertopic_df` when toggle='bertopic'.")

        merged = df.merge(bertopic_df, on="comment_id", how="inner")
        for _, row in merged.iterrows():
            comment_body = row["comment_body"]
            if isinstance(comment_body, str) and comment_body.strip():
                prompt = f"Topic: {row['bertopic_name']}. Text: {comment_body}"
                classification = classifier(
                    [prompt],
                    candidate_labels=topic_categories,
                    multi_label=True
                )[0]
                results.append({
                    "submission_id": row["submission_id"],
                    "comment_id": row["comment_id"],
                    "topic": row["bertopic_name"],
                    "labels": classification["labels"],
                    "scores": classification["scores"]
                })
    else:
        raise ValueError("Invalid toggle option. Use 'flair' or 'bertopic'.")

    return results

"""
PARTICIPANT SENTIMENT ANALYSIS
"""
def analyze_participant_sentiments(df, interaction_categories):
    """
    ALL COMMENTS ADDRESSED TO THEIR PARENT COMMENT/SUBMISSION, SO:
        THIS FUNCTION ANALYZES THE SENTIMENT EACH COMMENT EXPRESSES TOWARD ITS PARENT COMMENT/COMMENT'S AUTHOR.
    THE FUNCTION, LIKE analyze_topic_sentiments_toggle(), IS ROBUST TO PERSONALIZED SENTIMENT CATEGORIES.
    """
    sentiment_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # MERGE DF
    df_merged = df.merge(
        df[['comment_id', 'comment_author']],
        left_on='parent_id',
        right_on='comment_id',
        how='left',
        suffixes=('', '_parent')
    )

    results = []
    for idx, row in df_merged.iterrows():
        # ONLY RUN IF THE PARENT IS A COMMENT
        if row['parent_type'] == 'Comment' and row['comment_body'].strip():
            text = row['comment_body']
            target_author = row['comment_author_parent']
            classification = sentiment_pipeline(
                [text],
                candidate_labels=interaction_categories,
                multi_label=True
            )[0]
            results.append({
                "submission_id": row['submission_id'],
                "comment_id": row['comment_id'],
                "target_author": target_author,
                "labels": classification['labels'],
                "scores": classification['scores']
            })
    return results

"""
SAVE RESULTS TO EXCEL
"""
def save_results_to_excel(df_posts, topic_sentiments, participant_sentiments, base_file_name):
    """
    SAVES RESULTS TO EXCEL

    Disclaimer: Some of this code was written with the aid of LLMs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{base_file_name}_{timestamp}.xlsx"

    # CONVERT LISTS OF DICTS TO DATAFRAMES
    topic_sentiments_df = pd.DataFrame(topic_sentiments)
    participant_sentiments_df = pd.DataFrame(participant_sentiments)

    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # FULL CLEANED POSTS
        df_posts.to_excel(writer, sheet_name="Cleaned Posts", index=False)

        # TOPIC SENTIMENT RESULTS
        topic_sentiments_df.to_excel(writer, sheet_name="Topic Sentiments", index=False)

        # PARTICIPANT SENTIMENT RESULTS
        participant_sentiments_df.to_excel(writer, sheet_name="Participant Sentiments", index=False)

        # 4) MASTER SHEET (merges the OG sheet w/ results)
        master_df = pd.merge(df_posts, topic_sentiments_df, on=["submission_id", "comment_id"], how="left")
        master_df = pd.merge(master_df, participant_sentiments_df, on=["submission_id", "comment_id"], how="left")
        master_df.to_excel(writer, sheet_name="Master Sheet", index=False)

    print(f"Data saved to Excel file: {file_name}")

"""
MAIN
"""
# TOPIC SENTIMENT
bertopic_results_df, my_topic_model = extract_topics_bertopic(df_posts, max_topics=10)
topic_categories = ["agreeable", "neutral", "skeptical", "antagonistic"]

flair_sentiment = analyze_topic_sentiments_toggle(
    df_posts,
    topic_categories=topic_categories,
    toggle="flair"
)

bertopic_sentiment = analyze_topic_sentiments_toggle(
    df_posts,
    topic_categories=topic_categories,
    toggle="bertopic",
    bertopic_df=bertopic_results_df
)

# PARTICIPANT SENTIMENT
interaction_categories = ["respectful", "critical", "sarcastic", "demeaning", "friendly", "constructive"]
participant_sentiments = analyze_participant_sentiments(df_posts, interaction_categories)

# SAVE TO EXCEL
save_results_to_excel(
    df_posts,
    flair_sentiment,
    participant_sentiments,
    f"{output_path}/Reddit_Analysis_Results_flair"
)

save_results_to_excel(
    df_posts,
    bertopic_sentiment,
    participant_sentiments,
    f"{output_path}/Reddit_Analysis_Results_bertopic"
)