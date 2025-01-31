"""
Project: Reddit Rhetoric: Sentiment and Topic Insights from r/AskTrumpSupporters
Title: visualizations_v1.py
Author: Jacob Collier
Class: CSCI 1051a - Deep Learning
"""

"""
IMPORTS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

"""
LOADING DATASET
"""
# LOAD THE DATASET
file_path = "Path/To/Your/AnalysisResults.xlsx"
xls = pd.ExcelFile(file_path)
df = xls.parse("Cleaned Posts")  # Load the "Cleaned Posts" sheet

# MAKE SURE 'comment_body' EXISTS
if "comment_body" not in df.columns:
    raise ValueError("Dataset must have a 'comment_body' column.")

"""
CLEANING
"""
# REMOVE EMPTY / TOO SHORT COMMENTS
df = df[df["comment_body"].notna()]
df = df[df["comment_body"].str.len() > 5]

stop_words = set(stopwords.words('english'))

# SAME IMPLEMENTATION AS IN reddit_rhetoric_v1.py
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text

df["cleaned_comment"] = df["comment_body"].apply(clean_text)
documents = df["cleaned_comment"].tolist()

"""
EMBEDDING FOR BERTopic
"""
# LOAD SentenceTransformer MODEL
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FIT BERTopic MODEL
topic_model = BERTopic(embedding_model=embedding_model, nr_topics=10, verbose=True)
topics, probs = topic_model.fit_transform(documents)

# EXTRACT TOPIC EMBEDDINGS
topic_embeddings = topic_model.topic_embeddings_

if topic_embeddings is None:
    raise ValueError("BERTopic did not generate topic embeddings. Possibly not enough data to form distinct topics.")

topic_embeddings = np.array(topic_embeddings)

num_topics_found = topic_embeddings.shape[0]  # see num_topics

"""
TOPIC PCA PLOT

Disclaimer: Some of this code was written with the aid of LLMs.
"""
# SKIP PCA IF DATASET IS SMALL
if num_topics_found < 2:
    print(f"Only {num_topics_found} topic(s) found. Skipping 2D PCA.")
# PROCEED AS NORMAL IF DATASET HAS MANY TOPICS
else:
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(topic_embeddings)

    # CONVERT TO DATAFRAME
    pca_df = pd.DataFrame(reduced_embeddings, columns=["PC1", "PC2"])

    # GRAB TOPIC INDO
    topic_info = topic_model.get_topic_info()

    # ALIGN ROWS WITH CORRESPONDING TOPIC NUMBER 
    pca_df["topic"] = topic_info["Topic"][:len(pca_df)].values
    pca_df["topic_name"] = pca_df["topic"].map(topic_info.set_index("Topic")["Name"])

    # PLOT QUIVER
    plt.figure(figsize=(10, 6))
    unique_topic_names = pca_df["topic_name"].unique()
    legend_handles = {}
    num_topics = len(unique_topic_names)

    # CREATE A COLORMAP OF THE NUM OF TOPICS
    cmap = plt.cm.get_cmap('Spectral', num_topics)
    color_palette = {topic: cmap(i / num_topics) for i, topic in enumerate(unique_topic_names)}

    for topic_name in unique_topic_names:
        topic_data = pca_df[pca_df["topic_name"] == topic_name]
        color = color_palette[topic_name]
        plt.quiver(
            np.zeros(len(topic_data)),
            np.zeros(len(topic_data)),
            topic_data["PC1"],
            topic_data["PC2"],
            angles='xy', scale_units='xy', scale=1, color=color
        )
        legend_handles[topic_name] = mpatches.Patch(color=color, label=topic_name)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Vector Plot of BERTopic Topics")
    plt.legend(handles=legend_handles.values(), title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlim(reduced_embeddings[:, 0].min() - 0.05, reduced_embeddings[:, 0].max() + 0.05)
    plt.ylim(reduced_embeddings[:, 1].min() - 0.05, reduced_embeddings[:, 1].max() + 0.05)
    plt.show()

"""
SENTIMENT HEATMAPS

Disclaimer: Some of this code was written with the aid of LLMs.
"""
df_master = xls.parse("Master Sheet")

print("Master Sheet columns:", df_master.columns.tolist())

# TOPIC SENTIMENT HEATMAPS
unique_flairs = df_master["author_flair"].dropna().unique()
unique_topics = df_master["topic"].dropna().unique()

unique_sentiments_x = set()
for label_list_str in df_master["labels_x"].dropna():
    try:
        labels_list = eval(label_list_str)
        unique_sentiments_x.update(labels_list)
    except:
        pass

unique_sentiments_x = sorted(unique_sentiments_x)


for flair in unique_flairs[:5]:  # build a heatmap for each flair
    flair_df = df_master[df_master["author_flair"] == flair]

    topic_sentiment_matrix = pd.DataFrame(0.0, index=unique_topics, columns=unique_sentiments_x)
    count_matrix = pd.DataFrame(0, index=unique_topics, columns=unique_sentiments_x)

    for _, row in flair_df.iterrows():
        topic = row["topic"]
        labels_str = row["labels_x"]
        scores_str = row["scores_x"]

        if pd.notna(topic) and pd.notna(labels_str) and pd.notna(scores_str):
            try:
                labels_list = eval(labels_str)  # e.g. ["agreeable","skeptical"]
                scores_list = eval(scores_str)  # e.g. [0.53, 0.21]
                for i, lab in enumerate(labels_list):
                    if lab in topic_sentiment_matrix.columns and topic in topic_sentiment_matrix.index:
                        topic_sentiment_matrix.loc[topic, lab] += scores_list[i]
                        count_matrix.loc[topic, lab] += 1
            except:
                pass

    # CONVERT SUMS TO AVGS
    with np.errstate(divide='ignore', invalid='ignore'):
        topic_sentiment_matrix = topic_sentiment_matrix / count_matrix
    topic_sentiment_matrix = topic_sentiment_matrix.fillna(0)

    # PLOT HEATMAP
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        topic_sentiment_matrix.astype(float),
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("Sentiment Labels")
    plt.ylabel("Topics")
    plt.title(f"Heatmap of Topic Sentiments (Flair = {flair})")
    plt.show()

# PARTICIPANT SENTIMENT HEATMAPS
all_flairs = df_master["author_flair"].dropna().unique()

unique_sentiments_y = set()
for label_list_str in df_master["labels_y"].dropna():
    try:
        labels_list = eval(label_list_str)
        unique_sentiments_y.update(labels_list)
    except:
        pass
unique_sentiments_y = sorted(unique_sentiments_y)

comment_id_to_flair = df_master.set_index("comment_id")["author_flair"].to_dict()

author_to_flair = df_master.set_index("comment_author")["author_flair"].dropna().to_dict()

# BUILD A MATRIX FOR EACH SENTIMENT LABEL
flair_sentiment_matrices = {
    sentiment: pd.DataFrame(0.0, index=all_flairs, columns=all_flairs)
    for sentiment in unique_sentiments_y
}
flair_sentiment_counts = {
    sentiment: pd.DataFrame(0, index=all_flairs, columns=all_flairs)
    for sentiment in unique_sentiments_y
}

# ITERATE OVER EACH ROW THAT HAS PARTICIPANT-SENTIMEMNT COLUMNS
for _, row in df_master.iterrows():
    c_id = row["comment_id"]
    target = row["target_author"]
    labels_str = row["labels_y"]
    scores_str = row["scores_y"]

    # LOOK UP COMMENTER FLAIR AND TARGET FLAIR
    if (c_id in comment_id_to_flair) and (target in author_to_flair):
        c_flair = comment_id_to_flair[c_id]
        t_flair = author_to_flair[target]

        if pd.notna(labels_str) and pd.notna(scores_str):
            try:
                labels_list = eval(labels_str)
                scores_list = eval(scores_str)
                for i, lab in enumerate(labels_list):
                    flair_sentiment_matrices[lab].loc[c_flair, t_flair] += scores_list[i]
                    flair_sentiment_counts[lab].loc[c_flair, t_flair] += 1
            except:
                pass

# CONVERT SUMS TO AVGS
for sentiment, matrix in flair_sentiment_matrices.items():
    count_mat = flair_sentiment_counts[sentiment].replace(0, np.nan)
    matrix = matrix / count_mat
    matrix = matrix.fillna(0)
    flair_sentiment_matrices[sentiment] = matrix

# PLOT PARTICIPANT SENTIMENT MATRICES
for sentiment, matrix in flair_sentiment_matrices.items():
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        matrix.astype(float),
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("Target Flair Types")
    plt.ylabel("Commenter Flair Types")
    plt.title(f"'{sentiment}' Sentiment in Participant Interactions")
    plt.show()