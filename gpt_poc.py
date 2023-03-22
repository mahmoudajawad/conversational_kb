import glob
import json
import logging
import os
import pathlib
import sys

import openai
from dotenv import load_dotenv
from openai.embeddings_utils import cosine_similarity

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

ARTICLES: dict[str, str] = {}
EMBEDDINGS: dict[str, list[float]] = {}
MESSAGES: list[dict[str, str]] = []
ARTICLES_MATCH: list[str] = []


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    generate_articles_embeddings()

    logging.info("Articles and embeddings are loaded. AMA!")
    ama_loop()


def ama_loop() -> None:
    while True:
        logging.info("Empty or exit to exit")
        question = input("AMA: ").strip()
        if not question or question == "exit":
            logging.info("Talk to you later!")
            sys.exit(0)
        question_embeddings = get_embedding(question)
        embeddings_scores: dict[str, float] = {}
        logging.debug("Comparing question embeddings to articles..")
        for article, article_embeddings in EMBEDDINGS.items():
            embeddings_scores[article] = cosine_similarity(
                question_embeddings, article_embeddings
            )
        embeddings_scores_sorted = sorted(
            list(embeddings_scores.items()), key=lambda score: score[1], reverse=True
        )
        match_article = embeddings_scores_sorted[0][0]
        ARTICLES_MATCH.append(match_article)

        knowledge = ARTICLES[match_article]
        if ARTICLES_MATCH:
            last_match_article = ARTICLES_MATCH[-1]
            if last_match_article != match_article:
                knowledge += f"\n{ARTICLES_MATCH[last_match_article]}"

        logging.debug("Embeddings scores: %s", embeddings_scores_sorted)
        logging.debug("Resorting to find answer from: %s", match_article)

        MESSAGES.append({"role": "user", "content": question})

        messages = [
            {
                "role": "system",
                "content": f"You are a chat bot who is supposed to help users with tourism related questions. You should politely decline answering any question not related to tourism. You should always answer in the same language as the question is. Your only source of knowledge is following text: {knowledge}",
            },
            *MESSAGES[-10:],
        ]
        answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        response_text = answer["choices"][0]["message"]["content"]
        MESSAGES.append({"role": "assistant", "content": response_text})
        print(f"> {response_text}")


def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return result["data"][0]["embedding"]


def generate_articles_embeddings() -> None:
    logging.info("Attempting to check articles...")
    dirname = os.path.dirname(__file__)
    articles = glob.glob(os.path.join(dirname, "articles", "*.txt"))
    embeddings = glob.glob(os.path.join(os.path.join(dirname, "embeddings", "*.json")))

    for article in articles:
        article_name = pathlib.Path(article).name
        if os.path.isfile(f"embeddings/{article_name[:-4]}.json"):
            logging.info(
                "Article '%s' embeddings already generated. Skipping..", article_name
            )

            with open(article, encoding="UTF-8") as f:
                ARTICLES[article_name] = f.read()

            with open(f"embeddings/{article_name[:-4]}.json", encoding="UTF-8") as f:
                EMBEDDINGS[article_name] = json.loads(f.read())

            continue

        logging.info(
            "Article '%s' embeddings not generated. Generating..", article_name
        )
        with open(article, encoding="UTF-8") as f:
            article_content = f.read()
            ARTICLES[article_name] = article_content
            article_embeddings = get_embedding(article_content)
            EMBEDDINGS[article_name] = article_embeddings

        with open(f"embeddings/{article_name[:-4]}.json", "w", encoding="UTF-8") as f:
            f.write(json.dumps(article_embeddings))

        logging.info("  - Done")
