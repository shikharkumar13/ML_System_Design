# Machine Learning Systems

A practitioner-focused article series on what it actually takes to build, deploy, and maintain machine learning systems in production — based on *Designing Machine Learning Systems* by Chip Huyen (O'Reilly, 2022).

## Why This Series Exists

Most ML education stops at the model: train it, evaluate it on a clean test set, report an accuracy number. But in production, the model is a small fraction of the system — often less than 10% of the total engineering effort.

This series is for learners who have studied machine learning and built projects, but haven't yet had to answer the questions that show up the moment a model needs to serve real users: How do I get clean data flowing reliably? What happens when production data doesn't look like training data? How do I deploy this without breaking things? Why did a model that looked great offline fail in the real world?

Each article expands on a chapter of the book with deeper explanations, additional concepts, concrete industry examples, and a worked end-to-end scenario — written to show how a practicing ML engineer actually thinks, not just what the algorithms do.

## How to Read This Series

The articles build on each other and are best read in order. Concepts introduced early (training-serving skew, silent failures, business-metric alignment) get referenced and built upon throughout the rest of the series.

| # | Article | Core Question |
|---|---------|----------------|
| 1 | [Overview of Machine Learning Systems](./ml-systems-article-1.md) | What actually *is* a production ML system, and when should you use ML at all? |
| 2 | [How ML Systems Are Actually Built](./ml-systems-article-2.md) | How do you go from a business goal to a well-scoped, well-framed ML problem? |
| 3 | [Data Engineering for ML](./ml-systems-article-3.md) | Where does production data come from, and how is it stored, modeled, and moved? |
| 4 | [Training Data — Sampling, Labeling, and Hidden Traps](./ml-systems-article-4.md) | How do you choose what data to train on, and where do labels actually come from? |
| 5 | [Feature Engineering](./ml-systems-article-5.md) | How does raw data become model inputs, and how does data leakage quietly ruin models? |
| 6 | [Model Development and Offline Evaluation](./ml-systems-article-6.md) | How do you select, train, and rigorously evaluate a model before it ships? |
| 7 | [Deployment](./ml-systems-article-7.md) | How do models actually get served to users — and what happens when they're too slow? |
| 8 | [Monitoring and Continual Learning](./ml-systems-article-8.md) | How do you detect that a deployed model is silently failing? |
| 9 | [Continual Learning and Testing in Production](./ml-systems-article-9.md) | How do you keep models fresh and safely test updates on real traffic? |
| 10 | [ML Infrastructure](./ml-systems-article-10.md) | What's the actual technology stack underneath all of this? |
| 11 | [User Experience, Team Structure, and Responsible AI](./ml-systems-article-11.md) | How do humans, teams, and society fit into all of it? |

## What Makes This Series Different

- **Grounded in real failure modes.** Every concept is paired with a concrete scenario — a worked example, a documented industry incident, or a realistic debugging story — rather than left as an abstract definition.
- **Connected, not siloed.** Concepts introduced early (e.g., training-serving skew in Article 3) are deliberately revisited and built on in later articles (Articles 7, 8, and 10), the way they actually compound in a real system.
- **A running synthesis in every article.** Each article ends with an end-to-end "putting it together" scenario showing how that article's concepts combine in a single realistic project, plus a condensed key-takeaways summary.
- **Written for the gap between coursework and the job.** This isn't an introduction to ML algorithms — it assumes you already know how to train a model. It's about everything that happens before and after that step.

## Source Material

This series is a derivative, explanatory companion to:

> Huyen, Chip. *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications.* O'Reilly Media, 2022.

Each article corresponds to one chapter of the book (Articles 1–11 ≈ Chapters 1–11) and is intended as supplementary reading — expanding the book's concepts with additional explanation and examples for learners newer to production ML — not a replacement for the original text. Readers who find this series useful are strongly encouraged to read the full book.

## Who This Is For

- ML/Data Science practitioners who've completed coursework or built models but haven't shipped a production system
- Software engineers moving into ML-adjacent roles who need the production-systems context, not the algorithms
- Anyone preparing for ML system design interviews

## License / Attribution

This series is original written content created as a study companion and is not affiliated with or endorsed by Chip Huyen or O'Reilly Media. All concepts are credited to the source book; the explanations, examples, and scenarios in each article are original.

---

*Have feedback or found an error? Open an issue or a PR.*
