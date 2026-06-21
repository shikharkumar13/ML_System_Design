# Machine Learning Systems
## Article 2: How ML Systems Are Actually Built — From Business Goals to Deployment

*This is the second article in the **Machine Learning Systems** series, based on "Designing Machine Learning Systems" by Chip Huyen. If you haven't read Article 1, start there — it covers what an ML system actually is and why production ML is a fundamentally different world from research.*

---

In the first article, we established that a production ML system is far more than a model — it's an ecosystem of pipelines, infrastructure, monitoring, and people. 

Now let's go one level deeper: **how do you actually think about building one?**

This article covers the mental frameworks that experienced ML engineers use before writing a single line of model code. How do you translate a business problem into an ML problem? What does the development process actually look like? And why does the way you *frame* a problem often matter more than the algorithm you choose?

These are the things that separate engineers who build models from engineers who build *systems*.

---

## The Gap Between Business Goals and ML Metrics

Here is one of the most important realities of working in industry: **your manager does not care about your F1 score.**

What they care about is whether the system is making the company money, retaining customers, reducing costs, or improving some user experience that eventually leads to one of those outcomes.

This creates a translation problem that sits at the heart of every ML project. You need to convert a business objective into something a model can optimize — and then ensure that optimizing that ML metric actually moves the business metric in the right direction.

This sounds obvious. It is surprisingly hard to get right.

### A Classic Example

Suppose you're building an ad recommendation system. The business goal is **revenue**. You decide to optimize for **click-through rate (CTR)** because more clicks should mean more revenue, right?

Your model gets very good at predicting which ads get clicked. CTR goes up 12%. 

Revenue goes down.

What happened? Your model learned to recommend "clickbait" style ads — ones people click on out of curiosity but never purchase from. Advertisers see their conversion rates drop and reduce their spend. You optimized the proxy metric perfectly and made the real metric worse.

This is called **metric-objective misalignment** and it's more common than you'd think.

**The Amazon story:** In the early days of Amazon's recommendation engine, the team optimized for click-through rate. They later discovered that optimizing for *purchases* — not clicks — significantly changed which recommendations were made. A book someone clicks but doesn't buy is interesting. A book they buy is genuinely useful. These are very different signals.

### The Practicing Engineer's Approach

Before committing to an ML metric, experienced engineers ask:

1. **What is the ultimate business goal?** Revenue, retention, safety, cost reduction?
2. **What ML metric is a proxy for that goal?** Accuracy, AUC, NDCG, precision, recall?
3. **Under what conditions does optimizing this metric hurt the business goal?** (Always find an adversarial case.)
4. **How will we measure business impact after deployment?** What experiment design (A/B test, holdout group) will tell us if the model is actually working?

The answers to these questions should be written down and agreed upon with stakeholders *before* you start collecting data.

---

## The Four Non-Negotiable Requirements of ML Systems

Before building anything, you need to understand what "good" looks like for the system you're building. Every production ML system must satisfy four properties. These are not aspirational — they are baseline requirements.

### 1. Reliability

The system must perform correctly even when things go wrong — hardware failures, network timeouts, bad input data, code bugs.

The tricky part: **ML systems can fail silently.** Traditional software throws exceptions when something breaks. An ML model quietly returns a confident, wrong prediction. There's no stack trace. No alert. Just incorrect outputs flowing through your system.

**What this looks like in practice:**

Imagine a credit scoring model that silently starts receiving features with a different scaling than it was trained on — because someone updated the feature engineering pipeline without retraining the model. The model continues to score applicants, but its predictions are now unreliable. This could run for days before anyone notices the downstream impact on loan approval rates.

Reliability engineering for ML systems means:
- **Input validation:** Check that incoming data matches the expected schema, range, and distribution.
- **Output validation:** Flag predictions that fall outside expected ranges or that the model gives unusually high confidence to.
- **Fallback mechanisms:** When the ML model fails (latency spike, error rate surge), fall back to a simpler rule-based system rather than serving nothing.
- **Canary deployments:** Roll out a new model to 1–5% of traffic first. Watch for anomalies before full rollout.

### 2. Scalability

Your system needs to handle growth — in three dimensions that are often confused with each other:

**Growth in model complexity:** A model that runs fine on a single GPU might need distributed inference as it grows to billions of parameters. GPT-2 (2019) was 1.5B parameters. GPT-3 (2020) was 175B. A 100x jump in model size doesn't mean a 100x increase in infrastructure cost — it often means a qualitative shift in *how* you serve the model.

**Growth in traffic volume:** A fraud detection system might handle 1,000 transactions per second today and 50,000 per second in two years. Your serving infrastructure needs to scale horizontally (adding more machines) without requiring a full rebuild.

**Growth in model count:** This one surprises many learners. Large companies don't run one model — they run hundreds or thousands. Uber reportedly has thousands of ML models in production. Netflix has separate models for each country, device type, and user segment. Managing this many models requires MLOps infrastructure: automated retraining, versioning, deployment pipelines, and monitoring dashboards that scale beyond what any human can manually track.

**A useful mental test:** When designing your system, ask *"What breaks if traffic 10x's overnight?"* and *"What breaks if we need to manage 100 models instead of 1?"* The answers tell you where to invest in scalability.

### 3. Maintainability

ML systems are built by teams — not individuals — and those teams include people with very different backgrounds:

- **ML engineers** who understand model architecture and training
- **Data engineers** who build and maintain data pipelines
- **Backend engineers** who build serving infrastructure
- **Data scientists** who do exploratory analysis and feature research
- **Domain experts** (doctors, lawyers, financial analysts) who understand the business problem but not the code

A system is maintainable if each of these people can do their job without accidentally breaking everyone else's work.

**What poor maintainability looks like:**

A data scientist creates features in a Jupyter notebook, manually exports them to a CSV, and emails it to the ML engineer. The ML engineer trains a model, saves it to their local machine, and manually copies it to a server. Three months later, nobody knows which version of the features the deployed model was trained on. This is not hypothetical — this is how many early-stage ML projects operate.

**What good maintainability looks like:**

- Feature definitions are written as code, versioned, and stored in a **feature store** that both training and serving pipelines pull from.
- Experiments are tracked with tools like **MLflow** or **Weights & Biases** — every run records the code version, dataset version, hyperparameters, and metrics.
- Models are stored in a **model registry** with metadata about when they were trained, on what data, and by whom.
- Deployment is automated — a new model is promoted to production through a CI/CD pipeline, not by someone SSH-ing into a server.

### 4. Adaptability

Data changes. The world changes. Your model was trained on yesterday's reality, and it's deployed in today's. **Adaptability** is the system's capacity to detect when it's falling behind and update itself — ideally without service interruption.

This is where many teams underinvest. They spend months building a great model, deploy it, and then treat it as done. Six months later, performance has quietly degraded and nobody is sure why or when it started.

Adaptability requires:
- **Monitoring infrastructure** that continuously tracks model performance and data distributions.
- **Automated retraining pipelines** that can retrain and re-evaluate the model on fresh data on a schedule or when triggered by a performance drop.
- **Shadow mode deployment:** Running a new model in the background alongside the current model, comparing outputs, without actually serving the new model's predictions to users yet.

---

## The Iterative Reality: What ML Development Actually Looks Like

Every ML course teaches you a linear pipeline: get data → train model → evaluate → deploy. 

This is a lie of omission.

In reality, ML development is a messy, non-linear loop. You constantly discover problems that send you back to earlier stages. Here is a realistic example of building an ad click prediction model:

1. You decide to optimize for **impressions** (how many times an ad is shown).
2. You collect data and labels.
3. You engineer features and train a model.
4. During error analysis, you discover many prediction errors are caused by **wrong labels** — some ads were marked as shown when they actually failed to load. You go back and fix the labels.
5. You retrain.
6. Error analysis reveals your model **always predicts "don't show the ad"** — because 99.99% of the training examples are negative (ads that weren't clicked). The model learned to be pessimistic. You need to collect more positive examples and fix the class imbalance.
7. You retrain.
8. The model performs poorly on data from the past two days because trends have shifted. You update it on more recent data.
9. You retrain.
10. You deploy.
11. Business analysts notice **revenue is falling** because users aren't buying from the ads. You realize optimizing for impressions was the wrong objective. You switch to click-through rate.
12. You go back to step 1.

This is not a failure story — this is a normal story. Every loop teaches you something: about your data quality, your labels, your metric choice, your feature set, or your users. 

**The practicing engineer expects to loop.** The junior engineer is surprised by it and interprets each loop as a setback. The senior engineer budgets for it in project timelines from the start.

### The Six Canonical Stages (Simplified)

While the real process is messy, it helps to have a mental map. Most ML projects move through six broad stages:

**Stage 1 — Project Scoping**
Define the goals, constraints, success criteria, and stakeholders. Who will use this system? What does success look like in three months? What's the latency budget? What are the legal or fairness constraints? 

*This stage is where most projects fail — not because they build the wrong model, but because they never clearly defined what they were trying to build.*

**Stage 2 — Data Engineering**
Build pipelines to collect, store, and process the data you need. This often takes longer than any other stage. You'll discover that the data you assumed existed doesn't, or exists in a format that requires months of cleanup.

**Stage 3 — ML Model Development**
Extract features, build baselines, experiment with models, track experiments, and select the best candidate. Note the word "baseline" — experienced engineers always start with the simplest possible model and beat it before moving to complexity.

**Stage 4 — Deployment**
Package the model, build a serving layer, integrate it into the product, and run it against real traffic (initially on a small fraction). This stage involves far more software engineering than ML engineering.

**Stage 5 — Monitoring and Continual Learning**
Set up dashboards, alerts, and retraining pipelines. Watch for data drift, prediction drift, and business metric degradation. Retrain when needed.

**Stage 6 — Business Analysis**
Did the system actually move the business metrics it was supposed to? Run proper experiments (A/B tests), document findings, and feed insights back into the next iteration.

---

## Framing ML Problems: A Skill That's Underrated

Here's something that will feel counterintuitive: **how you frame an ML problem often matters more than which algorithm you use.**

The same real-world problem can be framed as multiple different ML tasks, and the choice of framing has enormous downstream consequences for data requirements, model complexity, latency, and maintainability.

### Classification vs. Regression

**Classification** assigns an input to one of several discrete categories. **Regression** produces a continuous numeric output.

Sometimes this choice is obvious. "Is this email spam?" is clearly classification. "What will this house sell for?" is clearly regression.

But often there's a choice — and it matters.

**Example:** Should a fraud detection system output a binary label ("fraud" / "not fraud") or a continuous fraud probability score (0.0 to 1.0)?

The binary output is simpler but inflexible. Different parts of the business may want to apply different thresholds — a high-value transaction team might want to review everything above 0.7, while the automated block list only activates at 0.95. If your model only outputs a binary label, you've baked a threshold decision into the model that should be a business decision made downstream.

Best practice: **output probabilities from your model and let downstream systems apply thresholds.** This gives you flexibility without retraining.

### Binary vs. Multiclass Classification

Binary classification has two possible outputs. Multiclass has more than two.

Binary is significantly simpler to train, evaluate, and debug. When you have a choice, bias toward binary. "Is this review positive?" is easier than "Is this review very positive, positive, neutral, negative, or very negative?" — and may be good enough.

**High cardinality multiclass** — problems with thousands of possible classes — are a special challenge. Classifying a customer query into one of 10,000 possible product categories, or identifying a song from a database of 50 million tracks, requires specialized architectures (embedding models, approximate nearest neighbor search) that are well beyond a simple softmax classifier.

### Multilabel Classification: The Often-Forgotten Case

In standard multiclass classification, each example belongs to exactly one class. In **multilabel** classification, each example can belong to multiple classes simultaneously.

**Example:** A news article can be tagged with "Technology," "Business," and "AI" all at once. A movie can be "Action," "Comedy," and "Thriller." A medical image can show multiple conditions.

Multilabel problems are common in production but underrepresented in courses because they're harder to benchmark cleanly. The key difference in implementation: instead of a softmax output (which forces probabilities to sum to 1), you use independent sigmoid outputs per label — each label is its own binary classification.

### The Power of Reframing

Here is a concrete example of how reframing can completely change the difficulty of a problem.

**Problem:** Predict which mobile app a user will open next.

**Framing A — Multiclass classification:** The model outputs a probability distribution over all possible apps. The class with the highest probability is the prediction.

*Problem:* You have 3 million apps. Your output layer has 3 million neurons. Training is slow. Evaluation is expensive. And whenever a new app is released, you have to add a new output neuron and retrain the entire model.

**Framing B — Regression (scoring):** The model takes a (user, app) pair as input and outputs a relevance score. You run the model on a candidate set of apps, rank by score, and recommend the top ones.

*Advantages:* The model is much smaller. Adding a new app means adding a new item to score, not retraining. You can update the candidate set without touching the model. This is essentially how recommender systems work at companies like TikTok and Spotify.

The underlying task is the same. The framing makes one approach feasible and the other impractical at scale.

### Handling Conflicting Objectives: Decoupling

A common mistake when building ML systems is trying to optimize for multiple conflicting goals in a single model.

**Example:** A social media feed ranking system needs to:
- Maximize engagement (likes, comments, shares)
- Minimize the spread of misinformation
- Filter harmful content
- Balance content from friends vs. followed accounts vs. ads

Trying to train one model with a weighted combination of all these objectives is a maintenance nightmare. The weights are arbitrary, the loss function is complex, and when business priorities shift (the company now wants to weight misinformation suppression more heavily), you have to retrain everything.

The better approach: **train separate models for each objective, and combine their scores in a downstream ranking layer.**

- Model A scores each post for predicted engagement.
- Model B scores each post for misinformation risk.
- Model C scores each post for harmful content probability.
- A rule-based or lightweight learned ranker combines these scores using business-defined weights.

Now, when priorities change, you adjust the weights in the ranker — not retrain your models. Each model is independently debuggable, evaluable, and improvable. This is how large feed ranking systems are actually built.

---

## Mind vs. Data: What Actually Drives ML Progress?

There's a long-running debate in the ML community: does progress come from smarter algorithms ("mind") or from more data?

The empirical answer from the last decade is fairly clear: **at scale, data wins.**

The "Bitter Lesson" (a famous essay by reinforcement learning pioneer Rich Sutton) argues that most advances in AI come not from clever domain-specific algorithms but from general methods that leverage more computation and data. GPT-4 is not smarter than GPT-2 because it has better architecture — it's smarter largely because it was trained on vastly more data with vastly more compute.

But this doesn't mean data quality is irrelevant. Two nuances matter for practitioners:

**Nuance 1: Scale requires resources most companies don't have.** The "data wins" insight applies at the frontier — when you have the compute budget of OpenAI or Google. For most companies building domain-specific ML systems, a carefully curated dataset of 100,000 high-quality examples will outperform a scraped dataset of 10 million noisy examples.

**Nuance 2: Garbage in, garbage out — at any scale.** More data of low quality makes your model confidently wrong in more situations. Data quality — correctness of labels, representativeness of the population, absence of leakage — is a multiplier on everything else.

**Practical heuristic:** Before scaling your dataset, clean the dataset you have. A 2016 study from Google found that reducing label noise in a training set by 10% often produced larger model improvement than doubling the dataset size.

---

## Putting It All Together: How a Senior Engineer Approaches a New ML Project

When a senior ML engineer is handed a new project, here's roughly what goes through their mind — before writing any code:

1. **What is the actual business goal, and how will we measure it?** (Don't start without this.)
2. **Is ML even the right tool? What's the simplest non-ML solution?**
3. **How can this problem be framed as an ML task?** Explore multiple framings and reason about their tradeoffs.
4. **What data exists? What data do we need? How long will it take to get labels?**
5. **What are the constraints?** Latency budget, compute budget, explainability requirements, regulatory requirements.
6. **What does "good enough" look like at each stage?** Define acceptance criteria before training begins.
7. **How will we monitor this in production?** Don't treat monitoring as an afterthought.
8. **What is the rollback plan if this model causes harm?**

These questions are not academic. They are the difference between a project that ships and delivers value and one that spends six months in a notebook and gets cancelled.

---

## Key Takeaways

- **Tie ML metrics to business metrics from day one.** Optimizing the wrong proxy metric can actively hurt business outcomes.
- **Every production ML system needs to be reliable, scalable, maintainable, and adaptable.** These are not nice-to-haves; they are baseline requirements.
- **ML development is iterative and non-linear.** Budget for loops. Expect to go back to earlier stages. Treat each loop as a learning opportunity.
- **How you frame the ML problem is often more important than which algorithm you use.** Reframing can make an infeasible problem tractable.
- **Decouple conflicting objectives.** Train separate models and combine at a ranking layer rather than trying to optimize one model for everything.
- **Data quality beats data quantity** for most companies operating below frontier scale.

---

## What's Next

In Article 3, we go deep into **data engineering for ML** — the stage that consumes the most time in any real ML project and is almost entirely absent from ML courses. We'll cover how production data pipelines are built, what data drift looks like in practice, how to think about labeling at scale, and the tools engineers use to make all of this manageable.

---
