# Machine Learning Systems
## Article 1: The Reality of ML in Production — What They Don't Teach You in Courses

*This is the first article in the **Machine Learning Systems** series, based on "Designing Machine Learning Systems" by Chip Huyen. This series is for practitioners who have built ML models and want to understand what happens when those models meet the real world.*

---

You've trained a model. Accuracy looks great. You're proud of it. Then someone asks: *"Can we put this in production?"*

That's where most ML courses go silent — and where this series begins.

The gap between a working Jupyter notebook and a reliable production ML system is enormous. This article lays the foundation: what an ML system actually is, when you should (and shouldn't) use ML, and why everything you learned in academic settings needs to be re-examined before you deploy.

---

## What Is a Machine Learning System, Really?

When most people hear "machine learning system," they picture the algorithm — a neural network, a gradient boosting model, a transformer. But the algorithm is, at best, 5–10% of the actual system.

A production ML system is an ecosystem with many interacting components:

- **Business requirements** — what problem are we actually solving, and how do we measure success?
- **Data infrastructure** — pipelines to collect, store, clean, and version data
- **Feature engineering** — transforming raw data into inputs the model can use
- **The model itself** — training, evaluation, and selection
- **Serving infrastructure** — getting predictions to users reliably and quickly
- **Monitoring and observability** — detecting when the model degrades or fails
- **Feedback loops** — capturing signals to retrain and improve the model over time
- **Human-in-the-loop workflows** — cases where a human must review model decisions

Think of the model as the engine in a car. It's critical, but without the chassis, wheels, fuel system, dashboard, and brakes, the engine alone goes nowhere. A practicing ML engineer spends far more time building and maintaining that surrounding infrastructure than tuning the model itself.

---

## Point 1: Knowing When to Use Machine Learning

Machine learning is not a Swiss army knife. It is a powerful but specific tool — one that learns complex patterns from existing data to make predictions on new, unseen data.

**The core insight:** Use ML when the rules are too complex, too numerous, or too dynamic to write by hand.

Consider spam detection. In 2002, you could write a list of rules: *"If the email contains 'Congratulations, you've won', flag it."* That worked for a while. But spammers adapt. The rules grew into thousands of conditions, became impossible to maintain, and still fell behind. ML solved this by learning patterns from millions of labeled examples and updating automatically as spam evolved.

### When ML is a good fit:

- **The problem is too complex for hand-coded rules.** Detecting fraudulent transactions involves hundreds of variables interacting in non-linear ways. No rule-based system can capture this reliably.
- **The task is highly repetitive and predictive.** Forecasting tomorrow's demand for a product, predicting which users will churn, ranking search results — these happen millions of times a day, and a model can do it consistently at scale.
- **The patterns change over time.** If what "normal" looks like shifts (user behavior, market conditions, language trends), an ML model can be retrained on new data rather than rewriting code.
- **You have data.** This is non-negotiable. ML systems are data hungry. Without sufficient, representative, labeled data, you cannot train a useful model.

### When ML is NOT a good fit:

- **You need a deterministic, explainable answer.** If a bank needs to explain exactly *why* a loan was rejected in a court of law, a black-box model may not be appropriate.
- **You have very little data.** A startup with 500 customer records shouldn't be building churn prediction models. Traditional analytics will serve better.
- **The cost of errors is too high and errors are unpredictable.** ML models fail in ways that are hard to anticipate. Safety-critical systems often require the reliability of rule-based software.
- **A simple rule works perfectly.** If you can write *"flag orders over $10,000 for review"* and it catches 98% of fraud, adding an ML model introduces complexity for marginal gain.

**A habit of practicing engineers:** Before proposing an ML solution, always ask *"What is the simplest non-ML solution, and why isn't it good enough?"* This forces you to justify the complexity ML introduces.

---

## Point 2: Real-World Use Cases

ML adoption has moved far beyond recommendation engines and search rankings. Today it touches nearly every industry.

### Consumer Applications

You interact with ML dozens of times a day without noticing:

- **Predictive keyboard** on your phone learns your typing style, shortcuts, and frequently used phrases.
- **Computational photography** — when your phone camera takes a "perfect" low-light photo, an ML model is running in milliseconds to denoise and enhance the image.
- **Voice assistants** use speech recognition (converting audio to text), intent classification (understanding what you want), and natural language generation (formulating a response) — three separate ML models chained together.
- **Social media feeds** use ranking models that decide, in real time, which of thousands of posts to show you first.

### Enterprise Applications

On the business side, ML is deeply embedded in operations:

- **Fraud detection (Fintech):** Every time you swipe a card, a model scores the transaction for fraud risk in under 100 milliseconds. Features include transaction amount, merchant category, location, time of day, and your historical behavior. A score above a threshold triggers a hold or a call.
- **Dynamic pricing (E-commerce, Airlines, Ride-sharing):** Uber's surge pricing is a real-time ML model balancing supply and demand. Airlines reprice seats hundreds of times a day based on booking pace, competitor prices, and historical fill rates.
- **Predictive maintenance (Manufacturing):** Sensors on factory equipment stream data continuously. An ML model flags when a machine is likely to fail *before* it actually does, saving millions in downtime and emergency repairs.
- **Content moderation (Social Media):** At the scale of billions of posts per day, no human team can review all content. ML classifiers flag potentially violating content for human review, prioritizing what humans see first.

### A Practitioner's Reality Check

These use cases look clean in articles. In practice, they are messy. The fraud detection model might be 15 models in a pipeline. The dynamic pricing system gets retrained daily on fresh data and has an entire team maintaining the feature store. When you read about a company "using ML," picture an engineering team of 5–50 people maintaining that system, not a single notebook.

---

## Point 3: ML in Research vs. ML in Production

This is the most important mindset shift you need to make.

Research ML and production ML are not just different environments — they optimize for fundamentally different goals.

### 1. What "Success" Means

In research, success is a better metric on a benchmark dataset. A 0.5% improvement in BLEU score on a translation task is publishable.

In production, success is *business impact*. A model that improves recommendation click-through rate by 2% might generate millions in additional revenue. But a model that reduces customer support call volume by 15% while being 3% less accurate might be preferred over the "better" model.

**Example:** Netflix has famously reported that their most accurate recommendation models are not always the ones they deploy. A model that recommends slightly less "optimal" content but keeps users engaged for longer wins, even if it loses on pure accuracy metrics.

### 2. Computational Priorities

In research: you want to train as fast as possible. GPUs, distributed training, mixed precision — all to reduce time-to-result.

In production: training speed matters much less. What matters is **inference latency** (how long it takes to get a prediction for a single request) and **throughput** (how many predictions you can serve per second).

A model that trains in 10 hours but serves predictions in 5ms is often better than one that trains in 2 hours but needs 50ms per prediction — especially at scale. 45ms difference per request, multiplied by 100 million requests per day, is the difference between your infrastructure bill and your profitability.

**Practical technique:** Model quantization, pruning, and distillation are all techniques used to make models faster at inference time, even if they sacrifice some accuracy. Practicing engineers must know these tools.

### 3. The Data Reality

Academic datasets are gifts. They are pre-cleaned, balanced, labeled by experts, and static — the same dataset for every researcher.

Production data is a war zone:

- **Missing values everywhere.** A sensor goes offline. A user doesn't fill in their age. An API call times out and the field is null.
- **Biased samples.** Your historical data reflects past decisions, which were made by humans with biases. A hiring model trained on past hire/reject decisions learns those biases.
- **Label noise.** In fraud detection, you might not know a transaction was fraudulent until 30 days later when the chargeback arrives. Until then, it's labeled "legitimate."
- **Distribution shift.** The data the model is deployed on looks different from the data it was trained on. This happens gradually (user demographics shift) or suddenly (COVID-19 changed every behavioral pattern overnight).
- **Constantly arriving.** Production data is a stream, not a snapshot. Your pipeline must handle new data continuously.

### 4. Stakeholder Conflicts

In research, you answer to your advisor or your paper reviewers. In production, you serve multiple stakeholders with conflicting priorities:

| Stakeholder | What They Want |
|---|---|
| ML Engineer | Expressive model, interesting technical problem |
| Product Manager | Feature ships on time, users love it |
| Infrastructure Team | Low computational cost, easy to maintain |
| Legal/Compliance | Explainable decisions, auditable logs |
| Business Leadership | Revenue impact, ROI on ML investment |

No model will make everyone happy. A large part of a practicing ML engineer's job is **negotiating and communicating tradeoffs** — not just building the best model.

### 5. Fairness and Interpretability Are Not Optional

In a research paper, you can note fairness as a "limitation" in the last paragraph. In production, a biased model causes real harm to real people.

**Example:** A hiring screening model trained on historical data at a tech company might learn that candidates from certain universities or with certain names are more likely to be hired (because that's what the historical data shows). Deployed at scale, it systematically disadvantages underrepresented groups. This is not a hypothetical — Amazon shut down an internal recruiting tool for exactly this reason.

Interpretability is required not just for ethics but for debugging. When a model gives a wrong prediction, you need to understand *why* to fix it. Tools like SHAP values, LIME, and attention visualization exist precisely for this reason.

---

## Point 4: ML Systems vs. Traditional Software

ML is often called "Software 2.0," and the distinction matters.

In traditional software (Software 1.0), an engineer writes explicit rules:

```python
def should_approve_loan(credit_score, income, debt):
    if credit_score > 700 and income / debt > 3:
        return True
    return False
```

The logic is human-written, interpretable, and deterministic. Given the same input, you always get the same output. When it breaks, there's usually a clear error.

In ML (Software 2.0), you provide input-output examples and the machine learns the rules:

```python
model = GradientBoostingClassifier()
model.fit(X_train, y_train)  # The "rules" are inside the model weights now
```

The learned rules are distributed across potentially millions of parameters. You can't read them. You can only probe the model's behavior.

### Why This Makes Maintenance Harder

Traditional software breaks loudly — exceptions, crashes, error codes. You know immediately that something is wrong.

**ML systems fail silently.** The model continues to return predictions with high confidence, but those predictions are wrong. This is the nightmare scenario in production:

- A demand forecasting model starts producing slightly optimistic forecasts. Nobody notices for three months. Warehouses overstock. Losses are in the millions.
- A content moderation model's accuracy drops after a new slang term emerges. Harmful content starts slipping through, but the model reports no errors.

This is why **monitoring is not optional** in production ML. You need to continuously track:

- Prediction distributions (are the model's outputs shifting?)
- Input data distributions (is the incoming data different from training data?)
- Business metrics (are downstream outcomes changing?)

### The Two-Codebase Problem

Traditional software has one artifact: the code. ML systems have two: the **code** and the **data**. Both can be sources of bugs, both need versioning, and both need to be reproducible.

If your model degrades and you want to debug it, you need to know:
- Which version of the code trained it?
- Which exact dataset was used?
- What were the hyperparameters?

Without discipline around **experiment tracking** and **data versioning** (tools like MLflow, DVC, Weights & Biases), debugging becomes archaeology.

---

## Point 5: The Iterative, Never-Done Nature of ML Systems

This is something courses rarely teach: **an ML system is never finished.**

A traditional software feature ships and is "done" unless requirements change. An ML model must be continuously monitored and periodically retrained because the world changes.

This has major organizational implications:

- You need **automated retraining pipelines** that can retrain the model on fresh data without manual intervention.
- You need **A/B testing infrastructure** to safely evaluate a new model version against the old one in production before fully switching.
- You need **rollback capability** — if a new model performs worse, you need to instantly revert.
- You need **human review workflows** for high-stakes decisions (medical diagnosis, loan approvals) even after automation.

**The practicing engineer's reality:** After a model ships, you don't move on. You own it. That means on-call alerts when monitoring thresholds are breached, quarterly retraining reviews, and constant conversations with product and data teams about what's changing.

---

## Putting It Together: What a Real ML Project Looks Like

Here's a rough timeline for a fraud detection model at a mid-size fintech:

1. **Weeks 1–3:** Understand business requirements. Define what "fraud" means (chargebacks? Disputes? Both?). Agree on success metrics with product and finance.
2. **Weeks 4–6:** Data audit. Discover that 3 years of transaction data exists, but 18 months of it has a labeling issue. Work with data engineering to fix the pipeline.
3. **Weeks 7–9:** Feature engineering. Build 50+ features from raw transactions. Half are later found to cause data leakage (using future information). Rebuild.
4. **Weeks 10–12:** Modeling. Train a baseline logistic regression, then gradient boosting. The neural network performs only marginally better at 10x the inference cost. You ship the gradient boosting model.
5. **Weeks 13–14:** Evaluation. Discover the model has lower recall on a certain geographic segment. Investigate — turns out training data underrepresents that region. Collect more data and retrain.
6. **Week 15:** Deploy behind a feature flag to 5% of traffic. Monitor closely.
7. **Week 16:** Full rollout. Set up dashboards and alerts.
8. **Month 4:** A new fraud pattern emerges. Model performance drops. Retrain.
9. **Ongoing:** Quarterly reviews, bias audits, infrastructure upgrades.

Notice: the model itself was built in about 3 weeks. The rest was everything else.

---

## Key Takeaways

- A production ML system is a sociotechnical system — algorithm, data pipelines, infrastructure, monitoring, and organizational workflows all intertwined.
- Use ML when patterns are too complex for rules, when scale demands automation, and when you have the data to learn from.
- Production ML optimizes for business outcomes, latency, reliability, fairness, and maintainability — not just accuracy.
- ML systems fail silently. Monitoring is not optional; it is as important as the model.
- An ML project is never "done." Continuous monitoring, retraining, and iteration are the job.

---

## What's Next

In the next article, we'll go deep into **data engineering for ML** — how production data pipelines are built, why your training data is probably not as clean as you think, and what tools practicing engineers use to manage data at scale.

---
