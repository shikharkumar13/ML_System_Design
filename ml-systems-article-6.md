# Machine Learning Systems
## Article 6: Model Development and Offline Evaluation — Building It Right Before You Ship It

*This is the sixth article in the **Machine Learning Systems** series, based on "Designing Machine Learning Systems" by Chip Huyen. Article 5 covered feature engineering — turning raw data into usable inputs. This article covers what happens with those features: how models are actually selected, trained, and evaluated by practicing engineers before anything gets near production.*

---

By the time you reach modeling, courses make it feel like the main event has finally arrived. In practice, this is where a different kind of discipline takes over — one that has very little to do with chasing the latest architecture and a lot to do with strategy, rigor, and resisting your own instincts.

This article covers how experienced engineers actually approach model selection (hint: it's rarely "reach for the biggest neural network"), the infrastructure needed to track and debug experiments, how training scales to massive models, and — just as important as building the model — how to evaluate it properly before it ever sees a real user.

---

## Model Selection and Development: Six Tips From Practitioners

There's a strong pull, especially for engineers newer to the field, toward reaching for the most sophisticated, state-of-the-art architecture available. In production, this instinct is usually wrong. Here's how experienced practitioners actually think about model selection.

### Tip 1: Avoid the State-of-the-Art Trap

A model being "state-of-the-art" means it achieves the best published performance on a specific benchmark dataset — not that it's the best choice for your specific problem, data, and constraints.

**Why this trap is so easy to fall into:** State-of-the-art models get papers, press coverage, and conference talks. There's a natural pull to want to use the newest, most impressive-sounding architecture. But a model that wins on ImageNet or a leaderboard benchmark was optimized for that benchmark's data distribution, evaluation metric, and computational assumptions — none of which necessarily match your production environment.

**Example:** A massive transformer-based architecture might top the leaderboard for a text classification benchmark, but if your production system needs to classify support tickets in under 20 milliseconds on commodity hardware, that model may be operationally unusable regardless of its accuracy — and a much smaller, "boring" model might be the only realistic choice.

### Tip 2: Start with the Simplest Models

Always build the simplest possible baseline first — logistic regression before a deep neural network, a single decision tree before a complex ensemble. This isn't just about humility; it serves several concrete engineering purposes:

- **It gives you a performance floor.** If your simple model already achieves 85% of the performance a complex model would, you have a strong, fast, interpretable baseline to compare everything against — and a real question of whether the added complexity is worth it.
- **It validates your whole pipeline faster.** A simple model trains in minutes, not hours, which means you find data bugs, leakage, and pipeline issues quickly, before you've sunk a week into training something complex on top of a broken foundation.
- **Simplicity is itself a feature in production.** Simpler models are easier to debug, faster to train and serve, and require less infrastructure — all real costs discussed in Article 5's "engineering good features" section apply just as much to model complexity itself.

**Practical takeaway:** Many experienced engineers treat "can a simple model solve this?" as the default starting hypothesis to disprove, not the default to skip past on the way to something more exciting.

### Tip 3: Avoid Human Biases in Evaluating Architectures

When comparing models, it's easy to unconsciously favor the architecture you're more familiar with, more excited about, or have personally invested more time tuning. This isn't a moral failing — it's a natural cognitive bias, and it quietly distorts model comparisons all the time.

**Example:** An engineer who spent two weeks carefully tuning a gradient-boosted tree model, but only spent one afternoon on a hastily configured neural network baseline, will almost certainly find the tree model "wins" — not necessarily because it's genuinely better suited to the problem, but because it received far more tuning effort and attention.

**Practical takeaway:** When comparing candidate models, ensure each one gets a comparable amount of tuning effort, comparable hyperparameter search budget, and ideally, evaluation by someone without a personal stake in which one wins. Document your comparison methodology before you start, not after you already have a favorite.

### Tip 4: Evaluate Models Using Learning Curves

A learning curve plots model performance (on both training and validation data) as a function of the amount of training data used. This single diagnostic tool answers a question that a single accuracy number never can: **will this model keep improving if I give it more data, or has it plateaued?**

**How to read it:**
- If validation performance is still climbing steeply as training data increases, the model is likely **data-starved** — collecting more data is probably the highest-leverage next step, more so than architecture changes.
- If validation performance has flattened while training performance stays high, the model has likely **plateaued** — more data won't help much, and the bottleneck is probably model capacity, feature quality, or label noise instead.
- A large and persistent gap between training and validation performance is a classic sign of **overfitting**, regardless of how much data you add.

**Practical takeaway:** Learning curves are one of the most underused diagnostic tools in industry. Before requesting a budget for a massive new data collection effort, or before reaching for a much larger model, plotting a learning curve can tell you in advance whether either investment is actually likely to pay off.

### Tip 5: Understand Your Trade-Offs

Every model choice involves trading one property for another. Experienced engineers think explicitly in these trade-off pairs rather than chasing a single metric in isolation:

- **Accuracy vs. interpretability:** A deep ensemble might outperform a single decision tree by 2% accuracy but be nearly impossible to explain to a regulator or a frustrated customer asking "why was I denied?"
- **Training speed vs. inference speed:** As discussed in Article 1, production usually cares far more about inference latency than training time — a model that's slow to train but fast to serve is often preferable to the reverse.
- **Model size vs. latency/memory footprint:** A larger model might be marginally more accurate but won't fit on the memory-constrained mobile device or edge sensor it needs to run on.
- **Bias vs. variance:** The classic statistical trade-off — an overly simple model underfits (high bias), an overly complex model overfits (high variance) — still governs every model choice underneath all the more "modern" framing.

**Practical takeaway:** There is no universally "best" model — only the model that best fits *your* specific combination of accuracy requirements, latency budget, interpretability needs, and infrastructure constraints. Naming these trade-offs explicitly, ideally in writing, is part of doing the job well — it turns an implicit, easily second-guessed decision into a documented, defensible one.

### Tip 6: Know Your Model's Assumptions

Every algorithm carries built-in assumptions about the data it's being applied to, and violating those assumptions silently degrades performance in ways that are hard to trace back to their root cause.

**Examples:**
- Linear regression assumes a linear relationship between features and the target, and assumes errors are normally distributed and have constant variance (homoscedasticity).
- Naive Bayes assumes features are conditionally independent given the class — an assumption that's almost never literally true, but the model can still work reasonably well even when it's violated, as long as you understand why it's a simplification, not a guarantee.
- Many classical models assume the training and serving data come from the same underlying distribution (the i.i.d. assumption) — an assumption directly threatened by the distribution shift problem discussed in earlier articles.

**Practical takeaway:** Before choosing a model, it's worth explicitly listing the assumptions it relies on and asking, honestly, whether your data actually satisfies them. A model performing poorly isn't always a sign you need a "better" model — sometimes it's a sign you picked a model whose assumptions don't match your data's actual structure.

---

## Ensembles: Combining Models for Better Performance

An ensemble combines predictions from multiple individual models (called **base learners**) to produce a single, typically more robust and accurate, final prediction. Ensembles are a staple of production ML, especially in domains like fraud detection and ranking, where small accuracy gains translate directly into real money.

### Bagging (Bootstrap Aggregating)

Train multiple instances of the same base model, each on a different **bootstrapped** sample of the training data (a random sample drawn *with replacement*, so some examples appear multiple times and others not at all in any given sample). Combine their predictions — typically by averaging (regression) or majority vote (classification).

**Why it works:** Each individual model, trained on a slightly different random subset of the data, will make somewhat different errors. Averaging across many such models cancels out a lot of this individual noise, reducing overall variance without needing more real data.

**Example:** Random Forest is the canonical bagging algorithm — it trains many individual decision trees, each on a bootstrapped sample, and each considering only a random subset of features at each split, then averages their predictions. This is precisely why Random Forests tend to be far less prone to overfitting than a single, deep decision tree.

### Boosting

Train base learners **sequentially**, where each new model is specifically trained to focus on correcting the mistakes (misclassified or high-error examples) of the previous models in the sequence.

**Why it works differently from bagging:** Rather than reducing variance by averaging independent models, boosting reduces *bias* by iteratively focusing the next model's attention precisely where the current ensemble is still getting things wrong.

**Example:** XGBoost and LightGBM — two of the most widely used production algorithms for tabular data — are both gradient boosting implementations. Each new tree in the ensemble is trained to predict the *residual error* of the existing ensemble, gradually reducing the overall error with each additional tree. This is a major reason boosting-based models are so dominant in production tabular ML — fraud detection, credit scoring, ad ranking — referenced throughout this series.

### Stacking

Train several different base models (which can be entirely different algorithm types — say, a logistic regression, a random forest, and a neural network, all trained on the same data), and then train a separate **meta-learner** whose job is to take the base models' predictions as its own input features and learn the optimal way to combine them.

**Why it's different from bagging and boosting:** Stacking explicitly learns *how* to weigh and combine diverse model types, rather than relying on simple averaging (bagging) or sequential error correction within a single algorithm family (boosting). This lets you leverage genuinely different model types — each potentially capturing different patterns in the data — in a single, principled combined system.

**Practical takeaway:** Ensembles reliably improve accuracy, but they come at a real production cost: every additional base model in the ensemble adds to your serving latency and infrastructure complexity, since you typically need to run all of them at prediction time. This is the same trade-off discussed in Article 5 around adding more features — more components in the system, even ones that genuinely help accuracy, are not free.

---

## Experiment Tracking and Versioning

ML development isn't a single training run — it's dozens or hundreds of iterations, each with slightly different data, features, hyperparameters, or code. Without disciplined tracking, you quickly lose the ability to answer the most basic question: *"which exact configuration produced this result?"*

### Experiment Tracking

This means systematically recording, for every training run:
- **Metrics over time:** Loss curves, accuracy, precision/recall, and any other relevant metric, tracked across training steps or epochs — not just the final number.
- **System performance:** GPU/CPU utilization, memory usage, training time. These matter for catching infrastructure issues (a sudden memory spike, a training job that's mysteriously slower than the last run) just as much as model quality.
- **Hyperparameters:** Learning rate, batch size, model architecture choices, regularization settings — anything that was configured for this specific run.

**Why this matters in practice:** Tools like **MLflow** and **Weights & Biases**, referenced earlier in this series, exist precisely so that when someone asks "why did the model from three weeks ago perform better than the one we have now," you have an actual, queryable answer rather than a vague memory of what might have changed.

### Versioning

Tracking metrics alone isn't enough — you also need to be able to **reproduce** any given experiment exactly, which requires versioning two things together:

- **Code versioning:** Standard practice via Git — but for ML specifically, this needs to capture not just model code but the exact preprocessing and feature engineering logic used for that run.
- **Data versioning:** Tools like **DVC (Data Version Control)** let you version datasets the way Git versions code — so you can say precisely "this model was trained on data version 1.4," and reliably retrieve that exact dataset later.

**Why both together matter:** A model's behavior is a function of *both* its code and its data. Versioning only the code while the underlying dataset silently changes underneath you (new data gets appended, old data gets cleaned up, a labeling fix gets applied) means you can never truly reproduce a past result — directly connecting back to the data lineage problem discussed in Article 4.

**Practical takeaway:** Reproducibility isn't a nice-to-have for ML teams — it's the only way to debug a regression, defend a model's behavior to an auditor, or simply understand why last quarter's model and this quarter's model behave differently.

---

## Debugging ML Models

Debugging traditional software usually starts from a clear signal: a stack trace, an error code, a crash. **ML models rarely give you this.** A model with a bug doesn't crash — it just trains, converges, and produces predictions that are wrong, often in ways that aren't immediately obvious. This is the modeling-stage version of the "silent failure" theme that's run throughout this entire series.

A few core, battle-tested debugging techniques:

**Start simple, then add complexity incrementally.** Don't build your full intended architecture, full feature set, and full training pipeline all at once and then try to debug all of it simultaneously when something goes wrong. Build the simplest possible working version first, confirm it behaves sensibly, then add complexity one piece at a time — features, regularization, architecture changes — so that when something breaks, you know exactly which change introduced the problem.

**Try to overfit a single batch of data.** This is one of the most effective and widely used sanity checks in the field. Take a tiny subset of your data — even a single batch — and try to get your model to memorize it perfectly (driving training loss very close to zero). If your model *can't* even overfit a tiny amount of data, that's a strong signal something is fundamentally broken in your implementation: a bug in the loss function, incorrectly shaped tensors, a learning rate that's wildly off, or labels that got shuffled out of alignment with their corresponding inputs somewhere in the pipeline. This test catches implementation bugs *before* you waste hours or days running a full, expensive training job on a broken setup.

**Set a random seed.** ML training involves multiple sources of randomness — weight initialization, data shuffling, dropout, data augmentation. Without fixing a random seed, you can't tell whether a difference in results between two runs reflects a genuine improvement from your change, or just random noise from a different random initialization. Fixing the seed during debugging and development isolates the variable you're actually trying to test.

**Practical takeaway:** Because ML systems fail silently, debugging has to be proactive and structural — built into how you develop, rather than something you do reactively after noticing a problem. The "overfit a single batch" test in particular is one of the highest-value, lowest-effort habits a practicing ML engineer can build.

---

## Distributed Training: Scaling Beyond One Machine

As models and datasets grow, a single machine eventually can't hold the model in memory, or training on a single GPU would simply take too long to be practical. Distributed training spreads the workload across multiple machines or devices. There are three core strategies.

### Data Parallelism

The model is copied identically onto multiple machines (or GPUs), and each copy trains on a different *slice* of the data simultaneously. After each step, the gradients computed by each copy are combined (typically averaged) and used to update all copies consistently, keeping them synchronized.

**When it's used:** This is the most common and straightforward form of distributed training, and the default choice when the model itself comfortably fits in memory on a single device, but you want to process more data, faster, by parallelizing across devices.

### Model Parallelism

Rather than copying the whole model everywhere, the model itself is split into pieces, with different pieces (different layers, or different parts of a layer) placed on different machines or devices.

**When it's used:** This becomes necessary when the model is simply too large to fit into the memory of a single device at all — a problem increasingly common with today's massive deep learning models, where even a single copy of the full model can exceed the memory capacity of one GPU.

### Pipeline Parallelism

A refinement on model parallelism: rather than having each device sit idle while waiting for the device before it in the model-parallel chain to finish its piece of the computation, pipeline parallelism breaks the training batch into smaller **micro-batches** and staggers them through the split model stages — so that while one device is processing micro-batch 2's portion of the model, the next device downstream can already be working on micro-batch 1's portion, keeping more devices busy simultaneously rather than sitting idle waiting their turn.

**Practical takeaway:** Most practicing ML engineers will use data parallelism far more often than model or pipeline parallelism, simply because most production models still fit comfortably on a single modern device. Model and pipeline parallelism become relevant primarily at the frontier of model scale — and increasingly so as large language models and other massive architectures become more common even outside of pure research settings.

---

## AutoML: Automating Parts of Model Development

AutoML refers to a spectrum of techniques for automating pieces of the model development process that would otherwise require manual, iterative human effort.

**Soft AutoML — Hyperparameter tuning:** Automatically searching over combinations of hyperparameters (learning rate, regularization strength, tree depth, number of layers) to find a well-performing configuration, using strategies like grid search, random search, or more efficient methods like Bayesian optimization. This is the most common and widely adopted form of AutoML in production teams today — most ML engineers use some form of automated hyperparameter tuning as a standard part of their workflow, via tools integrated into frameworks like scikit-learn, Ray Tune, or Optuna.

**Hard AutoML — Neural architecture search (NAS) and learned optimizers:** A much more ambitious and computationally expensive form of automation, where the system doesn't just tune hyperparameters for a fixed architecture — it searches over the architecture itself (number of layers, types of layers, how they connect), or even learns the optimization algorithm used to train the model, rather than relying on a fixed, hand-designed optimizer like Adam or SGD.

**Practical takeaway:** Soft AutoML (hyperparameter tuning) is genuinely standard practice and worth adopting broadly — it's relatively cheap and reliably improves results. Hard AutoML (NAS, learned optimizers) remains computationally expensive enough that it's primarily used by large research labs and major tech companies with significant compute budgets, rather than being a routine part of most production teams' workflows today.

---

## Offline Evaluation: Proving the Model Is Actually Good Before It Ships

Training a model that fits well on your test set is necessary, but far from sufficient. Before deployment, the model needs to survive a much more rigorous battery of evaluation — designed specifically to catch problems that a single overall accuracy number will never reveal.

### Baselines: Good Compared to What?

A model's raw accuracy number means almost nothing in isolation — it only becomes meaningful relative to a baseline. Practicing engineers compare against several types simultaneously:

- **Random baseline:** What would performance look like if predictions were made completely at random? Sets an absolute floor.
- **Simple heuristic baseline:** What does a basic, hand-coded rule achieve? (Connects back to Article 2's "what's the simplest non-ML solution" question.)
- **Zero rule baseline:** Always predict the majority class. As discussed in Article 4, on an imbalanced dataset (99.9% legitimate transactions), this trivially achieves 99.9% "accuracy" — making it an essential sanity check that immediately exposes whether a fancier model's seemingly impressive accuracy number is actually meaningful, or just reflective of class imbalance.
- **Human baseline:** How well do human experts perform on the same task? Relevant both as a performance benchmark and as a practical signal for deciding how much human oversight the system still needs after deployment.
- **Existing non-ML solution baseline:** If a rule-based or manual system is already in place, the new ML model needs to beat *it specifically* to justify the added complexity and maintenance cost of switching to ML at all.

**Practical takeaway:** A model report that states "our model achieves 94% accuracy" without stating what the zero-rule baseline achieves, or what the existing solution achieves, is functionally meaningless. Always report model performance *relative* to these baselines, not in isolation.

### Perturbation and Invariance Tests

**Perturbation tests** add small amounts of synthetic noise to test inputs (similar to the data augmentation techniques covered in Article 4) and check whether the model's predictions remain reasonably stable. If tiny, realistic variations in input data cause wildly different predictions, the model has likely learned a brittle decision boundary that won't hold up against the natural noise present in real-world production data.

**Invariance tests** check that changing a *sensitive* input — race, gender, zip code (often a proxy for race or income) — while holding everything else constant does **not** change the model's output. This is one of the most direct, practical tools for detecting bias before deployment.

**Example:** For a resume-screening model, an invariance test might take a single resume, change only the candidate's name from a traditionally male name to a traditionally female name (everything else held identical), and check whether the model's score for that resume changes. If it does, the model has learned to associate gender with qualification — a clear, actionable signal of bias that needs to be addressed before the model goes anywhere near production, directly connecting back to the fairness and interpretability requirements discussed in Article 1.

### Directional Expectation Tests

These tests check whether changing an input in a way that should obviously, predictably affect the output in a known direction actually does so.

**Example:** For a house price prediction model, increasing the square footage of a house (holding location, age, and everything else constant) should never *decrease* the predicted price. If it does, that's a strong signal that the model has learned some spurious, nonsensical relationship from the training data, rather than a sensible, domain-grounded one — and this kind of error often won't show up in an aggregate accuracy metric at all, since it can be a localized failure affecting only certain input combinations.

**Practical takeaway:** Perturbation, invariance, and directional expectation tests are collectively sometimes called "behavioral testing" — they probe specific, targeted aspects of model behavior the way a thorough QA process probes specific behaviors of traditional software, rather than relying purely on a single aggregate performance number.

### Model Calibration

A model is **well-calibrated** if its predicted probabilities actually match real-world frequencies. If a model predicts a 70% probability of rain across many different days, it should actually rain on roughly 70% of those days — not 40%, and not 95%.

**Why this matters specifically, beyond raw accuracy:** Two models can have identical classification accuracy (using a 0.5 threshold) while having wildly different calibration. A poorly calibrated model that outputs 95% confidence on predictions that are only actually correct 60% of the time is dangerous specifically in any system where the *probability itself* — not just the final classification — is used downstream. 

**Example:** Recall from Article 2 that fraud systems often use a probability score with multiple downstream thresholds — automatically blocking transactions above 0.95 confidence, routing transactions between 0.7 and 0.95 to human review, and allowing everything below 0.7. If the model is poorly calibrated, these threshold-based business decisions are built on meaningless numbers — a transaction scored 0.95 might genuinely only be fraudulent 60% of the time, completely undermining the carefully designed threshold strategy.

**Practical takeaway:** Whenever a model's output probability is used for anything beyond a single binary classification decision — risk-based routing, ranking, threshold-based business rules — calibration needs to be explicitly checked (often visualized using a reliability diagram/calibration curve) and, if necessary, corrected using techniques like Platt scaling or isotonic regression.

### Slice-Based Evaluation

A single overall accuracy number can hide dramatically different performance across different subgroups within your data. **Slice-based evaluation** means deliberately breaking down performance by relevant subgroups — by geography, by device type, by customer segment, by demographic group — rather than relying solely on the aggregate number.

**Why this matters:** A model with strong 92% overall accuracy might secretly be performing at 98% for your largest, majority user segment while performing at a dismal 60% for a smaller subgroup — a serious problem that the aggregate number completely conceals. This is exactly the kind of hidden disparity that the fairness concerns raised in Article 1 and Article 3's invariance tests are designed to catch in practice.

**Simpson's Paradox:** This is a striking and counterintuitive statistical phenomenon where a trend that's clearly visible in aggregated data completely disappears, or even *reverses direction*, when the same data is broken down into its underlying subgroups.

**Example:** Imagine a new model appears to have a higher overall approval rate for loans than the old model — looking like an improvement in access. But when you slice the data by income bracket, the new model actually approves loans at a *lower* rate than the old model within every single income bracket. The apparent overall improvement is an illusion, caused entirely by a shift in the *mix* of applicants across income brackets between the two time periods being compared, not by any genuine change in the model's behavior within any actual subgroup.

**Practical takeaway:** Slice-based evaluation isn't an optional, extra-credit step — it's often the only way to discover that a model is failing a specific, smaller-but-still-important subgroup of your users, a failure that will eventually surface in production regardless of how good the aggregate metric looked during offline evaluation.

---

## Putting It Together: Developing and Evaluating a Customer Churn Model

Here's how the concepts in this article come together in a realistic project:

1. **Model selection:** The team resists the urge to immediately reach for a deep learning architecture (avoiding the state-of-the-art trap) and starts with logistic regression as a baseline, then a gradient-boosted tree model — both trained and tuned with comparable effort to avoid bias in the comparison.
2. **Learning curves:** A learning curve reveals that the gradient-boosted model's validation performance is still climbing meaningfully as more training data is added — signaling that investing in collecting more historical customer data is likely to pay off more than further architecture tuning.
3. **Ensembling:** The final model stacks the gradient-boosted tree with a separate model trained specifically on customer support interaction data, combined via a lightweight meta-learner — since the two base models capture somewhat different signals.
4. **Tracking and versioning:** Every experiment run — hyperparameters, resulting metrics, the specific data version used — is logged via MLflow, and the training dataset itself is versioned via DVC, so any past result can be reproduced exactly months later.
5. **Debugging:** Early in development, the model fails to overfit even a tiny single batch of data — the team discovers a bug where customer IDs were accidentally being shuffled out of alignment with their churn labels during a join, completely scrambling the actual training signal.
6. **Offline evaluation against baselines:** The final model is compared not just in isolation, but explicitly against the zero-rule baseline (always predict "no churn," since churn is a minority class) and against the existing rule-based retention-flagging system the company currently uses — confirming the new model provides genuine, justified improvement over both.
7. **Invariance testing:** An invariance test reveals the model's churn predictions shift meaningfully when a customer's zip code is changed while holding all other features constant — a strong, actionable signal that zip code (acting as a proxy for income or race) is introducing unwanted bias, prompting the team to remove or carefully reweight that feature before deployment.
8. **Calibration check:** Since the business plans to use the model's probability score to prioritize which at-risk customers retention agents call first, the team explicitly checks and confirms calibration — verifying that customers scored at 80% churn risk actually do churn roughly 80% of the time.
9. **Slice-based evaluation:** Breaking down performance by customer tenure reveals the model performs noticeably worse for very new customers (under 3 months tenure) — flagging that a separate, specialized model or feature set may be needed for that specific subgroup, since they simply don't have enough historical behavioral data for the main model to work well.

Notice, again, how much of the real engineering rigor here lives in places that have nothing to do with picking an algorithm — tracking, debugging, baseline comparisons, calibration, and slicing. The actual model training call is, once again, a small fraction of the total effort.

---

## Key Takeaways

- Model selection should be strategic, not driven by hype: avoid the state-of-the-art trap, always start simple, control for your own bias when comparing architectures, use learning curves to guide whether more data or a different model is the better investment, and be explicit about trade-offs and assumptions.
- Ensembling (bagging, boosting, stacking) reliably improves accuracy but adds real serving cost — every additional base model adds latency and maintenance burden.
- Experiment tracking and data/code versioning together are what make ML development reproducible and debuggable; without both, you can't reliably answer "what changed?" months later.
- ML models fail silently, so debugging must be proactive: start simple, try to overfit a single batch to catch implementation bugs early, and fix random seeds to isolate genuine improvements from noise.
- Offline evaluation needs to go far beyond a single accuracy number: compare against meaningful baselines (especially the zero-rule baseline on imbalanced data), test with perturbation/invariance/directional tests, check calibration whenever probabilities feed downstream decisions, and always evaluate performance by slice to catch hidden disparities and avoid being misled by Simpson's paradox.

---

## What's Next

In Article 7, we move from offline evaluation to **deployment** — how models actually get shipped to production, deployment strategies like shadow deployment and canary releases, the difference between online and batch prediction, and how to manage the real-world infrastructure that serves a model's predictions to users.

---

*Machine Learning Systems Series | Article 6 of N*
*Based on "Designing Machine Learning Systems" by Chip Huyen (O'Reilly, 2022)*
