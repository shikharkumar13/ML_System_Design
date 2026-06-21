# Machine Learning Systems
## Article 5: Feature Engineering — Where Most of the Real Skill Lives

*This is the fifth article in the **Machine Learning Systems** series, based on "Designing Machine Learning Systems" by Chip Huyen. Article 4 covered how training data is sampled and labeled. This article covers what happens next: turning that raw, messy data into the actual inputs a model learns from.*

---

If you ask people outside the field what makes a model good, they'll usually say "the algorithm." Ask any practicing ML engineer, and most will tell you something closer to: *"the features."*

There's a well-worn saying in the field — *garbage in, garbage out.* A sophisticated model fed poorly engineered features will lose to a simple model fed well-engineered ones, almost every time. Feature engineering is the unglamorous, detail-obsessed work of turning raw data into signal a model can actually use — and it's where a huge amount of real-world ML skill is concentrated.

This article covers the standard toolkit of feature engineering operations, the single most dangerous failure mode in ML systems (data leakage), and how to actually evaluate whether a feature is pulling its weight or just adding cost and risk.

---

## Learned Features vs. Engineered Features

Deep learning is often described as "feature learning" — instead of a human manually deciding what matters, a neural network learns its own internal representations directly from raw pixels, raw audio, or raw text. This is genuinely one of deep learning's biggest contributions: in domains like computer vision and NLP, models now learn hierarchies of features (edges → shapes → objects, or characters → words → meaning) that used to require painstaking manual engineering.

So does this mean feature engineering is obsolete? **No — and this is a common misconception among people who learned ML primarily through deep learning courses.**

Here's why manual feature engineering remains essential in production:

1. **Most production ML systems are not deep learning systems.** Gradient-boosted trees (XGBoost, LightGBM) and other classical models still power a huge share of production use cases — fraud detection, credit scoring, recommendation ranking, ad click prediction — because they're faster to train, cheaper to serve, easier to interpret, and often perform just as well or better on structured/tabular data. These models don't learn features automatically; they rely entirely on the features you give them.
2. **Even deep learning systems benefit from engineered features.** A recommendation system using a neural network will still perform better if you hand it a well-constructed feature like "days since user's last purchase" rather than forcing the network to rediscover that pattern from raw timestamps.
3. **Domain knowledge is a shortcut the model can't discover on its own.** A fraud analyst knows that "transaction amount relative to the customer's typical spending" is a meaningful signal. A model starting from raw transaction amounts alone would need an enormous amount of data to rediscover this relationship — engineering the feature directly hands the model a shortcut grounded in real domain expertise.

**Practical takeaway:** The honest framing is that learned features and engineered features are complementary, not competing. Even the most advanced production systems typically combine both — feeding engineered features alongside raw inputs, or using engineered features as inputs to a deep learning model rather than relying purely on raw, unprocessed data.

---

## Common Feature Engineering Operations

These are the bread-and-butter techniques every practicing ML engineer applies repeatedly. They sound simple individually, but getting them right — and consistent between training and serving — is where much of the real engineering effort goes.

### Handling Missing Values

Real-world data is full of holes. A sensor goes offline. A user skips an optional form field. An upstream system fails to populate a field for a subset of records. You have two broad strategies:

**Deletion:**
- *Row deletion:* Remove any row (sample) that has a missing value in a relevant column. Simple, but if missingness is common, you can lose a large fraction of your dataset — and worse, if the *reason* data is missing correlates with the outcome you're predicting (e.g., users who are about to churn are less likely to fill out a satisfaction survey), deleting those rows introduces systematic bias into what remains.
- *Column deletion:* Remove an entire feature if too many of its values are missing. Useful when a feature is mostly useless anyway, but risks throwing away a feature that's only sparse for a specific, meaningful subpopulation.

**Imputation:** Fill in the missing value rather than removing the row.
- *Default values:* Filling with a constant like 0 or "unknown." Simple, but can introduce a misleading signal if 0 is also a legitimate, meaningful value for that feature elsewhere (e.g., filling a missing "number of prior purchases" with 0 makes a new customer indistinguishable from a customer whose purchase history failed to load).
- *Mean/median imputation:* Filling numeric values with the column's average or median. Common but flattens variance and can distort relationships, especially if missingness isn't random.

**A subtle production trap:** Imputation statistics (mean, median) must be computed from the *training set only*, then applied consistently to validation, test, and live production data. Recomputing the mean using the full dataset (including the test set) — or worse, recomputing it fresh every time new production data arrives — is a quiet but serious bug that introduces inconsistency between what the model was trained on and what it sees at serving time.

### Scaling

Most ML algorithms (with the notable exception of tree-based models) are sensitive to the *scale* of input features. If one feature ranges from 0–1 and another ranges from 0–1,000,000, gradient-based models will implicitly treat the second feature as far more important — not because it's more predictive, but purely because of its numeric magnitude.

**Common techniques:**
- **Min-max scaling (normalization):** Rescale values to a fixed range, typically 0 to 1, using the minimum and maximum observed values.
- **Standardization:** Transform values to have zero mean and unit variance (subtract the mean, divide by the standard deviation). This is the most commonly used default in production pipelines because it's less sensitive to outliers than min-max scaling.
- **Log transformation:** Apply a logarithm to heavily skewed features (like income, or transaction amount, which often have a long right tail with a few extremely large values). This compresses the scale of extreme values and often makes the feature's distribution closer to normal, which many models handle better.

**The same production trap applies here as with imputation:** scaling parameters (min, max, mean, standard deviation) must be calculated on the training set and then applied — not recalculated — on validation, test, and live data. This connects directly to the data leakage discussion below; it's one of the most common ways leakage quietly creeps into a pipeline.

### Discretization (Binning)

This converts a continuous or fine-grained feature into discrete buckets — for example, converting exact age (23, 24, 25...) into age groups (18–25, 26–35, 36–50, 50+), or converting exact income into income brackets.

**Why this helps:** A model — especially a simpler one like logistic regression — can struggle to find a smooth, accurate relationship across a wide continuous range, especially with limited data. Binning reduces the problem to learning a coefficient per bucket, which can generalize better when the relationship between the raw value and the outcome isn't actually linear (e.g., the spending behavior of a 24-year-old and a 26-year-old might be nearly identical, but a 24-year-old and a 54-year-old might be very different — exact age as a raw number doesn't capture this naturally).

**Tradeoff to be aware of:** Binning throws away precision. Two customers at the boundary of a bin (income of $49,999 and $50,001) might be treated identically to someone deep within the bin, while being treated very differently from someone just one dollar on the other side of the boundary. The choice of bin boundaries itself becomes a design decision that can meaningfully affect model behavior.

### Encoding Categorical Features

Categorical features (brand name, product category, country) need to be converted into a numeric form models can process. The standard approaches — one-hot encoding, label encoding — work fine when your set of categories is fixed and known in advance.

**The real production problem: categories change over time.** New brands join an e-commerce platform. New product SKUs are added daily. New merchant IDs appear in a payments system. If you one-hot encode based on the categories present at training time, your system has no defined behavior when a brand-new category shows up in production — at best it's ignored, at worst the system errors out or crashes.

**The hashing trick:** Instead of maintaining an explicit, ever-growing list of every category and updating it, you pass each category through a hash function that maps it to a fixed-size index space (say, 10,000 buckets), regardless of how many actual distinct categories exist or will ever exist.

**Why this works well in production:**
- The feature space size is fixed in advance — you never need to retrain your model's input layer just because a new brand was added.
- New, previously unseen categories are handled gracefully — they simply hash into one of the existing buckets rather than causing an error.
- It requires no ongoing maintenance of a category vocabulary list.

**The tradeoff:** Hash collisions — two different categories hashing to the same bucket — mean the model can't perfectly distinguish between them. In practice, with a sufficiently large hash space relative to the number of true categories, the impact of collisions is usually small and an acceptable price for the robustness gained. This is precisely why the hashing trick is heavily used at large tech companies with constantly evolving category sets (think: every advertiser ID, every product ID, every merchant ID on a large platform).

### Feature Crossing

This combines two or more existing features into a new one, in order to help a model capture interaction effects between them.

**Example:** Suppose you're predicting whether someone will buy winter coats. "Month" alone and "Location" alone might each be weakly predictive. But the *combination* — "Location = Minnesota AND Month = December" — is a much stronger signal than either feature individually. A feature cross like `location_x_month` explicitly hands the model this combined signal.

**Why this matters especially for simpler models:** Linear models (like logistic regression) can't naturally learn interaction effects between features on their own — they treat each feature's contribution independently. Feature crosses are a way of manually injecting nonlinearity into an otherwise linear model. Tree-based models can discover some interactions automatically through successive splits, but explicit crosses can still help them learn the relationship faster and more reliably with less data.

**The major tradeoff:** Crossing two categorical features multiplies their cardinalities together. Crossing a feature with 1,000 possible values with another feature with 1,000 possible values creates a new feature space with up to 1,000,000 possible combinations — most of which will have very few or zero training examples, leading straight back to a sparse, hard-to-learn feature space. Feature crossing is powerful but needs to be applied selectively, often combined with the hashing trick to keep the resulting space manageable.

### Positional Embeddings

For sequence data — text, time series, video frames — the *order* of elements often carries meaning that needs to be made explicit to the model. "The dog bit the man" and "the man bit the dog" contain identical words but mean very different things; the order is the signal.

Architectures like Transformers (heavily used in modern NLP and increasingly in computer vision) don't have any inherent sense of sequence order built into their core mechanism — unlike, say, a recurrent neural network, which processes inputs one step at a time and naturally has an implicit notion of order. So sequence order has to be explicitly injected as a feature: a **positional embedding**.

**Two main approaches:**
- **Learned positional embeddings:** The model learns a vector representation for each position (position 1, position 2, position 3...) during training, the same way it learns word or pixel embeddings. Flexible, but limited to sequence lengths seen during training.
- **Fixed positional embeddings:** Use a predetermined mathematical function — most famously, sine and cosine functions of different frequencies, as used in the original Transformer architecture — to generate position representations. These don't need to be learned and can, in principle, generalize to sequence lengths longer than what the model saw during training.

**Practical takeaway:** You don't need to implement positional embeddings from scratch in most production work — they're built into the architectures and libraries you'll use (Hugging Face Transformers, for example). But understanding *why* they exist matters: it explains why simply feeding a "bag of features" into certain architectures without explicit ordering information would silently destroy a huge amount of signal in any sequential or time-based data.

---

## Data Leakage: The Silent Killer of Production Models

If there's one concept from this entire article worth tattooing on your forehead before you ship anything to production, it's this one.

**Data leakage** happens when information that wouldn't actually be available at prediction time in the real world somehow makes its way into your training features — usually through some form of information about the label itself. The model learns to exploit this leaked information, performs beautifully in development and offline evaluation, and then falls apart in production, because that leaked information simply isn't there when the model has to make a real, live prediction.

This is, by a wide margin, one of the most common reasons a model that looked great in a notebook fails embarrassingly in production. It's also one of the hardest bugs to catch, because nothing in your code *crashes* — the model just quietly cheated during training.

### Common Causes

**1. Splitting time-correlated data randomly instead of chronologically.**

This is probably the single most common form of leakage in production ML. If your data has a time dimension (which most production data does), and you randomly shuffle it before splitting into train/test, you can end up training on data from *after* the point in time represented by some of your test examples — effectively letting the model "see the future" during training.

**Example:** You're predicting stock price movement and you randomly split five years of daily data into train/test. Your model might be trained on data from March 2024 and tested on data from January 2024 — meaning the model implicitly had access to patterns and market regimes that hadn't happened yet, relative to some of its test examples. The fix: always split time-series data **chronologically** — train on the past, test on the future, mimicking exactly how the model will actually be used in production.

**2. Scaling or imputing using statistics from the entire dataset (including test data).**

As mentioned above, if you compute the mean, standard deviation, min, or max for scaling/imputation using the *full* dataset before splitting into train/test, information about the test set's distribution leaks into the values used to transform the training set. The model gets a subtle, unfair hint about the test data's characteristics. The fix: compute these statistics **only** from the training set, then apply them, unchanged, to validation and test data.

**3. Handling missing values using global statistics.**

A close cousin of the above — imputing missing values using a mean or median calculated across the entire dataset (rather than the training split alone) leaks the same kind of information.

**4. Data duplication.**

If duplicate or near-duplicate records exist in your dataset and a duplicate ends up in both the training and test sets, the model can essentially "memorize" the answer for that exact record rather than learning a generalizable pattern — inflating your test performance in a way that won't hold up on genuinely new data in production.

**5. Group leakage.**

This occurs when multiple data points belonging to the same underlying entity or group end up split across both the train and test sets. 

**Example:** You're building a model to detect pneumonia from chest X-rays, and a single patient contributed three X-rays taken on different days, all in your dataset. If two of those X-rays land in the training set and one lands in the test set, the model may partially learn to recognize *that specific patient's* anatomy (bone structure, chest shape) rather than the general visual signature of pneumonia — inflating test performance in a way that won't generalize to new patients. The fix: split by *group* (patient, user, household) rather than by individual record, ensuring all of one entity's data stays entirely within one split.

### Detection: How to Actually Catch Leakage

Leakage rarely announces itself with an error message — it shows up as suspiciously, almost impossibly good model performance. Here's how practicing engineers hunt for it:

**1. Measure each feature's correlation with the label.** If a single feature has an unusually high, almost suspicious correlation with the target — especially one with no clear causal or domain-grounded reason for being so predictive — investigate it closely before trusting it. A feature that "predicts" loan default with 98% accuracy on its own is a major red flag, not a stroke of luck.

**2. Perform ablation studies.** Remove a feature (or small group of suspected features) and retrain. If model performance collapses dramatically when that one feature is removed, it's worth asking *why* that single feature carries so much weight — sometimes it's a genuinely powerful, legitimate signal, but often it's a leak.

**3. Strictly never use the test split for any tuning decisions.** This sounds obvious but is violated constantly in real projects. If you tune hyperparameters, select features, or make architecture decisions based on test set performance — even just glancing at it "one more time" before finalizing — you've effectively leaked test set information into your modeling decisions, even without touching the raw data at all. This is why a proper three-way split (train / validation / test) exists: validation is for all your iterative tuning, and the test set is touched exactly once, at the very end, as a final, honest check.

**Practical takeaway:** Whenever a model's offline performance looks unusually, almost too-good-to-be-true, the very first hypothesis to investigate isn't "we built a great model" — it's "we have a leak somewhere." Experienced engineers treat suspiciously excellent results with more suspicion than mediocre ones.

---

## Engineering Good Features: Knowing When to Stop

Once you understand the toolkit, a natural instinct kicks in: *add more features.* More information should mean a better model, right?

**Not necessarily — and this is a crucial production lesson.** Every feature you add comes with real costs:

- **Overfitting risk:** More features, especially with limited training data, increase the risk that the model learns spurious patterns specific to your training set rather than genuinely generalizable signal.
- **Latency cost:** Every feature needs to be computed at prediction time. A feature that requires an expensive database join or a slow external API call adds real, measurable milliseconds to your serving latency — and in a system serving millions of requests, that adds up to real infrastructure cost and a worse user experience.
- **Technical debt:** Every feature is a pipeline that needs to be built, monitored, and maintained — for both training and serving — indefinitely. A feature that contributes almost nothing to model performance, but still needs to be kept alive in the data pipeline, is dead weight that someone has to maintain forever.

So before adding (or keeping) a feature, you need to evaluate it along two distinct dimensions.

### Feature Importance

This measures how much a given feature actually contributes to the model's predictive performance.

**Built-in importance measures:** Many classical ML algorithms expose this directly. Tree-based models (like XGBoost or Random Forest) can report how much each feature reduced impurity across all the splits where it was used, giving a relative importance ranking essentially for free.

**Model-agnostic tools — SHAP:** For models where built-in importance isn't available, or where you want a more rigorous, theoretically grounded measure, tools like **SHAP (SHapley Additive exPlanations)** are the industry standard. SHAP is based on a concept from cooperative game theory: it calculates each feature's average marginal contribution to a prediction, across all possible combinations (coalitions) of features, giving a fair, consistent attribution of "credit" to each feature for a given prediction.

**Why SHAP matters beyond just ranking features:** SHAP doesn't just tell you that "transaction amount" is important overall — it can tell you *for this specific prediction*, how much each feature pushed the output up or down. This is hugely valuable for debugging individual wrong predictions ("why did the model flag this specific transaction as fraud?") and for explaining model decisions to non-technical stakeholders, regulators, or auditors — connecting directly back to the interpretability requirements discussed in Article 1.

**Practical takeaway:** Run feature importance analysis regularly, not just once at the end of model development. Features that were important six months ago may have become irrelevant as user behavior or the underlying data distribution shifts — feature importance should be part of your ongoing model monitoring, not a one-time check.

### Feature Generalization

A feature can be highly important on your current training and test data and still be a liability if it doesn't generalize well to new, unseen production data. Two things to check:

**Coverage:** What percentage of your samples actually have a non-missing value for this feature? A feature that's only populated for 5% of your data is, by definition, going to provide a meaningful signal for only a small slice of your predictions — and worse, if that 5% isn't randomly distributed (e.g., the feature is only available for users who completed an optional profile step), the feature might be implicitly encoding something else entirely (like "user engagement level") rather than what it appears to represent on the surface.

**Distribution across splits:** Compare the distribution of a feature's values between your training set and your test/production data. If a feature's distribution looks meaningfully different between training and the real world it will be deployed into, the patterns the model learned about that feature during training may simply not hold up in production. This is an early, proactive way to catch the kind of **distribution shift** discussed in Article 1 — checking for it *before* deployment, not waiting to discover it through degraded performance months later.

**Practical takeaway:** A genuinely good production feature scores well on both dimensions simultaneously — it's both *important* (it meaningfully improves predictions) and *generalizable* (it has solid coverage and a stable distribution that will hold up on new data). A feature that's important but doesn't generalize is fragile and will eventually hurt you. A feature that generalizes well but isn't actually important is just unnecessary weight on your pipeline. You want both.

---

## Putting It Together: Feature Engineering for a Loan Default Model

Here's how the concepts in this article come together in a realistic project:

1. **Learned vs. engineered:** The team uses a gradient-boosted tree model (XGBoost), not deep learning, since the data is tabular — meaning every single feature must be manually engineered; there's no automatic representation learning happening here.
2. **Missing values:** "Years at current employer" is missing for 8% of applicants (often self-employed individuals who don't have a single "employer"). Rather than deleting these rows, the team imputes a default category of "self-employed/other" — preserving the rows while encoding the *reason* for missingness as useful information in its own right, rather than just filling in an arbitrary number.
3. **Scaling:** "Annual income" is heavily right-skewed (a small number of applicants have very high incomes). The team applies a log transformation before standardizing, computing the scaling statistics from the training set only.
4. **Discretization:** "Credit score" (a continuous 300–850 range) is binned into standard industry risk tiers (poor, fair, good, excellent) alongside being kept as a raw continuous feature — giving the tree-based model both a smoothed and a precise version to split on.
5. **Encoding categoricals:** "Employer name" has tens of thousands of distinct values and grows continuously as new applicants list new employers. The team uses the hashing trick rather than maintaining an ever-growing vocabulary list.
6. **Feature crossing:** The team creates a cross between "loan purpose" and "loan amount bracket," since a $40,000 loan for debt consolidation carries different risk than a $40,000 loan for a home renovation, a distinction neither feature captures alone.
7. **Leakage check:** During development, a feature called "days since last payment reminder sent" turns out to be suspiciously predictive of default. Investigation reveals payment reminders are only sent *after* a payment is already late — meaning this feature was indirectly encoding the label itself. It's removed before the model goes anywhere near production.
8. **Feature evaluation:** Using SHAP, the team finds that "debt-to-income ratio" is consistently the single most important feature across predictions, while a feature for "number of credit inquiries in the last 6 months" has decent importance but only 60% coverage and a noticeably different distribution between training data (collected pre-pandemic) and current applications — flagging it for closer monitoring before being trusted in production.

Notice that almost none of this work involves the model itself. The XGBoost call at the end of this process is a single line of code. Everything that precedes it — the careful, detail-obsessed feature work — is where the actual engineering effort, and the actual model quality, comes from.

---

## Key Takeaways

- Engineered features remain essential even in the age of deep learning — most production systems are not pure deep learning systems, and even those that are still benefit from well-constructed, domain-informed features.
- The standard toolkit — handling missing values, scaling, binning, encoding, feature crossing, positional embeddings — sounds simple but requires careful, consistent implementation between training and serving to avoid subtle bugs.
- Data leakage is one of the most common and dangerous failure modes in ML systems: it makes a model look excellent offline and fail in production. The most frequent cause is random (rather than chronological) splitting of time-correlated data, and the best detection strategy is treating suspiciously great results with suspicion, not celebration.
- More features are not automatically better — every feature adds overfitting risk, serving latency, and long-term maintenance burden.
- A genuinely good feature must be evaluated on two axes: feature importance (does it actually improve predictions, measurable via built-in importance scores or SHAP) and feature generalization (does it have solid coverage and a stable distribution that will hold up in production, not just in your training set).

---

## What's Next

In Article 6, we shift from preparing data to **building models** — how to approach model selection in production (and why the "best" model on a leaderboard often isn't the right choice), the role of baselines and ensembling, and how experiment tracking and debugging actually work on real ML teams.

---

*Machine Learning Systems Series | Article 5 of N*
*Based on "Designing Machine Learning Systems" by Chip Huyen (O'Reilly, 2022)*
