# Machine Learning Systems
## Article 10: ML Infrastructure — The Layers That Make Everything Else Possible

*This is the tenth article in the **Machine Learning Systems** series, based on "Designing Machine Learning Systems" by Chip Huyen. Articles 1–9 covered the conceptual and engineering work of building, deploying, monitoring, and continually updating ML systems. This article steps back to look at the infrastructure layers underneath all of it — the storage, compute, environments, and platforms that make every previous article's advice actually executable at scale.*

---

Every technique discussed so far in this series — feature stores, shadow deployment, continual learning, distributed training — needs somewhere to actually run. This article is about that "somewhere": the layered infrastructure stack that sits underneath every production ML system, from the raw compute and storage at the bottom, up through the shared ML platform that lets an entire organization build and ship models without each team reinventing the same wheel.

A useful way to think about this article: if Articles 1–9 were about *what* to do, this one is about *what you build it on top of.*

---

## The Spectrum of ML Infrastructure

A common mistake — especially among engineers coming from a learning or research background — is to assume there's a single "correct" ML infrastructure stack that every serious company should be running. In reality, the right amount of infrastructure investment varies enormously, and depends almost entirely on production scale and organizational maturity.

**At one end of the spectrum:** a simple prototype, a one-off analysis, or an early proof-of-concept might genuinely need no dedicated infrastructure at all — a single engineer running a notebook on their laptop is a perfectly reasonable starting point.

**At the other end:** companies operating at the scale of Google, Meta, or Tesla — running thousands of models, training on petabytes of data, serving billions of predictions a day — build highly specialized, custom infrastructure, often developing internal tools that don't exist anywhere else, because off-the-shelf solutions simply weren't designed for problems at that scale.

**Where most companies actually fall:** Somewhere comfortably in the middle. They need what's often called **generalized infrastructure** — infrastructure robust enough to support several common ML applications operating at a genuinely meaningful, real production scale, without requiring the kind of bespoke, deeply custom engineering investment that only a handful of companies in the world can justify.

**Practical takeaway:** Before investing in any specific piece of infrastructure discussed in this article, the first question worth asking honestly is: *where on this spectrum does our company actually sit?* Over-investing in infrastructure built for Google-scale problems, when you're a 50-person startup with two models in production, is just as much of a mistake as under-investing and leaving your three-person ML team manually doing work that off-the-shelf tools could automate.

This generalized infrastructure is typically organized into four layers, each building on the one below it. We'll go through them from the ground up.

---

## Layer 1: Storage and Compute

This is the foundation everything else sits on — quite literally where your data lives, and what actually executes your code.

### Storage

This is the layer responsible for where data is collected, kept, and made accessible — the data lakes, data warehouses, and databases discussed extensively in Article 3.

**The good news here:** Storage has become heavily **commoditized**. Most companies today don't build their own storage systems from scratch — they rely on mature, reliable, well-understood cloud offerings: object/blob storage services (like Amazon S3) for raw, unstructured data, and managed cloud data warehouses (like Snowflake, BigQuery, or Redshift) for structured, query-optimized data.

**Practical takeaway:** Storage is rarely the layer where companies need to innovate or differentiate. The strategic decisions here are mostly about *which* commoditized provider to use, how to structure and organize data within it (connecting back to the data modeling discussion in Article 3), and cost management — not about building custom storage infrastructure from the ground up.

### Compute

This is the actual processing power that executes your jobs — data processing pipelines, model training runs, and serving infrastructure.

**A common, important misconception:** People often assume that more raw computational power, frequently measured in FLOPS (floating-point operations per second), is the most important number to optimize for. In practice, for most real ML workloads, **memory and the number of available cores** matter just as much, often more, than raw FLOPS — a chip with extremely high theoretical FLOPS but insufficient memory to hold your model and data simultaneously won't actually be usable for your workload at all, regardless of its theoretical peak performance.

**Practical takeaway:** When evaluating compute options, look past headline FLOPS numbers and ask concrete, workload-specific questions: Does this configuration have enough memory to hold the model and batch size you actually need? Does it have enough cores to parallelize your specific data processing pipeline effectively? Raw computational power that can't actually be applied to your specific bottleneck is a misleading metric to optimize around.

### Public Cloud vs. Private Data Centers

This is one of the more consequential, expensive decisions a growing ML organization makes, and it's not a permanent, one-time choice — many companies revisit it as they scale.

**The case for public cloud:** **Elasticity** is the headline advantage. Model training, in particular, is often extremely bursty — you might need a massive surge of compute for a few hours or days while training a large model, and then need almost none of that capacity for weeks afterward. Public cloud lets you pay only for what you use during that surge, rather than owning and maintaining expensive hardware that sits mostly idle the rest of the time.

**The case against — and "cloud repatriation":** As companies scale, the ongoing cost of running large, sustained workloads on public cloud infrastructure can become substantial enough to meaningfully impact profit margins — cloud providers' pricing is structured to be attractive at small to moderate scale, but the economics shift considerably once a company's compute usage becomes large and sustained rather than bursty and occasional. This has led to a real, increasingly discussed trend called **cloud repatriation** — companies that had moved significant workloads to public cloud providers deliberately moving some of them back to their own privately-owned and operated data centers, specifically once their scale and usage patterns make owning the hardware outright more cost-effective than continuing to rent it.

**Practical takeaway:** The "cloud is always cheaper/easier" assumption that's common among smaller companies and individual engineers doesn't hold uniformly at every scale. The right answer genuinely changes as a company grows — cloud's elasticity is most valuable for unpredictable, bursty, or rapidly scaling workloads, while owned infrastructure becomes increasingly attractive for large, stable, predictable workloads where the elasticity premium you're paying for stops being worth it.

---

## Layer 2: Development Environment

This is, somewhat surprisingly, one of the most consistently **underinvested** layers across the entire industry, despite having an outsized, direct impact on how productive engineers actually are day to day.

The **development environment** ("dev environment") is where ML engineers and data scientists actually write code, run experiments, and iterate — their editor, their local Python environment, their notebook server, the specific versions of every library and dependency they're using.

### Standardization

A surprising number of real, time-consuming bugs in ML projects aren't actually bugs in the model or the data at all — they're caused simply by two different developers (or a developer's laptop versus the production server) running slightly different package versions, different operating systems, or different underlying system configurations, leading to code that behaves subtly differently — or breaks outright — depending purely on whose machine it's running on.

**The increasingly common fix:** Moving away from a model where every individual engineer maintains their own, inevitably slightly different local development setup, toward standardized, often cloud-based development environments (cloud-hosted IDEs, shared development servers) where every engineer on a team works within a genuinely identical, centrally maintained, and version-controlled environment.

**Practical takeaway:** This might sound like a minor convenience, but the time cost of debugging "works on my machine" issues, multiplied across an entire team over the lifetime of a project, is often a significant, underappreciated source of wasted engineering effort — investing in genuine environment standardization tends to pay for itself surprisingly quickly.

### Containers

**Production environments are fundamentally different from a developer's local laptop** in a specific, important way: they're typically **stateless** (any given server instance can be destroyed and replaced at any moment without anything important being lost, because nothing important is stored on that specific instance) and need to **scale dynamically** (the number of running instances needs to flexibly grow and shrink based on real-time traffic, rather than staying fixed).

**Containers** (with **Docker** being the dominant, industry-standard tool) solve a specific, crucial problem this creates: how do you reliably take a development environment — with its exact specific code, dependencies, and configuration — and recreate that *exact same* environment, reliably and reproducibly, on any new server instance that gets spun up, without manually reinstalling and reconfiguring everything by hand each time?

**How to think about it:** A container is essentially a self-contained, portable "recipe" — a precise, version-controlled specification of an entire environment (code, dependencies, system libraries, configuration) — that can be used to spin up an identical, working instance of that environment on virtually any underlying machine, with extremely high confidence that it will behave exactly the same way it did during development and testing.

**Orchestrating containers at scale — Kubernetes:** Running a single container is straightforward. Running, coordinating, monitoring, and dynamically scaling **hundreds or thousands** of containers simultaneously — across many underlying physical or virtual machines, handling failures gracefully, and routing traffic appropriately — is a substantially harder problem. **Kubernetes** has become the de facto industry-standard tool for exactly this kind of large-scale container orchestration, and it's worth recognizing by name because it shows up constantly across real-world ML infrastructure discussions, even though most individual ML engineers won't need to configure it deeply themselves.

**Practical takeaway:** Containers are what make the "build once, deploy anywhere reliably" promise of modern ML infrastructure actually achievable in practice — connecting directly back to the deployment reliability and canary release discussions in Articles 2, 7, and 9, all of which implicitly depend on being able to reliably and consistently recreate a tested environment in production.

---

## Layer 3: Resource Management

ML workflows are rarely a single, isolated step. They're typically a whole sequence of interdependent steps — extract data, validate it, engineer features, train a model, evaluate it, deploy it — each of which depends on the previous step having completed successfully. This dependency structure is commonly represented as a **Directed Acyclic Graph (DAG)**: a graph where each node is a step in the workflow, edges represent dependencies between steps, and the "acyclic" part guarantees there are no circular dependencies that would make the workflow impossible to execute in any sensible order.

Managing the execution of these multi-step, dependency-laden workflows reliably and efficiently is the job of this layer.

### Cron vs. Schedulers vs. Orchestrators

These three terms are sometimes used loosely or interchangeably, but they describe genuinely distinct levels of capability, and understanding the distinction clarifies why more sophisticated tools were built on top of simpler ones.

**Cron:** The simplest, most familiar option — runs a specified job at fixed, predetermined times (every day at 2am, every hour, etc.). **Critical limitation:** cron has no real concept of dependencies between jobs. It can't natively express or enforce "only run step B after step A has successfully completed" — it just blindly fires off whatever's scheduled at whatever time was specified, regardless of whether any prerequisite work is actually done yet.

**Schedulers** (a tool like **Slurm** is a commonly cited example, particularly common in research and HPC computing contexts): These go a meaningful step further than cron by explicitly understanding and managing the **dependencies** between different jobs in a workflow — determining the correct *order* and *timing* for when each individual job should actually run, based on whether its prerequisite steps have genuinely completed successfully yet, rather than relying purely on a fixed clock time.

**Orchestrators** (with **Kubernetes**, introduced above, being the standard example): While schedulers focus on *when* and in what *order* jobs should run, orchestrators focus on a related but distinct question: *where* should the actual compute resources needed to run each job come from? Orchestrators handle provisioning, allocating, and managing the underlying compute resources (which specific machines, how much memory and CPU) that scheduled jobs actually need in order to execute.

**Practical takeaway:** In casual conversation, "scheduler" and "orchestrator" often get used somewhat interchangeably, and in practice modern tools frequently blend both responsibilities together. But understanding the conceptual distinction — *when/in what order* (scheduling) versus *where do the resources come from* (orchestration) — helps make sense of why the ML workflow tools discussed next typically need to handle both concerns simultaneously, rather than being purely one or the other.

### Data Science Workflow Tools

A category of tools has emerged specifically to combine scheduling and orchestration capabilities in a way that's purpose-built for the particular shape of data science and ML workflows (rather than being generic, general-purpose job schedulers adapted after the fact).

**Examples:** **Airflow**, **Argo**, **Kubeflow**, and **Metaflow** are commonly cited tools in this space. Each lets engineers define a multi-step ML workflow (often explicitly as a DAG, as described above) — covering steps like data extraction, validation, feature computation, training, and evaluation — and then handles reliably scheduling, executing, monitoring, and retrying that entire workflow, often with built-in support for ML-specific concerns like experiment tracking and artifact versioning (connecting back to Article 6) baked directly into the tool itself.

**Practical takeaway:** These tools exist specifically because generic scheduling/orchestration tools, while capable, weren't originally designed with the specific, recurring patterns of ML workflows in mind (versioned datasets, model artifacts, experiment tracking, the train→evaluate→deploy sequence). Most teams reaching a meaningful production scale eventually adopt one of these specialized tools rather than building and maintaining custom orchestration logic from scratch on top of more generic, lower-level building blocks.

---

## Layer 4: ML Platform

As an organization's ML usage matures and grows beyond a single team or a single model, a natural and important question emerges: how do we avoid every individual team rebuilding the same underlying infrastructure — deployment pipelines, model storage, feature computation — independently, redundantly, and often inconsistently, from scratch?

The answer many organizations converge on is building (or buying) a shared **ML Platform** — centralized infrastructure and tooling specifically designed to be used across many different ML applications and teams within the same organization, rather than being built bespoke for just one specific project.

### Model Deployment

Tools in this category (examples include **SageMaker** and **Seldon**) handle the genuinely substantial engineering work of taking a trained model and actually getting it safely into production — packaging the model and its dependencies, pushing it to a production serving environment, and exposing it through a well-defined endpoint that other systems can call to get either online (real-time) or batch predictions, directly building on the batch vs. online prediction concepts discussed in Article 7.

**Why this is valuable as shared infrastructure rather than something each team builds individually:** The core mechanics of "take a trained model, package it reliably, and expose it as a working prediction endpoint" are largely the same regardless of which specific team or which specific model is involved — making this a natural, high-leverage candidate for centralized, shared tooling rather than redundant, team-by-team reimplementation.

### Model Store

It's tempting to think that "storing a model" just means saving the trained model's binary weights file somewhere accessible. In practice, this is nowhere near sufficient for a model to be genuinely reproducible, debuggable, or maintainable over time.

A proper **model store** (with **MLflow**, introduced earlier in this series, being a commonly cited example) needs to track substantially more than just the raw weights:

- **The model's definition** (its architecture/code)
- **Its learned parameters** (the actual trained weights)
- **Its dependencies** (the exact library versions and environment it was trained and is expected to run within)
- **The code used to generate it** (the specific training script and configuration used)
- **Data lineage** (precisely which version of which dataset it was trained on, connecting directly back to the data versioning discussion in Article 6)

**Why all of this matters, not just the weights:** When a model fails or behaves unexpectedly in production months after it was deployed, debugging that failure effectively requires being able to answer — with confidence, not guesswork — exactly what code, what data, and what configuration actually produced this specific model. A model store that only saved the raw weights file would leave a team essentially unable to reliably reconstruct or debug that history at all, recreating exactly the kind of irreproducibility problem discussed in Article 6.

### Feature Store

This is one of the most consequential and most frequently referenced pieces of shared ML infrastructure across this entire series, and it's worth being precise about what problem it actually solves. A **feature store** (examples include **Feast** and **Tecton**) addresses three distinct problems simultaneously:

**1. Sharing and discovering features across teams.** Without a centralized feature store, it's common for multiple different teams across an organization to independently, redundantly compute very similar or even literally identical features — "average order value over 30 days" might get separately implemented (and subtly differently defined) by three different teams who have no visibility into each other's work. A feature store provides a centralized, searchable catalog where teams can discover and reuse features that have already been built and validated by others, rather than constantly reinventing the same wheel.

**2. Storing expensive feature computations.** Some features are genuinely expensive to compute — requiring complex joins across large datasets, or aggregations over long historical windows. Recomputing these from scratch every single time they're needed, across every team and every model that uses them, is wasteful. A feature store precomputes and caches these expensive features centrally, so they can be efficiently retrieved and reused rather than redundantly recalculated.

**3. Ensuring feature consistency between training and serving.** This is, arguably, the single most critical problem a feature store solves, and it directly addresses the **training-serving skew** problem that's been a recurring theme across this entire series (introduced in Article 3, and revisited in Articles 7 and 8). Without a feature store, it's common — almost inevitable at sufficient scale — for the batch pipeline computing features for *training* and the separate streaming pipeline computing the "same" features for real-time *serving* to be implemented independently, by different code, sometimes by different teams, and to subtly, quietly diverge from each other over time. A feature store provides a single, authoritative, consistently-computed definition of each feature, used identically by both the training pipeline and the serving pipeline, directly eliminating this entire category of silent, hard-to-debug production failure by construction rather than by hoping two independently-maintained implementations happen to stay in sync.

**Practical takeaway:** Given how many times training-serving skew has come up across this series as a cause of mysterious production failures, it's worth being explicit: a feature store isn't a "nice to have" convenience tool — for any organization running multiple ML models that share underlying features, it's one of the highest-leverage single pieces of infrastructure investment available specifically for eliminating an entire recurring, hard-to-debug category of production bugs at its root.

---

## Build vs. Buy

Once a team understands the four layers above, a practical, recurring question follows for almost every single piece: should we build this ourselves, in-house, or should we buy/adopt an existing vendor or open-source solution?

This isn't a question with a single universal answer — the right call depends on a combination of several real factors, each worth weighing explicitly:

**Company stage:** An early-stage startup with limited engineering resources and an urgent need to ship quickly is almost always better served by adopting existing, off-the-shelf tools wherever reasonably possible, rather than spending scarce engineering time building custom infrastructure that mature, well-tested solutions already provide. A large, well-resourced company operating at a scale where existing tools genuinely don't meet their specific needs has a much stronger case for building custom infrastructure.

**Whether the infrastructure is a core competitive advantage.** If a particular piece of infrastructure is directly, meaningfully tied to what actually makes the company's product genuinely differentiated and competitive in its market, investing in building and owning it in-house — rather than relying on the same shared, generic vendor tooling that competitors might also be using — can be a justified strategic investment. If it's genuinely undifferentiated, generic infrastructure that provides no real competitive advantage regardless of who built it (most teams' approach to basic model storage, for instance), there's usually little strategic upside to reinventing it from scratch in-house rather than adopting an existing solution.

**The maturity of available tools.** For some of the layers and problems discussed in this article, mature, well-tested, widely-adopted vendor or open-source solutions genuinely exist and work well — adopting them is usually the more pragmatic, lower-risk choice. For more novel, less-established needs where the available tooling is still genuinely immature, unreliable, or doesn't cover the specific use case well, building a custom solution in-house may be the only realistic, viable path forward, even if it's not anyone's first preference.

**Practical takeaway:** The build-vs-buy decision should be revisited periodically as a company evolves, not treated as a permanent, one-time choice. A reasonable, defensible "buy" decision made by a 20-person startup may become a genuinely reasonable "build" decision once that same company has grown to 2,000 employees with very different scale requirements, a much larger available engineering budget, and infrastructure needs that increasingly diverge from what generic, broadly-applicable vendor tools were ever designed to handle.

---

## Putting It Together: Infrastructure for a Growing ML Team

Here's how the concepts in this article come together as a company's ML practice matures over time:

1. **Early stage (no real infrastructure):** A two-person ML team trains and evaluates models locally on their laptops, manually deploying a single model behind a simple API. No dedicated storage layer, no containers, no orchestration — and at this stage, that's entirely appropriate; building out the full infrastructure stack this early would be a wasted, premature investment.
2. **Storage and compute:** As the team grows and starts training larger models on more data, they move to cloud blob storage and a managed data warehouse, and start using elastic cloud compute for bursty training jobs — paying for compute surges only when actually training, rather than maintaining idle owned hardware.
3. **Development environment:** As the team grows to 10+ engineers, "works on my machine" bugs start consuming real time. They standardize on a shared, containerized development environment using Docker, eliminating an entire recurring category of environment-mismatch bugs.
4. **Resource management:** With several multi-step ML workflows now running regularly (data extraction → feature computation → training → evaluation → deployment), the team adopts Airflow to define these workflows explicitly as DAGs, replacing a patchwork of fragile, manually-triggered cron jobs that couldn't properly express dependencies between steps.
5. **ML Platform — feature store first:** As the company now has five different models, several of which redundantly recompute very similar customer behavior features (with subtly different definitions causing occasional training-serving skew incidents), the team's highest-leverage next investment is a shared feature store — directly eliminating a recurring, costly source of silent production bugs that's been independently affecting multiple teams.
6. **Model store and deployment platform:** As the number of models and the frequency of retraining (connecting back to the continual learning maturity stages from Article 9) both continue to grow, the team adopts a proper model store and a standardized deployment tool, replacing ad hoc, per-team deployment scripts with a single, consistent, well-tracked deployment process across the whole organization.
7. **Build vs. buy revisited:** For most of these layers, the team adopts existing open-source or vendor tools rather than building from scratch — but for one very specific, latency-critical real-time feature computation need that's central to their core product's competitive differentiation, they make a deliberate, justified decision to build custom infrastructure in-house, since no existing off-the-shelf vendor solution adequately meets their specific, business-critical latency requirements.

Notice that this progression roughly mirrors the order the four layers were introduced in this article — storage and compute come first because literally nothing else can run without them, while the ML platform layer (feature stores, model stores, deployment tooling) becomes valuable specifically once there's enough genuine organizational scale and redundancy across teams to justify the investment in shared, centralized infrastructure.

---

## Key Takeaways

- The right amount of ML infrastructure investment depends entirely on production scale — most companies need "generalized infrastructure" supporting common ML applications at a reasonable scale, not the highly custom infrastructure built by companies operating at the very largest scale.
- Storage and compute form the foundation: storage is now heavily commoditized, while compute should be evaluated based on memory and core count for your actual workload, not misleading headline FLOPS numbers; the cloud-vs-owned-infrastructure decision shifts as a company's scale and workload predictability grow, sometimes prompting "cloud repatriation."
- The development environment is consistently underinvested in despite its outsized impact on engineering productivity; standardized, containerized environments (via Docker, often orchestrated at scale with Kubernetes) eliminate an entire category of "works on my machine" bugs.
- ML workflows are dependency-laden DAGs, not single isolated steps; cron handles fixed timing only, schedulers add dependency awareness, orchestrators handle where compute actually comes from, and purpose-built data science workflow tools (Airflow, Kubeflow, Metaflow, etc.) combine all of this specifically for ML use cases.
- A shared ML Platform — model deployment tooling, a proper model store (tracking far more than just weights), and especially a feature store (solving sharing, expensive recomputation, and training-serving skew simultaneously) — becomes valuable once an organization has enough scale and redundancy across teams to justify centralized, shared infrastructure investment.
- Build-vs-buy decisions should weigh company stage, whether the infrastructure is genuinely a competitive differentiator, and the real maturity of available tooling — and should be revisited periodically rather than treated as permanent.

---

## What's Next

This article completes our tour through the infrastructure layers underpinning every ML system discussed in this series. In our next and final article, we'll zoom out to **the human and organizational side of ML systems** — how to structure ML teams, the responsibilities split between data scientists, ML engineers, and platform/infrastructure engineers, and the broader question of how organizations build a genuinely sustainable, mature ML practice rather than a collection of one-off model projects.

---

*Machine Learning Systems Series | Article 10 of N*
*Based on "Designing Machine Learning Systems" by Chip Huyen (O'Reilly, 2022)*
