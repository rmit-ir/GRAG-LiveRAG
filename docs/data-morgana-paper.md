# Generating Diverse Q\&A Benchmarks for RAG Evaluation with DataMorgana 

Simone Filice<br>filice.simone@gmail.com<br>Technology Innovation Institute<br>Haifa, Israel<br>Zohar Karnin<br>zohar.karnin@tii.ae<br>Technology Innovation Institute<br>Haifa, Israel

Guy Horowitz<br>guy.horowitz@tii.ae<br>Technology Innovation Institute<br>Haifa, Israel<br>Liane Lewin-Eytan<br>liane.lewineytan@tii.ae<br>Technology Innovation Institute<br>Haifa, Israel<br>David Carmel<br>david.carmel@tii.ae<br>Technology Innovation Institute<br>Haifa, Israel

## ABSTRACT

Evaluating Retrieval-Augmented Generation (RAG) systems, especially in domain-specific contexts, requires benchmarks that address the distinctive requirements of the applicative scenario. Since real data can be hard to obtain, a common strategy is to use LLM-based methods to generate synthetic data. Existing solutions are general purpose: given a document, they generate a question to build a Q\&A pair. However, although the generated questions can be individually good, they are typically not diverse enough to reasonably cover the different ways real end-users can interact with the RAG system.

We introduce here DataMorgana, a tool for generating highly customizable and diverse synthetic Q\&A benchmarks tailored to RAG applications. DataMorgana enables detailed configurations of user and question categories and provides control over their distribution within the benchmark. It uses a lightweight two-stage process, ensuring efficiency and fast iterations, while generating benchmarks that reflect the expected traffic.

We conduct a thorough line of experiments, showing quantitatively and qualitatively that DataMorgana surpasses existing tools and approaches in producing lexically, syntactically, and semantically diverse question sets across domain-specific and generalknowledge corpora. DataMorgana will be made available to selected teams in the research community, as first beta testers, in the context of the upcoming SIGIR'2025 LiveRAG challenge to be announced in early February 2025.

## 1 INTRODUCTION

Retrieval-Augmented Generation (RAG) [12, 16] has recently gained a great deal of popularity, especially in specialized domains. It combines the strengths of Large Language Models (LLMs) with modern information retrieval by dynamically augmenting the LLM prompt with relevant information from an auxiliary corpus. This hybrid approach allows for more accurate and contextually relevant responses and thus mitigates LLMs limitations in handling specialized or frequently updated information.

Before adopting a RAG solution, however, it is critical to evaluate its effectiveness in the target environment, accounting not only for the environment's specific content (the RAG corpus) but also for its diverse types of users and their needs. Let us consider the typical RAG scenario of users asking questions over an enterprisespecialized corpus not memorized in the LLM. In order to evaluate
the RAG solution, in the absence of a real question/query log, the most common approach is to use an LLM to generate a Q\&A pair from a randomly selected document from the RAG corpus. The major risk in applying this approach indiscriminately is that such synthetic benchmarks lack diversity and might not reflect the actual questions users would ask.

To this purpose, we propose here a new approach to generate synthetic benchmarks with two key properties. Straightforward and flexible customization: Setting the way in which Q\&A pairs are generated is done via natural language descriptions, making customization accessible to non-technical users. Diverse generation: For both end-users and questions we allow defining multiple categorizations, along with their distribution within the benchmark, without being restricted by pre-defined options. These categorizations are jointly used to get a combinatorial number of possibilities to define Q\&A pairs, leading to highly diverse benchmarks.

We have implemented this approach in a tool called DataMorgana, which we introduce in this paper. DataMorgana is designed to be lightweight and easily configurable, allowing for rapid experimentation with custom question and end-user categories. In this paper, we focus on the question generation capabilities of DataMorgana, to demonstrate via quantitative experiments that it supports higher diversity than related tools or approaches.

Our key contributions in this work are as follows:

- We introduce DataMorgana, a synthetic benchmarks generation tool, emphasizing easy customization and high diversity.
- We guarantee the creation of high-coverage, highly diverse benchmarks, via a novel technique based on multiple enduser and question categorizations.
- Through a comprehensive series of experiments on different corpora, we demonstrate the superiority of DataMorgana in achieving a higher diversity of generated questions compared to existing benchmark generation methods, across lexical, syntactic, and semantic dimensions.


## 2 RELATED WORK

Recent advances in LLMs, with their tremendous zero-shot and few-shot generation capabilities, have led to many research efforts in creating synthetic test benchmarks for question answering $[4,9,11,26,29]$ and conversational dialog systems $[8,18]$. Ideally, an optimal test set would comprise a large set of real user questions from a query log, paired with "golden answers" provided by experts.

![img-0.jpeg](img-0.jpeg)

Figure 1: DataMorgana generation pipeline.

In the absence of a perfect test set, we seek to generate questions similar to those asked by real users, along with answers inferred from a data source. This task has received increasing interest in recent years from industrial and research communities because of its huge potential benefits in reducing the needed human labor in creating large-scale question-answer benchmarks. A comprehensive taxonomy of generation approaches can be found in [19, 31].

The common methodology for (question, answer) pairs generation is to follow the generate then filter paradigm. Given a corpus of documents, select at first a subset of documents; then, for each document leverage an LLM to generate some questions that can be answered by the given content. Next, ask the LLM to generate, for each of the questions, an answer, or a set of answers, based on the corresponding document. Finally, filter the generated (question, answer, document) tuples according to several criteria such as semantic similarity with golden questions, diversity, and more [29].

InPars [13] follows this paradigm, focusing on question generation while skipping the answer generation process. Via few-shot examples, an LLM is induced to generate relevant questions for a given document. Then, each (question, document) pair is scored according to their inner similarity and only highly scored pairs are selected. Prompagator [5], and more recently ARES [22], follow the same pipeline while keeping generated pairs only if the associated document appears on top of the result list when the question is submitted as a query to a given IR system. Shakeri et al. [26] generate questions and answers from input documents using a fine-tuned encoder-decoder model and then filter them based on the model's perplexity score. Yuan et al. [30] proposed a prompt-based approach to selecting high-quality questions from a set of LLM-generated candidates.

Uncontrolled generated content often tends to be monotonous and biased, hence limiting its applicability in downstream tasks [19]. The diversity of generated data is crucial for generating synthetic samples that mimic the diversified nature of real-world data, thereby preventing over-fitting and bias during model training or evaluation. Yoon and Bak [29] improve the diversity of generated questions by employing a recursive generation framework; they train a generative model (BART) to generate a question that differs from input questions, where the difference is measured with cosine similarity. At inference time previously generated questions are recursively fed back into the generation model to output different questions from reference questions. Eo et al. [10] enhance Q\&A type diversity by training the Q\&A generator to cover various types of questions per document, based on interrogative question words ([Who, When, What, Where, Why, How]).

Recent studies have suggested enhancing administrative control over the types of generated questions. In the Know Your RAG evaluation system [6], a taxonomy of question types is identified to cover different ways a user might interact with the system. The question generation process has three steps: (i) a document is decomposed into statements, (ii) depending on the question type, new statements are generated based on the previously extracted ones; and (iii) one statement is selected as a base information to generate a question. Each of these steps is done via invoking an LLM.

RAGAs [21] is a popular evaluation tool for RAG systems that additionally supports the generation of a synthetic Q\&A benchmark. At first, a knowledge graph (KG) is generated from the corpus by identifying entities, topics, and the relations between them; the test questions are then generated by an LLM based approach on the KG. Similarly to Know Your RAG, RAGAs considers different question types (single-hop vs multi-hop, specific versus abstract) as well as the user persona (senior, junior, etc.). This enriches the type of generated questions and improves diversity. DeepEval [7] is another evaluation tool that supports generating a synthetic Q\&A benchmark. To encourage diversity they enable an evolutionary process where the generated questions are used towards generating new questions (i.e., evolved) according to 1 of 7 pre-defined evolution methods. In contrast, DataMorgana, controls the diversity level with finer granularity via the customization of the question and user characteristics(See §3). In §5, we analyze the set of questions generated from the same corpus by a few of these systems, to compare their diversity with those generated by DataMorgana.

## 3 DATAMORGANA

DataMorgana is designed to generate synthetic benchmarks for training and testing primarily RAG systems and possibly other systems that require Q\&A benchmarks. It differs from other tools by offering configuration capabilities that allow to easily generate benchmarks with high diversity. It operates in two stages: a configuration stage, during which the DataMorgana admin user specifies their needs, and a generation stage during which DataMorgana leverages the input configuration to generate, with the assistance of an LLM, the desired benchmark.

### 3.1 Configuration Stage

The configuration stage allows for the definition of detailed categorizations and associated categories for both questions and end-users, which provide high-level information on the expected traffic of the RAG application. There can be as many categorizations of questions and users as need be, as long as categories within a single

categorization are mutually exclusive and each category is associated with its natural-language description and optionally its desired probability of distribution within the generated benchmark.

The configuration is defined in a JSON file that includes all necessary information to customize the generated benchmark as desired. For instance, if a user wants to generate factoid and non-factoid "experience" questions, as defined in the six types (i.e., instructions, reason, evidence-based, comparison, experience, and debate) of non-factoid questions suggested in [2], they will include in their configuration file the following fragment.

```
Question-Factuality Categorization
"categories": [
    {
    "name": "factoid",
    "probability": 0.25,
    "description":
    "A question seeking a specific, concise piece of information
    or a short fact about a particular subject, such as a
    name, date, or number."
    },
    "name": "non-factoid-experience",
    "probability": 0.75,
    "description":
    "A question to get advice or recommendations on a particular
    topic."
    }
]
```

Note that the desired probabilities of occurrence of each question category in the benchmark can be explicitly defined via the attribute probability. Table 1 details a set of general-purpose question categorizations and their respective categories, which can be used for most corpora.

End-user categorizations are defined analogously to question categorizations. The snippet below shows how to specify a categorization of end-users defining their expertise. We chose this categorization for end-users as our default general-purpose one since it applies as well to most corpora.

```
User-Expertise Categorization
"categories": [
    {
    "name": "expert",
    "probability": 0.50,
    "description": "a specialized user with deep understanding
    of the corpus."
    },
    {
    "name": "novice",
    "probability": 0.50,
    "description": "a regular user with no understanding of
    specialized terms."
    }
]
```

It is possible to define additional user categorizations depending on the RAG corpus. For instance, in a healthcare RAG application, one could add patient, doctor, and public health authority, or in a RAG-based embassy chatbot diplomat, student, worker, and tourist.

### 3.2 Generation Stage

The benchmark is built incrementally one Q\&A pair $\left(q_{i}, a_{i}\right)$ at a time. Each pair is generated by invoking an LLM with a prompt
automatically instantiated by DataMorgana according to the configuration file. Note that the structured parts of the configuration file (e.g., name, probability in each category) are used behind the scenes to instantiate the prompt DataMorgana builds, while the description value is inserted "as is" in the prompt. This gives a lot of freedom to DataMorgana users, who can iterate as needed with the description of categories when generating a benchmark ${ }^{1}$.

The generation process follows the steps depicted in Figure 1:
(1) A user category $u_{i}$ and a question category $c_{j}$ are selected per categorization according to their distribution probabilities, as specified in the configuration file. So if we use the general purpose question categorizations from Table 1 and the User Expertise Categorization detailed before this results in a combination of one user category and four question categories, $\left(u_{1}, c_{1}, \ldots, c_{k}\right)$. This tuple together with the naturallanguage description associated with each category is used to instantiate the prompt template. Note that, by allowing all combinations of categories, we enable a combinatorial number of options, resulting in a highly diverse benchmark.
(2) A document $d_{i}$ is sampled from the RAG corpus and added to the prompt.
(3) The chosen LLM is invoked with the instantiated prompt (a different prompt at each turn) to generate $k$ candidate question-answer pairs ${ }^{2}\left(q_{i}, a_{i}\right)$ about $d_{i}$.
(4) A filtering step is conducted to verify that these candidate pairs satisfy the constraints expressed in the prompt (e.g., be context-free), adhere to the categories specified by $u_{i}, c_{j}$, and that the question answers are faithful to $d_{i}$. If multiple pairs satisfy the quality requirements, one is sampled.

## Prompt Template

You are a user simulator that should generate [num_questions] candidate questions for starting a conversation.

The [num_questions] questions must be about facts discussed in the documents you will now receive. When generating the questions, assume that the real users you must simulate, as well as the readers of the questions, do not have access to these documents. Therefore, never refer to the author of the documents or the documents themselves. Also, assume that whoever reads the questions will read each question independently. The [num_questions] questions must be diverse and different from each other. Return only the questions without any preamble. Write each pair in a new line, in the following JSON format: ' $\{$ "question": <question>, "answer": <answer> $\}$ :

```
### The generated questions should be about facts from the
following document:
[document (d_i)]
### Each of the generated questions must reflect a user with
the following characteristics:
    - They must be [description of user category 1 (u_1)]
    - They must be [description of user category 2 (u_2)]
### Each of the generated questions must have the following
characteristics:
    - It must be [description of question category 1 (c_1)]
    - It must be [description of question category 2 (c_2)]
```

[^0]
[^0]:    ${ }^{1}$ Note that in this early release of DataMorgana, we are not enforcing consistency among categories. We will wait for our early beta testers to experiment with the tool before deciding whether this is a needed capability, or whether the LLM can handle such inconsistencies on its own, e.g., via the filtering stage described below.
    ${ }^{2}$ We set $k=3$ in our experiments.

Table 1: Question Categories. The examples in parentheses are for illustration only and are not necessarily part of the description to be used for generation.

| Categorization | Category | Description |
| :--: | :--: | :--: |
| Factuality | factoid | question seeking a specific, concise piece of information or a short fact about a particular subject, such as a name, date, or number (e.g., 'When was Napoleon born?). |
|  | open-ended | question inviting detailed or exploratory responses, encouraging discussion or elaboration. (e.g., 'what caused the French revolution?'). |
| Premise | direct | question that does not contain any premise or any information about the user) (e.g., 'what is the fee for speeding in Italy?') |
|  | with-premise | question starting with a very short premise, where the user reveals their needs or some information about himself (e.g., 'I have an H1-B visa for the United States. Is there a limit to how many times I can exit and enter the country in a year?). |
| Phrasing | concise-and-natural | phrased in the way people typically speak, reflecting everyday language use, without formal or artificial structure. It is a concise direct question consisting of less than 10 words (e.g., 'what's the weather like in Paris now?). |
|  | verbose-and-natural | phrased in the way people typically speak, reflecting everyday language use, without formal or artificial structure. It is a a relatively long question consisting of more than 9 words (e.g., 'I thought of visiting Paris this year, not sure when is the best time. How is it like in the summer?'). |
|  | short-search-query | phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of less than 7 words (e.g., 'Paris weather August'). |
|  | long-search-query | phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of more than 6 words (e.g., 'Paris, France temperature humidity climate summer vs fall'). |
| Linguistic variation | similar-to-document | phrased using the same terminology and phrases appearing in the document (e.g., for the document 'The Amazon River has an average discharge of about 215,000-230,000 m3/s', 'what is the average discharge of the Amazon river'). |
|  | distant-from-document | phrased using terms completely different from the ones appearing in the document (e.g., for a document 'The Amazon River has an average discharge of about 215,000-230,000 m3/s', 'How much water run through the Amazon?'). |

Note that this two-stage methodology is simple and lightweight by design. We intentionally try to avoid approaches with a costly pre-processing stage (e.g., building a knowledge graph [21] or performing heavy analysis on the document [6]) or multiple invocations for post-processing (e.g., evolving a question [7]) Also, note that we describe here only the initial features of DataMorgana that will be used for the SIGIR'2025 LiveRAG Challenge. Additional capabilities are planned to be added for additional types of benchmarks in the near future.

## 4 EXPERIMENTAL SETTINGS

A common way of evaluating the quality of synthetic data is via fidelity, diversity, and generalization (see [1] and references within). Fidelity measures the quality of a model's synthetic samples, and Diversity is the extent to which these samples cover the full variability of real samples. Generalization applies to processes like GANs where the generation process is based on a training set of real examples, hence does not apply to our setting. In our setting of question generation, fidelity translates to the quality of individual questions: Each generated question should be fluent, coherent, relevant to the target application, and realistic. In other words, it should represent a plausible way a real user could interact with the system. Diversity means that the generated questions cover all or at least many of the questions asked by humans.

In our analysis, we decided not to focus on fidelity. The reason is that recent powerful LLMs (e.g., Claude-3.5-Sonnet or GPT-4) are known to excel in generative tasks and produce high-quality text,
matching and perhaps exceeding human level [3]; therefore the individual question quality is typically extremely high, regardless of the specific generation strategy adopted. To further confirm this assumption, in preliminary studies, we manually annotated $\sim 200$ individual questions generated by DataMorgana powered by Claude-3.5-Sonnet in terms of text quality and relevance to the document used to generate each question. We observed close to perfect results. We note that for Q\&A pairs, fidelity includes the quality and specifically correctness of the answer. In a preliminary analysis, we observed that the answers generated by DataMorgana are typically faithful to the original document and that the Q\&A filtering stage helps remove bad generations. However, our focus here is on the quality of the questions rather than answers, hence we do defer further investigation of the answer quality to future research.

Our analysis below focuses on the diversity/coverage aspect, which is still an open problem due to the tendency of LLMs to generate obvious responses to input prompts, which in our scenarios means they mostly generate specific types of questions and neglect other interaction types.

### 4.1 Baselines

We compare DataMorgana with the following synthetic data generation methods:

- Vanilla: this strategy repeatedly uses the same exact process to generate questions from different documents, namely the LLM instructions appearing in the prompt are always the

same, and the only part that varies is the input document. This baseline corresponds to the usage of DataMorgana without configuration. This is probably the most common strategy to generate synthetic benchmarks [4, 17, 27, 28].

- Know Your RAG: we re-implemented the solution proposed by de Lima et al. [6] described in Section 2. The original solution generates four question types: Single-fact, reasoning, summary, and unanswerable questions. We excluded the latter since, while it is fitting for reading comprehension, it is too challenging in a RAG context (especially with large RAG corpora) to guarantee that no document in the corpus can answer the question.
- DeepEval: we chose DeepEval [7] as a representative of unplished commercial solutions. We used its default setting that enables evolving questions with one step of evolution, where the type is drawn uniformly at random from the 7 possible evolutions. We chose DeepEval since it is quite adopted, their git repo has 4.3 K stars and 358 forks, and their data generation code is easy to run and flexible enough to allow generating multiple questions per document (required for the experiments below).
For a fair comparison, all tested generation methods leverage Claude-3.5 Sonnet v2 ${ }^{3}$ with default parameters as LLM backbone.


### 4.2 Experiments Corpora

To demonstrate the capabilities of DataMorgana, we generated synthetic data from two different corpora:

- COVID-19 Open Research Dataset (CORD-19) [6]: this corpus contains scientific papers on COVID-19 and related historical coronavirus research. To allow us to compare with human-generated questions, we selected the 147 articles that biomedical experts used when generating the 2019 questions appearing in the COVID-QA dataset [20]. This healthcare scenario serves to show how DataMorgana can be easily configured to adapt to a domain-specific corpus.
- Wikipedia: Wikipedia is a free online encyclopedia that contains millions of articles about general human knowledge. To allow us to compare with human-generated questions, we considered the real user questions along with the Wikipedia passages containing their answer as they appear in the NQ dataset [14]. More specifically, we used the 2889 questions appearing in the test set of the open version of NQ [15]. This dataset serves us to show the effectiveness of DataMorgana in general-purpose scenarios.


### 4.3 Configuration of Question and User Categorizations

In both scenarios, we used the question categorizations and categories as detailed in Table 1. Regarding the user categorizations, in the Wikipedia scenario, we employed our default (general-purpose) user categorization. For the CORD-19 corpus, we designed a categorization specific to the healthcare scenario:

[^0]- patient: a regular patient who uses the system to get basic health information, symptom checking, and guidance on preventive care.
- medical-doctor: a medical doctor who needs to access some advanced information.
- clinical-researcher: a clinical researcher who uses the system to access population health data, conduct initial patient surveys, track disease progression patterns, etc.
- public-health-authority: a public health authority who uses the system to manage community health information dissemination, be informed on health emergencies, etc.


## 5 ASSESSING DIVERSITY

### 5.1 Qualitative Diversity Exploration

To get an initial feeling of the diversity characterizing synthetic benchmarks ${ }^{4}$, we report in Table 2 a random set of questions about different articles from the CORD-19 corpus, generated by different methods.

The first set of questions is generated by the Vanilla approach. In this case, we do not provide detailed instructions to the LLM, which therefore is left completely free in its question generation process. The resulting questions are all in natural language, and most of them appear very specific, relatively long, and characterized by sophisticated terminology, reflecting an inherent bias of the LLM towards these types of questions.

The second and third blocks report a random sample of questions generated by Know Your RAG and DeepEval. Similarly to the Vanilla solution, these methods do not allow control of the style of the question, and consequently it is exposed to the inherent LLM bias, which tends to generate detailed and long natural questions containing sophisticated terminology. The diversification introduced by their taxonomy can be appreciated by the presence of many comparative questions (e.g., What's the difference between TIV, QIV, and LAIV flu vaccines, and which one provides the best protection?). However, even if in DataMorgana we did not explicitly prompt the model to generate comparative questions, in some occasions the model generates such questions (e.g., How deadly was COVID compared to SARS and MERS?).

The fourth set of questions, obtained with DataMorgana, exhibits larger diversity, with respect to the user and question categories used in the generation phase. For instance, the question phrasing categorization contributes to creating long and short questions, as well as questions in natural form and expressed as a web search query. Similarly, it is possible to appreciate questions having a premise (e.g., I live in a tropical area. When do most flu cases happen?), basic questions having a more simplistic terminology, typical of patients (e.g., Are there new ways to make better vaccines?), questions from public authorities (e.g., What scientific evidence do we have to counter the claims that COVID-19 was created in a lab, so I can properly address community concerns about this? or questions which we can expect from researchers (e.g., How many genetic differences

[^1]
[^0]:    ${ }^{3}$ https://www.anthropic.com/claude/sonnet

[^1]:    ${ }^{4}$ Unless explicitly mentioned, all the experiments reported in this paper are conducted so that for each document in the corpus we generate the exact same number of questions appearing in COVID-QA or NQ-open datasets. In most cases, there is a single question associated with each document but on rare occasions, this number can be much larger (i.e., up to 125 for a document in the COVID-QA dataset).

Table 2: Random Sample of questions generated by different methods. Sophisticated terms are highlighted in bold; questions in search query format are underlined.

| Model | Random Sample of Questions |
| :--: | :--: |
| Vanilla | How common are ex-infections in people who have influenza, and why is this important for treatment? <br> How do humans typically get infected with hantavirus, and what activities put people at higher risk of infection? <br> How do humans typically get infected with pathogenic arenaviruses? <br> How does the protein Prohibitin (PHB) affect the life cycle of the lymphocytic choriomeningitis virus? <br> What are the current limitations of seasonal influenza vaccines that make them less effective than desired? <br> What are the main approaches being explored for developing a universal influenza vaccine using viral vectors? <br> What are the main clinical symptoms and warning signs of severe adenovirus type 55 infection in otherwise healthy adults? <br> What is the mortality rate for MERS and how does it compare to SARS? <br> What specific protective equipment and safety measures were required for healthcare workers conducting CT scans of COVID-19 patients? <br> What were the main routes of transmission for SARS-CoV-2 in the early stage of the outbreak in Wuhan, and which one was more significant? |
| Know Your RAG | By how much did pneumonia deaths in children decrease between 2000-2015 due to new vaccines? <br> How do virus-vectored flu vaccines compare to traditional vaccines in terms of safety and immune response? <br> How does 2-bromopalmitic acid affect hantavirus host cell mineralization patterns? <br> How does EGR1 deficiency affect BIRC5 expression during VEEV infection? <br> What genetic relationship was found between SAIBK virus and COVID-19 vaccine strains during the early pandemic studies? <br> What genetic similarities does the French BCoV strain share with Asian coronavirus strains? <br> What safer alternative to live virus can be used for arenavirus neutralization testing? <br> What starting material did the engineered E. coli platform use to generate glucose-1-phosphate for UDP-sugar synthesis? <br> What was Germany's COVID-19 infection rate compared to other European countries during early pandemic interventions in March 2020? <br> What was the mortality rate of HCPS cases in South America during 1993-2009? |
| DeepEval | How did World War 1's social and economic conditions make the Spanish flu pandemic more deadly, leading to over 20 million deaths? <br> How do environmental factors like habitat fragmentation, climate patterns, and seasons affect hantavirus outbreaks and rodent populations in the Americas? <br> How do respiratory viruses affect the airways? <br> How do viral infections change our body's immune response and inflammation levels when symptoms get worse? <br> How would Australian Japanese biomedical research collaboration be different today if the AIFII and ComBio conferences had never taken place? <br> How would scientists use VP1 sequencing and viral testing to identify meningitis infections if an outbreak happened today? <br> What are the average and highest percentage increases in COVID-19 cases predicted for China by FPASSA-ANFIS? <br> What are the advantages and challenges of using Ad5 as a vaccine vector, particularly regarding stability, storage, delivery, and immunity issues? <br> What's the difference between TIV, QIV, and LAIV flu vaccines, and which one provides the best protection? <br> Which caspases are activated, and at what concentrations, when HT-29 cells are treated with Cu2 compared to untreated cells? |
| DataMorgans | Are there new ways to make better vaccines? <br> death rate 1918 influenza young adults <br> hospital screening protocols during coronavirus early outbreak <br> How deadly was COVID compared to SARS and MERS? <br> How many genetic differences are there between the human coronavirus that causes COVID-19 and its closest known relative found in bats? <br> I live in a tropical area. When do most flu cases happen? <br> transmission rate comparison between respiratory viruses <br> What factors increase risk of hantavirus outbreaks? <br> What scientific evidence do we have to counter the claims that COVID-19 was created in a lab, so I can properly address community concerns about this? <br> What were the main symptoms of early COVID-19 cases? |
| Humans | How does MARS-COV differ from SARS-COV? <br> How was HFRS first brought to the attention of western medicine? <br> What animal models exist for both the asymptomatic carriage of PUUV and SNV? <br> What can respiratory viruses cause? <br> What is MERS mostly known as? <br> What is RANBP2? <br> What is the transmission of MERS-CoV is defined as? <br> What reduces the antimicrobial activities of alveolar macrophages? <br> What regulates the broad, but less specific, virus-cell interaction in a hepatitis B infection? <br> Where did SARS-CoV-2 originate? |

are there between the human coronavirus that causes COVID-19 and its closest known relative found in bats?)

Another observation we can derive from Table 2 is that the Vanilla solution tends to repeatedly use some word expressions across multiple questions, for instance, most of the questions start with What are/is/was/were or How do/does, while in DataMorgana this is less frequent. To better quantify this phenomenon, as in [25], we use syntactic templates, i.e., Part-of-Speech (PoS) tag sequences, that can capture structural repetitions better than lexical patterns. In particular, we use spacy ${ }^{5}$ to extract the PoS tags of the generated questions and consider the first five PoS of each question as its syntactic template. After generating a Q\&A benchmark over the

CORD-19 corpus using the various solutions (baselines, DataMorgana, etc.) we discussed before, we grouped the generated questions based on their syntactic template. In the benchmark generated with the Vanilla method, we found 573 distinct templates. This number increases to 859 and 933 with DeepEval and Know Your RAG, respectively, and gets the best result of 1248 with DataMorgana.

Table 3 reports the most frequent syntactic PoS templates appearing in the generated questions, as well as their frequency and the cumulative frequency of the three common templates. Unsurprisingly, the frequent templates are typically associated with What questions, and this is also in line with human-generated questions, where this type of question is the most frequent. An important aspect to consider is that in the Vanilla strategy, the top-3 frequent templates cumulatively account for $\sim 16 \%$ of the entire benchmark, similarly as in the human-generated COVID-QA dataset, while

Table 3: Most common PoS template appearing in the generated questions for each method. The bold letter groups (WP, VBP, etc) represent standard part-of-speech tags. We list the frequency of the most common (Top 1) pattern and the cumulative frequency of the three most common (Top 1-3) patterns over the CORD-19 corpus.

| Model | Most Common Starting Pattern | Top 1 frequency | Top 1-3 frequency | Example Questions of Top Pattern |
| :--: | :--: | :--: | :--: | :--: |
| Vanilla | WP VBP DT JJ NNS | 181/2019 (9.0\%) | 321/2019 (15.9\%) | What are the main symptoms... <br> What are the main differences... <br> What are the typical symptoms... |
| Know Your RAG | WP VBD DT NN NN | 56/2019 (2.8\%) | 140/2019 (6.9\%) | What was the gender distribution... <br> What was the survival rate... <br> What was the detection rate... |
| DeepEval | WP VBD DT NNS IN | 99/2019 (4.9\%) | 255/2019 (12.6\%) | What are the effects of... <br> What are the advantages of... <br> What are the differences in... |
| DataMorgana | WP VBP DT JJ NNS | 56/2019 (2.8\%) | 118/2019 (5.8\%) | What are the clinical applications... <br> What are the typical signs... <br> What are the main types... |
| Humans | WP VBZ DT NN IN | 200/2019 (9.9\%) | 337/2019 (16.7\%) | What is an example of... <br> What is the difference between... <br> What is the structure of... |

with DataMorgana this number is only $\sim 6 \%$. We argue that this discrepancy should not be seen as a deficiency of DataMorgana because the human experts who generated the COVID-QA datasets were not genuine users of Q\&A systems but volunteers requested to create questions for a dataset.

### 5.2 Quantitative Diversity Evaluation

5.2.1 Diversity Metrics. To estimate the diversity of the generated benchmark $B$, we use the following metrics, as suggested in [24]:

- N-Gram Diversity (NDG) Score: this score represents the ratio of the unique n-gram counts to all n-gram counts in the benchmark. It is a widely used metric to compute lexical diversity. Following Shaib et al. [24] we use up to $n=4$ grams. More formally:

$$
N D G(B)=\sum_{n=1}^{4} \frac{\# \text { unique } \mathrm{n}-\mathrm{grams} \text { in } B}{\# \mathrm{n}-\mathrm{grams} \text { in } B}
$$

- Self-Repetition Score (SRS): this metric was introduced by Salkar et al. [23] and it counts the number of questions that contain at least one $n$-gram (we use $n=4$ ) that also appears in another question in the benchmark. We define the repetition score for a dataset as the number of questions containing repeating n-grams divided by the total number of questions in that benchmark.
- Compress Ratio (CR): The compression ratio is the ratio between the size of the file of the benchmark, to the size of its compressed file, using gzip. High compression ratios imply more redundancy, i.e., less diversity:

$$
C R(B)=\frac{\# \text { size of } B}{\# \text { size of compressed } B}
$$

We refer to this metric as word-CR, when applied to the file containing the generated questions, and it measures the lexical diversity of the benchmark. Conversely, we use PoS-CR to refer to the same metric applied to the Part-of-Speech tag sequence of the questions. In this case, the metric provides an estimate of the syntactic diversity of the benchmark.

- Homogenization Score (HS): this score computes the average similarity between all question pairs in the benchmark, more formally:

$$
H S(B)=\frac{1}{|B|(|B|-1)} \sum_{q, q^{\prime}<B \mid q \neq q^{\prime}} \operatorname{sim}\left(q, q^{\prime}\right)
$$

Where sim is a similarity function between two questions, which we compute by using the cosine similarity of the question embeddings obtained by running the all-MiniLM-L6-v2 sentence encoder from the Sentence Transformer package ${ }^{6}$. We call embeddings-HS the resulting homogenization score that we use to compute semantic diversity.
As discussed in [24] these metrics often do not correlate, since they capture different diversity dimensions.
5.2.2 Experimental Results. Tables 4 and 5 report the diversity scores of the benchmarks obtained with DataMorgana and the other methods described in Section 4.1. Furthermore, as an ablation study, we also run DataMorgana without user categorizations, namely DM w/o user cat. and without question categorizations, namely DM w/o question cat.. Finally, we also report the diversity scores of the human-generated benchmarks, as an additional reference point, but as indicated in Section 5.1, not as a gold standard of diversity.

Overall, the Vanilla solution achieves the worst results. This confirms our hypothesis that repeatedly using the same general LLM prompt without specifying the desired question characteristics exposes to the inherent bias of the LLM towards some types of questions; this consequently results in low diversity. DataMorgana improves the Vanilla results for all metrics (in a statistically significant ${ }^{7}$ manner).

DataMorgana is also generally better than Know Your RAG, especially in syntactic and semantic metrics, with differences that are statistically significant for all metrics for the COVID-QA case, and for all metrics but SRS and word-CR for the Wikipedia scenario. The gap between DataMorgana and DeepEval is less pronounced, but there are still statistically significant differences in NDG, SRS, and

[^0]
[^0]:    ${ }^{6}$ https://www.sbert.net/
    ${ }^{7}$ We computed Student's t-test on bootstrapped samples and obtained p-values $=0.01$ in all metrics.

Table 4: Diversity scores of the COVID-QA dataset and different synthetic datasets containing the same number of questions associated with each of the 147 clinical articles appearing in the COVID-QA dataset. In bold are the best diversity scores among the synthetic datasets (excluding ablation studies).

|  | lexical diversity |  |  | syntactic diversity <br> PoS-CR $(\downarrow)$ | semantic diversity <br> embeddings-HS $(\downarrow)$ |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Model | NGD $(\uparrow)$ | SRS $(\downarrow)$ | word-CR $(\downarrow)$ |  |  |
| Vanilla | 1.517 | 0.920 | 5.576 | 7.861 | 0.301 |
| Know Your RAG | 2.358 | 0.613 | 3.879 | 6.271 | 0.265 |
| DeepEval | 2.415 | 0.644 | 3.535 | 5.885 | 0.251 |
| DataMorgana | 2.536 | 0.372 | 3.701 | 5.583 | 0.249 |
| DM w/o question cat. | 1.777 | 0.908 | 4.746 | 6.945 | 0.296 |
| DM w/o user cat. | 2.484 | 0.401 | 3.725 | 5.648 | 0.247 |
| Humans | 2.484 | 0.365 | 3.380 | 6.212 | 0.182 |

Table 5: Diversity scores of the open-NQ dataset and different synthetic datasets containing the same number of questions associated with each of the 2682 Wikipedia Passages appearing in the open-NQ dataset. In bold are the best diversity scores among the synthetic datasets (excluding ablation studies).

|  | lexical diversity |  |  | syntactic diversity <br> PoS-CR $(\downarrow)$ | semantic diversity <br> embeddings-HS $(\downarrow)$ |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Model | NGD $(\uparrow)$ | SRS $(\downarrow)$ | word-CR $(\downarrow)$ |  |  |
| Vanilla | 2.662 | 0.533 | 2.665 | 5.824 | 0.068 |
| Know Your RAG | 2.981 | 0.144 | 2.488 | 5.864 | 0.074 |
| DeepEval | 2.879 | 0.371 | 2.477 | 5.631 | 0.067 |
| DataMorgana | 3.016 | 0.140 | 2.502 | 5.397 | 0.052 |
| DM w/o question cat. | 2.722 | 0.529 | 2.662 | 5.832 | 0.064 |
| DM w/o user cat. | 2.999 | 0.138 | 2.509 | 5.394 | 0.053 |
| Humans | 2.585 | 0.357 | 2.775 | 5.753 | 0.016 |

Pos-CR in both scenarios, and in embedding-HS in the Wikipedia questions.

From the ablation studies it is clear that most of the diversity improvement of DataMorgana is due to the question categorizations: the contribution of user categorization is marginal in the COVID-QA case, where we use four different user categories, and negligible for the open-NQ corpus, where instead we use only two user categories. One of the reasons for the discrepancy between the impact of the question categorizations and the user categorizations is their respective sizes: while we use a single user categorization having at most four categories (as in the COVID-QA case), we apply four question categorizations in both corpora, resulting in a combination of 32 different joint question categories. In addition, the impact of user categorization is likely to vary with the RAG application, depending on the homogeneity of users, while question categorization can be applied to any corpus and is likely to remain significant regardless of the specific RAG application.

Another aspect we need to consider while interpreting the diversity scores is that most of the metrics, especially the lexical ones, tend to favor questions using sophisticated terms; the reason is that these terms are typically extremely specific and as such appear very few times in the generated benchmark. On the opposite, questions using simpler terminology are penalized since they contain fewer distinct words and phrases. As we can notice in Table 2, most of the questions generated by the Vanilla approach contain very sophisticated terminology, reflecting the LLM bias towards their utilization. DataMorgana, by using user categories such as
patient in the COVID-QA case, or non-expert in the open-NQ case, mitigates this bias and produces a nice mixture of simple and sophisticated questions. Unfortunately, the metrics we use do not capture this type of diversity and actually penalize it ${ }^{8}$. Therefore, we believe there is a need to explore new diversity metrics, and we leave this for future work.

As a further experiment, Figure 2 reports how diversity changes when increasing the number of generated questions per document. When we generate a single question per document we are basically enforcing topical diversity, since each question is about a different document. Nevertheless, DataMorgana achieves better diversity than other solutions demonstrating that it inherently counter-balances the tendency of LLMs to generate repeated lexical or syntactic patterns. By increasing the number of questions per document, the diversity of the resulting benchmark naturally decreases, as there are multiple questions about the same topic. However, the gap between DataMorgana and the other solutions increases, demonstrating the effectiveness of the proposed method.

Similar observations can be derived from Figure 3. In this case, we fix to 500 the total number of questions in the benchmark, and we increase the number of documents used to generate them, from 20 (which means that we generate 25 questions per document) to 147 (where we generate at most four questions per document). Using more documents allows for more topical diversity, and this

[^0]
[^0]:    ${ }^{8}$ To verify this, we compared the diversity of a benchmark consisting of only questions from simulated expert users with a benchmark containing only questions from simulated non-expert users. The benchmark from expert users resulted more diverse in all adopted metrics.

![img-1.jpeg](img-1.jpeg)

Figure 2: Lexical (NDG, the higher the better - on the left) and syntactic (PoS-CR, the lower the better - on the right) diversity of synthetic benchmarks when increasing the number of questions generated for each of the 147 documents in the COVID-QA dataset. Similar trends are observed with other metrics.
![img-2.jpeg](img-2.jpeg)

Figure 3: Lexical (NDG, the higher the better - on the left) and syntactic (PoS-CR, the lower the better - on the right) diversity of synthetic benchmarks containing 500 questions generated from an increasing number of documents in the COVID-QA dataset. Similar trends are observed with other metrics.
justifies the diversity increment captured by lexical metrics. However, DataMorgana guarantees very high diversity regardless of the number of used documents and it is consistently better than other methods.

## 6 CONCLUSION

In this work, we introduce DataMorgana, a benchmark generation tool that offers simple, yet rich configuration capabilities to tailor synthetic benchmarks to the expected traffic of a RAG application.

DataMorgana takes as input a JSON configuration file that abstractly describes, via a semi-structured categorization representation, the expected questions and end users of the RAG application. It then automatically builds the appropriate prompts to be fed to an LLM in order to generate synthetic questions while providing good coverage for questions and users according to the configuration file.

Through qualitative and quantitative analyses, we demonstrated that the questions generated by DataMorgana are significantly more diverse than those generated by other related question generation tools or approaches, which typically leave the choice of question
type to the LLM or use internal mechanisms for controlling question diversity.

While DataMorgana was originally planned for RAG systems evaluation, it can support any application that might benefit from a high-quality and diverse Q\&A benchmark. We intend to introduce in the near future additional capabilities for generating other types of benchmarks, such as synthetic conversations, as well as additional controlling mechanisms, such as document sampling. DataMorgana will be made available to selected teams in the research community, as first beta testers, in the context of the upcoming SIGIR 2025 LiveRAG challenge ${ }^{9}$, before releasing it more widely.

## REFERENCES

[1] Ahmed Alaa, Boris Van Breugel, Evgeny S Saveliev, and Mihaela van der Schaar. 2022. How faithful is your synthetic data? sample-level metrics for evaluating and auditing generative models. In International Conference on Machine Learning. PMLR, 290-306.
[2] Valeriia Bolotova, Vladislav Blinov, Falk Scholer, W. Bruce Croft, and Mark Sanderson. 2022. A Non-Factoid Question-Answering Taxonomy. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). Association for Computing

[^0]
[^0]:    ${ }^{9}$ https://sigir2025.dei.unipd.it/

Machinery, New York, NY, USA, 1196-1207. https://doi.org/10.1145/3477495. 3551926
[3] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Eze Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 (2023).
[4] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 17754-17762.
[5] Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guo, Keith Hall, and Ming-Wei Chang. 2022. Promptagator: Few-shot Dense Retrieval From 8 Examples. In The Eleventh International Conference on Learning Representations.
[6] Rafael Teixeira de Lima, Shulsham Gupta, Cesar Berrospi, Lokesh Mishra, Michele Dolfi, Peter Staar, and Panagiotis Vagenas. 2024. Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems. arXiv preprint arXiv:2411.19710 (2024).
[7] DeepEval. 2025. DeepEval: Synthesizers. https://docs.confident-ai.com/docs/ synthesizer-introduction
[8] Xuan Long Do, Bowei Zou, Liangming Pan, Nancy Chen, Shafiq Joty, and Aiti Aw. 2022. CoMS-CQG: Context and History Selection for Conversational Question Generation. In Proceedings of the 29th International Conference on Computational Linguistics. 580-591.
[9] Chenhe Dong, Ying Shen, Shiyang Lin, Zhenzhou Lin, and Yang Deng. 2023. A unified framework for contextual and factoid question generation. IEEE Transactions on Knowledge and Data Engineering 36, 1 (2023), 21-34.
[10] Sugyeong Eo, Hyeonsook Moon, Jinsung Kim, Yuna Hur, Jeongwook Kim, SongEun Lee, Changwoo Chun, Sungsoo Park, and Heuiseok Lim. 2023. Towards Diverse and Effective Question-Answer Pair Generation from Children Storybooks. In Findings of the Association for Computational Linguistics: ACL 2023, Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 6100-6115. https: //doi.org/10.18633/v1/2023.findings-acl.580
[11] Zichu Fei, Qi Zhang, Tao Gui, Di Liang, Sirui Wang, Wei Wu, and Xuan-Jing Huang. 2022. CQG: A simple and effective controlled generation framework for multi-hop question generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 6896-6906.
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinku Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2023).
[13] Vitor Jeronymo, Luiz Bonifacio, Hugo Abonizio, Marzich Fadaee, Roberto Lotulo, Jakub Zavrel, and Rodrigo Nogueira. 2023. Inpare-v2: Large language models as efficient dataset generators for information retrieval. arXiv preprint arXiv:2301.01820 (2023).
[14] Tom Kwiatkowski, Jeminmaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Keleey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452-466. https://doi.org/10.1162/tacl_a_00276
[15] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent Retrieval for Weakly Supervised Open Domain Question Answering. In Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, Florence, Italy, 6086-6096. https://doi.org/10.18653/ v1/P19-1612.
[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33 (2020), 9459-9474.
[17] Jiarui Li, Ye Yuan, and Zehua Zhang. 2024. Enhancing llm factual accuracy with rag to counter hallucinations: A case study on domain-specific queries in private knowledge-hases. arXiv preprint arXiv:2403.10446 (2024).
[18] Yansuang Ling, Fei Cai, Honghui Chen, and Maarten de Rijke. 2020. Leveraging context for neural question generation in open-domain dialogue systems. In Proceedings of The Web Conference 2020. 2486-2492.
[19] Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. 2024. On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. In Findings of the Association for Computational Linguistics ACL 2024. 11063-11082.
[20] Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. 2020. COVID-QA: A Question Answering Dataset for COVID-19. In Proceedings of the 1st Workshop on NLP for COVID-19 at ACL 2020, Karin Verspoor, Kevin Bretonnel Cohen, Mark Dredze, Emilio Ferrara, Jonathan May, Robert Munro, Cecile Paris, and Byron Wallace (Eds.). Association for Computational Linguistics, Online. https://aclanthology.org/2020.nlpcovid19-acl.18/
[21] RAGAS. 2025. Ragas: Testset Generation for RAG. https://docs.ragas.io/en/ stable/concepts/test_data_generation/rag/
[22] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 338-354.
[23] Nikita Salkar, Thomas Trikalinos, Byron Wallace, and Ani Nenkova. 2022. SelfRepetition in Abstractive Neural Summarizers. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Vol- ume 2: Short Papers), Yulan He, Heng Ji, Sujian Li, Yang Liu, and Chua-Hui Chang (Eds.). Association for Computational Linguistics, Online only, 341-350. https://doi.org/10.18653/v1/2022.aacl-short. 42
[24] Chantal Shaib, Joe Barrow, Jinding Sun, Alexa F. Stu, Byron C. Wallace, and Ani Nenkova. 2024. Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores. arXiv:2403.00553 [cs.CL] https://arxiv.org/abs/ 2403.00553
[25] Chantal Shaib, Yanai Elazar, Junyi Jessy Li, and Byron C. Wallace. 2024. Detection and Measurement of Syntactic Templates in Generated Text. arXiv:2407.00211 [cs.CL] https://arxiv.org/abs/2407.00211
[26] Siamak Shakeri, Cicero Nogueira dos Santos, Henghui Zhu, Patrick Ng, Feng Nan, Zhiguo Wang, Ramesh Nallapati, and Bing Xiang. 2020. End-to-End Synthetic Data Generation for Domain Adaptation of Question Answering Systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, Online, 5445-5460. https://doi.org/ 10.18653/v1/2020.emnlp-main. 439
[27] Shuting Wang, Jiongnan Liu, Shiren Song, Jiehan Cheng, Yaqi Fu, Peidong Guo, Kun Fang, Yutao Zhu, and Zhicheng Dou. 2024. Domainrag: A chinese benchmark for evaluating domain-specific retrieval-augmented generation. arXiv preprint arXiv:2406.05654 (2024).
[28] Shuting Wang, Jiejun Tan, Zhicheng Dou, and Ji-Rong Wen. 2024. OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain. arXiv preprint arXiv:2412.13018 (2024).
[29] Hokeun Yoon and JinYeong Bak. 2023. Diversity Enhanced Narrative Question Generation for Storybooks. In The 2023 Conference on Empirical Methods in Natural Language Processing.
[30] Xingdi Yuan, Tong Wang, Yen-Hsiang Wang, Emery Fine, Rania Abdelghani, Hélène Sauzion, and Pierre-Yves Oudeyer. 2023. Selecting Better Samples from Pre-trained LLMs: A Case Study on Question Generation. In Findings of the Association for Computational Linguistics: ACL 2023. 12952-12965.
[31] Ruqing Zhang, Jiafeng Guo, Lu Chen, Yixing Fan, and Xueqi Cheng. 2021. A review on question generation from natural language text. ACM Transactions on Information Systems (TOIS) 40, 1 (2021), 1-43.