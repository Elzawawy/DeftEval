# DeftEval
<p align="center">
  <img src="https://www.kdnuggets.com/wp-content/uploads/nlp-word-cloud.jpg"/>
</p>

## Understanding The Problem
**Definition Extraction** (DE) is the task to extract textual definitions from naturally occurring text. It is **gaining popularity** as a prior step for constructing taxonomies, ontologies, automatic glossaries or dictionary entries. These fields of application motivate greater interest in well-formed encyclopedic text from which to extract definitions, and therefore DE for academic or lay discourse has received less attention. 

Historically, **Definition extraction** has been a popular topic in NLP research for well more than a decade, but has been limited to well defined, structured, and narrow conditions. In reality, natural language is complicated, and complicated data requires both complex solutions and data that reflects that reality.

### Problem's Background
Definitions are a well-studied topic, which traces back to the Aristotelian genus et differentia model of a definition, where the defined term (definiendum) is described by mentioning its immediate superordinate, usually a hypernym (genus), and the cluster of words that differentiate such definiendum from others of its class (definiens). 

Furthermore, additional research has elaborated on different criteria to take into consideration when deciding what is a definition: either by looking at their degree of formality (Trimble, 1985), the extent to which they are specific to an instance of an object or to the object itself (Seppal¨ a, 2009), the semantic relations holding between definiendum and concepts included in the definiens (Alarcon et al., 2009; Schumann, 2011), ´ the fitness of a definition for target users (Bergenholtz and Tarp, 2003; Fuertes-Olivera, 2010) or their stylistic and domain features (Velardi et al., 2008)

### Problem's Applications
It has received notorious attention for its potential application to **glossary generation** (Muresan and Klavans, 2002; Park et al.,2002), **terminological databases** (Nakamura and Nagao, 1988), **question answering systems** (Saggion and Gaizauskas, 2004; Cui et al., 2005), for supporting terminological applications (Meyer, 2001; Sierra et al., 2006), **e-learning** (Westerhout and Monachesi, 2007), and more recently for **multilingual paraphrase extraction** (Yan et al., 2013), **ontology learning** (Velardi et al., 2013) or **hypernym discovery** (Flati et al., 2014).

###  Problem's Related Work
The earliest attempts focused on **lexico-syntactic pattern-matching, either by looking at cue verbs** (Rebeyrolle and Tanguy, 2000; Saggion and Gaizauskas, 2004; Sarmento et al., 2006; Storrer and Wellinghoff, 2006), **or other features like
punctuation or layout** (Muresan and Klavans, 2002; Malaise et al., 2004; S ´ anchez and M ´ arquez, 2005; ´
Przepiorkowski et al., 2007; Monachesi and Westerhout, 2008).

As for supervised settings, let us refer to (Navigli and Velardi, 2010), who **propose a generalization of word lattices for identifying definitional components and ultimately identifying definitional text fragments.** Finally, **more complex morphosyntactic patterns** were used by (Boella et al., 2014), who model single tokens as relations over the sentence syntactic dependencies.
Or unsupervised approaches (Reiplinger et al., 2012) **benefit from hand crafted definitional patterns.**

## Understanding the dataset
The DEFT corpus contains roughly 7,000 sets of 3-sentence groupings extracted from textbooks of various topics from cnx.org. Each sentence set reflects a context window around which the author of the original text marked a bolded word to indicate a key term. For annotation reasons, the bolded words are not included in the annotated corpus, though you may find them from the textbooks themselves. Each grouping may have multiple term-definition pairs or none at all - it is simply a context window around a likely location for a term.

Train and dev data is provided to you in a CONLL-like tab-deliniated format. Each line represents a token and its features. A single blank line indicates a sentence break; two blank lines indicates a new 3-sentence context window. All context windows begin with a sentence id followed by a period. These are treated as tokens in the data. Each token is represented by the following features:
<p align="center">
  <b><i>[TOKEN] [SOURCE] [START_CHAR] [END_CHAR] [TAG] [TAG_ID] [ROOT_ID] [RELATION]</i></b>
</p>

Where: 

* **SOURCE** is the source .txt file of the excerpt
* **START_CHAR/END_CHAR** are char index boundaries of the token
* **TAG** is the label of the token (O if not a B-[TAG] or I-[TAG])
* **TAG_ID** is the ID associated with this TAG (0 if none)
* **ROOT_ID** is the ID associated with the root of this relation (-1 if no relation/O tag, 0 if root, and TAG_ID of root if not the root)
* **RELATION** is the relation tag of the token (0 if none).

## Understanding Dataset Folder Structure 
- `deft_corpus\data\deft_files`: contains the dataset files itself. It has two splits divided into two subfolders: 
  - `train`: For training split. 
  - `dev`: For development split (used as testing data for evaluation when submitting on website in Training Phase).
- `deft_corpus\data\reference_files`: Are used in the Codalab pipeline for evaluation purposes. When you submit your predictions via Codalab, these are the exact files that the scoring program evaluates your submission against.
- `deft_corpus\data\source_txt`: The original sentences extracted from the textbooks used in the dataset. The source_txt has 80 files full of sentences for training and 68 files for development with less sentences per file.
The deft_files have nearly the same files names as in source_text. 
- `deft_corpus\task1_convertor.py`: This script is used to convert from the sequence/relation labeling format to classification format.This produces files in the following tab-delineated format: **[SENTENCE]  [HAS_DEF]**. 
This is intended for Subtask 1: Sentence Classification.


## Understanding Annotation Schema
The DEFT annotation schema is comprised of terms and definitions, as well as various auxiliary tags which aid in identifying complex or long-distance relationships between a term-definition pair. With the exception of "implicit" definitions (defined below), all terms are linked in some way to a definition or alias term.

### Tag Full Schema

* **Term:** A primary term.
* **Alias Term:** A secondary or less common name for the primary term. Links to a term tag.
* **Ordered Term:** Multiple terms that have matching sets of definitions which cannot be separated from each other without creating a non-contiguous sequence of tokens. (Eg. x and y represent positive and negative versions of definition z, respectively)
* **Referential Term:** An NP reference to a previously mentioned term tag. Typically this/that/these + NP following a sentence boundary.
* **Definition:**	A primary definition of a term. May not exist without a matching term.
* **Secondary Definition:** Supplemental information that may qualify as a definition sentence or phrase, but crosses a sentnece boundary.
* **Ordered Definition:**	Multiple definitions that have matching sets of terms which cannot be separated from each other. See Ordered Term.
* **Referential Definition:**	NP reference to a previously mentioned definition tag. See Referential Term.
* **Qualifier:** A specific date, location, or other condition under which the definition holds true. Typically seen at the clause level.

### Relation Full Schema
* **Direct-defines**	Links definition to term.
* **Indirect-defines**	Links definition to referential term or term to referential definition.
* **Refers-to**	Links referential term to term or referential definition to definition.
* **AKA**	Links alias term to term.
* **Supplements**	Links secondary definition to definition, or qualifier to term.

## Undertsanding the competition
DeftEval is split into three subtasks,
- **Subtask 1: Sentence Classification**, Given a sentence, classify whether or not it contains a definition. This is the traditional definition extraction task.

- **Subtask 2: Sequence Labeling**, Label each token with BIO tags according to the corpus' tag specification.

- **Subtask 3: Relation Classification**, Given the tag sequence labels, label the relations between each tag according to the corpus' relation specification.

Test data will be evaluated in the following CONLL-2003-like formats:

- Subtask 1: Sentence Classification 
  - **[SENTENCE] [BIN_TAG]** Where the binary tag is 1 if the sentence contains a definition and 0 if the sentence does not contain a definition.

- Subtask 2: Sequence Labeling
  - **[TOKEN] [SOURCE] [START_CHAR] [END_CHAR] [TAG]**

- Subtask 3: Relation Extraction
  - **[TOKEN] [SOURCE] [START_CHAR] [END_CHAR] [TAG] [TAG_ID] [ROOT_ID] [RELATION]** Where ROOT_ID is -1 if there is no relation, 0 if the token is part of the root, and TAG_ID of the root if the token points to the root.

## Resources
1. [*Weakly Supervised Definition Extraction (Luis Espinosa-Anke, Francesco Ronzano and Horacio Saggion), Proceedings of Recent Advances in Natural Language Processing, pages 176–185,Hissar, Bulgaria, Sep 7–9 2015.*](https://www.aclweb.org/anthology/R15-1025.pdf)

2. [*DEFT: A corpus for definition extraction in free- and semi-structured text, Sasha Spala, Nicholas A. Miller, Yiming Yang, Franck Dernoncourt, Carl Dockhorn*](https://www.aclweb.org/anthology/W19-4015/) 
  - [Check the Github Repo.](https://github.com/adobe-research/deft_corpus)
  
3. [*CodaLab DeftEval 2020 (SemEval 2020 - Task 6), Organized by sspala.*](https://competitions.codalab.org/competitions/20900#learn_the_details)
