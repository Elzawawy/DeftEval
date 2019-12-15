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

## Undertsanding the competition
DeftEval is split into three subtasks,
- **Subtask 1: Sentence Classification**, Given a sentence, classify whether or not it contains a definition. This is the traditional definition extraction task.

- **Subtask 2: Sequence Labeling**, Label each token with BIO tags according to the corpus' tag specification.

- **Subtask 3: Relation Classification**, Given the tag sequence labels, label the relations between each tag according to the corpus' relation specification.

## Resources
1. [*Weakly Supervised Definition Extraction (Luis Espinosa-Anke, Francesco Ronzano and Horacio Saggion), Proceedings of Recent Advances in Natural Language Processing, pages 176–185,Hissar, Bulgaria, Sep 7–9 2015.*](https://www.aclweb.org/anthology/R15-1025.pdf)
