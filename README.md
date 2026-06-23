# BioRAG Demo

Interactive Streamlit demo for **BioRAG: A Systematic Ablation Study of Retrieval Strategies for Biomedical Question Answering** (BioNLP Workshop @ ACL 2026).

The app has three tabs:

| Tab | What it shows |
|-----|---------------|
| Paper Results | Table 2 + Figures 2–3 from the paper; clinical deployment guidance |
| Live Demo | Ask any neurology question and compare answers across 6 retrieval variants in real time |
| Ablation Explorer | Interactive Table 3 (bootstrap CIs) and Table 6 (k-sensitivity) |

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Groq key
streamlit run app.py
```

A free Groq API key is available at [console.groq.com](https://console.groq.com).  
The corpus is 1,954 PubMed neurology abstracts (2015–2024); questions outside that scope may return low-confidence answers.

## Retrieval variants in the live demo

| Variant | Strategy |
|---------|----------|
| V1 Naive | Dense similarity (BioMedBERT) |
| V2 HyDE | Hypothetical document embedding |
| V3 Reranker | Dense + cross-encoder rerank |
| V4 HyDE + Reranker | HyDE candidate pool + cross-encoder rerank |
| V5 Multi-Query | Query decomposition + rerank |
| V7 Hybrid | BM25 + dense RRF fusion + cross-encoder rerank |

V6 Self-RAG is excluded from the live demo due to Groq free-tier rate limits (~48 s/query); its evaluation results appear in the Paper Results and Ablation Explorer tabs.

## Citation

```
@inproceedings{bhojani2026biorag,
  title     = {BioRAG: A Systematic Ablation Study of Retrieval Strategies for Biomedical Question Answering},
  author    = {Bhojani, Krushil and Waghmare, Mayank and Nandyala, Hima Bindu and Hirpara, Krupali and Yadav, Shekhar and Malhan, Aakash},
  booktitle = {Proceedings of the BioNLP Workshop at ACL 2026},
  year      = {2026}
}
```
