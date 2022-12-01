# BillBoard
To submit UniEval to [Bidimensional Leaderboards](https://nlp.cs.washington.edu/billboard/#tasks/cnndm/metrics.html) for summarization, we provide the relevant code here.

The input should contain three files, `source-file.jsonl`, `generator-output.jsonl`, and `reference-file.jsonl`. Then please run the following script:
```
./run.sh
```
The results will be presented in five files, representing the scores of each model output in different dimensions (*fluency*, *coherence*, *consistency*, *relevance* and *overall*).
