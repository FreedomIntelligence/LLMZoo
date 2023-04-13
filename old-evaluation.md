## 🧐 Evaluation and Benchmark

We provide a bilingual, multidimensional comparison across different open-source models with ours.
See [here](llmzoo/eval/README.md) for detailed information regarding the evaluation metrics and criteria.

* Zh

The pair-wise comparison of `Phoenix-chat-7b` model with others.
|  Zh |  Model Score | Phoenix Score |
| --- | --- | --- |
| ChatGPT | 8.41 | 7.35   |
| ChatGLM-6B | 7.56  | 7.15   |
| Baidu-Wenxin |  7.38  | 7.15  | 
| Chinese-Alpaca-13b | 6.02  | 7.54  |
| BELLE-7b-2m | 5.91  |  7.25  |
| Chinese-Alpaca-7b | 5.51  | 7.15  |

 



|  Zh | General | Coherence | Diversity  | Relevance |
| --- | --- | --- | --- | --- |
| Phoenix-chat-7b | #1  |  #1 | #1  |  #1 |
| Chimera-chat-13b |  #2 |  #3  | #3  | #3  |
| Chimera-chat-7b |  #3 |  #2  | #2  | #2  |
| Chinese-Alpaca-7b |  #4 |  #4 | #4  |  #4 |

* En

|  En | General | Coherence | Diversity  | Relevance |
| --- | --- | --- | --- | --- |
| Vicuna-7b |  #1 |  #3 | #1  |  #1 |
| Phoenix-chat-7b | #2  |  #2 | #2  |  #3 |
| Chimera-chat-7b |  #3 |  #1  | #3  | #2  |


|  En | General | Coherence | Diversity  | Relevance |
| --- | --- | --- | --- | --- |
| Vicuna-13b |  #1 |  #2 | #1  |  #1 |
| Chimera-chat-13b |  #2 |  #1  | #2  | #2  |


* 
<!-- | Zh | General | Coherence | Diversity  | Relevance |
| --- | --- | --- | --- | --- |
| ChatGPT | 8.49 | 9.86  | 8.96 | 9.64  |
| Phoenix-chat-7b |  6.49 | 6.08  | 6.49 | 6.85 |
| Chimera-chat-7b | 5.07  | 4.75  | 4.51 |  4.36 |
| Chimera-chat-13b | 4.25  | 3.99  | 4.70  | 4.10 |
 -->
