Before projecting an activation into PC space, you should scale it first using the scaler.

You can also get the cosine similarity of a vector with any PC, if you only care about the direction.

Three models currently available

* **gemma-2-27b**: google/gemma-2-27b-it  
  * No system prompt  
* **qwen-3-32b**: Qwen/Qwen3-32B (thinking disabled)  
* **llama-3.3-70b**: meta-llama/Llama-3.3-70B-Instruct

Viewer: [https://lu-christina.github.io/persona-subspace/viewer/prompting/](https://lu-christina.github.io/persona-subspace/viewer/prompting/)

**ROLE** data in {model}/roles\_240 

* [275 role instructions](https://github.com/lu-christina/persona-subspace/tree/master/roles/data/instructions) with 5 positive system prompts each  
* **responses/** are generated with the role instruction \+ [shared list of 240 questions](https://github.com/lu-christina/persona-subspace/blob/master/traits/data/questions_240.jsonl)  
  * each role has 1200 responses: 240 questions x 5 system prompts  
  * **Keys: system\_prompt, label (pos), prompt\_index, conversation, question\_index, question**  
* **extract\_scores/** contains gpt-4.1-mini labels for each response ([prompt here](https://github.com/lu-christina/persona-subspace/blob/master/roles/prompts.py))  
  * **0** if the model clearly refuses to role-play and is the AI assistant. Refusal to answer the question itself should not count as this category.  
  * **1** if the model does not necessarily refuse to role-play and answers the question but responds in the style of an AI assistant (polite, succinct, uses bulletpoints).  
  * **2** if the model identifies as itself (an AI assistant, Gemma, an LLM, etc.) but has some attributes of the role (e.g. altruistic AI for the role altruist).  
  * **3** if the model is fully playing the role, such as not mentioning being an AI or giving itself another name.  
  * **Key**: **pos\_p{prompt\_index}\_q{question\_index}**  
* **vectors/** contains mean role vectors of shape (n\_layers, hidden\_dims)  
  * **pos\_0, pos\_1, pos\_2, pos\_3**: mean activations for pos prompt type by label  
  * **pos\_all**: mean of all pos activations regardless of label  
* **default\_vectors.pt**  
  * **pos\_1**: Mean activations where keys start with 'pos\_' and score \== 1  
  * **default\_1**: Mean all activations from default system prompted responses  
  * **all\_1:** Weighted mean of pos\_1 and default\_1   
* **pca/** contains layer{layer}\_pos23.pt which tells you what layer and vector types they were generated with  
  * **layer**: what layer  
  * **roles:** role labels for the vectors used  
    * **pos\_2**: list of strings  
    * **pos\_3**: list of strings  
  * **vectors**: vectors used in PCA  
    * **pos\_2**: list of hidden\_dim tensors  
    * **pos\_3**: list of hidden\_dim tensors  
  * **pca\_transformed**: np matrix of transformed vectors (n\_vectors, n\_components)  
  * **variance\_explained**: explained variance ratio  
  * **n\_components**: int  
  * **pca**: sklearn PCA  
  * **scaler**: sklearn StandardScaler

**TRAIT** data in {model}/traits\_240

* [240 trait instructions](https://github.com/lu-christina/persona-subspace/tree/master/traits/data/instructions) with 5 positive system prompts and 5 negative system prompts each  
* **responses/** are generated with the trait instruction \+ [shared list of 240 questions](https://github.com/lu-christina/persona-subspace/blob/master/traits/data/questions_240.jsonl)  
  * each trait has 2400 responses: 240 questions x 5 system prompts x 2 {positive or negative prompted}  
  * **Keys: system\_prompt, label (pos|neg), prompt\_index, conversation, question\_index, question**  
* **extract\_scores/** contains gpt-4.1-mini scores for each response ([prompt here](https://github.com/lu-christina/persona-subspace/blob/master/traits/0_prompts.py))  
  * A score from 0 (trait not present at all) to 100 (trait strongly present)  
  * **Key**: {**pos|neg}\_p{prompt\_index}\_q{question\_index}**  
* **vectors/** contains mean trait vectors of shape (n\_layers, hidden\_dims)  
  * **pos\_neg**: mean(pos) \- mean(neg) for all pairs  
  * **pos\_neg\_50**: mean(pos) \- mean(neg) for pairs with score difference \> 50  
  * **pos\_70**: mean(pos) for all positive responses with score \>= 70  
  * **pos\_40\_70**: mean(pos) for all positive responses with score \>= 40 and \< 70  
* **pca/** contains layer{layer}\_pos-neg50.pt which tells you what layer and vector types they were generated with  
  * **layer**: what layer  
  * **roles:** role labels for the vectors used  
    * **pos\_neg\_50**: list of strings  
  * **vectors**: vectors used in PCA  
    * **pos\_neg\_50**: list of hidden\_dim tensors  
  * **pca\_transformed**: np matrix of transformed vectors (n\_vectors, n\_components)  
  * **variance\_explained**: explained variance ratio  
  * **n\_components**: int  
  * **pca**: sklearn PCA  
  * **scaler**: sklearn StandardScaler

**To look at the PCs**

* Repo: [https://github.com/lu-christina/persona-subspace-plots](https://github.com/lu-christina/persona-subspace-plots)  
* Actually viewing plots: [https://lu-christina.github.io/persona-subspace-plots/{model\_name}/{roles|traits}/pc{i+1}.html](https://lu-christina.github.io/persona-subspace-plots/{model_name}/{roles|traits}/pc{i+1}.html)

**My interp**

* **Qwen Role PCs** \= \['Assistant-like ↔ role-playing', "mystical/transcendent ↔ mundane/irreverent", "empathetic/vulnerable ↔ analytical/predatory", "concrete/practical ↔ abstract/ideological", "thinking/passive ↔ doing/active", "creative/expressive ↔ rigid/constrained"\]  
* **Qwen Trait PCs** \= \["expressive/irreverent ↔ controlled/professional", "analytical ↔ intuitive", "accessible/practical ↔ esoteric/complex", "active/flexible ↔ passsive/rigid", "questioning ↔ confident", "indirect/diplomatic ↔ direct/assertive"\]  
  