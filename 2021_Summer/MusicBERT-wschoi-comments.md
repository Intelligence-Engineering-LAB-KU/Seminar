**논문의 구조를 읽고 작성 시 항상 염두해 둘 것"

- 밑밥: Similar to natural language, music is usually represented in symbolic data format with sequential tokens, and some methods from NLP can be adopted for symbolic music understanding.

   - => 이게 없으면 ACL에 투고할 수가 없음. 

- challenge: 그럼 bert 쓰면 되느냐? 그건 아니지. 그냥 갖다쓰면 안돼. 문제가 달라.
   > However, it is challenging to directly apply the pre-training techniques from NLP to symbolic music because of the difference between natural text data and symbolic music data. 

- c1) First, since music songs are more structural (e.g., bar, position) and diverse (e.g., tempo, instrument, and pitch), encoding symbolic music is more complicated than natural language.
  - => 3.2  OctupleMIDI Encoding (제안)
  - => (ablation) Table 5. Results of different encoding methods. “Accom.” represents accompaniment suggestion task.

- c2) Second, due to the complicated encoding of symbolic music, the pre-training mechanism (e.g., the masking strategy like the masked language model in BERT) should be carefully designed to avoid information leakage in pre-training
   - => 3.3 Masking Strategy
   - => (ablation) Table 6: Results of different masking strategies.
