# Pathovoyage 脳 RAG 鎺ュ叆璁捐锛坴0.1锛?
> 鐩爣锛氬湪涓嶇牬鍧忕幇鏈夊鑸富寰幆锛坄agent/runner.py`锛変笌涓変緵搴斿晢瀹㈡埛绔殑鍓嶆彁涓嬶紝涓鸿瘖鏂摼璺紩鍏?> **鍙绱€佸彲寮曠敤銆佸彲瀹¤** 鐨勭梾鐞嗙煡璇嗭紝鎻愬崌鍒ゆ嵁璐ㄩ噺涓庢渶缁堣瘖鏂殑鍙京鎶ゆ€с€?
---

## 0. TL;DR

1. **鍙锛屼絾涓嶈鎶婁綘缁欑殑閭ｄ釜浠撳簱褰撳簱瑁呰繘鍘?*鈥斺€斿畠鍜屾湰椤圭洰鍦ㄤ緷璧栥€侀鍩熴€佽鍙瘉涓婇兘鏄啿绐佺殑銆?2. 浣犵粰鐨?`SNOWTEAM2023/MedRAG`锛圵WW'25锛夋槸 **鐥囩姸/EHR 鈫?鐤剧梾** 鐨勭煡璇嗗浘璋辫瀵?RAG 鐮旂┒鑴氭湰锛?   涓庡畠鍚屽悕鐨勮繕鏈変竴涓?**MedRAG Toolkit**锛圓CL'24 Findings / MIRAGE锛夛紝閭ｄ釜鎵嶆彁渚?   PubMed / StatPearls / Textbooks / Wikipedia 璇枡涓?4 绉嶆绱㈠櫒銆?3. 鎺ㄨ崘鎷嗘垚涓夊眰鐭ヨ瘑锛?*K-RAG锛堟寚鍗?鏁欑涔︽绱級+ KG-RAG锛堢粨鐩磋偁鐥呯悊鏈綋锛? C-RAG锛堢浉浼肩梾渚嬶級**銆?4. 绗竴鏈熷彧鍋?**K-RAG 鐨勮瘝娉曟绱?+ 2 涓帴鍏ョ偣**锛堣鍒掋€佷徊瑁侊級锛?鈥? 澶╁彲璺戦€氾紝榛樿鍏抽棴銆侀浂鍥炲綊銆?
---

## 1. 涓や釜 MedRAG 鐨勮鲸鏋?
| | `SNOWTEAM2023/MedRAG`锛堜綘缁欑殑閾炬帴锛?| `Teddy-XiongGZ/MedRAG` Toolkit |
|---|---|---|
| 璁烘枃 | WWW'25 *KG-Elicited Reasoning for Healthcare Copilot* | ACL'24 Findings *Benchmarking RAG for Medicine* |
| 杈撳叆 | 鎮ｈ€呬富璇?鐥囩姸/娲诲姩鍙楅檺锛圖DXPlus 鍚堟垚 EHR锛?| 鍖诲鑰冮銆佷复搴婇棶棰?|
| 妫€绱㈠璞?| **鐩镐技鐥呬緥**锛圗HR embedding锛? 鐤剧梾-琛ㄧ幇 **鐭ヨ瘑鍥捐氨** | **鏂囩尞鐗囨**锛圥ubMed/StatPearls/Textbooks/Wikipedia锛?|
| 妫€绱㈠櫒 | 鍚戦噺鐩镐技 + KG 閭诲眳 | BM25 / Contriever / SPECTER / MedCPT |
| 杈撳嚭 | 閴村埆璇婃柇 + 娌荤枟寤鸿 + **杩介棶闂** | 甯?snippet 鍑哄鐨勭瓟妗?|
| 浠ｇ爜褰㈡€?| 鐮旂┒鑴氭湰锛坄main.py`銆乣KG_Retrieve.py`锛夛紝鏃犲寘缁撴瀯 | 鍙鐢ㄧ殑 `MedRAG` 绫?|
| 璁稿彲 | **CC BY-NC 4.0锛堢鍟嗙敤锛?* | 浠ｇ爜寮€婧愶紱璇枡鍚勮嚜璁稿彲锛圫tatPearls 涓?CC BY-NC-ND锛?|
| 瀵规湰椤圭洰浠峰€?| **鎺ㄧ悊鑼冨紡**锛欿G 鎵╁睍 + 閴村埆瀵圭収 + 杩介棶 | **璇枡涓庢绱㈠櫒** |

缁撹锛?*鑼冨紡鍊?SNOWTEAM锛岃鏂?妫€绱㈠櫒鍙傝€?Toolkit锛岃惤鍦颁唬鐮佽嚜宸卞啓銆?*

---

## 2. 涓轰粈涔堜笉鑳界洿鎺ユ妸 SNOWTEAM 鐨勪粨搴撹杩涙潵

| 鍐茬獊椤?| 瀹冪殑瑕佹眰 | 鏈」鐩姹?| 鍚庢灉 |
|---|---|---|---|
| torch | `2.4.1`锛堥粯璁?cu121 杞瓙锛?| `>=2.7`锛坈u128锛孯TX 50 绯?sm_120 蹇呴渶锛?| 瑁呬笂鍗抽檷绾э紝PLIP 璺戜笉鍔?|
| numpy | `1.26.4` | `>=2.2.6,<2.4` | 鍐茬獊锛宲ip 瑙ｄ笉鍑哄彲琛岃В |
| transformers | `4.45.1` | `>=4.46` | 涓?PLIP 鍔犺浇璺緞鍙兘涓嶅吋瀹?|
| faiss | `faiss-cpu 1.8.0` | 鏈紩鍏?| 棰濆閲嶄緷璧栵紝鍙负 1 涓绱㈠櫒涓嶅€煎緱 |
| 棰嗗煙 | DDXPlus 51 涓垵绾ц瘖鏂?+ 鎱㈢棝 EHR | 缁撶洿鑲犵檶 **褰㈡€佸浜氬瀷 / 鍒嗙骇 / 鍒嗘湡** | 瀹冪殑 KG 鑺傜偣瀵规垜浠棤鎰忎箟 |
| 褰㈡€?| 鑴氭湰 + 鍏ㄥ眬 `authentication.py` 鍗犱綅 key + `config.py` | 鍖呭寲 `src/giant`锛岄厤缃蛋 pydantic-settings | 鏃犳硶 import 澶嶇敤 |
| 璁稿彲 | CC BY-NC 4.0 | 骞冲彴鍖栧ぇ姒傜巼鍟嗙敤 | **娉曞緥椋庨櫓锛屽繀椤荤粫寮€** |

**鍙鍋氭硶**锛氭妸瀹冪殑 4 涓€屾€濊矾銆嶉噸鍐欐垚鎴戜滑鐨勬ā鍧楋紙鎬濊矾涓嶅彈鐗堟潈淇濇姢锛屼唬鐮佷笌鎻愮ず璇嶆枃鏈噸鏂版挵鍐欙級銆?
---

## 3. 鍙€熼壌鐨勫洓涓ā寮?
1. **鍒嗗眰鑱氬悎寤哄浘**锛歚Level1 澶х被 鈫?Level2 浜氱被 鈫?Level3 鍏蜂綋璇婃柇`锛屾瘡涓妭鐐规寕銆岃〃鐜?鍒ゆ嵁銆嶆弿杩般€?   鏄犲皠鍒扮梾鐞嗭細`缁撶洿鑲犵檶 鈫?鑵虹檶/榛忔恫鑵虹檶/鍗版垝缁嗚優鐧?楂撴牱鐧屸€?鈫?WHO 鍏蜂綋浜氬瀷`锛岃妭鐐规寕闀滀笅鍒ゆ嵁銆?2. **KG 璇卞鎺ㄧ悊**锛氭绱㈠懡涓妭鐐瑰悗锛屾部鍥炬墿灞曞埌**鍏勫紵鑺傜偣**锛屾樉寮忕敓鎴愩€屽繀椤讳笌涔嬪尯鍒嗙殑 N 涓瘖鏂?+ 鍚勮嚜鍏抽敭鐗瑰緛銆嶃€?   杩欐鏄梾鐞嗛壌鍒瘖鏂殑褰㈡€侊紙榛忔恫鑵虹檶 vs 榛忔恫婀?vs 鍗版垝缁嗚優锛夈€?3. **閴村埆宸紓澧炲己锛坉iagnostic difference锛?*锛氭妸鏄撴贩瀵瑰仛鎴愭垚瀵规潯鐩杩涙彁绀鸿瘝锛屾瘮娉涙硾鐨勩€岃瘖鏂爣鍑嗐€嶆湁鏁堛€?4. **杩介棶鐢熸垚锛坒ollow-up questions锛?*锛氫俊鎭笉瓒虫椂鍏堥棶 2鈥? 涓緞娓呴棶棰橈紱骞冲彴渚у彲鍙樻垚銆屽緟琛ュ厖涓村簥淇℃伅銆嶆竻鍗曘€?
---

## 4. 鐩爣鏋舵瀯

```
                      鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ KnowledgeService锛堟噿鍔犺浇銆佸彲娉ㄥ叆锛?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? User question 鈹€鈹€鈹€鈹€鈹€鈹€鈻垛攤 K-RAG  鎸囧崡/鏁欑涔︾墖娈? BM25 + MedCPT dense 鈫?RRF 铻嶅悎          鈹?                      鈹?KG-RAG 缁撶洿鑲犵梾鐞嗘湰浣?  鑺傜偣鍛戒腑 鈫?閭诲眳鎵╁睍 鈫?閴村埆鐐?            鈹傗攢鈹€鈹? Historical cases 鈹€鈹€鈹€鈻垛攤 C-RAG  鐩镐技鐥呬緥/杞ㄨ抗    PLIP 鎴栧垏鐗?embedding + 缁撴瀯鍖栬繃婊?       鈹? 鈹?                      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹? 鈹?                                                                                          鈻?                                                                    EvidenceBundle锛堝甫 id/鏉ユ簮/鍒嗘暟锛?                                                                                          鈹?        鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        鈻?                              鈻?                            鈻?  鈶?瑙勫垝闃舵                     鈶?浼氳瘖闃舵                    鈶?浠茶闃舵
  plan_retrieval /             姣忎釜 crop 鐨?Patho-R1          arbitrate_patho_r1
  plan_observation             銆屽垽鎹崱銆嶏紙鈮?50 璇嶏級          銆屾寚鍗楄瘉鎹€嶅尯鍧?+ 寮曠敤鏍￠獙
        鈹?                              鈹?                            鈹?        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                                        鈻?                         Trajectory.rag_evidence[] + Turn.evidence_ids[]
                                        鈻?                            鏈€缁堣瘖鏂紙鍙拷婧埌鍏蜂綋鎸囧崡鏉℃枃锛?```

### 4.1 浜斾釜鎺ュ叆鐐癸紙鏂囦欢绾э級

| # | 浣嶇疆 | 鐜扮姸 | 鎺ュ叆鍚?|
|---|---|---|---|
| 鈶?| `llm/plip_planner.py:270 plan_retrieval` | 绾?LLM 鎶婇棶棰樼炕鎴愭绱㈡枃鏈紱缁村害鏉冮噸鍑粡楠?| 鎰忓浘鍒嗙被鍚庡厛妫€绱㈣闂鐨勭梾鐞嗗垽鎹紝鐢?evidence 鐗囨绾︽潫鐢熸垚鐨勬绱㈢淮搴︼紙鏈鏇存帴杩戠梾鐞嗚涔夛紝PLIP 鍙洖鏇村噯锛?|
| 鈶?| `llm/plip_planner.py:297 plan_observation` | `criteria` 鐢?Qwen 鍑┖鐢熸垚锛屾崲棰樺嵆婕?| `criteria` 鏉ヨ嚜鎸囧崡 evidence锛宍ObservationPlan` 鏂板 `evidence: list[Evidence]`锛堥粯璁ょ┖锛屽悜鍚庡吋瀹癸級 |
| 鈶?| `llm/patho_r1_client.py:20 _CONSULT_PROMPT`锛坈riteria 娉ㄥ叆鍦?:127锛?| 姣忎釜 crop 甯︿竴浠借嚜閫犲垽鎹?| 鏀逛负**鍒ゆ嵁鍗?*锛?鈥? 鏉℃寚鍗楄鐐?+ 2 鏉￠壌鍒偣 + 褰撳墠鍊嶇巼浠诲姟锛汦vidence id 闅?`Turn` 钀藉簱 |
| 鈶?| `agent/runner.py:1224 _arbitrate_patho_r1` 鈫?`plip_planner.py:318 arbitrate_patho_r1` | 浠茶鑰呭彧鐪嬩笓瀹舵枃瀛?| 澧炲姞 `# 鎸囧崡璇佹嵁` 鍖哄潡锛涜姹傛瘡鏉＄粨璁烘爣娉?`[E1]`锛涜緭鍑哄悗鍋?*寮曠敤鏍￠獙**锛堟棤鍑哄鐨勬柇瑷€闄嶆潈/鍓旈櫎锛?|
| 鈶?| `agent/trajectory.py:66鈥?7` + `cli/main.py:57 run` | 杞ㄨ抗鍙湁 plip_plan / patho_r1_plan / arbitration | 杞ㄨ抗鏂板 `rag_evidence`锛汣LI 鏂板 `--rag/--no-rag`銆乣--rag-k`銆乣--rag-mode`锛屽瓙鍛戒护 `giant knowledge build|status|search` |

### 4.2 涓轰粈涔堜紭鍏?鈶犫憿鈶?鑰屼笉鏄厛鎺?鈶?
- 鈶紙Patho-R1 鍒ゆ嵁锛夋槸**褰撳墠鏈€寮辩殑涓€鐜?*锛氫笓瀹舵ā鍨嬪緢鍚冦€屼綘璁╁畠鐪嬩粈涔堛€嶏紝鑰岀幇鍦ㄨ繖浠姐€屼粈涔堛€嶆槸鍙︿竴涓?LLM 缂栫殑銆?- 鈶ｏ紙浠茶锛夋槸鏈€缁堢瓟妗堢殑鍑哄彛锛屾帴璇佹嵁鏀剁泭鏈€澶т笖**涓€娆¤皟鐢?*锛屾垚鏈彲鎺с€?- 鈶★紙姣?crop 涓€娆′細璇婏級璋冪敤閲?= 姝ユ暟锛屾敞鍏ュ繀椤绘瀬搴﹀帇缂╋紙鈮?00 tokens锛夛紝鍚﹀垯寤惰繜/鎴愭湰绾挎€т笂娑ㄣ€?
---

## 5. 鏁版嵁缁撴瀯锛堟柊澧烇紝鍏ㄩ儴 frozen pydantic锛?
```python
class Passage(BaseModel, frozen=True):
    id: str                      # "WHO-CRC-2019#3.2.1"
    title: str                   # "WHO Classification of Tumours: Digestive System"
    source: str                  # "who" | "csco" | "nccn" | "textbook" | "case"
    section: str | None = None
    text: str
    score: float = 0.0
    meta: dict[str, str] = {}    # edition / year / url

class EvidenceBundle(BaseModel, frozen=True):
    query: str
    stage: Literal["planning", "consult", "arbitration"]
    passages: list[Passage] = []
    discriminators: list[str] = []   # KG 浜у嚭鐨勩€屽繀椤诲尯鍒嗐€嶆潯鐩?    corpus_version: str = ""
```

`Turn` 澧炲姞 `evidence_ids: list[str] = []`锛沗Trajectory` 澧炲姞 `rag_evidence: list[Passage] = []`銆?
---

## 6. 妯″潡甯冨眬

```
src/giant/knowledge/
  __init__.py      # KnowledgeService 闂ㄩ潰锛堟噿鍔犺浇锛宺unner 閫氳繃 AgentConfig 娉ㄥ叆锛?  schemas.py       # Passage / EvidenceBundle / KGNode
  corpus.py        # build_index(): data/corpus/*.jsonl 鈫?鍒嗗潡 鈫?绱㈠紩钀界洏
  retriever.py     # BM25Retriever | DenseRetriever(MedCPT) | HybridRRF
  kg.py            # ColorectalKG: data/kg/colorectal.yaml 鈫?涓夊厓缁?+ 閭诲眳鎵╁睍
  cards.py         # criteria_card(bundle, bucket) -> str锛堝帇缂╁埌 token 棰勭畻鍐咃級
  service.py       # retrieve_for_planning / _consult / _arbitration锛堝甫 diskcache锛?data/corpus/*.jsonl
data/kg/colorectal.yaml
```

`pyproject.toml` 澧炲姞鍙€?extra锛堟部鐢ㄧ幇鏈?`plip` extra 鐨勫仛娉曪級锛?
```toml
[project.optional-dependencies]
rag = ["rank_bm25>=0.2.2", "jieba>=0.42.1"]
```

dense 妫€绱㈠鐢ㄥ凡鏈夌殑 `torch`/`transformers`锛坄--extra plip` 宸插甫锛夛紝涓嶅紩鍏?faiss锛氳鏂欏彧鏈夊嚑鍗冩潯鏃讹紝
numpy 鏆村姏鍐呯Н瓒冲锛坄np.argsort(-X @ q)`锛夛紝閬垮厤鏂伴噸渚濊禆銆?
---

## 7. 妫€绱㈢瓥鐣ヤ笌涓婁笅鏂囬绠?
| 闃舵 | 妫€绱?| 娉ㄥ叆閲?| 澶囨敞 |
|---|---|---|---|
| 瑙勫垝 | top-6 鈫?鍘嬫垚 5 鏉¤鐐?| 鈮?.5k tokens | 鍙窇涓€娆?|
| 浼氳瘖锛堟瘡 crop锛?| 涓嶆绱紝澶嶇敤瑙勫垝闃舵鐨?*鍒ゆ嵁鍗?*锛屾寜鍊嶇巼鍒囨 | 鈮?00 tokens | 姣?run 鍙畻涓€娆★紝闅?crop 澶嶇敤 |
| 浠茶 | top-8锛堥棶棰?涓撳缁撹鍏抽敭璇嶏級 | 鈮?k tokens | 鍙窇涓€娆?|

- 铻嶅悎锛歊RF锛坄score = 危 1/(60+rank)`锛夛紝BM25 涓?dense 鍚勫彇 top-20銆?- 涓枃锛欱M25 闇€鍒嗚瘝锛孭0 鐢?`jieba`锛涗篃鍙厛鐢ㄣ€屽瓧绗?bigram銆嶉伩鍏嶆柊渚濊禆銆?- 缂撳瓨锛氬凡鏈?`diskcache` 渚濊禆锛宬ey = `(corpus_version, mode, stage, query_hash, k)`銆?- 鎴愭湰锛氭湰鍦版绱?鈮?涓€娆?VLM 璋冪敤锛涚湡姝ｅ閲忔槸娉ㄥ叆鐨?token锛堜及绠?+8%~15%/run锛屽彇鍐充簬 top-k锛夈€?
---

## 8. 鍒嗛樁娈靛疄鏂?
**P0锛?鈥? 澶╋紝鍙珛鍗抽獙璇佷环鍊硷級**
- 鎵嬪伐绛栧睍 30鈥?0 鏉＄粨鐩磋偁鐥呯悊璇枡锛圵HO 浜氬瀷鍒ゆ嵁銆丆SCO/NCCN 瑕佺偣銆佸父瑙侀壌鍒偣锛夛紝鑻辨枃+涓枃瀵圭収銆?- BM25 + jieba锛宍KnowledgeService.retrieve()` 鎵撻€氥€?- 鎺?鈶犫憿鈶ｏ細瑙勫垝闃舵娉ㄥ叆銆佷徊瑁佹敞鍏?+ 寮曠敤鏍￠獙锛沗--rag/--no-rag` 寮€鍏炽€?- 杞ㄨ抗钀?`rag_evidence`锛沗giant knowledge build|status`銆?
**P1锛堢害 1 鍛級**
- MedCPT dense + RRF 铻嶅悎锛涘垽鎹崱鍘嬬缉閫昏緫锛汯G YAML锛堜簹鍨嬪眰绾?+ 閴村埆瀵癸級涓庨偦灞呮墿灞曘€?- 鎺?鈶★紙姣?crop 鍒ゆ嵁鍗★級锛涘紩鐢ㄦ牎楠屾寚鏍囧寲銆?
**P2锛堝钩鍙板寲锛?*
- C-RAG锛氬巻鍙插垏鐗?杞ㄨ抗寤虹储寮曪紙PLIP 鍥惧悜閲忔垨杞ㄨ抗鎽樿鍚戦噺锛夛紝鐩镐技鐥呬緥 few-shot + 涓€鑷存€ф牎楠屻€?- 杩介棶鑺傜偣锛堝€?SNOWTEAM 鐨?follow-up 鑼冨紡锛夛細闂淇℃伅涓嶈冻鏃惰緭鍑恒€屽緟琛ュ厖涓村簥淇℃伅銆嶃€?- 涓?`refactor/langgraph-orchestration-v1` 瀵归綈锛歊AG 浣滀负涓€涓嫭绔嬭妭鐐规敞鍏ワ紝涓嶆敼 State 濂戠害銆?
---

## 9. 璇勬祴涓庡洖褰?
1. **閰嶅瀵圭収**锛氬悓涓€鎵瑰垏鐗囪窇 `--rag` / `--no-rag`锛屾瘮杈?answer銆乤greement銆乧ost銆乴atency銆?2. **寮曠敤蹇犲疄搴?*锛氫徊瑁佽緭鍑洪噷姣忔潯鏂█鑳藉惁鏄犲皠鍒?`[E#]`锛涙棤鍑哄姣斾緥浣滀负鏍稿績鎸囨爣锛堢洰鏍?<10%锛夈€?3. **妫€绱㈣川閲?*锛氫汉宸ユ爣娉?20 棰?脳 top-6 鍛戒腑鐜囷紙P@3銆丮RR锛夈€?4. **鍥炲綊**锛氶粯璁?`RAG_ENABLED=false`锛宍ObservationPlan`/`Turn` 鏂板瓧娈靛甫榛樿鍊硷紝纭繚 langgraph 鍒嗘敮鐨?parity 娴嬭瘯涓嶅彈褰卞搷銆?
---

## 10. 鍚堣涓庤鍙?
| 璧勪骇 | 璁稿彲 | 缁撹 |
|---|---|---|
| SNOWTEAM2023/MedRAG 浠ｇ爜 | CC BY-NC 4.0锛?*绂佸晢鐢?*锛?| 鍙€熸€濊矾锛屼唬鐮佷笌鎻愮ず璇嶅叏閮ㄩ噸鍐?|
| MedRAG Toolkit 璇枡 | 浠ｇ爜寮€婧愶紱Textbooks 婧愯嚜 MedQA锛堝師涔︾増鏉冿級銆丼tatPearls 涓?CC BY-NC-ND銆乄ikipedia CC BY-SA銆丳ubMed 鎽樿鍙?NLM 鏉℃ | 鐮旂┒鍙敤锛涘晢鐢ㄥ繀椤绘浛鎹负鎺堟潈璇枡 |
| 鎺ㄨ崘鑷缓璇枡 | WHO 娑堝寲绯荤粺鑲跨槫鍒嗙被锛堟巿鏉冨壇鏈級銆丆SCO/NCCN 鎸囧崡锛堟巿鏉冿級銆侀櫌鍐呰劚鏁忕梾鐞嗘姤鍛?| 鍟嗙敤鍞竴骞插噣璺緞 |
| 鎮ｈ€呮暟鎹紙C-RAG锛?| 鈥?| 鍒囩墖涓庢姤鍛婇渶鑴辨晱銆佽闂帶鍒躲€佸璁℃棩蹇?|

---

## 11. 椋庨櫓娓呭崟

| 椋庨櫓 | 褰卞搷 | 缂撹В |
|---|---|---|
| 妫€绱㈠櫔澹版薄鏌撲徊瑁?| 寮曞叆閿欒缁撹 | `RAG_MIN_SCORE` 闃堝€?+ 寮曠敤鏍￠獙 + 榛樿鍏抽棴 |
| Patho-R1 涓婁笅鏂囪鎾戠垎锛堟湰鍦?7B锛宍max_tokens=256`锛?| 杈撳嚭閫€鍖?| 鍒ゆ嵁鍗＄‖鎬?鈮?00 tokens锛岃秴闀挎埅鏂?|
| 璇枡鐗堟湰婕傜Щ | 缁撹涓嶅彲澶嶇幇 | `corpus_version` 鍐欏叆杞ㄨ抗 |
| 涓?langgraph 鍒嗘敮鍐茬獊 | 鍚堝苟鍥伴毦 | RAG 鍋氭垚娉ㄥ叆渚濊禆锛屼笉鏀?State 濂戠害 |
| 鍟嗙敤璁稿彲 | 娉曞姟闃绘柇 | 璇枡璧拌嚜寤?鎺堟潈璺嚎锛堣 搂10锛?|

---

## 12. P0 瀹炵幇鐘舵€侊紙宸插畬鎴愶級

### 12.1 鏂板鏂囦欢

| 鏂囦欢 | 浣滅敤 |
|---|---|
| `src/giant/knowledge/schemas.py` | `Passage` / `EvidenceBundle` / `KnowledgeStats`锛坒rozen pydantic锛?|
| `src/giant/knowledge/tokenize.py` | 杞婚噺鍒嗚瘝锛氳嫳鏂囧皬鍐?鍘诲仠鐢ㄨ瘝+鏋佺畝璇嶅共锛涗腑鏂囨寜瀛楃浜屽厓缁?|
| `src/giant/knowledge/lexicon.py` | 涓啋鑻辩梾鐞嗘湳璇〃锛堢害 120 鏉★級锛屼腑鏂囬棶棰?鈫?鑻辨枃妫€绱㈣瘝锛岀绾垮彲澶嶇幇 |
| `src/giant/knowledge/bm25.py` | 鑷寘鍚?Okapi BM25 鍊掓帓绱㈠紩锛堟棤绗笁鏂逛緷璧栵級 |
| `src/giant/knowledge/corpus.py` | 鍘诲櫔锛堣瘝绮樿繛妫€娴嬶級銆佸彞瀛愮骇婊戠獥鍒囧垎銆佽鏂欐寚绾?|
| `src/giant/knowledge/selection.py` | 鎸?`id_range` / `regex` / 鍏ㄩ噺鐨勮婧愰€夋嫨瑙勫垯 |
| `src/giant/knowledge/cards.py` | 鍒ゆ嵁鍗★紙鈮?00 瀛楃锛変笌缂栧彿璇佹嵁鍧楋紙`[E1]鈥锛?|
| `src/giant/knowledge/service.py` | `KnowledgeService` 闂ㄩ潰锛氭噿鍔犺浇銆佹绱€佺浉瀵瑰垎鏁版埅鏂?|
| `data/corpus/sources.json` | 璇枡绛栧睍瑙勫垯锛堝彲瀹¤銆佸彲澶嶇幇锛?|
| `data/corpus/curated_colorectal.jsonl` | 14 鏉′汉宸ュ垽鎹崱锛堜簹鍨?鍒嗙骇/鍒嗘湡/鍒囩紭/LVI/PNI/鍑鸿娊/MMR锛夛紝**寰呯梾鐞嗕笓瀹跺鏍?* |
| `data/corpus/textbooks_colorectal.jsonl` | 鏋勫缓浜х墿锛?41 鐗囨 / 247k 瀛楃 / 鎸囩汗 `6b21d2539d8e`锛?|
| `tests/test_knowledge.py` | 20 涓绾挎祴璇曪紙鍚彁绀鸿瘝娉ㄥ叆鏂█锛?|

### 12.2 鎺ュ叆鐐癸紙淇敼鐨勬枃浠讹級

| 浣嶇疆 | 鏀瑰姩 |
|---|---|
| `config.py` | 鏂板 `RAG_*` 閰嶇疆锛堥粯璁ゅ叧闂級 |
| `cli/main.py` | 鏂板 `giant knowledge build/status/search`锛沗run` 鏂板 `--rag/--no-rag`銆乣--rag-corpus`銆乣--rag-k` |
| `cli/runners.py` | 鎸夊紑鍏虫瀯閫?`KnowledgeService`锛屾敞鍏?`AgentConfig`锛涢绛涘墠鍏堟绱㈡湳璇?|
| `agent/runner.py` | `AgentConfig.knowledge`锛沗_retrieve_evidence()`锛涜鍒掗樁娈垫绱㈠苟鍐欏叆杞ㄨ抗锛涗徊瑁佸墠鑱斿悎妫€绱紱姣忎釜 crop 鐨?`Turn.evidence_ids` |
| `llm/plip_planner.py` | `plan_retrieval/plan_observation/arbitrate_patho_r1` 鎺ュ彈 `evidence`锛屾妸璇佹嵁娉ㄥ叆鎻愮ず璇?|
| `agent/trajectory.py` | `Trajectory.rag_evidence`銆乣rag_corpus_version`锛沗Turn.evidence_ids` |

### 12.3 鐢ㄦ硶

```bash
# 1) 涓嬭浇璇枡锛堝凡涓嬭浇 3 涓簮鏂囦欢鍒?data/rag_raw/锛?#    鏉ヨ嚜 huggingface.co/MedRAG 鐨?textbooks 瀛愰泦
# 2) 鏋勫缓
giant knowledge build
# 3) 璇曟绱紙涓枃鍙敤锛?giant knowledge search "鑲跨槫鍑鸿娊鎬庝箞鍒嗙骇" --k 3
# 4) 甯︾煡璇嗘绱㈣窇璇婃柇
giant run slide.svs -q "缁勭粐瀛︿簹鍨嬫槸浠€涔堬紵" -p qwen --patho-r1 --rag
```

### 12.4 妫€绱㈠疄娴?
| 涓枃闂 | Top-1 鍛戒腑 | 鍒嗘暟 |
|---|---|---|
| 鍒囩紭鏄惁骞插噣锛圧0锛?| Curated锛歊0/R1 瀹氫箟 + 鐩磋偁 CRM 鈮? mm | 15.4 |
| 鏈夋病鏈夎剦绠＄槫鏍?| Curated锛歀VI 瀹氫箟涓庨鍚庢剰涔?| 11.8 |
| 鑲跨槫鍑鸿娊鎬庝箞鍒嗙骇 | Curated锛欱d1/Bd2/Bd3 鐑偣璁℃暟 | 23.8 |
| 榛忔恫鑵虹檶鐨勭粍缁囧浜氬瀷鏄粈涔?| Curated锛歐HO 绗?5 鐗堜簹鍨嬫竻鍗?| 40.6 |
| T 鍒嗘湡鐪嬫蹈娑︽繁搴﹀悧 | Schwartz锛歍 鍒嗘湡鏄穻宸寸粨杞Щ鏈€寮洪娴嬪洜瀛?| 15.2 |

### 12.5 璇枡韪╁潙璁板綍锛堥噸瑕侊級

1. **Robbins 鐨勬秷鍖栭亾绔犺妭鍦?PDF 鎶藉彇涓崯鍧?*锛氱害 16% 鐗囨璇嶈竟鐣屼涪澶憋紙`Rupturedovarianmucinoustumors鈥锛夛紝
   BM25 鏃犳硶鍛戒腑銆傝繖绫荤墖娈垫寜銆岀┖鏍煎崰姣?< 8% 鎴栬瘝闀夸腑浣嶆暟 > 16銆嶅垽鎹涪寮冿紝鍏ㄤ功浠呬繚鐣欏己鐗瑰緛鍙ャ€?2. **鍏抽敭璇嶅叏灞€杩囨护涓嶅彲闈?*锛歚colon` 浼氬懡涓€岀洿鑲犳椿妫€鏌ユ穩绮夋牱鍙樸€嶃€傛敼鐢?*绔犺妭浣嶇疆瀹氫綅**鈥斺€?   缁撶洿鑲犺瘝棰戝湪鏂囨。椤哄簭涓婄殑杩炵画瀵嗛泦娈碉紙Schwartz 缁撶洿鑲犵珷鑺傜殑瀵嗗害鏄叾浠栧尯鍧楃殑 60 鍊嶏級銆?3. **涓枃娈佃惤澶╃劧鏃犵┖鏍?*锛屼笉鑳藉鐢ㄥ櫔澹板垽鎹紱涓枃鎸夊瓧绗︿簩鍏冪粍寤虹储寮曪紝閬垮厤寮曞叆鍒嗚瘝鍣ㄣ€?4. **涓嫳娣锋帓鏌ヨ鐨勮鐩栫巼闂ㄦ浼氳鏉€**锛氭敼涓恒€岀浉瀵规渶楂樺垎 25%銆嶆埅鏂紝璇█鏃犲叧銆?5. MedRAG 鐨?StatPearls / PubMed / Wikipedia 璇枡浠呮湁鍏冩暟鎹紙9.9 涓囩瘒銆?3.9M 鐗囨閲忕骇锛夛紝
   鏈湴涓嶅彲琛岋紱P0 鍙敤 textbooks銆?
### 12.6 宸茬煡闄愬埗 / P1 寰呭姙

- 鍙仛 **BM25 璇嶆硶妫€绱?*锛沝ense锛圡edCPT锛? RRF 铻嶅悎鏈疄鐜般€?- **鈶?姣?crop 鍒ゆ嵁鍗℃湭鎺?*锛氬垽鎹崱宸茶兘鐢熸垚锛屼絾灏氭湭閫?crop 娉ㄥ叆 Patho-R1锛堥绠椾笌寤惰繜闇€瀹炴祴锛夈€?- 寮曠敤鏍￠獙鍙湁鎻愮ず璇嶇害鏉燂紝灏氭棤銆屾棤鍑哄鏂█銆嶇殑鑷姩鍓旈櫎涓庢寚鏍囩粺璁°€?- 璇枡浠?Schwartz 澶栫绔犺妭涓轰富锛?*缁勭粐瀛︽弿杩板亸钖?*锛汸1 搴旇ˉ WHO 鍒嗙被鍘熸枃鎴栨巿鏉冩寚鍗椼€?- 14 鏉?curated 鍒ゆ嵁闇€鐥呯悊鍖荤敓澶嶆牳鍚庢柟鍙綔涓轰复搴婁緷鎹€?- 绔埌绔紙鍚湡瀹?LLM 璋冪敤锛夋湭楠岃瘉锛氭湰鏈烘棤 DashScope Key銆佹棤 Patho-R1 鏈嶅姟銆佹湭瑁?torch/PLIP銆?
### 12.7 楠岃瘉璁板綍

```
$ python -m pytest tests/test_knowledge.py -q
20 passed in 2.98s

$ python -m compileall -q src/giant          # exit 0
$ python -m giant.cli.main knowledge status
鐗囨: 541 / 瀛楃: 247348 / 鎸囩汗: 6b21d2539d8e
璇簮: Curated_CRC_Criteria, Pathology_Robbins, Pathoma_Husain, Surgery_Schwartz
```

楠岃瘉鐢ㄧ幆澧冿細`<repo>/.venv`锛圥ython 3.12.7锛岃 pydantic/pytest/typer/pydantic-settings/
structlog/rich/aiolimiter/tenacity/openai/anthropic/httpx/pillow/numpy/opencv-python-headless/scipy/
diskcache/scikit-image/openslide-python/openslide-bin锛夛紝杩愯闇€ `PYTHONPATH=src`銆?
---

## 13. P0.5锛氭绱㈤椄闂?+ 缁撴瀯鍖栦徊瑁侊紙鏋舵瀯涓庤鏂欏潎鏈敼鍔級

鐩爣鏄湪**涓嶅姩涓诲惊鐜粨鏋勩€佷笉鎹㈣鏂?*鐨勫墠鎻愪笅锛屾彁鍗囦袱浠朵簨锛氬噯纭巼锛堜笉娉ㄥ叆閿欑煡璇嗭級涓?鍙В閲婃€э紙缁撹鍙粺璁°€佸彲杩芥函锛夈€?
### 13.1 鏂板/鏀瑰姩

| 鏂囦欢 | 鏀瑰姩 |
|---|---|
| `knowledge/gate.py`锛堟柊锛?| 纭畾鎬ф绱㈤椄闂細闂閿氱偣瑕嗙洊 + **璇枡鍩熻寖鍥?*鍒ゅ畾锛涗笉璋冪敤 LLM锛岄浂鎴愭湰銆佸彲澶嶇幇 |
| `knowledge/schemas.py` | `CitationReport` / `WeightedEvidence` / `ArbitrationResult` |
| `knowledge/cards.py` | `citation_check()`锛氱粺璁?`[E#]` 鐨?used / unused / invalid / coverage |
| `llm/plip_planner.py` | `arbitrate_structured()`锛氳姹?JSON 杈撳嚭骞惰В鏋愶紱澶辫触閫€鍖栦负绾枃鏈紱`arbitrate_patho_r1` 淇濈暀涓哄吋瀹瑰寘瑁?|
| `agent/runner.py` | `_retrieve_evidence()` 妫€绱㈠悗杩囬椄闂ㄥ苟璁板綍鍒ゅ畾锛涗徊瑁佹敼鐢ㄧ粨鏋勫寲缁撴灉锛涜建杩规柊澧?`rag_gate` / `final_citation` |
| `agent/trajectory.py` | 鏂板 `rag_gate: list[dict]`銆乣final_citation: dict | None` |

### 13.2 闂搁棬鐨勫疄娴嬭涓猴紙鐪熷疄璇枡锛?41 鐗囨锛?
| 闂 | 鍒ゅ畾 | 鎷︽埅鍘熷洜 | 娉ㄥ叆鏉℃暟 |
|---|---|---|---|
| 缁勭粐瀛︿簹鍨嬫槸浠€涔堬紵 | relevant | 閿氱偣瑕嗙洊 5 | 3/6 |
| 鑲跨槫鍑鸿娊鎬庝箞鍒嗙骇 | relevant | 閿氱偣瑕嗙洊 6 | 5/6 |
| 鍒囩紭鏄惁骞插噣锛圧0锛?| relevant | 閿氱偣瑕嗙洊 3 | 6/6 |
| 缁撶洿鑲犵檶鑲濊浆绉荤殑鍒囩紭瑕佹眰 | relevant | 鍚粨鐩磋偁锛屾斁琛?| 6/6 |
| 鑲鸿吅鐧岀殑 EGFR 绐佸彉鐜囨槸澶氬皯锛?| **irrelevant** | `out_of_scope_organ=鑲篳 | 0 |
| 涔宠吅鐧岀殑鍒嗗瓙鍒嗗瀷鏈夊摢浜涳紵 | **irrelevant** | `out_of_scope_organ=涔宠吅` | 0 |
| 鑳冪檶鐨?Lauren 鍒嗗瀷 | **irrelevant** | `out_of_scope_organ=鑳僠 | 0 |

鍏抽敭鏁欒锛?*鍙潬璇嶉潰瑕嗙洊鎸′笉浣忚法鍣ㄥ畼鍣０**鈥斺€斻€岃偤鑵虹檶鐨勭粍缁囧浜氬瀷銆嶅拰
銆岀粨鐩磋偁鑵虹檶鐨勭粍缁囧浜氬瀷銆嶅叡浜?adenocarcinoma / histology / subtype 涓変釜閿氱偣銆?蹇呴』鍙犲姞璇枡鍩熻寖鍥村垽瀹氾紙璇枡鏄粨鐩磋偁涓撶敤鐨勶紝杩欎竴鐐瑰啓鍦?`sources.json` 閲岋級銆?
### 13.3 缁撴瀯鍖栦徊瑁佺殑瀛楁

```json
{
  "answer": "娴告鼎鎬ч粡娑茶吅鐧?,
  "weighted_evidence": [{"region": "(70000,65000,12000,12000)", "opinion": "榛忔恫婀?50%",
                         "weight": 0.8, "why": "浣庡€?楂樺€嶅叡 3 涓尯鍩熶竴鑷?}],
  "conflicts": ["鍖哄煙 A 鍒ら珮鍒嗗寲锛屽尯鍩?B 鍒ら粡娑茶吅鐧岋紙鑲跨槫寮傝川鎬э級"],
  "minority": ["鍖哄煙 A 鐨勯珮鍒嗗寲缁撹"],
  "citations": ["E1", "E2"]
}
```

瑙ｆ瀽澶辫触鏃惰嚜鍔ㄩ€€鍖栦负銆屽師鏂囧嵆绛旀銆嶏紙`structured=false`锛夛紝涓嶉樆鏂祦绋嬶紱
`citations` 鍙帴鍙?`E\\d+` 褰㈠紡锛岄潪娉曠紪鍙疯涓㈠純骞惰鍏?`invalid`銆?
### 13.4 寮曠敤缁熻锛堝彲瑙ｉ噴鎬х殑閲忓寲鍙ｅ緞锛?
```
answer: 娴告鼎鎬ч粡娑茶吅鐧岋紙渚濇嵁 [E1] 鐨勪簹鍨嬫竻鍗曚笌 [E2] 鐨勫垎绾ф爣鍑嗭級
鈫?used=[E1,E2]  unused=[E3,E4,E5,E6]  invalid=[]  coverage=0.333
```

`coverage` 涓?`invalid` 鏄袱涓彲鐩存帴杩涜瘎娴嬭〃鐨勬寚鏍囷細鍓嶈€呰　閲忋€岃瘉鎹鐢ㄨ捣鏉ユ病鏈夈€嶏紝
鍚庤€呮槸**骞昏缂栧彿**淇″彿锛堝紩鐢ㄤ簡涓嶅瓨鍦ㄧ殑 E9锛夈€?
### 13.5 灏氭湭鍋氾紙闇€瑕佺湡瀹?API 鎵嶈兘楠岃瘉锛?
- SED 绗?1 姝ワ紙閫愭潯鐙珛璇勪及 agree/uncertain/disagree锛夊皻鏈帴鍏ワ紱
- 缁撴瀯鍖?JSON 杈撳嚭鐨?*瀹為檯鎴愬姛鐜?*鏈煡鈥斺€旂绾垮彧鏈夎В鏋愬櫒鐨勫崟鍏冩祴璇曪紱
- 璇佹嵁鍏呭垎鎬у畧鍗紙answer 鍓嶆鏌ュ尯鍩?鍊嶆暟锛変粛鏄彁绀鸿瘝绾︽潫锛屾湭鍙樻垚浠ｇ爜锛?- `--runs` 鐨勫 rollout 浠嶅彧鐢ㄤ簬瀛楃涓叉姇绁紝鏈仛绛栫暐钂搁銆?

