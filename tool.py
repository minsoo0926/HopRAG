import sys
import os
from openai import OpenAI
import json
import pickle
from config import *
import torch
import re
import time

from sentence_transformers import SentenceTransformer
from modelscope import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple, Dict, Set

_SPACY_NLP = None
_SPACY_LOAD_ATTEMPTED = False

def sparse_similarity(a:Set, b:Set):
    return len(a.intersection(b))/len(a.union(b))

def try_run(func, *args, **kwargs):
    retry = 0
    while retry < max_try_num:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retry += 1
            with open(exception_log_path, "a") as file:
                file.write(f"Exception: {e}\n")
                file.write(f"Commandline: {sys.argv}\n")
            time.sleep(0)
    else:
        with open(exception_log_path, "a") as file:
            file.write(f"--FAIL--\n")
            file.write(f"Commandline: {sys.argv}\n")
        #exit(555)
        return None,None

def replace_newlines(match):
    # Replace \n and \r in the matched string
    return match.group(0).replace('\n', '\\n').replace('\r', '\\r')

# def clean_json_str(json_str: str) -> str:
#     """
#     The generated JSON format may be non-standard, perform replacement processing first.
#     :param json_str:
#     :return:
#     """
#     # Remove code block markers ```
#     # Replace None with null in the JSON string

#     if "```json" not in json_str:
#         index=json_str.index('{')
#         json_str = json_str[index:]
#         json_str = "```json"+json_str
#     json_str = json_str.replace("None","null")
#     if not json_str.startswith('```') and '```' in json_str:
#         json_str = '```'+json_str.split('```')[1]
#     json_str = json_str.split('}')[0]+'}'
#     if json_str.startswith("```") and not json_str.endswith("```"):
#         json_str += "```"
#     match = re.search(r'```json(.*?)```', json_str, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#     match = re.search(r'```(.*?)```', json_str, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#     # Replace \n and \r in the matched string
#     json_str = re.sub( r'("(?:\\.|[^"\\])*")', replace_newlines, json_str)
#     # Remove trailing commas after key-value pairs
#     json_str = re.sub(r',\s*}', '}', json_str)
#     json_str = re.sub(r',\s*]', ']', json_str)
#     # Restore the missing commas
#     json_str = re.sub(r'\"\s+\"', '\",\"', json_str)
#     # Inplacement of True and False
#     json_str = json_str.replace("True","true")
#     json_str = json_str.replace("False","false")
#     return json_str

def clean_json_str(json_str: str) -> str:
    """
    Robustly extract a JSON object from LLM output.
    Works even if the model does not use ```json fences.
    """
    if json_str is None:
        return ""

    json_str = str(json_str).strip()

    # Remove Qwen-style thinking block if included in content.
    json_str = re.sub(r"<think>.*?</think>", "", json_str, flags=re.DOTALL).strip()

    # Remove markdown fences if present.
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    # Normalize quotes sometimes produced by LLMs.
    json_str = json_str.replace("“", '"').replace("”", '"').replace("`", "")

    # Convert Python-style literals to JSON-style literals.
    json_str = json_str.replace("None", "null")
    json_str = json_str.replace("True", "true")
    json_str = json_str.replace("False", "false")

    # Extract first JSON object.
    start = json_str.find("{")
    end = json_str.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return ""

    json_str = json_str[start:end + 1]

    # Remove trailing commas.
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # Replace raw newlines inside strings.
    json_str = re.sub(r'("(?:\\.|[^"\\])*")', replace_newlines, json_str)

    return json_str

# def txt2obj(text):  
#     try: 
#         text = clean_json_str(text) 
#         text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
#         text = text.replace('`','').replace('“','"').replace('”','"')
#         return json.loads(text) 
#     except Exception as e:
#         if LOG:
#             print(e)
#         return None

def txt2obj(text):
    try:
        text = clean_json_str(text)

        if not text:
            return None

        text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        return json.loads(text)

    except Exception as e:
        if LOG:
            print("txt2obj error:", e)
            print("raw response:", text)
        return None




# def get_title_keywords_eng(title_template, doc,query_generator)->Tuple[str,Set[str]]:
#     chat = []
#     chat.append({"role": "user", "content": title_template.format(doc_content=doc)})
#     title, chat = get_chat_completion(chat, keys=["Title"],model=query_generator,max_tokens=4096)
#     if len(title)==0:
#         title=doc[:20]
#     keywords=get_ner_eng(title)
#     if len(keywords)==0:
#         keywords=title.replace(',',"").replace('，',"").replace('。','').replace('.','')
#         keywords=set(keywords)
#         return title,keywords
#     return title,set(keywords)

# def get_title_keywords_eng(title_template, doc, query_generator) -> Tuple[str, Set[str]]:
#     chat = []
#     chat.append({"role": "user", "content": title_template.format(doc_content=doc)})

#     title, chat = get_chat_completion(
#         chat,
#         keys=["Title"],
#         model=query_generator,
#         # max_tokens=4096
#         max_tokens = 512
#     )

#     # Fallback if Qwen output parsing failed
#     if title is None:
#         title = doc[:80].replace("\n", " ").strip()

#     if isinstance(title, list):
#         title = title[0] if len(title) > 0 else doc[:80].replace("\n", " ").strip()

#     title = str(title).strip()

#     if len(title) == 0:
#         title = doc[:80].replace("\n", " ").strip()

#     keywords = get_ner_eng(title)

#     if len(keywords) == 0:
#         keywords = title.replace(",", " ").replace("，", " ").replace("。", " ").replace(".", " ").split()

#     return title, set(keywords)

def get_title_keywords_eng(title_template, doc, query_generator) -> Tuple[str, Set[str]]:
    # Debug / local-small-model mode:
    # Do not call LLM for title generation.
    title = doc[:80].replace("\n", " ").strip()

    if len(title) == 0:
        title = "untitled document"

    keywords = get_ner_eng(doc)

    if len(keywords) == 0:
        keywords = title.replace(",", " ").replace("，", " ").replace("。", " ").replace(".", " ").split()

    return title, set(keywords)

# def get_question_list(extract_template, sentences,query_generator)->List[str]:
#     chat = []
#     chat.append({"role": "user", "content": extract_template.format(sentences=sentences)})
#     question_list, chat = get_chat_completion(chat, keys=["Question List"],model=query_generator,max_tokens=4096)
#     return question_list

# def get_question_list(extract_template, sentences, query_generator) -> List[str]:
#     # Debug / local-small-model mode:
#     # Do not call LLM, but return one dummy question so that HopBuilder creates nodes.
#     if isinstance(sentences, list):
#         text = " ".join(sentences)
#     else:
#         text = str(sentences)

#     text = text.replace("\n", " ").strip()

#     if len(text) == 0:
#         return ["What information is provided in this passage?"]

#     short_text = text[:120]
#     return [f"What information is provided in this passage about {short_text}?"]

def get_question_list(extract_template, sentences, query_generator) -> List[str]:
    if isinstance(sentences, list):
        text = " ".join(sentences)
    else:
        text = str(sentences)

    text = text.replace("\n", " ").strip()

    if len(text) == 0:
        return ["What information is provided in this passage?"]

    # 너무 긴 prompt를 막기 위해 잘라줌
    text = text[:800]

    prompt = f"""
/no_think
Generate exactly two simple factual questions about the passage.
Return only a valid JSON object.
Do not include thinking, reasoning, explanation, markdown, or code fences.
The format must be exactly:
{{"Question List":["question 1","question 2"]}}

Passage:
{text}
"""

    chat = [{"role": "user", "content": prompt}]

    question_list, chat = get_chat_completion(
        chat,
        keys=["Question List"],
        model=query_generator,
        max_tokens=256
    )

    if question_list is None:
        return [f"What information is provided in this passage about {text[:120]}?"]

    if isinstance(question_list, str):
        question_list = [question_list]

    if not isinstance(question_list, list) or len(question_list) == 0:
        return [f"What information is provided in this passage about {text[:120]}?"]

    return [str(q).strip() for q in question_list if str(q).strip()][:2]

# def get_ner_eng(text):
#     from paddlenlp import Taskflow
#     ner_task = Taskflow("pos_tagging")
#     results = ner_task(text)
#     filtered = []
#     for result in results:
#         entity, mode = result
#         if mode not in [
#             "w",  # Punctuation marks
#             "c",  # Conjunctions
#             "f",  # Directional words
#             "ad", # Adverbs
#             "q",  # Quantifiers
#             "u",  # Particles
#             "s",  # Locative words
#             "vd", # Verbal adverbs
#             "an", # Noun-adjective compound
#             "r",  # Pronouns
#             "xc", # Other function words
#             "vn", # Noun-verb compounds
#             "d",  # Adverbs
#             "p",  # Prepositions
#         ]:
#             filtered.append(entity)
#     filtered = list(set(filtered))
#     return filtered

def _dedupe_keywords(keywords, max_keywords=20):
    deduped = []
    seen = set()
    for keyword in keywords:
        keyword = re.sub(r"\s+", " ", str(keyword)).strip(" \t\r\n.,;:()[]{}\"'")
        if not keyword:
            continue
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(keyword)
        if len(deduped) >= max_keywords:
            break
    return deduped

def _load_spacy_nlp():
    global _SPACY_NLP, _SPACY_LOAD_ATTEMPTED
    if _SPACY_LOAD_ATTEMPTED:
        return _SPACY_NLP
    _SPACY_LOAD_ATTEMPTED = True
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["textcat"])
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP

def _regex_keywords(text, max_keywords=20):

    text = str(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text)

    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "were", "his", "her", "its", "into", "about", "after", "before",
        "which", "what", "when", "where", "who", "how", "why", "has", "had",
        "have", "not", "but", "you", "your", "their", "they", "them", "than",
        "then", "there", "here", "also", "other", "such", "only", "more"
    }

    keywords = []
    for w in words:
        lw = w.lower()
        if len(lw) >= 3 and lw not in stopwords:
            keywords.append(w)

    return _dedupe_keywords(keywords, max_keywords=max_keywords)

def get_ner_eng(text):
    text = str(text)
    if len(text.strip()) == 0:
        return []

    nlp = _load_spacy_nlp()
    if nlp is None:
        return _regex_keywords(text)

    doc = nlp(text[:3000])
    entity_labels = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE"
    }

    candidates = []
    candidates.extend(ent.text for ent in doc.ents if ent.label_ in entity_labels)
    candidates.extend(
        chunk.text
        for chunk in doc.noun_chunks
        if 2 <= len(chunk.text.strip()) <= 80
    )

    if not candidates:
        return _regex_keywords(text)

    return _dedupe_keywords(candidates, max_keywords=20)

def load_embed_model(model_name):
    if model_name in embed_model_dict:
        return SentenceTransformer(embed_model_dict[model_name],device=llm_device)  #
    else:
        raise NotImplementedError
    
def load_language_model(model_name):
    # 如果llm已经本地部署了，直接用openai接口调用
    for sign in deployment_sign:
        if sign in model_name:
            return model_name
    # 否则用transformer框架加载模型
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=llm_device) #
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

def load_rerank_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(llm_device)
    return model,tokenizer

def get_doc_embeds(documents, model):
    with torch.no_grad():
        embeddings = model.encode(documents, normalize_embeddings=True, device=llm_device).tolist() # 
    return embeddings

def _get_chat_completion(chat, return_json=True, model=default_gpt_model, max_tokens=4096, keys=None):
    print("LLM CALLED:", model)
    if not isinstance(chat, list):
        chat = [{"role": "user", "content": chat}]
    local_deployed = False
    for sign in deployment_sign:
        if sign in model:
            local_deployed = True
            current_personal_key = deployment_sign[sign]['key']
            current_personal_base = deployment_sign[sign]['base']
            break
    if type(model)== str and local_deployed:
        client = OpenAI(api_key=current_personal_key, base_url=current_personal_base)
        completion_kwargs = {
            "model": model,
            "messages": chat,
            "response_format": {"type": "json_object" if return_json else "text"},
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        if "qwen3" in model.lower():
            completion_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        chat_completion = client.chat.completions.create(**completion_kwargs)
        # print(chat_completion.choices[0].message.content)
        chat = chat + [{"role": "assistant", "content": chat_completion.choices[0].message.content}]
        response = chat_completion.choices[0].message.content
    elif type(model)==tuple:
        model,tokenizer=model
        text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
    else:
        raise NotImplementedError
    if not return_json:
        return response, chat
    # obj = txt2obj(response)
    # obj = tuple([obj[key] for key in keys if key in obj]) #
    # return *obj, chat
    obj = txt2obj(response)

    if keys is None:
        return obj, chat

    if obj is None or not isinstance(obj, dict):
        return *([None] * len(keys)), chat

    values = []
    for key in keys:
        # 혹시 key가 '"Title"'처럼 들어오는 경우도 방어
        normalized_key = str(key).strip().strip('"').strip("'")

        if normalized_key in obj:
            values.append(obj[normalized_key])
        elif key in obj:
            values.append(obj[key])
        else:
            values.append(None)

    return *values, chat

def get_chat_completion(chat, return_json=True, model=default_gpt_model, max_tokens=4096, keys=None):
    return try_run(_get_chat_completion, chat, return_json, model, max_tokens, keys)

def _torch_accel_device():
    if llm_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if llm_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return None

def pending_dot_answerable(pending_df,answerable_df):
    pending=np.array(pending_df['embedding'].tolist())
    answerable=np.array(answerable_df['embedding'].tolist())
    accel_device = _torch_accel_device()
    if accel_device is not None:
        pending=torch.as_tensor(pending, dtype=torch.float32, device=accel_device)
        answerable=torch.as_tensor(answerable, dtype=torch.float32, device=accel_device)
        dense_similarity=pending.mm(answerable.T).cpu().numpy()
    else:
        dense_similarity=pending.dot(answerable.T)
    outcome=dense_similarity.flatten().tolist()
    del pending,answerable,dense_similarity
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return outcome

def sparse_similarities_df(df)->Dict[Tuple[str,str],float]:
    if os.path.exists('/path/to/cache/sparse_similarities_result.pkl'):
        with open('/path/to/cache/sparse_similarities_result.pkl','rb') as file:
            return pickle.load(file)
    docs_keywords=df['keywords'].astype(str).unique()
    sparse_similarities={}
    for i in range(len(docs_keywords)):
        for j in range(i,len(docs_keywords)):  
            sparse_similarities[(docs_keywords[i],docs_keywords[j])]=sparse_similarity(set(eval(docs_keywords[i])),set(eval(docs_keywords[j])))
            sparse_similarities[(docs_keywords[j],docs_keywords[i])]=sparse_similarities[(docs_keywords[i],docs_keywords[j])]
    return sparse_similarities


if __name__ == "__main__":
    print(get_chat_completion([{"role": "user", "content": "What is the capital of China? reply in json format {\"Answer\":\"\"}"}], keys=["Answer"], model=default_gpt_model, max_tokens=4096))
