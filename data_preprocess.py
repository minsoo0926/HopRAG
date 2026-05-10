import json
from tqdm import tqdm
import jsonlines
import os
import re


def make_safe_filename(name: str) -> str:
    """
    Convert a document title/id into a cross-platform safe filename.
    This avoids Windows-invalid characters such as ?, :, *, etc.
    """
    name = str(name)

    # Replace invalid filename characters on Windows/macOS/Linux-safe usage.
    name = re.sub(r'[<>:"/\\|?*]', '_', name)

    # Remove leading/trailing whitespace and dots.
    name = name.strip().strip(".")

    # Avoid empty filename.
    if not name:
        name = "untitled"

    return name


def make_unique_filename(base_name: str, used_names: set) -> str:
    """
    Avoid collisions after sanitization.
    Example:
      "A/B" -> "A_B"
      "A?B" -> "A_B"
    These would collide, so we append suffixes.
    """
    candidate = base_name
    count = 1

    while candidate in used_names:
        candidate = f"{base_name}_{count}"
        count += 1

    used_names.add(candidate)
    return candidate


def process_data(source_path, docs_dir, output_path):
    doc2id = {}

    # Open and load the source data
    data = []
    if source_path.endswith(".jsonl"):
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))
    else:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Process the entries and create text files for documents
    for temp in tqdm(data):
        context = temp["context"]

        for title, sentences in context:
            doc = "\n\n".join(sentences)

            # Keep the original title/id as metadata.
            if doc not in doc2id:
                doc2id[doc] = title

    # Ensure the docs_dir exists
    os.makedirs(docs_dir, exist_ok=True)

    used_filenames = set()
    filename_mapping = {}

    # Write each document to a text file
    for doc, original_id in doc2id.items():
        safe_base = make_safe_filename(original_id)
        safe_id = make_unique_filename(safe_base, used_filenames)

        filename_mapping[safe_id] = original_id

        file_path = os.path.join(docs_dir, f"{safe_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc)

    # Save mapping from safe filename to original title/id
    mapping_path = os.path.join(docs_dir, "filename_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(filename_mapping, f, ensure_ascii=False, indent=2)

    print(f"done: all text files saved to directory {docs_dir}")
    print(f"done: filename mapping saved to {mapping_path}")

    # Write the data to a jsonlines file
    if source_path.endswith(".json"):
        with jsonlines.open(output_path, mode="w") as writer:
            for result in data:
                writer.write(result)


def process_data_musique(source_path, docs_dir):
    os.makedirs(docs_dir, exist_ok=True)

    doc2id = {}
    count = 0

    with open(source_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    id2txt = {}

    for temp in data:
        qid = temp["id"]
        paragraphs = temp["paragraphs"]

        for dic in paragraphs:
            idx = dic["idx"]
            text = dic["paragraph_text"]
            txtname = qid + "_" + str(idx)

            if text not in doc2id:
                doc2id[text] = txtname
            else:
                count += 1
                txtname = doc2id[text]

            if qid in id2txt:
                id2txt[qid].append(txtname)
            else:
                id2txt[qid] = [txtname]

    unique_txtname = []
    for txtnames in id2txt.values():
        unique_txtname += txtnames

    assert len(set(unique_txtname)) == len(doc2id)

    used_filenames = set()
    filename_mapping = {}
    old_to_safe = {}

    for text, original_id in doc2id.items():
        safe_base = make_safe_filename(original_id)
        safe_id = make_unique_filename(safe_base, used_filenames)

        old_to_safe[original_id] = safe_id
        filename_mapping[safe_id] = original_id

        file_path = os.path.join(docs_dir, f"{safe_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    # Update id2txt to use safe filenames
    safe_id2txt = {}
    for qid, txtnames in id2txt.items():
        safe_id2txt[qid] = [old_to_safe[name] for name in txtnames]

    with open(source_path.replace(".jsonl", "_id2txt.json"), "w", encoding="utf-8") as f:
        json.dump(safe_id2txt, f, ensure_ascii=False, indent=2)

    mapping_path = os.path.join(docs_dir, "filename_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(filename_mapping, f, ensure_ascii=False, indent=2)


def main_hotpot_2wiki(
    source_path="quickstart_dataset/hotpot_example.jsonl",
    docs_dir="quickstart_dataset/hotpot_example_docs",
):
    output_path = source_path.replace(".json", ".jsonl") if source_path.endswith(".json") else source_path
    process_data(source_path, docs_dir, output_path)


def main_musique(
    source_path="quickstart_dataset/musique_example.jsonl",
    docs_dir="quickstart_dataset/musique_example_docs",
):
    process_data_musique(source_path, docs_dir)


if __name__ == "__main__":
    main_hotpot_2wiki()