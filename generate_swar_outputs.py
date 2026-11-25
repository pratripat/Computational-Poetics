# -*- coding: utf-8 -*-
"""
generate_swar_outputs.py:
Generates prosodic scansion outputs (chunks and brackets) for all words in
'स्वर व्यंजन Word List.docx' using the updated Hindi prosody pipeline (R1-R5).

Outputs generated:
  1. scansion_output_swar.txt:
     Swar words in proper Hindi dictionary collation order (matching the collation
     order of Pronunciation_Consonants_Amisha.txt).
  2. scansion_output_swar_doc_order.txt:
     Swar words in the original order of the docx file.
  3. scansion_output_full_swar_vyanjan.txt:
     Complete unified dictionary (Swar words followed by Vyanjan words from
     Corrected_scansion_output_oneline_Consonants.txt in appropriate dictionary order).
"""

import zipfile
import xml.etree.ElementTree as ET
import re
import json
import sys
import os

from hindi_prosody_pipeline import predict, load_gold_data, normalize_word, repair_json_content

# Collation order matching Hindi Varnamala / Shabdakosh
DEV_ORDER = [
    'ं', 'ँ', 'ः',
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'ऑ',
    'क', 'क़', 'ख', 'ख़', 'ग', 'ग़', 'घ', 'ङ',
    'च', 'छ', 'ज', 'ज़', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ड़', 'ढ', 'ढ़', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'फ़', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ॅ', 'ॉ',
    '्', '़'
]
CHAR_TO_INDEX = {c: i for i, c in enumerate(DEV_ORDER)}

def hindi_collation_key(word):
    """Generates a sorting tuple for Hindi dictionary order."""
    return [CHAR_TO_INDEX.get(c, 999) for c in word]


def extract_words_from_docx(docx_path="स्वर व्यंजन Word List.docx"):
    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    paragraphs = []
    for p in tree.iterfind('.//w:p', namespaces):
        texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
        if texts:
            t = "".join(texts).strip()
            if t:
                paragraphs.append(t)

    words = []
    prev_headword = ""
    for idx, p in enumerate(paragraphs):
        # Normalize dashes
        p = p.replace('–', '-').replace('—', '-')

        # Handle template lines like "word" : {...} or "word" {...}
        m_json = re.match(r'^[“"‘\']([^"”’\':{]+)[”"’\']?\s*[:{]', p)
        if m_json:
            w_list = [m_json.group(1).strip()]
        else:
            quoted_items = re.findall(r'[“"‘\']([^"”’\']+)[”"’\']', p)
            if len(quoted_items) > 1:
                w_list = [item.strip() for item in quoted_items if item.strip()]
            else:
                w_list = [p.strip()]

        for raw_w in w_list:
            w = re.sub(r'["“”‘’]', '', raw_w).strip()
            if not w:
                continue

            # Split space-separated multiple words if present (e.g. "अनुष्ठान अनुसंधान")
            sub_words = [sw.strip() for sw in w.split(' ') if sw.strip()]
            for sw in sub_words:
                # Handle suffix sub-entries
                if sw == 'वाद' and prev_headword == 'अन्योन्याश्रय':
                    sw = 'अन्योन्याश्रयवाद'
                elif sw in ('मानव', 'व्यवस्था', 'सचिव') and prev_headword.startswith('अर्थ'):
                    sw = 'अर्थ' + sw
                elif sw in ('घातक', 'चारितात्मक', 'चालित') and prev_headword.startswith('आत्म'):
                    sw = 'आत्म' + sw

                if sw in ('अन्योन्याश्रय', 'अर्थ', 'आत्म'):
                    prev_headword = sw
                elif not any(sw.startswith(prefix) for prefix in ('अन्योन्याश्रय', 'अर्थ', 'आत्म')):
                    prev_headword = ""

                words.append(sw)

    return words


def process_word(word, lexicon=None, extra_lexicon=None):
    """Run prosodic pipeline and construct exact matching word chunks."""
    if '-' in word:
        subparts = word.split('-')
        all_chunks = []
        all_groups = []
        for p_idx, part in enumerate(subparts):
            syls, weights, groups = predict(part, lexicon=lexicon, extra_lexicon=extra_lexicon)
            part_chunks = []
            s_i = 0
            for g in groups:
                c = "".join(syls[s_i + j]['text'] for j in range(len(g)))
                part_chunks.append(c)
                s_i += len(g)
            if p_idx < len(subparts) - 1:
                part_chunks[-1] += '-'
            all_chunks.extend(part_chunks)
            all_groups.extend(groups)
        assert "".join(all_chunks) == word, f"Hyphen mismatch for {word}: '{''.join(all_chunks)}' != '{word}'"
        assert len(all_chunks) == len(all_groups), f"Count mismatch for {word}: {all_chunks} vs {all_groups}"
        return {
            'word': all_chunks,
            'scansion': all_groups
        }

    syls, weights, groups = predict(word, lexicon=lexicon, extra_lexicon=extra_lexicon)

    # Reconstruct chunks matching each bracket group
    chunks = []
    s_idx = 0
    for g in groups:
        chunk_text = "".join(syls[s_idx + j]['text'] for j in range(len(g)))
        chunks.append(chunk_text)
        s_idx += len(g)

    # Invariants verification
    assert len(chunks) == len(groups), f"Chunk count mismatch for {word}: {chunks} vs {groups}"
    reconstructed = "".join(chunks)
    assert reconstructed == word, f"Reconstruction mismatch for {word}: '{reconstructed}' != '{word}'"

    return {
        'word': chunks,
        'scansion': groups
    }


def write_oneline_json(filepath, data_dict):
    """Writes in the exact one-line JSON dictionary format used in the project."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('{\n')
        total = len(data_dict)
        for i, (k, v) in enumerate(data_dict.items()):
            comma = ',' if i < total - 1 else ''
            entry_json = json.dumps(v, ensure_ascii=False)
            f.write(f'  "{k}": {entry_json}{comma}\n')
        f.write('}\n')


def main():
    print("Loading gold lexicon and extra lexicon...")
    merged, extra = load_gold_data(
        'Pronunciation_Consonants_Amisha.txt',
        'Corrected_scansion_output_oneline_Consonants.txt',
        'extra_lexicon.json'
    )

    print("Extracting words from 'स्वर व्यंजन Word List.docx'...")
    raw_words = extract_words_from_docx('स्वर व्यंजन Word List.docx')
    print(f"Total words extracted from docx: {len(raw_words)}")

    # Deduplicate while preserving first appearance order
    unique_words_doc_order = list(dict.fromkeys(raw_words))
    print(f"Unique words: {len(unique_words_doc_order)}")

    print("\nScanning all Swar words through Hindi Prosody Pipeline...")
    doc_order_outputs = {}
    for idx, w in enumerate(unique_words_doc_order):
        doc_order_outputs[w] = process_word(w, lexicon=merged, extra_lexicon=extra)
        if (idx + 1) % 500 == 0 or idx + 1 == len(unique_words_doc_order):
            print(f"  Processed {idx + 1}/{len(unique_words_doc_order)} words...")

    # 1. Output in original document order
    doc_order_path = "scansion_output_swar_doc_order.txt"
    write_oneline_json(doc_order_path, doc_order_outputs)
    print(f"\n[Generated] {doc_order_path} ({len(doc_order_outputs)} words in original doc order)")

    # 2. Output in proper Hindi dictionary collation order
    sorted_words = sorted(unique_words_doc_order, key=hindi_collation_key)
    sorted_outputs = {w: doc_order_outputs[w] for w in sorted_words}
    sorted_path = "scansion_output_swar.txt"
    write_oneline_json(sorted_path, sorted_outputs)
    print(f"[Generated] {sorted_path} ({len(sorted_outputs)} words in Hindi dictionary order)")

    # 3. Output unified dictionary: Swar words followed by Vyanjan words
    with open('Corrected_scansion_output_oneline_Consonants.txt', 'r', encoding='utf-8') as f:
        corr_data = json.loads(repair_json_content(f.read()))

    full_unified = {}
    for w in sorted_words:
        full_unified[w] = sorted_outputs[w]
    for w, v in corr_data.items():
        if w not in full_unified:
            v_clean = {
                'word': [c.strip() for c in v.get('word', [])],
                'scansion': v.get('scansion', [])
            }
            if w == 'करकराहट':
                v_clean['scansion'] = [[1, 1], [1], [2], [1, 1]]
            full_unified[w] = v_clean

    unified_path = "scansion_output_full_swar_vyanjan.txt"
    write_oneline_json(unified_path, full_unified)
    print(f"[Generated] {unified_path} ({len(full_unified)} total words: {len(sorted_outputs)} Swar + {len(full_unified)-len(sorted_outputs)} Vyanjan)")

    # Summary preview
    print("\n" + "=" * 70)
    print("SAMPLE PREVIEW (First 20 Swar Outputs in Dictionary Order):")
    print("=" * 70)
    for i, (k, v) in enumerate(list(sorted_outputs.items())[:20]):
        print(f'  "{k}": {json.dumps(v, ensure_ascii=False)}')


if __name__ == '__main__':
    main()
