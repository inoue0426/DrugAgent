import os
import xml.etree.ElementTree as ET


def findall_no_ns(element, tag):
    """
    Find all direct child elements with a given tag, ignoring namespaces.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        Parent element to search under
    tag : str
        Tag name without namespace to find

    Returns
    -------
    list[Element]
        List of matched child elements
    """
    return [child for child in element if child.tag.endswith(tag)]


def find_no_ns(element, tag):
    """
    Find the first direct child element with a given tag, ignoring namespaces.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element
        Parent element to search under
    tag : str
        Tag name without namespace to find

    Returns
    -------
    Element or None
        The first matched child element or None if not found
    """
    for child in element:
        if child.tag.endswith(tag):
            return child
    return None


def get_all_text(elem):
    """
    Recursively extract all text from an element and its descendants.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        Element to extract text from

    Returns
    -------
    list[str]
        List of text strings extracted from the element and its children
    """
    texts = []
    if elem.text and elem.text.strip():
        texts.append(elem.text.strip())
    for child in elem:
        texts.extend(get_all_text(child))
        if child.tail and child.tail.strip():
            texts.append(child.tail.strip())
    return texts


# --- Dictionary of target section aliases ---
TARGET_SECTION_ALIASES = {
    "Abstract": ["abstract", "summary"],  # Abstract is handled separately
    "Introduction": ["introduction", "background", "context", "aims", "purpose"],
    "Methods": [
        "methods",
        "methodology",
        "materials and methods",
        "experimental procedures",
        "experimental methods",
        "materials & methods",
        "methods and materials",
        "methods and experimental procedures",
        "experimental section",
        "animal models",
        "cell culture",
        "immunoblotting",
        "rna extraction",
        "eu incorporation assay",
        "comet assay",
        "immunofluorescence",
        "pharmacological treatments",
        "amyloid beta preparation",
        "dna fiber analysis",
        "cell viability assays",
        "neurite degeneration index",
        "publicly available data set analysis",
        "statistical analysis",
        "animals",
        "primary neuronal culture",
        "immunoblotting of murine cortical lysates",
        "human embryonic stem cell culture",
        "rna extraction and library preparation",
        "immunoblotting for primary neuronal cortical cultures",
        "list of antibodies used for western blotting",
        "list of antibodies used for if",
        "immunofluorescence (if)",
        "list of antibodies used for if",
        "compounds",
        "reagents and instruments",
    ],
    "Results": [
        "results",
        "findings",
        "experimental results",
        "data",
        "observations",
        "outcome",
    ],
    "Discussion": [
        "discussion",
        "conclusions",
        "conclusion",
        "summary and conclusion",
        "outlook",
        "limitations",
    ],
    "References": ["references", "bibliography", "literature cited"],
    "Materials": [
        "materials",
        "material",
        "reagents",
        "chemicals",
        "apparatus",
        "equipment",
        "cell lines",
        "antibodies",
        "constructs",
        "plasmids",
        "list of antibodies",
        "pharmacological treatments",
    ],
    "Experiment": [
        "experiments",
        "experimental",
        "experimental section",
        "experimental setup",
        "assay",
        "assays",
        "testing",
        "procedure",
        "procedures",
        "trials",
        "study",
        "investigation",
        "analysis",
        "demonstration",
    ],
}


def extract_sections(xml_path, target_section_aliases=TARGET_SECTION_ALIASES):
    """
    Extract specified sections' text from a PMC XML file using given aliases.

    Parameters
    ----------
    xml_path : str
        Path to the PMC XML file.
    target_section_aliases : dict
        Dictionary mapping target section names to lists of common aliases.

    Returns
    -------
    dict
        Dictionary mapping target section names to their extracted text.
        Sections not found are not included.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return {}
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_path}")
        return {}

    section_texts = {}

    # 1. Extract Abstract (usually under <front>/<article-meta>/<abstract>)
    abstract_elem = root.find(".//{*}abstract")
    if abstract_elem is not None:
        abstract_text = "\n".join(get_all_text(abstract_elem)).strip()
        section_texts["Abstract"] = abstract_text
    # else:
    #     print("Abstract not found.")

    # 2. Extract sections from <body>/<sec>
    body = find_no_ns(root, "body")
    if body is None:
        body = root.find(".//{*}body")
        if body is None:
            print(
                f"Error: 'body' tag not found in {xml_path}. Skipping body processing."
            )

    if body:
        secs = findall_no_ns(body, "sec")

        for sec in secs:
            title_elem = find_no_ns(sec, "title")

            if title_elem is not None and title_elem.text is not None:
                current_section_title = title_elem.text.strip().lower()
            else:
                current_section_title = ""

            for target_name, aliases in target_section_aliases.items():
                # Skip Abstract and References here since they are handled separately
                if target_name in ["Abstract", "References"]:
                    continue

                # Skip if section already extracted
                if target_name in section_texts:
                    continue

                if any(alias in current_section_title for alias in aliases):
                    texts = get_all_text(sec)
                    full_text = "\n".join(texts).strip()
                    section_texts[target_name] = full_text
                    # print(f"DEBUG: Found '{target_name}' with title '{current_section_title}'")
                    break

    # 3. Extract References (usually under <back>/<ref-list>)
    ref_list_elem = root.find(".//{*}ref-list")
    if ref_list_elem is not None:
        references_text = "\n".join(get_all_text(ref_list_elem)).strip()
        section_texts["References"] = references_text
    # else:
    #     print("References not found.")

    return section_texts
