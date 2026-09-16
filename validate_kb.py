from pathlib import Path
import re


# Folder containing our knowledge-base documents
KB_FOLDER = Path("knowledge_base")


# The 12 topics required by the project brief
REQUIRED_DOCUMENTS = [
    "ticket_priority.txt",
    "sla_severity.txt",
    "escalation_matrix.txt",
    "refund_compensation.txt",
    "communication_channels.txt",
    "business_hours.txt",
    "repeat_complaints.txt",
    "service_credit.txt",
    "feedback_collection.txt",
    "vip_handling.txt",
    "outage_communication.txt",
    "ticket_data_retention.txt",
]


def count_sentences(text):
    """
    Count sentences using ., !, or ? as sentence endings.
    """
    sentences = re.findall(r"[.!?]+(?=\s|$)", text)
    return len(sentences)


def validate_kb():

    print("=" * 60)
    print("KNOWLEDGE BASE VALIDATION")
    print("=" * 60)

    # Check that the folder exists
    if not KB_FOLDER.exists():
        print("\n❌ knowledge_base folder does not exist.")
        return

    print(f"\nKB folder found: {KB_FOLDER.resolve()}")

    # Find all text files
    txt_files = list(KB_FOLDER.glob("*.txt"))

    print(f"Total .txt documents found: {len(txt_files)}")

    # Check document count
    if len(txt_files) >= 12:
        print("✅ At least 12 documents found.")
    else:
        print("❌ Fewer than 12 documents found.")

    print("\nDocument Validation")
    print("-" * 60)

    all_valid = True

    # Check each required document
    for filename in REQUIRED_DOCUMENTS:

        file_path = KB_FOLDER / filename

        if not file_path.exists():
            print(f"❌ {filename} — MISSING")
            all_valid = False
            continue

        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"❌ {filename} — EMPTY")
            all_valid = False
            continue

        sentence_count = count_sentences(text)

        if 2 <= sentence_count <= 5:
            print(
                f"✅ {filename:<35} "
                f"{sentence_count} sentences"
            )
        else:
            print(
                f"❌ {filename:<35} "
                f"{sentence_count} sentences "
                f"(must be 2–5)"
            )
            all_valid = False

    print("\n" + "=" * 60)

    if all_valid and len(txt_files) >= 12:
        print("✅ KNOWLEDGE BASE VALIDATION PASSED")
    else:
        print("❌ KNOWLEDGE BASE VALIDATION FAILED")

    print("=" * 60)


if __name__ == "__main__":
    validate_kb()